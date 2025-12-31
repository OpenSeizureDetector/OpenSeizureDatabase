#!/usr/bin/env python3

import argparse
from re import X
import sys
import os
import json
import importlib
from urllib.parse import _NetlocResultMixinStr
#from tkinter import Y
import sklearn.model_selection
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime as dt

import gc

import pandas as pd
import imblearn

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libosd.osdDbConnection
import libosd.dpTools
import libosd.osdAlgTools
import libosd.configUtils


def calcProcessTime(starttime, cur_iter, max_iter):
    # From https://stackoverflow.com/questions/44926127/calculating-the-amount-of-time-left-until-completion
    telapsed = time.time() - starttime
    testimated = (telapsed/(cur_iter+1))*(max_iter)
    finishtime = starttime + testimated
    finishtime = dt.datetime.fromtimestamp(finishtime).strftime("%H:%M:%S")  # in time
    lefttime = testimated-telapsed  # in seconds
    iterTime = (telapsed/(cur_iter+1))
    return (int(telapsed), int(lefttime), finishtime, iterTime)

def type2id(typeStr):
    if typeStr.lower() == "seizure":
        id = 1
    elif typeStr.lower() == "false alarm":
        id = 0
    elif typeStr.lower() == "nda":
        id = 0
    else:
        id = 2
    return id


def getUserCounts(df):
    counts = df['userId'].value_counts()
    total = counts.sum()
    props = counts/total
    return(props)

def analyseDf(df):
    props=getUserCounts(df)
    print("analyseDf(): Distribution by user for all data")
    print(props)

    props=getUserCounts(df[df['type']==1])
    print("analyseDf(): Distribution by user for seizure data")
    print(props)



def loadCsv(inFname, debug=False):
    '''
    loadCsv - read osdb csv file into a pandas dataframe.
    if inFname is None, reads from stdin.
    '''
    TAG = "augmentData.loadCsv()"
    if inFname is not None:
        print("%s: reading from file %s" % (TAG, inFname))
        inFile = inFname
    else:
        inFile = sys.stdin

    df = pd.read_csv(inFile, low_memory=False)

    #print(df)
    if (debug): print("%s: returning %d datapoints" % (TAG, len(df)))

    return(df)


def getSeizureNonSeizureDfs(df):
    ''' returns two datasets, seizure, non-seizure from dataframe df
    '''
    seizuresDf = df[df['type']==1]
    nonSeizureDf = df[df['type']!=1]
    return (seizuresDf, nonSeizureDf)


def _build_event_index(df, id_col='eventId'):
    """Return ordered list of event ids and a mapping id->group DataFrame.
    Preserves the original order of events as they appear in df.
    """
    event_ids = []
    event_groups = {}
    # groupby with sort=False preserves first-seen order
    for eid, grp in df.groupby(id_col, sort=False):
        event_ids.append(eid)
        event_groups[eid] = grp
    return event_ids, event_groups


def _make_new_id(orig_id, counter, numeric_max=None):
    """Generate a new unique id when duplicating events.
    If orig_id is numeric and numeric_max provided, return numeric id > numeric_max.
    Otherwise append a suffix to the original id.
    """
    try:
        # treat ints and numpy ints as numeric
        if numeric_max is not None:
            return numeric_max + counter
        # fallback for non-numeric: append suffix
        return f"{orig_id}__dup{counter}"
    except Exception:
        return f"{orig_id}__dup{counter}"



def userAug(df):
    ''' implement user augmentation to oversample at the event level and balance users
    It expects df to be a pandas dataframe representation of a flattened osdb dataset.
    Each duplicated event keeps the same number of rows as the original, with a new eventId suffix.
    '''
    seizuresDf, nonSeizureDf = getSeizureNonSeizureDfs(df)
    if 'eventId' in seizuresDf.columns:
        seizuresDf = seizuresDf.copy()
        seizuresDf['eventId'] = seizuresDf['eventId'].astype(str)
    if 'eventId' not in seizuresDf.columns:
        print("userAug(): eventId column missing; returning original dataframe")
        return df

    # Build event index for seizure events
    event_ids, event_groups = _build_event_index(seizuresDf, id_col='eventId')
    # Map userId per event (use first row of the event)
    event_users = {eid: event_groups[eid].iloc[0]['userId'] for eid in event_ids}

    # Count events per user and determine target count (max user count)
    user_event_counts = {}
    for eid, uid in event_users.items():
        user_event_counts[uid] = user_event_counts.get(uid, 0) + 1
    if len(user_event_counts) == 0:
        return df
    target_count = max(user_event_counts.values())

    out_groups = []
    # Keep originals
    for eid in event_ids:
        out_groups.append(event_groups[eid].copy())

    # Duplicate events for users with fewer events
    dup_counters = {eid: 0 for eid in event_ids}
    # Group events by user for sampling
    events_by_user = {}
    for eid, uid in event_users.items():
        events_by_user.setdefault(uid, []).append(eid)

    rng = np.random.default_rng(0)
    for uid, count in user_event_counts.items():
        needed = target_count - count
        if needed <= 0:
            continue
        user_events = events_by_user[uid]
        for _ in range(needed):
            # Sample an event from this user (with replacement)
            src_event = rng.choice(user_events)
            dup_counters[src_event] += 1
            synthetic_id = f"{src_event}-{dup_counters[src_event]}"
            grp_copy = event_groups[src_event].copy()
            grp_copy['eventId'] = synthetic_id
            out_groups.append(grp_copy)

    # Combine seizure and non-seizure data back into single dataframe to return
    xResamp = pd.concat(out_groups, ignore_index=True)

    print("userAug(): Event-level distribution after user augmentation")
    analyseDf(xResamp)
    df = pd.concat([nonSeizureDf, xResamp], ignore_index=True)
    return(df)


def noiseAug(df, noiseAugVal, noiseAugFac, debug=False):
    ''' Implement noise augmentation of seizure events in dataframe df
     It expects df to be a pandas dataframe representation of a flattened osdb dataset.
     noiseAugVal is the amplitude (in mg) of the noise applied.
     noiseAugFac is the number of *new events* generated for each input event.
     Each augmented event keeps the same number of rows as the source event.
     If 3D acceleration data exists and is non-zero, applies noise to X, Y, Z components
     and recalculates magnitude. Otherwise, applies noise directly to magnitude.
    '''
    tStart = time.time()
    seizuresDf, nonSeizureDf = getSeizureNonSeizureDfs(df)
    if 'eventId' in seizuresDf.columns:
        seizuresDf = seizuresDf.copy()
        seizuresDf['eventId'] = seizuresDf['eventId'].astype(str)
    else:
        print("noiseAug(): ERROR: eventId column missing - can not group by eventId")
        exit(-1)
    if(debug): print(seizuresDf.columns)
    accStartCol = seizuresDf.columns.get_loc('M001')-1
    accEndCol = seizuresDf.columns.get_loc('M124')+1
    eventIdCol = seizuresDf.columns.get_loc('eventId')

    # Check if 3D acceleration columns exist in the dataframe
    has3DColumns = 'X000' in seizuresDf.columns and 'Y000' in seizuresDf.columns and 'Z000' in seizuresDf.columns
    accXStartCol, accXEndCol, accYStartCol, accYEndCol, accZStartCol, accZEndCol = None, None, None, None, None, None
    if has3DColumns:
        accXStartCol = seizuresDf.columns.get_loc('X000')
        accXEndCol = seizuresDf.columns.get_loc('X124') + 1
        accYStartCol = seizuresDf.columns.get_loc('Y000')
        accYEndCol = seizuresDf.columns.get_loc('Y124') + 1
        accZStartCol = seizuresDf.columns.get_loc('Z000')
        accZEndCol = seizuresDf.columns.get_loc('Z124') + 1
        print("noiseAug(): 3D acceleration columns detected - will check each event for 3D data")

    event_ids, event_groups = _build_event_index(seizuresDf, id_col='eventId')
    if len(event_ids) == 0:
        return df
    print("noiseAug(): Augmenting %d seizure events with factor %d" % (len(event_ids), noiseAugFac))

    out_groups = []
    processed_events = 0
    for eid in event_ids:
        grp = event_groups[eid]
        out_groups.append(grp.copy())  # keep original event

        # Check if this event has valid 3D data (per event)
        use3D_event = False
        if has3DColumns:
            # Evaluate 3D data across the whole event
            accX_vals = pd.to_numeric(grp.iloc[:, accXStartCol:accXEndCol].stack(), errors='coerce').fillna(0)
            accY_vals = pd.to_numeric(grp.iloc[:, accYStartCol:accYEndCol].stack(), errors='coerce').fillna(0)
            accZ_vals = pd.to_numeric(grp.iloc[:, accZStartCol:accZEndCol].stack(), errors='coerce').fillna(0)
            use3D_event = (accX_vals.sum() != 0 or accY_vals.sum() != 0 or accZ_vals.sum() != 0)

        for dup in range(1, noiseAugFac + 1):
            aug_rows = []
            for _, row in grp.iterrows():
                outRow = []
                # Copy metadata columns
                for i in range(0, accStartCol):
                    if i == eventIdCol:
                        outRow.append(f"{eid}-{dup}")
                    else:
                        outRow.append(row.iloc[i])

                if use3D_event:
                    # Apply noise to 3D acceleration and recalculate magnitude
                    xArr = pd.to_numeric(row.iloc[accXStartCol:accXEndCol], errors='coerce').fillna(0).to_numpy(dtype=np.float64)
                    yArr = pd.to_numeric(row.iloc[accYStartCol:accYEndCol], errors='coerce').fillna(0).to_numpy(dtype=np.float64)
                    zArr = pd.to_numeric(row.iloc[accZStartCol:accZEndCol], errors='coerce').fillna(0).to_numpy(dtype=np.float64)

                    noiseX = np.random.normal(0, noiseAugVal, xArr.shape)
                    noiseY = np.random.normal(0, noiseAugVal, yArr.shape)
                    noiseZ = np.random.normal(0, noiseAugVal, zArr.shape)

                    xAugmented = xArr + noiseX
                    yAugmented = yArr + noiseY
                    zAugmented = zArr + noiseZ

                    magAugmented = np.sqrt(xAugmented**2.0 + yAugmented**2.0 + zAugmented**2.0)

                    outRow.extend(magAugmented.tolist())
                    outRow.extend(xAugmented.tolist())
                    outRow.extend(yAugmented.tolist())
                    outRow.extend(zAugmented.tolist())

                    # Append any remaining columns after acceleration data (e.g., note)
                    for i in range(accZEndCol, len(row)):
                        outRow.append(row.iloc[i])
                else:
                    accArr = row.iloc[accStartCol:accEndCol]
                    inArr = np.array(accArr)
                    noiseArr = np.random.normal(0, noiseAugVal, inArr.shape)
                    outArr = inArr + noiseArr

                    outRow.extend(outArr.tolist())

                    for i in range(accEndCol, len(row)):
                        outRow.append(row.iloc[i])

                aug_rows.append(outRow)

            aug_group = pd.DataFrame(aug_rows, columns=seizuresDf.columns)
            out_groups.append(aug_group)

        processed_events += 1
        if processed_events % 50 == 0:
            tElapsed, tRem, tCompletion, tIter = calcProcessTime(tStart, processed_events, len(event_ids))
            sys.stdout.write("events=%d, tIter=%.1f ms, elapsed: %s(s), time left: %s(s), estimated finish time: %s\r" % (processed_events, tIter*1000., tElapsed, tRem, tCompletion))
            sys.stdout.flush()
            gc.collect()

    print("noiseAug() - Creating dataframe")
    sys.stdout.flush()
    gc.collect()
    if len(out_groups) > 0:
        augDf = pd.concat(out_groups, ignore_index=True)
    else:
        augDf = seizuresDf
    if (debug): print("noiseAug() - augDf=", augDf)
    if (debug): print("noiseAug() nonSeizureDf=", nonSeizureDf)

    print("noiseAug() - Concatenating dataframe")
    sys.stdout.flush()
    df = pd.concat([augDf, nonSeizureDf], ignore_index=True)
    if (debug): print("df=",df)
    return(df)

def phaseAug(df, phase_step=1, debug=False):
    ''' Implement phase augmentation of seizure events in dataframe df
     It expects df to be a pandas dataframe representation of a flattened osdb dataset.
    For each input event, concatenate its samples in time order and generate one augmented event by
    sliding a 125-sample window with stride phase_step across the concatenated signal.
    The augmented event contains as many datapoints (rows) as windows that fit without padding.
    '''
    seizuresDf, nonSeizureDf = getSeizureNonSeizureDfs(df)
    if 'eventId' in seizuresDf.columns:
        seizuresDf = seizuresDf.copy()
        seizuresDf['eventId'] = seizuresDf['eventId'].astype(str)
    else:
        print("phaseAug(): ERROR: eventId column missing.")
        exit(-1)

    accStartCol = seizuresDf.columns.get_loc('M001')-1
    accEndCol = seizuresDf.columns.get_loc('M124')+1
    eventIdCol = seizuresDf.columns.get_loc('eventId')
    
    # Check if 3D acceleration columns exist
    has3DColumns = 'X000' in seizuresDf.columns and 'Y000' in seizuresDf.columns and 'Z000' in seizuresDf.columns
    accXStartCol, accXEndCol, accYStartCol, accYEndCol, accZStartCol, accZEndCol = None, None, None, None, None, None
    if has3DColumns:
        accXStartCol = seizuresDf.columns.get_loc('X000')
        accXEndCol = seizuresDf.columns.get_loc('X124') + 1
        accYStartCol = seizuresDf.columns.get_loc('Y000')
        accYEndCol = seizuresDf.columns.get_loc('Y124') + 1
        accZStartCol = seizuresDf.columns.get_loc('Z000')
        accZEndCol = seizuresDf.columns.get_loc('Z124') + 1
        print("phaseAug(): 3D acceleration columns detected - will apply phase augmentation to all channels")
    else:
        print("phaseAug(): No 3D acceleration columns detected - will apply phase augmentation to magnitude only")

    event_ids, event_groups = _build_event_index(seizuresDf, id_col='eventId')

    if len(event_ids) == 0:
        print("phaseAug(): ERROR:  No seizure events found; returning original dataframe")
        exit(-1)

    datetime_col = None
    for cand in ['dataTime', 'datetime', 'timestamp', 'time']:
        if cand in seizuresDf.columns:
            datetime_col = cand
            break
    if datetime_col is not None:
        print(f"phaseAug(): Using datetime column '{datetime_col}' for event ordering and spacing checks")
    else:
        print("phaseAug(): No datetime column found; proceeding without time-ordering or spacing checks")

    # Loop through each event
    print(f"phaseAug(): Augmenting {len(event_ids)} seizure events with phase step {phase_step}")
    out_groups = []
    for eid in event_ids:
        grp = event_groups[eid]
        out_groups.append(grp.copy())  # keep original event

        # Validate and sort by datetime if available; ensure at least 5s spacing
        if datetime_col is not None:
            grp = grp.copy()
            # Add a column with parsed datetime, and sort by datetime
            grp['_dt'] = pd.to_datetime(grp[datetime_col], errors='coerce')
            grp = grp.sort_values(by='_dt')
            # check spacing if we have at least two valid timestamps
            if grp['_dt'].notna().sum() >= 2:
                diffs = grp['_dt'].diff().dt.total_seconds()
                # Find rows where spacing from previous row is < 4s (excluding first row which has NaN diff)
                bad_spacing_mask = (diffs < 4.0) & (diffs.notna())
                if bad_spacing_mask.any():
                    num_dropped = bad_spacing_mask.sum()
                    print(f"phaseAug(): Dropping {num_dropped} row(s) from event {eid} due to overlapping data.")
                    # Keep only rows with adequate spacing (or first row)
                    grp = grp[~bad_spacing_mask].reset_index(drop=True)
            grp = grp.drop(columns=['_dt'])
            if len(bad_spacing_mask) > 0 and bad_spacing_mask.any():
                print(f"phaseAug(): After dropping overlapping rows, event {eid} has {len(grp)} rows.")

        print("phaseAug(): Processing event %s with %d rows" % (eid, len(grp)))
        acc_len = accEndCol - accStartCol
        step = max(1, int(phase_step))

        # Build concatenated magnitude and (optional) 3D arrays in event order
        mag_concat = pd.to_numeric(grp.iloc[:, accStartCol:accEndCol].to_numpy().reshape(-1), errors='coerce')
        mag_concat = np.nan_to_num(mag_concat, nan=0.0)
        x_concat = y_concat = z_concat = None
        if has3DColumns:
            x_concat = pd.to_numeric(grp.iloc[:, accXStartCol:accXEndCol].to_numpy().reshape(-1), errors='coerce')
            y_concat = pd.to_numeric(grp.iloc[:, accYStartCol:accYEndCol].to_numpy().reshape(-1), errors='coerce')
            z_concat = pd.to_numeric(grp.iloc[:, accZStartCol:accZEndCol].to_numpy().reshape(-1), errors='coerce')
            x_concat = np.nan_to_num(x_concat, nan=0.0)
            y_concat = np.nan_to_num(y_concat, nan=0.0)
            z_concat = np.nan_to_num(z_concat, nan=0.0)

        print(f"phaseAug(): event {eid}: concatenated magnitude array length={len(mag_concat)}")
        total_len = len(mag_concat)
        if total_len < acc_len:
            continue  # not enough data to form a window

        n_windows = 1 + (total_len - acc_len) // step
        aug_rows = []
        base_row = grp.iloc[0]

        print(f"phaseAug(): event {eid}: generating {n_windows-1} augmented windows (excluding original)")
        # Start from w=1 since w=0 duplicates the original event
        for w in range(1, n_windows):
            start = w * step
            end = start + acc_len
            mag_slice = mag_concat[start:end]

            outRow = []
            for i in range(0, accStartCol):
                if i == eventIdCol:
                    outRow.append(f"{eid}-{w}")
                else:
                    outRow.append(base_row.iloc[i])

            outRow.extend(mag_slice.tolist())

            if has3DColumns:
                x_slice = x_concat[start:end]
                y_slice = y_concat[start:end]
                z_slice = z_concat[start:end]
                outRow.extend(x_slice.tolist())
                outRow.extend(y_slice.tolist())
                outRow.extend(z_slice.tolist())

            endCol = accZEndCol if has3DColumns else accEndCol
            for i in range(endCol, len(base_row)):
                outRow.append(base_row.iloc[i])

            aug_rows.append(outRow)

        if len(aug_rows) > 0:
            aug_group = pd.DataFrame(aug_rows, columns=seizuresDf.columns)
            out_groups.append(aug_group)

        print("phaseAug(): Completed event %s" % eid)

    augDf = pd.concat(out_groups, ignore_index=True)
    if (debug): print("phaseAug() - augDf=", augDf)

    df = pd.concat([augDf, nonSeizureDf], ignore_index=True)
    if (debug): print("df=",df)
    return(df)


def augmentSeizureData(configObj, dataDir=".", debug=False):
    '''
    Given a pandas dataframe of osdb data,
    Apply data augmentation to the seizure data and return a new, extended data frame.
    It uses augmentation functions defined in the  augmentData module.

    The following configuration object values are used:
      - useNoiseAugmentation:  boolean - if True, noise Augmentation is applied to seizure rows.
      - noiseAugmentationFactor: int - number of copies of each seizure row to create with random noise applied
      - noiseAugmentationValue: int - amplitude of random noise to apply with noise augmentation.
      - usePhaseAugmentation:  boolean - if True, phase augmentation is applied to seizure rows.
      - useUserAugmentation:  boolean - if True, user augmentation is applied to seizure rows.
      - oversample: boolean - if not 'None', applies oversampling to balance the seizure and non-seizure rows.
                        Valid values are 'none', 'random' and 'smote'
      - undersample: boolean - if not 'None', applies undersampling to balance the seizure and non-seizure rows.
                        Valid values are 'none' and 'random'

    '''
    TAG = "augmentData.augmentSeizureData()"
    trainCsvFname = configObj['dataFileNames']['trainDataFileCsv']
    trainAugCsvFname = configObj['dataFileNames']['trainAugmentedFileCsv']
    useNoiseAugmentation = configObj['dataProcessing']['noiseAugmentation']
    noiseAugmentationFactor = configObj['dataProcessing']['noiseAugmentationFactor']
    noiseAugmentationValue = configObj['dataProcessing']['noiseAugmentationValue']
    usePhaseAugmentation = configObj['dataProcessing']['phaseAugmentation']
    phaseAugmentationStep = configObj['dataProcessing'].get('phaseAugmentationStep', 1)
    useUserAugmentation = configObj['dataProcessing']['userAugmentation']
    oversample = configObj['dataProcessing']['oversample']
    undersample = configObj['dataProcessing']['undersample']   

    trainCsvFnamePath = os.path.join(dataDir, trainCsvFname)
    trainAugCsvFnamePath = os.path.join(dataDir, trainAugCsvFname)
    if (debug): print("%s: trainCsvFnamePath=%s" % (TAG, trainCsvFnamePath))
    if (debug): print("%s: trainAugCsvFnamePath=%s" % (TAG, trainAugCsvFnamePath))
    if (trainCsvFname is None):
        print("%s: No input file specified.  Exiting." % TAG)
        sys.exit(1)

    print("%s: Loading data from file %s." % (TAG, trainCsvFnamePath))
    df = loadCsv(trainCsvFnamePath,debug)
    print("augmentData:  Loaded training data - Columns are:", df.columns)
    #df.to_csv("before_aug.csv")
    print("Applying Augmentation....")

    if usePhaseAugmentation:
        print("Phase Augmentation...")
        if (debug): print("%s: %d datapoints. Applying Phase Augmentation to Seizure data" % (TAG, len(df)))
        augDf = phaseAug(df, phase_step=phaseAugmentationStep)
        df = augDf
        #df.to_csv("after_phaseAug.csv")

    if useUserAugmentation:
        print("User Augmentation...")
        if (debug): print("%s: %d datapoints. Applying User Augmentation to Seizure data" % (TAG, len(df)))
        augDf = userAug(df)
        df = augDf
        #df.to_csv("after_userAug.csv")

    if useNoiseAugmentation: 
        print("Noise Augmentation...")
        if (debug): print("%s: %d datapoints.  Applying Noise Augmentation - factor=%d, value=%.2f%%" % (TAG, len(df), noiseAugmentationFactor, noiseAugmentationValue))
        augDf = noiseAug(df, 
                                    noiseAugmentationValue, 
                                    noiseAugmentationFactor, 
                                    debug=False)
        df = augDf
        #df.to_csv("after_noiseAug.csv")

    #print("After applying augmentation, columns are:",df.columns)

    print("Data Augmentation complete.")


    # Oversample Data to balance positive and negative data -- operate on whole events
    if (oversample is not None and oversample.lower()!="none"):
        print("Oversampling (event-level)...")
        # build event index and groups
        event_ids, event_groups = _build_event_index(df, id_col='eventId')
        # event-level labels: use the first row's 'type' for each event (consistent with extractor)
        X_events = []
        y_events = []
        for eid in event_ids:
            grp = event_groups[eid]
            X_events.append([eid])
            # take first row 'type' as event label
            try:
                ev_type = int(grp.iloc[0]['type'])
            except Exception:
                ev_type = int(grp['type'].iloc[0])
            y_events.append(ev_type)

        if (oversample.lower() == "random"):
            print("%s: %d events: Using Random Oversampling" % (TAG, len(X_events)))
            event_oversampler = imblearn.over_sampling.RandomOverSampler(random_state=0)
        elif (oversample.lower() == "smote"):
            print("%s: %d events: Using SMOTE Oversampling" % (TAG, len(X_events)))
            # SMOTE is not meaningful on single-feature event ids; fall back to random
            event_oversampler = imblearn.over_sampling.RandomOverSampler(random_state=0)
        else:
            print("%s: Not Using Oversampling" % TAG)
            event_oversampler = None

        if event_oversampler is not None:
            # fit_resample expects array-like X; we'll use event index as X
            res_X, res_y = event_oversampler.fit_resample(X_events, y_events)
            # res_X contains event-id placeholders (possibly duplicated)
            # We will reconstruct df by concatenating the corresponding event groups in order
            new_rows = []
            dup_counters = {}
            for rec in res_X:
                orig_eid = rec[0]
                grp = event_groups[orig_eid]
                # if this is a duplicated event (more occurrences in res_X than in event_ids),
                # assign a new eventId to avoid identical ids for separate duplicated events
                dup_counters.setdefault(orig_eid, 0)
                dup_counters[orig_eid] += 1
                dup_count = dup_counters[orig_eid]
                if dup_count == 1:
                    # first occurrence: keep original eventId
                    out_grp = grp.copy()
                else:
                    # subsequent duplicates: make a copy and assign synthetic eventIds
                    out_grp = grp.copy()
                    synthetic_eventId = f"{orig_eid}-{dup_count-1}"
                    out_grp = out_grp.copy()
                    out_grp['eventId'] = synthetic_eventId
                new_rows.append(out_grp)

            # concatenate while preserving the resampled event order
            if len(new_rows) > 0:
                df = pd.concat(new_rows, ignore_index=True)
            else:
                df = pd.DataFrame(columns=df.columns)
        #df.to_csv("after_oversample.csv")


    # Undersample data to balance positive and negative data
    if (undersample is not None and undersample.lower() != "none"):
        print("Under Sampling (event-level)...")
        # build event index and groups
        event_ids, event_groups = _build_event_index(df, id_col='eventId')
        X_events = []
        y_events = []
        for eid in event_ids:
            grp = event_groups[eid]
            X_events.append([eid])
            try:
                ev_type = int(grp.iloc[0]['type'])
            except Exception:
                ev_type = int(grp['type'].iloc[0])
            y_events.append(ev_type)

        if (undersample.lower() == "random"):
            print("Using Random Event Undersampling")
            event_undersampler = imblearn.under_sampling.RandomUnderSampler(random_state=0)
        else:
            print("%s: Not using undersampling" % TAG)
            event_undersampler = None

        if event_undersampler is not None:
            res_X, res_y = event_undersampler.fit_resample(X_events, y_events)
            new_rows = []
            # res_X are kept event ids; just concatenate corresponding groups
            for rec in res_X:
                orig_eid = rec[0]
                grp = event_groups[orig_eid]
                new_rows.append(grp.copy())
            if len(new_rows) > 0:
                df = pd.concat(new_rows, ignore_index=True)
            else:
                df = pd.DataFrame(columns=df.columns)
        #df.to_csv("after_underample.csv")

    #print("after undersampling, columns are:",df.columns)
    
    # Sort by eventId to group original and synthetic events together
    if 'eventId' in df.columns:
        print("Sorting data by eventId to group synthetic events with originals...")
        df['eventId'] = df['eventId'].astype(str)
        df = df.sort_values(by='eventId').reset_index(drop=True)
                
    print("Saving augmented data file to %s" % trainAugCsvFnamePath)
    df.to_csv(trainAugCsvFnamePath)
    print("%s: saved %d datapoints to file %s" % (TAG, len(df), trainAugCsvFnamePath))

    return


def balanceTestData(configObj, debug=False):
    '''
    Produce a balanced test data file using random over or undersampling as specified
    in the config file.

    The following configuration object values are used:
      - oversample: boolean - if not 'None', applies oversampling to balance the seizure and non-seizure rows.
                        Valid values are 'none', 'random' and 'smote'
      - undersample: boolean - if not 'None', applies undersampling to balance the seizure and non-seizure rows.
                        Valid values are 'none' and 'random'
    '''
    TAG = "augmentData.balanceTestData()"
    testCsvFname = configObj['testDataFileCsv']
    testBalCsvFname = configObj['testBalancedFileCsv']
    oversample = libosd.configUtils.getConfigParam("oversample", configObj)
    undersample = libosd.configUtils.getConfigParam("undersample", configObj)

    print("%s: Loading data from file %s." % (TAG, testCsvFname))
    df = loadCsv(testCsvFname,debug)
    # Oversample Data at event-level to balance positive and negative data
    if (oversample is not None and oversample.lower()!="none"):
        print("Oversampling (event-level)...")
        event_ids, event_groups = _build_event_index(df, id_col='eventId')
        X_events = []
        y_events = []
        for eid in event_ids:
            grp = event_groups[eid]
            X_events.append([eid])
            try:
                ev_type = int(grp.iloc[0]['type'])
            except Exception:
                ev_type = int(grp['type'].iloc[0])
            y_events.append(ev_type)

        if (oversample.lower() == "random"):
            print("%s: %d events: Using Random Oversampling" % (TAG, len(X_events)))
            event_oversampler = imblearn.over_sampling.RandomOverSampler(random_state=0)
        elif (oversample.lower() == "smote"):
            print("%s: %d events: Using SMOTE Oversampling" % (TAG, len(X_events)))
            event_oversampler = imblearn.over_sampling.RandomOverSampler(random_state=0)
        else:
            print("%s: Not Using Oversampling" % TAG)
            event_oversampler = None

        if event_oversampler is not None:
            res_X, res_y = event_oversampler.fit_resample(X_events, y_events)
            new_rows = []
            dup_counters = {}
            for rec in res_X:
                orig_eid = rec[0]
                grp = event_groups[orig_eid]
                dup_counters.setdefault(orig_eid, 0)
                dup_counters[orig_eid] += 1
                dup_count = dup_counters[orig_eid]
                if dup_count == 1:
                    out_grp = grp.copy()
                else:
                    out_grp = grp.copy()
                    synthetic_eventId = f"{orig_eid}-{dup_count-1}"
                    out_grp = out_grp.copy()
                    out_grp['eventId'] = synthetic_eventId
                new_rows.append(out_grp)
            if len(new_rows) > 0:
                df = pd.concat(new_rows, ignore_index=True)
            else:
                df = pd.DataFrame(columns=df.columns)
    # Undersample data at event-level to balance positive and negative data
    if (undersample is not None and undersample.lower() != "none"):
        print("Under Sampling (event-level)...")
        event_ids, event_groups = _build_event_index(df, id_col='eventId')
        X_events = []
        y_events = []
        for eid in event_ids:
            grp = event_groups[eid]
            X_events.append([eid])
            try:
                ev_type = int(grp.iloc[0]['type'])
            except Exception:
                ev_type = int(grp['type'].iloc[0])
            y_events.append(ev_type)

        if (undersample.lower() == "random"):
            print("Using Random Event Undersampling")
            event_undersampler = imblearn.under_sampling.RandomUnderSampler(random_state=0)
        else:
            print("%s: Not using undersampling" % TAG)
            event_undersampler = None

        if event_undersampler is not None:
            res_X, res_y = event_undersampler.fit_resample(X_events, y_events)
            new_rows = []
            for rec in res_X:
                orig_eid = rec[0]
                grp = event_groups[orig_eid]
                new_rows.append(grp.copy())
            if len(new_rows) > 0:
                df = pd.concat(new_rows, ignore_index=True)
            else:
                df = pd.DataFrame(columns=df.columns)
    
    # Sort by eventId to group original and synthetic events together
    if 'eventId' in df.columns:
        print("Sorting test data by eventId to group synthetic events with originals...")
        df['eventId'] = df['eventId'].astype(str)
        df = df.sort_values(by='eventId').reset_index(drop=True)
                
    print("Saving augmented data file")
    df.to_csv(testBalCsvFname)
    print("%s: saved %d datapoints to file %s" % (TAG, len(df), testBalCsvFname))

    return




def main():
    print("flattenOsdb.main()")
    parser = argparse.ArgumentParser(description='Perform data augmentation on a flattened (.csv) version of the OpenSeizureDatabase data')
    parser.add_argument('-i', default=None,
                        help='Input filename (uses stdin if not specified)')
    parser.add_argument('-o', default='dfAug.csv',
                        help='Output filename (default dfAug.csv)')
    parser.add_argument('--debug', action="store_true",
                        help='Write debugging information to screen')
    parser.add_argument('-u', action="store_true",
                        help='Apply User Augmentation')
    parser.add_argument('-n', action="store_true",
                        help='Apply Noise Augmentation')
    parser.add_argument('-p', action="store_true",
                        help='Apply Phase Augmentation')
    argsNamespace = parser.parse_args()
    args = vars(argsNamespace)
    print(args)



    df = loadCsv(args['i'], args['debug'])
    if (args['u']):
        df = userAug(df)
        print("userAug returned df")
        analyseDf(df)
    if (args['n']):
        df = noiseAug(df,10,3,args['debug'])
        print("noiseAug returned df")
        analyseDf(df)
    if (args['p']):
        df = phaseAug(df, phase_step=25, debug=args['debug'])
        print("phaseAug returned df")
        analyseDf(df)

    print("Saving augmented data file to %s" % args['o'])
    df.to_csv(args['o'])
    print("Saved %d datapoints to file %s" % (len(df), args['o']))

if __name__ == "__main__":
    main()

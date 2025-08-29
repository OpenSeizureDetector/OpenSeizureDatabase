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

    df = pd.read_csv(inFile)

    #print(df)
    if (debug): print("%s: returning %d datapoints" % (TAG, len(df)))

    return(df)


def getSeizureNonSeizureDfs(df):
    ''' returns two datasets, seizure, non-seizure from dataframe df
    '''
    seizuresDf = df[df['type']==1]
    nonSeizureDf = df[df['type']!=1]
    return (seizuresDf, nonSeizureDf)


def _build_event_index(df, id_col='id'):
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
    ''' implement user augmentation to oversample data to balance the contributions of different users
    It expects df to be a pandas dataframe representation of a flattened osdb dataset.
    '''
    seizuresDf, nonSeizureDf = getSeizureNonSeizureDfs(df)
    y = seizuresDf['userId']
    ros = imblearn.over_sampling.RandomOverSampler(random_state=0)
    xResamp, yResamp = ros.fit_resample(seizuresDf, y)
    print("userAug(): Distribution of seizure data after user augmentation")
    analyseDf(xResamp)
    # Combine seizure and non-seizure data back into single dataframe to return
    df = pd.concat([nonSeizureDf,xResamp])
    return(df)


def noiseAug(df, noiseAugVal, noiseAugFac, debug=False):
    ''' Implement noise augmentation of the seizue datapoints in dataframe df
     It expects df to be a pandas dataframe representation of a flattened osdb dataset.
     noiseAugVal is the amplitude (in mg) of the noise applied.
     noiseAugFac is the number of augmented datapoints generated for each input datapoint.
    '''
    tStart = time.time()
    seizuresDf, nonSeizureDf = getSeizureNonSeizureDfs(df)
    augDf = seizuresDf
    if(debug): print(seizuresDf.columns)
    accStartCol = seizuresDf.columns.get_loc('M001')-1
    accEndCol = seizuresDf.columns.get_loc('M124')+1
    print("noiseAug(): Augmenting %d seizure datpoints.  accStartCol=%d, accEndCol=%d" % (len(seizuresDf), accStartCol, accEndCol))
    outLst = []
    for n in range(0,len(seizuresDf)):
        if (debug): print("n=%d" % n)
        rowArr = seizuresDf.iloc[n]
        if (debug): print("rowArrLen=%d" % len(rowArr), type(rowArr), rowArr)
        accArr = rowArr.iloc[accStartCol:accEndCol]
        if (debug): print("accArrLen=%d" % len(accArr), type(accArr), accArr)
        inArr =np.array(accArr)
        if(debug): print(inArr.shape)
        for j in range(0,noiseAugFac):
            noiseArr = np.random.normal(0,noiseAugVal,inArr.shape)
            outArr = inArr + noiseArr
            noiseArr = None
            outRow = []
            for i in range(0,accStartCol):
                outRow.append(rowArr.iloc[i])
            outRow.extend(outArr.tolist())
            outLst.append(outRow)
            outRow = None
            noiseArr = None
            # gc.collect()
        inArr = None
        rowArr = None
        accArr = None
        #gc.collect()
        # Update every 1000 datapoints and run garbage collector to try to save memory.
        if (n % 1000 == 0):
            tElapsed, tRem, tCompletion, tIter = calcProcessTime(tStart,n,len(seizuresDf))
            sys.stdout.write("n=%d, tIter=%.1f ms, elapsed: %s(s), time left: %s(s), estimated finish time: %s\r" % (n, tIter*1000., tElapsed,tRem,tCompletion))
            sys.stdout.flush()
            gc.collect()
    print("noiseAug() - Creating dataframe")
    sys.stdout.flush()
    gc.collect()
    augDf = pd.DataFrame(outLst, columns=nonSeizureDf.columns)
    outLst = None
    if (debug): print("noiseAug() - augDf=", augDf)
    if (debug): print("noiseAug() nonSeizureDf=", nonSeizureDf)

    print("noiseAug() - Concatenating dataframe")
    sys.stdout.flush()
    df = pd.concat([seizuresDf, augDf, nonSeizureDf])
    if (debug): print("df=",df)
    return(df)

def phaseAug(df, debug=False):
    ''' Implement phase augmentation of the seizue datapoints in dataframe df
     It expects df to be a pandas dataframe representation of a flattened osdb dataset.
    '''
    seizuresDf, nonSeizureDf = getSeizureNonSeizureDfs(df)

    accStartCol = seizuresDf.columns.get_loc('M001')-1
    accEndCol = seizuresDf.columns.get_loc('M124')+1
    eventIdCol = seizuresDf.columns.get_loc('id')
    #print("accStartCol=%d, accEndCol=%d" % (accStartCol, accEndCol))
    outLst = []
    lastAccArr = None
    lastEventId = seizuresDf.iloc[0].iloc[eventIdCol]
    for n in range(0,len(seizuresDf)):
        rowArr = seizuresDf.iloc[n]
        eventId = rowArr.iloc[eventIdCol]
        # the Dataframe is a list of datapoints, so we have to look for events changing
        if (eventId != lastEventId):
            lastEventId = eventId
            lastAccArr = None
        accArr = rowArr.iloc[accStartCol:accEndCol]
        if (lastAccArr is not None):
            # Make one long list from two consecutive rows.
            combArr = lastAccArr.tolist().copy()
            combArr.extend(accArr)
            for n in range(0,len(accArr)):
                outArr = combArr[n:n+len(accArr)]
                outRow = []
                for i in range(0,accStartCol):
                    outRow.append(rowArr.iloc[i])
                outRow.extend(outArr)
                outLst.append(outRow)
                outArr = None,
                outRow = None
        lastAccArr = accArr.copy()
        rowArr = None
        accArr = None
    augDf = pd.DataFrame(outLst, columns=nonSeizureDf.columns)
    if (debug): print("phaseAug() - augDf=", augDf)

    df = pd.concat([seizuresDf, augDf, nonSeizureDf])
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
        augDf = phaseAug(df)
        df = augDf
        df.to_csv("after_phaseAug.csv")

    if useUserAugmentation:
        print("User Augmentation...")
        if (debug): print("%s: %d datapoints. Applying User Augmentation to Seizure data" % (TAG, len(df)))
        augDf = userAug(df)
        df = augDf
        df.to_csv("after_userAug.csv")

    if useNoiseAugmentation: 
        print("Noise Augmentation...")
        if (debug): print("%s: %d datapoints.  Applying Noise Augmentation - factor=%d, value=%.2f%%" % (TAG, len(df), noiseAugmentationFactor, noiseAugmentationValue))
        augDf = noiseAug(df, 
                                    noiseAugmentationValue, 
                                    noiseAugmentationFactor, 
                                    debug=False)
        df = augDf
        df.to_csv("after_noiseAug.csv")

    print("After applying augmentation, columns are:",df.columns)



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
            # compute numeric_max for id generation when numeric ids are used
            numeric_ids = [eid for eid in event_ids if isinstance(eid, (int, float, np.integer, np.floating))]
            numeric_max = int(max(numeric_ids)) if len(numeric_ids) > 0 else None
            dup_counters = {}
            for rec in res_X:
                orig_eid = rec[0]
                grp = event_groups[orig_eid]
                # if this is a duplicated event (more occurrences in res_X than in event_ids),
                # assign a new id to avoid identical ids for separate duplicated events
                dup_counters.setdefault(orig_eid, 0)
                dup_counters[orig_eid] += 1
                dup_count = dup_counters[orig_eid]
                if dup_count == 1:
                    # first occurrence: keep original id
                    out_grp = grp.copy()
                else:
                    # subsequent duplicates: make a copy and assign new ids
                    out_grp = grp.copy()
                    new_id = _make_new_id(orig_eid, dup_count-1, numeric_max)
                    out_grp = out_grp.copy()
                    out_grp['id'] = new_id
                new_rows.append(out_grp)

            # concatenate while preserving the resampled event order
            if len(new_rows) > 0:
                df = pd.concat(new_rows, ignore_index=True)
            else:
                df = pd.DataFrame(columns=df.columns)
        df.to_csv("after_oversample.csv")


    # Undersample data to balance positive and negative data
    if (undersample is not None and undersample.lower() != "none"):
        print("Under Sampling (event-level)...")
        # build event index and groups
        event_ids, event_groups = _build_event_index(df, id_col='id')
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
        df.to_csv("after_underample.csv")

    print("after undersampling, columns are:",df.columns)
                
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
        event_ids, event_groups = _build_event_index(df, id_col='id')
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
            numeric_ids = [eid for eid in event_ids if isinstance(eid, (int, float, np.integer, np.floating))]
            numeric_max = int(max(numeric_ids)) if len(numeric_ids) > 0 else None
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
                    new_id = _make_new_id(orig_eid, dup_count-1, numeric_max)
                    out_grp['id'] = new_id
                new_rows.append(out_grp)
            if len(new_rows) > 0:
                df = pd.concat(new_rows, ignore_index=True)
            else:
                df = pd.DataFrame(columns=df.columns)
    # Undersample data at event-level to balance positive and negative data
    if (undersample is not None and undersample.lower() != "none"):
        print("Under Sampling (event-level)...")
        event_ids, event_groups = _build_event_index(df, id_col='id')
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
        df = noiseAug(df)
        print("noiseAug returned df")
        analyseDf(df)
    if (args['p']):
        df = phaseAug(df)
        print("phaseAug returned df")
        analyseDf(df)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import json
import sys
import os
import importlib
import dateutil.parser
import numpy as np
import csv

sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
#import libosd.analyse_event
import libosd.webApiConnection
import libosd.osdDbConnection
import libosd.osdAppConnection
import libosd.dpTools
import libosd.configUtils


def _resolve_existing_path(path, search_dirs=None):
    """Resolve a path by checking common base directories.

    Resolution order:
    - absolute path
    - path as-provided (relative to CWD)
    - each directory in search_dirs joined with path
    """
    if path is None:
        return None
    if not isinstance(path, str):
        path = str(path)
    path = path.strip()
    if path == "":
        return None

    if os.path.isabs(path) and os.path.exists(path):
        return path
    if os.path.exists(path):
        return path

    if search_dirs:
        for base_dir in search_dirs:
            if base_dir is None:
                continue
            candidate = os.path.join(base_dir, path)
            if os.path.exists(candidate):
                return candidate

    return None


def _load_training_event_ids_from_csv(csv_path):
    """Return a set of event IDs (as strings) from a flattened trainData CSV."""
    event_ids = set()
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader, None)
        if headers is None:
            return event_ids
        headers = [_clean_csv_header(h) for h in headers]

        # Find an event id column (trainData.csv uses 'eventId')
        event_idx = None
        for col in ['eventId', 'EventID', 'event_id', 'id']:
            if col in headers:
                event_idx = headers.index(col)
                break
        if event_idx is None:
            event_idx = 0

        for row in reader:
            if not row:
                continue
            if event_idx >= len(row):
                continue
            ev = row[event_idx]
            if ev is None:
                continue
            ev = str(ev).strip()
            if ev == "":
                continue
            event_ids.add(ev)
    return event_ids


def _load_training_event_ids_from_json(json_path):
    """Return a set of event IDs (as strings) from a trainData JSON file."""
    event_ids = set()
    with open(json_path, 'r') as f:
        obj = json.load(f)

    # OSDB format is a list of event objects.
    if isinstance(obj, dict):
        # Be tolerant of alternative container formats.
        if 'events' in obj and isinstance(obj['events'], list):
            events = obj['events']
        else:
            raise ValueError(f"Unsupported JSON structure in {json_path}: expected a list or dict with 'events' list")
    elif isinstance(obj, list):
        events = obj
    else:
        raise ValueError(f"Unsupported JSON structure in {json_path}: expected a list of events")

    for ev_obj in events:
        if not isinstance(ev_obj, dict):
            continue

        ev_id = ev_obj.get('id', None)
        if ev_id is None:
            # Fallback: some variants embed eventId in the first datapoint
            dps = ev_obj.get('datapoints', None)
            if isinstance(dps, list) and len(dps) > 0 and isinstance(dps[0], dict):
                ev_id = dps[0].get('eventId', None)

        if ev_id is None:
            continue
        event_ids.add(str(ev_id).strip())

    return event_ids


def load_training_event_ids(train_data_path, search_dirs=None, debug=False):
    """Load training event IDs from a trainData.csv or trainData.json file.

    Returns a set of event IDs as *strings*.
    """
    resolved = _resolve_existing_path(train_data_path, search_dirs=search_dirs)
    if resolved is None:
        raise FileNotFoundError(
            f"excludeTrainingEvents file not found: '{train_data_path}'. "
            f"Searched CWD and: {search_dirs}"
        )
    if debug:
        print(f"Loading training event IDs from {resolved}")

    lower = resolved.lower()
    if lower.endswith('.csv'):
        ids = _load_training_event_ids_from_csv(resolved)
    elif lower.endswith('.json'):
        ids = _load_training_event_ids_from_json(resolved)
    else:
        raise ValueError(f"excludeTrainingEvents must be a .csv or .json file: {resolved}")

    if debug:
        print(f"Loaded {len(ids)} unique training event IDs")
    return ids


def exclude_training_events_from_osd(osd, train_data_path, search_dirs=None, debug=False):
    """Remove any events from `osd` whose IDs appear in the training file.

    Returns a list of event IDs (in the database's native type) that were removed.
    """
    training_ids = load_training_event_ids(train_data_path, search_dirs=search_dirs, debug=debug)
    if not training_ids:
        return []

    current_ids = osd.getEventIds()
    remove_ids = [ev_id for ev_id in current_ids if str(ev_id) in training_ids]

    if remove_ids:
        if debug:
            print(f"Removing {len(remove_ids)} events used in training")
        osd.removeEvents(remove_ids)
    return remove_ids


def _clean_csv_header(header):
    """Normalise CSV header names.

    Some generated CSVs may contain embedded newlines in quoted headers
    (e.g. "M0\n00"), which breaks downstream column lookup.
    """
    if header is None:
        return header
    if not isinstance(header, str):
        header = str(header)
    return header.replace('\ufeff', '').replace('\r', '').replace('\n', '').strip()


OTHERS_INDEX = 0
ALL_INDEX = 1
FALSE_INDEX = 2
NDA_INDEX = 3

def csvRowToEvent(row, headers, eventId, eventsByIdDict, debug=False):
    """
    Convert a CSV row (produced by flattenData.py) to a datapoint object
    and add it to the event in eventsByIdDict.
    
    If the event doesn't exist yet, it will be created.
    
    Args:
        row: List of values from CSV row
        headers: List of header names
        eventId: The event ID for this row
        eventsByIdDict: Dictionary mapping eventId to event objects
        debug: Debug flag
    """
    # Create a mapping of header name to row value (tolerate short rows)
    rowDict = {headers[i]: row[i] for i in range(min(len(headers), len(row)))}
    
    # If this is the first datapoint for this event, create the event object
    if eventId not in eventsByIdDict:
        try:
            typeStr = rowDict.get('typeStr', 'Unknown')
            # Remove quotes if present
            if typeStr.startswith('"') and typeStr.endswith('"'):
                typeStr = typeStr[1:-1]
            # Split "type/subType" format
            if '/' in typeStr:
                eventType, subType = typeStr.split('/', 1)
            else:
                eventType = typeStr
                subType = ''
            
            # Use the first datapoint's time as the event time
            dataTime = rowDict.get('dataTime', '')
            
            eventsByIdDict[eventId] = {
                'id': eventId,
                'userId': rowDict.get('userId', ''),
                'type': eventType,
                'subType': subType,
                'dataTime': dataTime,  # Set event dataTime from first datapoint
                'desc': f"Event {eventId} ({eventType})",
                'datapoints': []
            }
        except Exception as e:
            print(f"Error creating event from CSV row: {e}", file=sys.stderr)
            return
    
    # Create the datapoint object
    try:
        # Extract basic fields
        dataTime = rowDict.get('dataTime', '')
        alarmState = int(float(rowDict.get('osdAlarmState', 0)))
        specPower = float(rowDict.get('osdSpecPower', 0))
        roiPower = float(rowDict.get('osdRoiPower', 0))
        hr = rowDict.get('hr', '')
        if hr and hr != '':
            hr = int(float(hr))
        else:
            hr = -1
        o2sat = rowDict.get('o2sat', '')
        if o2sat and o2sat != '':
            o2sat = int(float(o2sat))
        else:
            o2sat = -1
        
        # Extract 1D acceleration (M000-M124)
        rawData = []
        for n in range(125):
            colName = f"M{n:03d}"
            val = rowDict.get(colName, '')
            if val is None or val == '':
                # Support feature-history CSVs where columns are named like M000_t-0
                colNameHist = f"M{n:03d}_t-0"
                val = rowDict.get(colNameHist, '')
            if val and val != '':
                rawData.append(float(val))
            else:
                rawData.append(None)
        
        # Extract 3D acceleration (X/Y/Z 000-124)
        rawData3D = []
        for n in range(125):
            for axis in ["X", "Y", "Z"]:
                colName = f"{axis}{n:03d}"
                val = rowDict.get(colName, '')
                if val and val != '':
                    rawData3D.append(float(val))
                else:
                    rawData3D.append(None)
        
        datapoint = {
            'dataTime': dataTime,
            'alarmState': alarmState,
            'specPower': specPower,
            'roiPower': roiPower,
            'hr': hr,
            'o2Sat': o2sat,
            'rawData': rawData,
            'rawData3D': rawData3D,
            'maxVal': 0,
            'minVal': 0,
            'maxFreq': 0,
            'alarmPhrase': ''
        }
        
        eventsByIdDict[eventId]['datapoints'].append(datapoint)
    except Exception as e:
        print(f"Error creating datapoint from CSV row: {e}", file=sys.stderr)

def loadCsvFile(csvFname, debug=False):
    """
    Load a CSV file produced by flattenData.py and convert it to a dictionary
    of event objects, similar to what would be loaded from JSON files.
    
    Args:
        csvFname: Path to the CSV file
        debug: Debug flag
    
    Returns:
        Dictionary mapping eventId to event objects
    """
    eventsByIdDict = {}
    
    if not os.path.exists(csvFname):
        print(f"Error: CSV file {csvFname} does not exist", file=sys.stderr)
        return eventsByIdDict
    
    with open(csvFname, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)  # Read header row
        headers = [_clean_csv_header(h) for h in headers]
        
        for rowIdx, row in enumerate(reader):
            if not row or all(v == '' for v in row):  # Skip empty rows
                continue
            
            eventId = row[0]  # First column is eventId
            csvRowToEvent(row, headers, eventId, eventsByIdDict, debug=debug)
    
    if debug:
        print(f"Loaded {len(eventsByIdDict)} events from CSV file {csvFname}")
        for eventId, event in eventsByIdDict.items():
            print(f"  Event {eventId}: {len(event.get('datapoints', []))} datapoints")
    
    return eventsByIdDict

def loadDataFiles(dataFiles, dbDir=None, debug=False):
    """
    Load data files, which can be either OSDB JSON files or CSV files from flattenData.py.
    
    Args:
        dataFiles: List of file paths to load
        dbDir: Optional cache directory for OSDB files (also used to resolve relative paths)
        debug: Debug flag
    
    Returns:
        OsdDbConnection object with loaded events
    """
    osd = libosd.osdDbConnection.OsdDbConnection(cacheDir=dbDir, debug=debug)

    for fname in dataFiles:
        if fname.lower().endswith('.csv'):
            # CSV files are opened directly from the filesystem.
            if dbDir and not os.path.isabs(fname):
                filePath = os.path.join(dbDir, fname)
            else:
                filePath = fname

            if debug:
                print(f"Loading CSV file: {filePath}")
            csvEvents = loadCsvFile(filePath, debug=debug)

            for eventId, eventObj in csvEvents.items():
                osd.addEvent(eventObj)
            print(f"loaded {len(csvEvents)} events from CSV file {filePath}")
        else:
            # JSON files are loaded via OsdDbConnection, which handles cacheDir.
            if os.path.isabs(fname):
                filePath = fname
                useCacheDir = False
            else:
                filePath = fname
                useCacheDir = True

            if debug:
                print(f"Loading JSON file: {filePath} (useCacheDir={useCacheDir})")
            eventsObjLen = osd.loadDbFile(filePath, useCacheDir=useCacheDir)
            print(f"loaded {eventsObjLen} events from file {filePath}")
    
    return osd

def runTest(configObj, debug=False, configPath=None):
    print("runTest - configObj="+json.dumps(configObj))
    if ('dbDir' in configObj.keys()):
        dbDir = configObj['dbDir']
    else:
        dbDir = None

    configDir = None
    if configPath:
        try:
            configDir = os.path.dirname(os.path.abspath(configPath))
        except Exception:
            configDir = None

    invalidEvents = configObj['invalidEvents']
    print("invalid events", invalidEvents)

    # Load each of the data files (can be OSDB JSON or CSV from flattenData.py)
    osd = loadDataFiles(configObj['dataFiles'], dbDir=dbDir, debug=debug)
    osd.removeEvents(invalidEvents)
    filterCfg = configObj['eventFilters']
    print("filterCfg=", filterCfg)

    # Optional: exclude any events used during training to avoid train/test contamination
    # when evaluating against a newer version of the database.
    train_data_path = filterCfg.get('excludeTrainingEvents', None)
    if train_data_path:
        search_dirs = [d for d in [configDir, dbDir, os.getcwd()] if d]
        removed = exclude_training_events_from_osd(
            osd,
            train_data_path,
            search_dirs=search_dirs,
            debug=debug,
        )
        print(f"Excluded {len(removed)} training events")

    osd.listEvents()
    

    eventIdsLst = osd.getFilteredEventsLst(
            includeUserIds = filterCfg['includeUserIds'],
            excludeUserIds = filterCfg['excludeUserIds'],
            includeTypes = filterCfg['includeTypes'],
            excludeTypes = filterCfg['excludeTypes'],
            includeSubTypes = filterCfg['includeSubTypes'],
            excludeSubTypes = filterCfg['excludeSubTypes'],
            includeDataSources = filterCfg['includeDataSources'],
            excludeDataSources = filterCfg['excludeDataSources'],
            includeText = filterCfg['includeText'],
            excludeText = filterCfg['excludeText'],
            require3dData= filterCfg['require3dData'],
            requireHrData= filterCfg['requireHrData'],
            requireO2SatData= filterCfg['requireO2SatData'],
            debug = True

    )

    print("%d events remaining after applying filters" % len(eventIdsLst))
    print(eventIdsLst)
    

    
    # Create an instance of the relevant Algorithm class for each algorithm
    # specified in the configuration file.
    # They are imported dynamically so we do not need to have 'import'
    # statements for all the possible algorithm classes in this file.
    algs = []
    algNames = []
    for algObj in configObj['algorithms']:
        print(algObj['name'], algObj['enabled'])
        if (algObj['enabled']):
            moduleId = algObj['alg'].split('.')[0]
            classId = algObj['alg'].split('.')[1]
            print("Importing Module %s" % moduleId)
            module = importlib.import_module(moduleId)

            algObj['settings']['name'] = algObj['name']
            settingsStr = json.dumps(algObj['settings'])
            print("settingsStr=%s (%s)" % (settingsStr, type(settingsStr)))
            algs.append(eval("module.%s(settingsStr, debug)" % (classId)))
            algNames.append(algObj['name'])
        else:
            print("Algorithm %s is disabled in configuration file - ignoring"
                  % algObj['name'])

    

    # Run each event through each algorithm
    tcResults, tcResultsStrArr, expandedAlgNames = testEachEvent(eventIdsLst, osd, algs, algNames, debug=debug)
    saveResults2("output", tcResults, tcResultsStrArr, eventIdsLst, osd, expandedAlgNames)
    
    #allSeizureResults, allSeizureResultsStrArr = testEachEvent(osdAll, algs, debug)
    #saveResults("allSeizureResults.csv", allSeizureResults, allSeizureResultsStrArr, osdAll, algs, algNames, True)
    
    #falseAlarmResults, falseAlarmResultsStrArr = testEachEvent(osdFalse, algs, debug)
    #saveResults("falseAlarmResults.csv", falseAlarmResults, falseAlarmResultsStrArr, osdFalse, algs, algNames, False)

    #summariseResults(tcResults, allSeizureResults, falseAlarmResults, algNames)

def getEventVal(eventObj, elemId):
    if (elemId in eventObj.keys()):
        return eventObj[elemId]
    else:
        return None

def getEventAlarmState(eventObj, debug=False):
    ''' scan through the datapoints and check the highest (non-manual alarm) alarm state
    over the event.
    Returns the alarm state number.'''
    alarmStateTextLst = ['OK', 'WARN', 'ALARM']
    maxAlarmState = 0
    if ('datapoints' in eventObj):
        for dp in eventObj['datapoints']:
            #if (debug): print(dp)
            dpTimeStr = dp['dataTime']
            dpTimeSecs = libosd.dpTools.dateStr2secs(dpTimeStr)
            alarmState = libosd.dpTools.getParamFromDp('alarmState',dp)
            if alarmState == 1 and maxAlarmState == 0:
                maxAlarmState = 1
            if alarmState == 2:
                maxAlarmState = 2
    return (maxAlarmState)
                 



def createZeroDatapoint():
    """
    Create a datapoint with all zero acceleration data.
    Used to transition between events or reset algorithms.
    """
    return {
        'dataTime': '2000-01-01T00:00:00Z',  # Dummy timestamp
        'alarmState': 0,
        'specPower': 0,
        'roiPower': 0,
        'hr': -1,
        'o2Sat': -1,
        'rawData': [0] * 125,
        'rawData3D': [0] * 375,
        'maxVal': 0,
        'minVal': 0,
        'maxFreq': 0,
        'alarmPhrase': 'Reset'
    }

def sendZeroDataTransition(alg, eventId, nDatapoints=6, debug=False):
    """
    Send zero-data datapoints to an algorithm for clean transition between events.
    This is particularly important for streaming algorithms (e.g., DeviceAlg)
    to flush their internal buffers and reset state gracefully.
    
    Args:
        alg: Algorithm instance
        eventId: Event ID (for logging purposes)
        nDatapoints: Number of zero datapoints to send (default 6)
        debug: Debug flag
    """
    if debug:
        print(f"Sending {nDatapoints} zero datapoints to {alg.__class__.__name__} for event transition")
    
    for i in range(nDatapoints):
        zeroDatapoint = createZeroDatapoint()
        rawDataStr = libosd.dpTools.dp2rawData(zeroDatapoint, debug=False)
        if rawDataStr is not None:
            try:
                retVal = alg.processDp(rawDataStr, eventId)
                # Don't count these transition datapoints in results
                if debug:
                    sys.stdout.write("Z")
            except Exception as e:
                if debug:
                    print(f"Warning: Error processing zero datapoint: {e}")
        sys.stdout.flush()


def _try_int(val):
    if val is None:
        return None
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, float):
        return int(val)
    if isinstance(val, str):
        v = val.strip()
        if v == "":
            return None
        try:
            return int(float(v))
        except Exception:
            return None
    return None


def _iter_device_subalg_states(retObj, baseAlgName):
    """Yield (slot_name, valid, state_int) for sub-algorithm states in device /data JSON."""
    if not isinstance(retObj, dict):
        return

    # Scalar per-algorithm states like osdAlgState, flapAlgState, cnnAlgState, etc.
    for key, val in retObj.items():
        if not isinstance(key, str):
            continue
        if not key.endswith('AlgState'):
            continue
        state = _try_int(val)
        yield (f"{baseAlgName}.{key}", True, state)

    # ML models reported as parallel arrays.
    model_names = retObj.get('mlModelNames', None)
    model_states = retObj.get('mlModelStates', None)
    model_active = retObj.get('mlModelActive', None)

    if isinstance(model_names, list) and isinstance(model_states, list):
        for i in range(min(len(model_names), len(model_states))):
            name = str(model_names[i])
            active = True
            if isinstance(model_active, list) and i < len(model_active):
                active = bool(model_active[i])
            state = _try_int(model_states[i])
            yield (f"{baseAlgName}.ml.{name}", bool(active), state)

def testEachEvent(eventIdsLst, osd, algs, algNames,  debug=False):
    """
    for each event in the OsdDbConnection 'osd', run each algorithm in the
    list 'algs', where each item in the algs list is an instance of an SdAlg
    class.
    Returns a numpy array of the outcome of each event for each algorithm.
    """
    # Now we loop through each event in the eventsList and run the event
    # through each of the specified algorithms.
    # we collect statistics of the number of alarms and warnings generated
    # for each event for each algorithm.
    # result[e][a][s] is the count of the number of datapoints in event e giving status s using algorithm a.

    nEvents = len(eventIdsLst)
    nAlgs = len(algs)
    nStatus = 5  # 0=OK, 1=WARNING, 2=ALARM etc.

    # Dynamic results that can expand when the device returns additional
    # per-algorithm state fields.
    slot_names = []
    slot_index = {}
    results_counts = [[] for _ in range(nEvents)]  # [event][slot][status]
    resultsStrArr = [[] for _ in range(nEvents)]   # [event][slot] -> status string

    def ensure_slot(name, fill_event_no=None, fill_len=0):
        if name in slot_index:
            return slot_index[name]
        idx = len(slot_names)
        slot_names.append(name)
        slot_index[name] = idx
        for ev in range(nEvents):
            results_counts[ev].append([0] * nStatus)
            resultsStrArr[ev].append("_")
        if fill_event_no is not None and fill_len and fill_event_no < nEvents:
            resultsStrArr[fill_event_no][idx] = "_" + ("." * int(fill_len))
        return idx

    for eventNo in range(0, nEvents):
        eventId = eventIdsLst[eventNo]
        eventObj = osd.getEvent(eventId, includeDatapoints=True)
        print("Analysing event %s (%s, userId=%s, desc=%s)" % (eventId, eventObj['type'], eventObj['userId'], eventObj['desc']))

        for algNo in range(0, nAlgs):
            alg = algs[algNo]
            baseName = algNames[algNo]
            baseIdx = ensure_slot(baseName)

            print("Processing Algorithm %d: %s (%s): " % (algNo, baseName, alg.__class__.__name__))
            alg.resetAlg()
            if alg.__class__.__name__ == 'DeviceAlg':
                sendZeroDataTransition(alg, eventId, nDatapoints=6, debug=debug)

            sys.stdout.write("Looping through Datapoints: ")
            sys.stdout.flush()
            lastDpTimeSecs = 0
            lastDpTimeStr = ''
            dpCounter = 0

            if 'datapoints' in eventObj:
                for dp in eventObj['datapoints']:
                    dpTimeStr = dp['dataTime']
                    dpTimeSecs = libosd.dpTools.dateStr2secs(dpTimeStr)
                    alarmState = libosd.dpTools.getParamFromDp('alarmState', dp)
                    if debug:
                        print("%s, %.1fs, alarmState=%d" % (dpTimeStr, dpTimeSecs - lastDpTimeSecs, alarmState))

                    # Skip manual alarm datapoint duplicates
                    if alarmState == 5:
                        if debug:
                            print("Skipping Manual Alarm datapoint (duplicate)")
                            print("alarmStatus=%s  %s, %s, %d" % (alarmState, dpTimeStr, lastDpTimeStr, (dpTimeSecs - lastDpTimeSecs)))
                        continue

                    rawDataStr = libosd.dpTools.dp2rawData(dp, debug=False)
                    if debug:
                        print("rawDataStr =", rawDataStr if rawDataStr is not None else "None")
                    if rawDataStr is None:
                        print("Invalid datapoint in event %s" % eventId)
                        continue

                    # Count this datapoint for per-alg status strings
                    dpCounter += 1

                    retVal = alg.processDp(rawDataStr, eventId)
                    retObj = json.loads(retVal)

                    # Base (voted) alarm status
                    if retObj.get('valid', True):
                        statusVal = _try_int(retObj.get('alarmState', None))
                        if statusVal is not None and 0 <= statusVal < nStatus:
                            results_counts[eventNo][baseIdx][statusVal] += 1
                            resultsStrArr[eventNo][baseIdx] = "%s%d" % (resultsStrArr[eventNo][baseIdx], statusVal)
                            sys.stdout.write("%d" % statusVal)
                        else:
                            resultsStrArr[eventNo][baseIdx] = "%s." % (resultsStrArr[eventNo][baseIdx])
                            sys.stdout.write(".")
                    else:
                        resultsStrArr[eventNo][baseIdx] = "%s." % (resultsStrArr[eventNo][baseIdx])
                        sys.stdout.write(".")

                    # Per-subalgorithm states from device JSON
                    for sub_name, sub_valid, sub_state in _iter_device_subalg_states(retObj, baseName):
                        sub_idx = ensure_slot(sub_name, fill_event_no=eventNo, fill_len=dpCounter - 1)
                        if sub_valid and sub_state is not None and 0 <= sub_state < nStatus:
                            results_counts[eventNo][sub_idx][sub_state] += 1
                            resultsStrArr[eventNo][sub_idx] = "%s%d" % (resultsStrArr[eventNo][sub_idx], sub_state)
                        else:
                            resultsStrArr[eventNo][sub_idx] = "%s." % (resultsStrArr[eventNo][sub_idx])

                    # Debug output for OSD algorithm
                    if alg.__class__.__name__ == 'OsdAlg' and debug:
                        sys.stdout.write(" - specPower=%.0f (%.0f), roiPower=%.0f (%.0f), roiRatio=%.0f (%.0f), alarmState=%.0f (%.0f)\n" % (
                            retObj['specPower'], dp['specPower'],
                            retObj['roiPower'], dp['roiPower'],
                            retObj['roiRatio'], dp['roiRatio'],
                            retObj['alarmState'], dp['alarmState']
                        ))

                    lastDpTimeSecs = dpTimeSecs
                    lastDpTimeStr = dpTimeStr
                    sys.stdout.flush()
            else:
                print("Skipping Event with no datapoints")

            sys.stdout.write("\n")
            sys.stdout.flush()
            print("Finished Algorithm %d (%s): " % (algNo, alg.__class__.__name__))
            sys.stdout.write("\n")
            sys.stdout.flush()

    results = np.array(results_counts, dtype=float)
    return results, resultsStrArr, slot_names
    

def type2index(typeStr, subTypeStr=None):
    retVal = OTHERS_INDEX
    if (typeStr.lower() == "nda"):
        retVal = NDA_INDEX
    elif (typeStr.lower() == "false alarm"):
        retVal = FALSE_INDEX
    elif (typeStr.lower() == "seizure"):
        retVal = ALL_INDEX
    return(retVal)

def saveResults2(outFileRoot, results, resultsStrArr, eventIdsLst, osd, algNames):
    print("saveResults2")
    nEvents = len(eventIdsLst)

 
    outputs = ["","","",""]
    outputs[OTHERS_INDEX] = "otherEvents"
    outputs[ALL_INDEX] = "allSeizures"
    outputs[FALSE_INDEX] = "falseAlarms"
    outputs[NDA_INDEX] = "nda"

    # Open one file for each class of event that we analyse 
    #     (TC seizures, all seizures, false alarms and NDA)
    outfLst = []
    for output in outputs:
        fname = "%s_%s.csv" % (outFileRoot, output)
        file = open(fname,"w")
        outfLst.append(file)

    # Write file headers
    lineStr = "eventId, date, type, subType, userId, datasource"
    nAlgs = results.shape[1]
    for algNo in range(0,nAlgs):
        lineStr = "%s, %s" % (lineStr, algNames[algNo])
    lineStr = "%s, reported" % lineStr
    for algNo in range(0,nAlgs):
        lineStr = "%s, %s" % (lineStr, algNames[algNo])
    lineStr = "%s, desc" % lineStr
    print(lineStr)
    for outf in outfLst:
        if outf is not None:
            outf.write(lineStr)
            outf.write("\n")

    # Zero counts for each algorithm
    NTP = np.zeros(nAlgs+1)
    NTN = np.zeros(nAlgs+1)
    NFP = np.zeros(nAlgs+1)
    NFN = np.zeros(nAlgs+1)

    # Loop through each event in turn
    #correctCount = [0] * (nAlgs+1)
    correctCount = np.zeros((len(outfLst), nAlgs+1))
    totalCount = np.zeros(len(outfLst))

    for eventNo in range(0,nEvents):
        eventId = eventIdsLst[eventNo]
        eventObj = osd.getEvent(eventId, includeDatapoints=False)
        outputIndex = type2index(eventObj['type'])
        if (eventObj['type'].lower()=="seizure"):
            expectAlarm=True
        else:
            expectAlarm=False
        totalCount[outputIndex] += 1
        lineStr = "%s, %s, %s, %s, %s" % (
            eventId, 
            eventObj['dataTime'], 
            eventObj['type'], 
            eventObj['subType'], 
            eventObj['userId'])
        if ('dataSourceName' in eventObj):
            lineStr = "%s, %s" % (lineStr, eventObj['dataSourceName'])
        else:
            lineStr = "%s, %s" % (lineStr, "unknown")


        for algNo in range(0,nAlgs):
            # Increment count of correct results
            # If the correct result is to alarm
            if (results[eventNo][algNo][2]>0):
                # The algorithm generated an alarm
                if (expectAlarm):
                    correctCount[outputIndex, algNo] += 1
                    NTP[algNo] += 1
                else:
                    NFP[algNo] += 1
            # If correct result is NOT to alarm
            if (results[eventNo][algNo][2]==0):
                if (expectAlarm):
                    NFN[algNo] += 1
                else:
                    correctCount[outputIndex, algNo] += 1
                    NTN[algNo] += 1

            # Set appropriate alarm phrase
            if results[eventNo][algNo][2] > 0:
                lineStr = "%s, ALARM" % (lineStr)
            elif results[eventNo][algNo][1] > 0:
                lineStr = "%s, WARN" % (lineStr)
            else:
                lineStr = "%s, ----" % (lineStr)

        # Record the 'as reported' result from OSD when the data was generated.
        alarmPhrases = ['----','WARN','ALARM','FALL','unused','MAN_ALARM',"NDA"]
        reportedAlarmState = getEventAlarmState(eventObj=eventObj, debug=False)
        lineStr = "%s, %s" % (lineStr, alarmPhrases[reportedAlarmState])
        if (reportedAlarmState==2):
            if (expectAlarm):
                correctCount[outputIndex, nAlgs] += 1
                NTP[nAlgs] += 1
            else:
                NFP[nAlgs] += 1
        if (reportedAlarmState!=2):
            if (expectAlarm):
                NFN[nAlgs] += 1
            else:
                correctCount[outputIndex, nAlgs] += 1
                NTN[nAlgs] += 1

        for algNo in range(0,nAlgs):
            lineStr = "%s, %s" % (lineStr, resultsStrArr[eventNo][algNo])

        lineStr = "%s, \"%s\"" % (lineStr, eventObj['desc'])
        print(lineStr)

        if outfLst[outputIndex] is not None:
            outfLst[outputIndex].write(lineStr)
            outfLst[outputIndex].write("\n")
  

    for outputIndex in range(0,len(outfLst)):
        outf = outfLst[outputIndex]
        if outf is not None:
            lineStr = "#Total, , , ,"
            for algNo in range(0,nAlgs+1):
                lineStr = "%s, %d" % (lineStr, totalCount[outputIndex])
            print(lineStr)
            outf.write(lineStr)
            outf.write("\n")
            
            lineStr = "#Correct Count, , , ,"
            for algNo in range(0,nAlgs+1):
                lineStr = "%s, %d" % (lineStr,correctCount[outputIndex, algNo])
            print(lineStr)
            outf.write(lineStr)
            outf.write("\n")

            lineStr = "#Correct Prop, , , ,"
            for algNo in range(0,nAlgs+1):
                denom = totalCount[outputIndex]
                if denom > 0:
                    prop = 1.0 * correctCount[outputIndex, algNo] / denom
                else:
                    prop = float('nan')
                lineStr = "%s, %.2f" % (lineStr, prop)
            print(lineStr)
            outf.write(lineStr)
            outf.write("\n")
            
            outf.close()
            print("Output written to file %s" % outputs[outputIndex])

    # Write summary statistics file.
    outf = open("testRunner_Summary.txt","w")
    outf.write("TestRunner Summary\n\n")
    for algNo in range(0,nAlgs+1):
        if algNo < nAlgs:
            outf.write("Algorithm %d: %s\n" % (algNo, algNames[algNo]))
        else:
            outf.write("Algorithm %d: reported\n" % algNo)
        outf.write("  NTP = %d\n" % NTP[algNo])
        outf.write("  NFP = %d\n" % NFP[algNo])
        outf.write("  NTN = %d\n" % NTN[algNo])
        outf.write("  NFN = %d\n" % NFN[algNo])
        outf.write("\n")
        if (NTP[algNo]+NFN[algNo]) > 0:
            outf.write("TPR = %.1f%%\n" % (100.*NTP[algNo]/(NTP[algNo]+NFN[algNo])))
        else:
            outf.write("TPR Not Calculated - no positive samples\n");
        if (NTN[algNo]+NFP[algNo]) > 0:
            outf.write("TNR = %.1f%%\n" % (100.*NTN[algNo]/(NTN[algNo]+NFP[algNo])))
        else:
            outf.write("TNR not calculated - no negative samples\n")
        outf.write("\n")
    outf.close()

def saveResults(outFile, results, resultsStrArr, osd, algs, algNames,
                expectAlarm=True):
    print("Displaying Results")
    eventIdsLst = osd.getEventIds()
    nEvents = len(eventIdsLst)
    print("Displaying %d Events" % nEvents)

    with open(outFile,"w") as outf:
        lineStr = "eventId, type, subType, userId"
        nAlgs = len(algs)
        for algNo in range(0,nAlgs):
            lineStr = "%s, %s" % (lineStr, algNames[algNo])
        lineStr = "%s, reported" % lineStr
        for algNo in range(0,nAlgs):
            lineStr = "%s, %s" % (lineStr, algNames[algNo])
        lineStr = "%s, desc" % lineStr
        print(lineStr)
        outf.write(lineStr)
        outf.write("\n")

        correctCount = [0] * (nAlgs+1)
        print(correctCount)
        for eventNo in range(0,nEvents):
            eventId = eventIdsLst[eventNo]
            eventObj = osd.getEvent(eventId, includeDatapoints=False)
            lineStr = "%s, %s, %s, %s" % (eventId, eventObj['type'], eventObj['subType'], eventObj['userId'])
            for algNo in range(0,nAlgs):
                # Increment count of correct results
                # If the correct result is to alarm
                if (results[eventNo][algNo][2]>0 and expectAlarm):
                    correctCount[algNo] += 1
                # If correct result is NOT to alarm
                if (results[eventNo][algNo][2]==0 and not expectAlarm):
                    correctCount[algNo] += 1

                # Set appropriate alarm phrase
                if results[eventNo][algNo][2] > 0:
                    lineStr = "%s, ALARM" % (lineStr)
                elif results[eventNo][algNo][1] > 0:
                    lineStr = "%s, WARN" % (lineStr)
                else:
                    lineStr = "%s, ----" % (lineStr)

            # Record the 'as reported' result from OSD when the data was generated.
            alarmPhrases = ['OK','WARN','ALARM','FALL','unused','MAN_ALARM',"NDA"]
            lineStr = "%s, %s" % (lineStr, alarmPhrases[eventObj['osdAlarmState']])
            if (eventObj['osdAlarmState']==2 and expectAlarm):
                correctCount[nAlgs] += 1
            if (eventObj['osdAlarmState']!=2 and not expectAlarm):
                correctCount[nAlgs] += 1

            for algNo in range(0,nAlgs):
                lineStr = "%s, %s" % (lineStr, resultsStrArr[eventNo][algNo])

            lineStr = "%s, \"%s\"" % (lineStr, eventObj['desc'])
            print(lineStr)
            outf.write(lineStr)
            outf.write("\n")

        lineStr = "#Total, ,"
        for algNo in range(0,nAlgs+1):
            lineStr = "%s, %d" % (lineStr, nEvents)
        print(lineStr)

        lineStr = "#Correct Count, ,"
        for algNo in range(0,nAlgs+1):
            lineStr = "%s, %d" % (lineStr,correctCount[algNo])
        print(lineStr)
        outf.write(lineStr)
        outf.write("\n")

        lineStr = "#Correct Prop, , , ,"
        for algNo in range(0,nAlgs+1):
            lineStr = "%s, %.2f" % (lineStr,1.*correctCount[algNo]/nEvents)
        print(lineStr)
        outf.write(lineStr)
        outf.write("\n")

    print("Output written to file %s" % outFile)



def getResultsStats(results, expectAlarm=True):
    correctPropLst = []
    #print(results.shape)
    nEvents = results.shape[0]
    nAlgs = results.shape[1]

    correctCount = [0] * (nAlgs)
    correctPropLst = [0] * (nAlgs)
    for eventNo in range(0,nEvents):
        for algNo in range(0,nAlgs):
            # Increment count of correct results
            # If the correct result is to alarm
            if (results[eventNo][algNo][2]>0 and expectAlarm):
                correctCount[algNo] += 1
            # If correct result is NOT to alarm
            if (results[eventNo][algNo][2]==0 and not expectAlarm):
                correctCount[algNo] += 1


    for algNo in range(0,nAlgs):
        correctPropLst[algNo] = 1. * correctCount[algNo] / nEvents
    
    return(nEvents, nAlgs, correctPropLst)

    

def summariseResults(tcResults, allSeizuresResults, falseAlarmResults,
                     algNames):
    print("Results Summary")

    nTcEvents, nAlgs, tcStats = getResultsStats(tcResults, True)
    nAllSeizuresEvents, nAlgs, allSeizuresStats = getResultsStats(allSeizuresResults, True)
    nFalseAlarmEvents, nAlgs, falseAlarmStats = getResultsStats(falseAlarmResults, False)
    
    nAlgs = tcResults.shape[1]

    lineStr = "Category"
    for algNo in range(0,nAlgs):
        lineStr = "%s, %s" % (lineStr, algNames[algNo])
    print(lineStr)

    lineStr = "tcSeizures"
    for algNo in range(0,nAlgs):
        lineStr = "%s, %.2f" % (lineStr, tcStats[algNo])
    print(lineStr)
    lineStr = "allSeizures"
    for algNo in range(0,nAlgs):
        lineStr = "%s, %.2f" % (lineStr, allSeizuresStats[algNo])
    print(lineStr)
    lineStr = "falseAlarms"
    for algNo in range(0,nAlgs):
        lineStr = "%s, %.2f" % (lineStr, falseAlarmStats[algNo])
    print(lineStr)
    

    

def main():
    print("testRunner.main()")
    parser = argparse.ArgumentParser(description='Seizure Detection Test Runner')
    parser.add_argument('--config', default="testConfig.json",
                        help='name of json file containing test configuration')
    #parser.add_argument('--out', default="output",
    #                    help='name of output CSV file')
    parser.add_argument('--debug', action="store_true",
                        help='Write debugging information to screen')
    argsNamespace = parser.parse_args()
    args = vars(argsNamespace)
    print(args)


    configObj = libosd.configUtils.loadConfig(args['config'])
    print("configObj=",configObj)
    # Load a separate OSDB Configuration file if it is included.
    if ("osdbCfg" in configObj):
        osdbCfgFname = libosd.configUtils.getConfigParam("osdbCfg",configObj)
        print("Loading separate OSDB Configuration File %s." % osdbCfgFname)
        osdbCfgObj = libosd.configUtils.loadConfig(osdbCfgFname)
        # Merge the contents of the OSDB Configuration file into configObj
        configObj = configObj | osdbCfgObj

    print("configObj=",configObj)

    runTest(configObj, args['debug'], configPath=args.get('config'))
    


if __name__ == "__main__":
    main()

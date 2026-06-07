"""io_utils.py – Data loading utilities for testRunner.

Handles:
- Loading OSDB JSON event files via OsdDbConnection
- Loading flattened CSV files produced by flattenData.py
- Filtering out training events to prevent train/test contamination
"""
import os
import sys
import csv
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libosd.osdDbConnection
import libosd.dpTools


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------

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


def _clean_csv_header(header):
    """Normalise CSV header names.

    Some generated CSVs may contain embedded newlines in quoted headers
    (e.g. "M0\\n00"), which breaks downstream column lookup.
    """
    if header is None:
        return header
    if not isinstance(header, str):
        header = str(header)
    return header.replace('\ufeff', '').replace('\r', '').replace('\n', '').strip()


# ---------------------------------------------------------------------------
# Training-event exclusion
# ---------------------------------------------------------------------------

def _load_training_event_ids_from_csv(csv_path):
    """Return a set of event IDs (as strings) from a flattened trainData CSV."""
    event_ids = set()
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader, None)
        if headers is None:
            return event_ids
        headers = [_clean_csv_header(h) for h in headers]

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

    if isinstance(obj, dict):
        if 'events' in obj and isinstance(obj['events'], list):
            events = obj['events']
        else:
            raise ValueError(
                f"Unsupported JSON structure in {json_path}: "
                "expected a list or dict with 'events' list"
            )
    elif isinstance(obj, list):
        events = obj
    else:
        raise ValueError(
            f"Unsupported JSON structure in {json_path}: expected a list of events"
        )

    for ev_obj in events:
        if not isinstance(ev_obj, dict):
            continue
        ev_id = ev_obj.get('id', None)
        if ev_id is None:
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
        raise ValueError(
            f"excludeTrainingEvents must be a .csv or .json file: {resolved}"
        )

    if debug:
        print(f"Loaded {len(ids)} unique training event IDs")
    return ids


def exclude_training_events_from_osd(osd, train_data_path, search_dirs=None,
                                     debug=False):
    """Remove any events from `osd` whose IDs appear in the training file.

    Returns a list of event IDs (in the database's native type) that were removed.
    """
    training_ids = load_training_event_ids(
        train_data_path, search_dirs=search_dirs, debug=debug
    )
    if not training_ids:
        return []

    current_ids = osd.getEventIds()
    remove_ids = [ev_id for ev_id in current_ids if str(ev_id) in training_ids]

    if remove_ids:
        if debug:
            print(f"Removing {len(remove_ids)} events used in training")
        osd.removeEvents(remove_ids)
    return remove_ids


# ---------------------------------------------------------------------------
# CSV event loading
# ---------------------------------------------------------------------------

def csvRowToEvent(row, headers, eventId, eventsByIdDict, debug=False):
    """Convert a CSV row (produced by flattenData.py) to a datapoint object
    and add it to the event in eventsByIdDict.

    If the event doesn't exist yet, it will be created.
    """
    rowDict = {headers[i]: row[i] for i in range(min(len(headers), len(row)))}

    if eventId not in eventsByIdDict:
        try:
            typeStr = rowDict.get('typeStr', 'Unknown')
            if typeStr.startswith('"') and typeStr.endswith('"'):
                typeStr = typeStr[1:-1]
            if '/' in typeStr:
                eventType, subType = typeStr.split('/', 1)
            else:
                eventType = typeStr
                subType = ''

            dataTime = rowDict.get('dataTime', '')
            eventsByIdDict[eventId] = {
                'id': eventId,
                'userId': rowDict.get('userId', ''),
                'type': eventType,
                'subType': subType,
                'dataTime': dataTime,
                'desc': f"Event {eventId} ({eventType})",
                'datapoints': []
            }
        except Exception as e:
            print(f"Error creating event from CSV row: {e}", file=sys.stderr)
            return

    try:
        dataTime   = rowDict.get('dataTime', '')
        alarmState = int(float(rowDict.get('osdAlarmState', 0)))
        specPower  = float(rowDict.get('osdSpecPower', 0))
        roiPower   = float(rowDict.get('osdRoiPower', 0))

        hr = rowDict.get('hr', '')
        hr = int(float(hr)) if (hr and hr != '') else -1
        o2sat = rowDict.get('o2sat', '')
        o2sat = int(float(o2sat)) if (o2sat and o2sat != '') else -1

        rawData = []
        for n in range(125):
            colName = f"M{n:03d}"
            val = rowDict.get(colName, '')
            if val is None or val == '':
                val = rowDict.get(f"M{n:03d}_t-0", '')
            rawData.append(float(val) if (val and val != '') else None)

        rawData3D = []
        for n in range(125):
            for axis in ["X", "Y", "Z"]:
                val = rowDict.get(f"{axis}{n:03d}", '')
                rawData3D.append(float(val) if (val and val != '') else None)

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
    """Load a CSV file produced by flattenData.py and convert it to a dict of
    event objects, similar to what would be loaded from JSON files.

    Returns:
        Dictionary mapping eventId → event object
    """
    eventsByIdDict = {}

    if not os.path.exists(csvFname):
        print(f"Error: CSV file {csvFname} does not exist", file=sys.stderr)
        return eventsByIdDict

    with open(csvFname, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        headers = [_clean_csv_header(h) for h in headers]

        for row in reader:
            if not row or all(v == '' for v in row):
                continue
            eventId = row[0]
            csvRowToEvent(row, headers, eventId, eventsByIdDict, debug=debug)

    if debug:
        print(f"Loaded {len(eventsByIdDict)} events from CSV file {csvFname}")
        for eventId, event in eventsByIdDict.items():
            print(f"  Event {eventId}: {len(event.get('datapoints', []))} datapoints")

    return eventsByIdDict


# ---------------------------------------------------------------------------
# Unified loader (JSON or CSV)
# ---------------------------------------------------------------------------

def loadDataFiles(dataFiles, dbDir=None, debug=False):
    """Load data files – either OSDB JSON or CSV files from flattenData.py.

    Returns:
        OsdDbConnection object with all events loaded.
    """
    osd = libosd.osdDbConnection.OsdDbConnection(cacheDir=dbDir, debug=debug)

    for fname in dataFiles:
        if fname.lower().endswith('.csv'):
            filePath = os.path.join(dbDir, fname) if (dbDir and not os.path.isabs(fname)) else fname
            if debug:
                print(f"Loading CSV file: {filePath}")
            csvEvents = loadCsvFile(filePath, debug=debug)
            for eventId, eventObj in csvEvents.items():
                osd.addEvent(eventObj)
            print(f"loaded {len(csvEvents)} events from CSV file {filePath}")
        else:
            filePath = fname
            useCacheDir = not os.path.isabs(fname)
            if debug:
                print(f"Loading JSON file: {filePath} (useCacheDir={useCacheDir})")
            eventsObjLen = osd.loadDbFile(filePath, useCacheDir=useCacheDir)
            print(f"loaded {eventsObjLen} events from file {filePath}")

    return osd

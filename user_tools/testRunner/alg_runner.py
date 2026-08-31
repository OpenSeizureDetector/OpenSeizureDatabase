"""alg_runner.py – Algorithm execution helpers for testRunner.

Provides:
- Event-level alarm state inspection
- Zero-data transition helper (for streaming algorithms)
- Main ``testEachEvent`` loop that drives events through algorithm instances
- Checkpoint/resume support for crash recovery
"""
import sys
import json
import numpy as np
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libosd.dpTools


# ---------------------------------------------------------------------------
# Checkpoint/Resume Support
# ---------------------------------------------------------------------------

def save_checkpoint(checkpoint_path, eventNo, nEvents, results_counts, resultsStrArr,
                    slot_names, perDpDataLst, nAlgs, nStatus):
    """Save checkpoint data to resume test after a crash."""
    checkpoint_data = {
        'eventNo': eventNo,
        'nEvents': nEvents,
        'nAlgs': nAlgs,
        'nStatus': nStatus,
        'slot_names': slot_names,
        'results_counts': [[[int(results_counts[e][s][st]) 
                            for st in range(nStatus)] 
                           for s in range(len(results_counts[e]))]
                          for e in range(len(results_counts))],
        'resultsStrArr': resultsStrArr,
        'perDpDataLst': perDpDataLst,
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    print(f"Checkpoint saved at event {eventNo}/{nEvents}")


def load_checkpoint(checkpoint_path):
    """Load checkpoint data to resume from a previous run."""
    if not os.path.exists(checkpoint_path):
        return None
    
    try:
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        print(f"Checkpoint loaded: resuming from event {checkpoint_data['eventNo']}/{checkpoint_data['nEvents']}")
        return checkpoint_data
    except Exception as e:
        print(f"Warning: Failed to load checkpoint: {e}")
        return None


# ---------------------------------------------------------------------------
# Progress and ETA reporting
# ---------------------------------------------------------------------------

def format_event_desc(eventObj):
    """Format event description, removing redundant 'Event xxx' prefix."""
    desc = eventObj.get('desc', '')
    # Remove "Event XXXXX (Type)" pattern at the start if present
    import re
    desc = re.sub(r'Event\s+\d+\s*\(\w+\)\s*', '', desc).strip()
    return desc


def print_event_progress(eventNo, nEvents, eventObj, startTime):
    """Print event analysis with progress, ETA, and better description."""
    eventId = eventObj.get('eventId', eventObj.get('id', 'unknown'))
    eventType = eventObj.get('type', 'unknown')
    subType = eventObj.get('subType', '')
    userId = eventObj.get('userId', 'unknown')
    desc = format_event_desc(eventObj)
    
    # Calculate progress and ETA
    elapsed = time.time() - startTime
    eventsProcessed = eventNo - (0 if eventNo == 0 else eventNo)  # Current event count
    if eventNo > 0:
        avgTimePerEvent = elapsed / (eventNo)
        remainingEvents = nEvents - eventNo
        estimatedRemaining = avgTimePerEvent * remainingEvents
        eta_str = f" (ETA: {int(estimatedRemaining // 60)}m {int(estimatedRemaining % 60)}s)"
    else:
        eta_str = ""
    
    # Format type/subType
    typeStr = eventType
    if subType:
        typeStr = f"{eventType}/{subType}"
    
    # Format description with clean output
    if desc:
        desc_str = f", {desc}"
    else:
        desc_str = ""
    
    print("Event %d/%d: %s (userId=%s%s)%s%s" % (
        eventNo, nEvents, eventId, userId, desc_str, 
        f" [{typeStr}]" if typeStr else "", eta_str))



# ---------------------------------------------------------------------------
# Event helpers
# ---------------------------------------------------------------------------

def getEventVal(eventObj, elemId):
    if elemId in eventObj.keys():
        return eventObj[elemId]
    return None


def getEventAlarmState(eventObj, debug=False):
    """Scan through the datapoints and return the highest (non-manual) alarm
    state reached over the course of the event.
    """
    maxAlarmState = 0
    if 'datapoints' in eventObj:
        for dp in eventObj['datapoints']:
            dpTimeSecs = libosd.dpTools.dateStr2secs(dp['dataTime'])
            alarmState = libosd.dpTools.getParamFromDp('alarmState', dp)
            if alarmState == 1 and maxAlarmState == 0:
                maxAlarmState = 1
            if alarmState == 2:
                maxAlarmState = 2
    return maxAlarmState


# ---------------------------------------------------------------------------
# Zero-data transition (flush streaming algorithm buffers between events)
# ---------------------------------------------------------------------------

def createZeroDatapoint():
    """Create a datapoint with all-zero acceleration data."""
    return {
        'dataTime': '2000-01-01T00:00:00Z',
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
    """Send ``nDatapoints`` zero-acceleration datapoints to ``alg`` so that
    its internal state is cleanly flushed before the next event.

    This is particularly important for streaming algorithms (e.g. DeviceAlg).
    """
    if debug:
        print(f"Sending {nDatapoints} zero datapoints to "
              f"{alg.__class__.__name__} for event transition")

    for _ in range(nDatapoints):
        zeroDatapoint = createZeroDatapoint()
        rawDataStr = libosd.dpTools.dp2rawData(zeroDatapoint, debug=False)
        if rawDataStr is not None:
            try:
                alg.processDp(rawDataStr, eventId)
                if debug:
                    sys.stdout.write("Z")
            except Exception as e:
                if debug:
                    print(f"Warning: Error processing zero datapoint: {e}")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
    """Yield (slot_name, valid, state_int) for sub-algorithm states in a
    device /data JSON response.
    """
    if not isinstance(retObj, dict):
        return

    # Scalar per-algorithm states: osdAlgState, flapAlgState, cnnAlgState, …
    for key, val in retObj.items():
        if not isinstance(key, str):
            continue
        if not key.endswith('AlgState'):
            continue
        state = _try_int(val)
        yield (f"{baseAlgName}.{key}", True, state)

    # ML models reported as parallel arrays
    model_names  = retObj.get('mlModelNames', None)
    model_states = retObj.get('mlModelStates', None)
    model_active = retObj.get('mlModelActive', None)

    if isinstance(model_names, list) and isinstance(model_states, list):
        for i in range(min(len(model_names), len(model_states))):
            name   = str(model_names[i])
            active = True
            if isinstance(model_active, list) and i < len(model_active):
                active = bool(model_active[i])
            state = _try_int(model_states[i])
            yield (f"{baseAlgName}.ml.{name}", bool(active), state)


def _extract_alg_metric(retObj):
    """Extract the most informative scalar metric from an algorithm result dict.

    Priority: pSeizure (ML) → roiRatio (computed or direct) → alarmRatio.

    Returns ``(value, label_string)`` or ``(None, None)`` if unavailable.
    """
    p = retObj.get('pSeizure', None)
    if p is not None:
        try:
            return float(p), 'pSeizure (0-1)'
        except (TypeError, ValueError):
            pass

    rp = retObj.get('roiPower', None)
    sp = retObj.get('specPower', None)
    if rp is not None and sp is not None:
        try:
            rp_f, sp_f = float(rp), float(sp)
            if sp_f > 0:
                return rp_f / sp_f, 'roiRatio'
        except (TypeError, ValueError):
            pass

    rr = retObj.get('roiRatio', None)
    if rr is not None:
        try:
            return float(rr), 'roiRatio'
        except (TypeError, ValueError):
            pass

    ar = retObj.get('alarmRatio', None)
    if ar is not None:
        try:
            return float(ar), 'alarmRatio'
        except (TypeError, ValueError):
            pass

    return None, None


# ---------------------------------------------------------------------------
# Main algorithm test loop
# ---------------------------------------------------------------------------

def testEachEvent(eventIdsLst, osd, algs, algNames, debug=False, checkpoint_path=None):
    """Run each event through each algorithm and collect results.

    For each event/algorithm pair, counts the number of datapoints that
    produced each alarm status (0=OK, 1=WARNING, 2=ALARM, …).

    Also collects per-datapoint data (acceleration, algorithm metrics, alarm
    states) for the summary report.

    Args:
        eventIdsLst: list of event IDs to process
        osd: OSDB object with event data
        algs: list of algorithm instances
        algNames: list of algorithm names
        debug: enable debug output
        checkpoint_path: optional path to checkpoint file for resume support

    Returns:
        results (np.ndarray): shape [nEvents, nSlots, nStatus]
        resultsStrArr (list): [event][slot] status history string
        slot_names (list): ordered slot labels
        perDpDataLst (list): per-event dict of raw plot data
    """
    nEvents = len(eventIdsLst)
    nAlgs   = len(algs)
    nStatus = 5   # 0=OK, 1=WARNING, 2=ALARM, …

    slot_names    = []
    slot_index    = {}
    results_counts = [[] for _ in range(nEvents)]
    resultsStrArr  = [[] for _ in range(nEvents)]
    perDpDataLst   = []
    
    # Try to load checkpoint for resume
    startEventNo = 0
    if checkpoint_path:
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint:
            startEventNo = checkpoint['eventNo']
            # Restore state from checkpoint
            slot_names = checkpoint['slot_names']
            slot_index = {name: i for i, name in enumerate(slot_names)}
            resultsStrArr = checkpoint['resultsStrArr']
            perDpDataLst = checkpoint['perDpDataLst']
            
            # Restore results_counts
            results_counts_data = checkpoint['results_counts']
            for eventNo in range(len(results_counts_data)):
                results_counts[eventNo] = [row[:] for row in results_counts_data[eventNo]]
            
            # Initialize empty slots for remaining events
            for eventNo in range(len(results_counts)):
                while len(results_counts[eventNo]) < len(slot_names):
                    results_counts[eventNo].append([0] * nStatus)
                while len(resultsStrArr[eventNo]) < len(slot_names):
                    resultsStrArr[eventNo].append("_")

    def ensure_slot(name, fill_event_no=None, fill_len=0):
        if name in slot_index:
            return slot_index[name]
        idx = len(slot_names)
        slot_names.append(name)
        slot_index[name] = idx
        for ev in range(nEvents):
            if ev < len(results_counts):
                results_counts[ev].append([0] * nStatus)
            if ev < len(resultsStrArr):
                resultsStrArr[ev].append("_")
        if fill_event_no is not None and fill_len and fill_event_no < nEvents:
            if fill_event_no < len(resultsStrArr) and idx < len(resultsStrArr[fill_event_no]):
                resultsStrArr[fill_event_no][idx] = "_" + ("." * int(fill_len))
        return idx

    # Track start time for ETA calculations
    startTime = time.time()

    for eventNo in range(startEventNo, nEvents):
        eventId  = eventIdsLst[eventNo]
        eventObj = osd.getEvent(eventId, includeDatapoints=True)
        
        # Print progress with ETA (1-based event numbering for display)
        print_event_progress(eventNo + 1, nEvents, eventObj, startTime)

        # ---- Pre-collect per-event data for the summary report ----
        try:
            _eventStartSecs = libosd.dpTools.dateStr2secs(eventObj['dataTime'])
        except Exception:
            _eventStartSecs = None

        eventPerDpData = {
            'eventId': eventId,
            'timestamps': [],
            'accelMag': [],
            'dpTimestamps': [],
            'reportedAlarmStates': [],
            'algOutputs': {
                algName: {'alarmStates': [], 'metrics': [], 'metricName': None}
                for algName in algNames
            }
        }

        if 'datapoints' in eventObj and _eventStartSecs is not None:
            for _dp in eventObj['datapoints']:
                _as  = libosd.dpTools.getParamFromDp('alarmState', _dp)
                _rds = libosd.dpTools.dp2rawData(_dp, debug=False)
                if _as == 5 or _rds is None:
                    continue
                _dpTs = libosd.dpTools.dateStr2secs(_dp['dataTime']) - _eventStartSecs
                eventPerDpData['dpTimestamps'].append(_dpTs)
                eventPerDpData['reportedAlarmStates'].append(
                    int(_as) if _as is not None else 0)
                for _n, _v in enumerate(_dp.get('rawData', [])):
                    if _v is not None:
                        eventPerDpData['timestamps'].append(_dpTs + _n / 25.0)
                        eventPerDpData['accelMag'].append(float(_v))
        # ---- End pre-collection ----

        for algNo in range(nAlgs):
            alg      = algs[algNo]
            baseName = algNames[algNo]
            baseIdx  = ensure_slot(baseName)

            print("Processing Algorithm %d: %s (%s): " % (
                algNo, baseName, alg.__class__.__name__))
            print("Resetting algorithm state for new event")
            alg.resetAlg()
            if alg.__class__.__name__ == 'DeviceAlg':
                print("Sending zero-data transition to flush DeviceAlg state")
                sendZeroDataTransition(alg, eventId, nDatapoints=6, debug=debug)

            sys.stdout.write("Looping through Datapoints: ")
            sys.stdout.flush()
            lastDpTimeSecs = 0
            lastDpTimeStr  = ''
            dpCounter      = 0

            if 'datapoints' in eventObj:
                for dp in eventObj['datapoints']:
                    dpTimeStr  = dp['dataTime']
                    dpTimeSecs = libosd.dpTools.dateStr2secs(dpTimeStr)
                    alarmState = libosd.dpTools.getParamFromDp('alarmState', dp)
                    if debug:
                        print("%s, %.1fs, alarmState=%d" % (
                            dpTimeStr, dpTimeSecs - lastDpTimeSecs, alarmState))

                    if alarmState == 5:
                        if debug:
                            print("Skipping Manual Alarm datapoint (duplicate)")
                        continue

                    rawDataStr = libosd.dpTools.dp2rawData(dp, debug=False)
                    if debug:
                        print("rawDataStr =",
                              rawDataStr if rawDataStr is not None else "None")
                    if rawDataStr is None:
                        print("Invalid datapoint in event %s" % eventId)
                        continue

                    dpCounter += 1
                    retVal = alg.processDp(rawDataStr, eventId)
                    retObj = json.loads(retVal)

                    # Collect per-dp algorithm output for the report
                    _algOut = eventPerDpData['algOutputs'].get(baseName)
                    if _algOut is not None:
                        _algOut['alarmStates'].append(
                            _try_int(retObj.get('alarmState', 0)) or 0)
                        _metric, _metricName = _extract_alg_metric(retObj)
                        _algOut['metrics'].append(_metric)
                        if _metricName and not _algOut['metricName']:
                            _algOut['metricName'] = _metricName

                    # Base (voted) alarm status
                    if retObj.get('valid', True):
                        statusVal = _try_int(retObj.get('alarmState', None))
                        if statusVal is not None and 0 <= statusVal < nStatus:
                            results_counts[eventNo][baseIdx][statusVal] += 1
                            resultsStrArr[eventNo][baseIdx] += str(statusVal)
                            sys.stdout.write("%d" % statusVal)
                        else:
                            resultsStrArr[eventNo][baseIdx] += "."
                            sys.stdout.write(".")
                    else:
                        resultsStrArr[eventNo][baseIdx] += "."
                        sys.stdout.write(".")

                    # Per-sub-algorithm states from device JSON
                    for sub_name, sub_valid, sub_state in \
                            _iter_device_subalg_states(retObj, baseName):
                        sub_idx = ensure_slot(sub_name,
                                              fill_event_no=eventNo,
                                              fill_len=dpCounter - 1)
                        if sub_valid and sub_state is not None \
                                and 0 <= sub_state < nStatus:
                            results_counts[eventNo][sub_idx][sub_state] += 1
                            resultsStrArr[eventNo][sub_idx] += str(sub_state)
                        else:
                            resultsStrArr[eventNo][sub_idx] += "."

                    if alg.__class__.__name__ == 'OsdAlg' and debug:
                        sys.stdout.write(
                            " - specPower=%.0f (%.0f), roiPower=%.0f (%.0f), "
                            "roiRatio=%.0f (%.0f), alarmState=%.0f (%.0f)\n" % (
                                retObj['specPower'], dp['specPower'],
                                retObj['roiPower'],  dp['roiPower'],
                                retObj['roiRatio'],  dp['roiRatio'],
                                retObj['alarmState'], dp['alarmState']
                            ))

                    lastDpTimeSecs = dpTimeSecs
                    lastDpTimeStr  = dpTimeStr
                    sys.stdout.flush()
            else:
                print("Skipping Event with no datapoints")

            sys.stdout.write("\n")
            sys.stdout.flush()
            print("Finished Algorithm %d (%s): " % (algNo, alg.__class__.__name__))
            sys.stdout.write("\n")
            sys.stdout.flush()

        perDpDataLst.append(eventPerDpData)
        
        # Save checkpoint after each event completes
        if checkpoint_path:
            save_checkpoint(checkpoint_path, eventNo + 1, nEvents, results_counts, 
                          resultsStrArr, slot_names, perDpDataLst, nAlgs, nStatus)

    results = np.array(results_counts, dtype=float)
    return results, resultsStrArr, slot_names, perDpDataLst

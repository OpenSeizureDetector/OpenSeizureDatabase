#!/usr/bin/env python3
"""
flattenData.py - Convert OpenSeizureDatabase (OSDB) JSON events to CSV format for neural network training.

OVERVIEW
--------
This module converts seizure events from JSON format (as stored in OSDB) to CSV format suitable for
training neural networks (specifically CNN-LSTM models). Each event is decomposed into individual
datapoints, with optional filtering based on seizure time windows.

KEY FEATURES
------------

1. JSON to CSV Conversion:
   - Reads event objects from JSON format
   - Extracts datapoints (5-second accelerometer windows) from events
   - Converts to CSV rows for machine learning pipeline
   - Supports both 1D (magnitude) and 3D (X/Y/Z channels) acceleration data

2. SeizureTimes Constraint (Configurable):
   - Optional filtering of datapoints based on seizureTimes
   - SeizureTimes are offsets (in seconds) from the earliest datapoint's END time
   - Enables models to focus on seizure-related data or include pre-seizure context
   - Default seizureTimes [-30, 30] covers 30s before to 30s after earliest datapoint
   - Configurable margin extends window for LSTM temporal context

 3. Data Validation & Gap Handling (Optional):
    - Can validate datapoints for gaps and missing data
    - Gaps are NEVER filled with synthetic data: missing spans are left as
      time discontinuities in the rows (if validate=True). Downstream rolling
      buffers do NOT restart at the discontinuity - the next valid datapoint
      concatenates onto the buffer, matching the production device, so model
      windows may span a gap exactly as they do on-device.

CONFIGURATION PARAMETERS
------------------------

config dict (passed to flattenOsdb() and process_event_obj()):

  useSeizureTimesConstraint: bool (default: False)
    - Enable/disable filtering based on seizureTimes
    - When True, only datapoints within seizure window are included
    - When False (default), all valid datapoints are included (backward compatible)

  seizureTimeMarginSeconds: float (default: 10.0)
    - Extends the seizure time window before and after for context
    - Window becomes [seizure_start - margin, seizure_end + margin]
    - Useful for LSTM models that need temporal context
    - Example: seizureTimes=[0, 10] with margin=5 gives window [-5, 15]

  dcNormalisation: bool (default: False)
    - Rescale surrogate-magnitude events (early firmware |x|+|y|+|z| baselines
      > threshold mg, default 1100 mg) back to a ~1000 mg still level, matching
      newer true-vector-magnitude data. Events at/below the threshold are left
      untouched, so current on-device data (1 g ~= 1000 mg offset) needs no
      runtime transform.

  dcNormalisationSurrogateThreshold: float (default: 1100.0)
    - Event still-level (lower-quartile of per-datapoint M means, mg) above
      which the surrogate rescale triggers.
    - Deliberately 1100 mg, not exactly 1000 mg: healthy modern recordings sit
      at 1000 mg +/- device bias (~tens of mg), so a 1000 mg trigger would
      pointlessly rescale normal events. Surrogate |x|+|y|+|z| still levels
      span ~1000..1732 mg by orientation; levels already near 1000 mg need no
      correction anyway.

SEIZURETIME SEMANTICS
---------------------

SeizureTimes Format:
  - [start_offset, end_offset] in seconds
  - Offsets are relative to the EARLIEST DATAPOINT'S END TIME in the event
  - Start offset can be negative (includes data before earliest datapoint)

Example:
  If earliest datapoint ends at 2022-01-01 12:00:10 and seizureTimes=[0, 20]:
    - Seizure window is [2022-01-01 12:00:10, 2022-01-01 12:00:30]
    - Datapoint at 12:00:15 (within window) → INCLUDED
    - Datapoint at 12:00:35 (outside window) → EXCLUDED

  If seizureTimes=[-30, 30] (default when missing):
    - Window is [-30s, +30s] from earliest_dt_end
    - Covers 60-second window: 30s before to 30s after earliest datapoint
    - Useful for capturing pre-seizure patterns and post-seizure effects

DATA FORMAT
-----------

Input (JSON):
  Event object with:
    - id: Event ID
    - userId: User ID
    - type: Event type (e.g., 'Seizure')
    - subType: Event subtype (e.g., 'Tonic-Clonic')
    - seizureTimes: [start_offset, end_offset] (optional, uses default if missing)
    - datapoints: List of datapoint objects, each with:
      - dataTime: ISO 8601 timestamp of datapoint END (UTC)
      - rawData: List of 125 magnitude values (1 per 200ms @ 25Hz sampling)
      - rawData3D: List of 375 values (X, Y, Z: 125 each)
      - alarmState, hr, o2Sat, etc.

Output (CSV rows):
  Each row contains (510 values total):
    - Header fields (10): eventId, userId, typeStr, type, dataTime, alarmState, specPower, roiPower, hr, o2Sat
    - Raw acceleration data (125 + 375 = 500):
      - rawData (125): Magnitude values
      - rawData3D (375): X, Y, Z channel data

DATA INTEGRITY GUARANTEES
-------------------------

- Full Row Inclusion: When a datapoint overlaps with seizure window, the ENTIRE row is included
  (no partial inclusion or truncation of acceleration data)
- Data Completeness: Each output row always contains full 125 rawData + 375 rawData3D values
- No Data Corruption: Filtering is based on temporal overlap only; data values are never modified

USAGE
-----

As a module:
  from user_tools.nnTraining2 import flattenData
  
  config = {
    'useSeizureTimesConstraint': True,
    'seizureTimeMarginSeconds': 10
  }
  rows = flattenData.process_event_obj(event, validate=False, config=config)

Command line:
  python flattenData.py -i events.json -o training_data.csv --debug

"""

import argparse
import sys
import os
import statistics
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libosd.osdDbConnection
import libosd.configUtils
import json

def type2id(typeStr):
    ''' convert the event type string typeStr into an integer representing the high level event type.
        false alarm or nda = 0
        seizure = 1
        other type = 2
        '''
    if typeStr.lower() == "seizure":
        id = 1
    elif typeStr.lower() == "false alarm":
        id = 0
    elif typeStr.lower() == "nda":
        id = 0
    else:
        id = 2
    return id


def parse_datatime(datatime_str):
    """
    Robustly parse dataTime string with multiple format support.
    
    Args:
        datatime_str: String representation of datetime
    
    Returns:
        datetime object or None if parsing fails
    """
    if not datatime_str:
        return None
    
    # Try ISO 8601 format first (most common in modern systems)
    # Handle both with and without 'Z' suffix, and with/without microseconds
    iso_formats = [
        "%Y-%m-%dT%H:%M:%SZ",       # ISO 8601 with Z (UTC): 2022-05-09T14:30:00Z
        "%Y-%m-%dT%H:%M:%S",        # ISO 8601 without Z: 2022-05-09T14:30:00
        "%Y-%m-%dT%H:%M:%S.%fZ",    # ISO 8601 with microseconds and Z
        "%Y-%m-%dT%H:%M:%S.%f",     # ISO 8601 with microseconds
    ]
    
    for fmt in iso_formats:
        try:
            return datetime.strptime(datatime_str, fmt)
        except (ValueError, TypeError):
            continue
    
    # Common legacy formats
    formats = [
        "%d-%m-%Y %H:%M:%S",      # DD-MM-YYYY HH:MM:SS
        "%Y-%m-%d %H:%M:%S",      # YYYY-MM-DD HH:MM:SS
        "%d/%m/%Y %H:%M:%S",      # DD/MM/YYYY HH:MM:SS
        "%Y/%m/%d %H:%M:%S",      # YYYY/MM/DD HH:MM:SS
        "%d-%m-%Y %H:%M:%S.%f",   # With milliseconds
        "%Y-%m-%d %H:%M:%S.%f",   # With milliseconds
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(datatime_str, fmt)
        except (ValueError, TypeError):
            continue
    
    return None


def _prepare_seizure_time_constraint(eventObj, valid_datapoints, config=None, debug=False):
    """
    Prepare seizure time constraint information for filtering datapoints.
    
    The seizureTimes in the event are offsets (in seconds) from the event start time.
    The event start time is determined from the earliest datapoint's timestamp minus
    the datapoint duration (5 seconds).
    
    Args:
        eventObj (dict): Event object that may contain 'seizureTimes' field
        valid_datapoints (list): List of (datetime, dp_dict) tuples, sorted by time
        config (dict): Configuration dict with 'useSeizureTimesConstraint' and 'seizureTimeMarginSeconds'
        debug (bool): Enable debug logging
    
    Returns:
        tuple: (constraint_active, event_start_dt, seizure_start_s, seizure_end_s, margin_s)
               or (False, None, None, None, None) if constraint not active
    """
    # Check if constraint is enabled
    if not config or not config.get('useSeizureTimesConstraint', False):
        return (False, None, None, None, None)
    
    # Get seizureTimes, using default if not provided
    seizure_times = eventObj.get('seizureTimes', None)
    if not seizure_times or len(seizure_times) < 2:
        # Use default: [-30, 30] means 30 seconds before and after event start
        seizure_times = [-30.0, 30.0]
        if debug:
            print(f"  [seizureTimes constraint] Event {eventObj.get('id')} has no seizureTimes - using default {seizure_times}")
    
    seizure_start_s = float(seizure_times[0])
    seizure_end_s = float(seizure_times[1])
    margin_s = float(config.get('seizureTimeMarginSeconds', 0))
    
    # We need the event start time to convert absolute datetimes to relative offsets
    # This will be set later if we're in validated path, or calculated in non-validated path
    if debug:
        print(f"  [seizureTimes constraint] Event {eventObj.get('id')}: seizureTimes=[{seizure_start_s}, {seizure_end_s}], margin={margin_s}s")
    
    return (True, None, seizure_start_s, seizure_end_s, margin_s)


def _calculate_event_start_dt(earliest_dt_end, config=None):
    """
    Calculate the event start datetime from the earliest datapoint end time.
    
    Since we don't have the actual event start time, we make a reasonable assumption:
    the event started well before the first datapoint was collected. This ensures
    that seizureTimes offsets (like [10, 20]) are interpreted correctly.
    
    Args:
        earliest_dt_end (datetime): End time of the earliest datapoint
        config (dict): Configuration dict (may include event start offset assumptions)
    
    Returns:
        datetime: The calculated event start time (start of first datapoint's 5-sec window)
    """
    # Earliest datapoint spans [earliest_dt_end - 5s, earliest_dt_end]
    # So event effectively starts 5 seconds before the earliest datapoint end
    event_start_dt = earliest_dt_end - timedelta(milliseconds=5000)
    
    return event_start_dt


def _datapoint_in_seizure_window(dt_end, earliest_dt_end, seizure_start_s, seizure_end_s, margin_s):
    """
    Check if a datapoint overlaps with the seizure time window.
    
    The seizureTimes are defined as offsets (seconds) relative to the earliest
    datapoint's END time. This function checks if the datapoint has any overlap
    with the seizure time window (including margin).
    
    Args:
        dt_end (datetime): End time of the datapoint to check
        earliest_dt_end (datetime): End time of the earliest datapoint in the event
        seizure_start_s (float): Seizure start offset in seconds (relative to earliest_dt_end)
        seizure_end_s (float): Seizure end offset in seconds (relative to earliest_dt_end)
        margin_s (float): Margin in seconds to extend window before/after seizure
    
    Returns:
        bool: True if datapoint overlaps with [earliest_dt_end + seizure_start - margin, 
              earliest_dt_end + seizure_end + margin]
    """
    # Calculate datapoint start time (5-second window)
    DATAPOINT_DURATION_MS = 125 * 40.0  # 5000ms = 5 seconds
    dt_start = dt_end - timedelta(milliseconds=DATAPOINT_DURATION_MS)
    
    # Calculate absolute seizure window in datetime
    # seizureTimes are offsets from the earliest datapoint END time
    seizure_start_abs = earliest_dt_end + timedelta(seconds=seizure_start_s - margin_s)
    seizure_end_abs = earliest_dt_end + timedelta(seconds=seizure_end_s + margin_s)
    
    # Check for overlap between datapoint [dt_start, dt_end] and seizure window
    # Intervals [a1, a2] and [b1, b2] overlap if a1 < b2 AND a2 > b1
    # Use >= for the end points to include boundary touches
    return dt_start < seizure_end_abs and dt_end >= seizure_start_abs


def dp2row(ev, dp, header=False):
    '''Convert event and datapoint to a flat row for CSV.'''
    rowLst = []
    if header:
        rowLst = [
            "eventId", "userId", "typeStr", "type", "dataTime", "osdAlarmState",  "osdSpecPower", "osdRoiPower", "hr", "o2sat"
        ]
        # 1D acceleration
        for n in range(125):
            rowLst.append(f"M{n:03d}")
        # 3D acceleration
        for axis in ["X", "Y", "Z"]:
            for n in range(125):
                rowLst.append(f"{axis}{n:03d}")
        return rowLst

    # Data row
    # Normalize eventId to string to handle both int and string IDs consistently
    rowLst.append(str(ev.get('id', '')))
    rowLst.append(ev.get('userId', ''))
    rowLst.append('"%s/%s"' % (ev.get('type', ''), ev.get('subType', '')))
    rowLst.append(type2id(ev.get('type', '')))
    rowLst.append(dp.get('dataTime', ''))
    rowLst.append(dp.get('alarmState', ''))
    rowLst.append(dp.get('specPower', ''))
    rowLst.append(dp.get('roiPower', ''))
    rowLst.append(dp.get('hr', ''))
    rowLst.append(dp.get('o2Sat', ''))

    # 1D acceleration
    rawData = (dp.get('rawData', [None]*125))
    rowLst.extend(rawData if rawData else [None]*125)

    # 3D acceleration
    rawData3D = dp.get('rawData3D', [None]*375)
    # Split into X, Y, Z
    accX = rawData3D[::3] if rawData3D else [None]*125
    accY = rawData3D[1::3] if rawData3D else [None]*125
    accZ = rawData3D[2::3] if rawData3D else [None]*125
    rowLst.extend(accX)
    rowLst.extend(accY)
    rowLst.extend(accZ)

    return rowLst


def _has_accelerometer_data(dp):
    """Return True if the datapoint carries any accelerometer samples.

    We consider data present if either `rawData` (magnitude) or `rawData3D`
    contains at least one non-None value. Empty lists, missing fields, or all
    None values are treated as absent.
    """
    raw = dp.get('rawData') if isinstance(dp, dict) else None
    if raw:
        try:
            if any(x is not None for x in raw):
                return True
        except Exception:
            pass

    raw3 = dp.get('rawData3D') if isinstance(dp, dict) else None
    if raw3:
        try:
            if any(x is not None for x in raw3):
                return True
        except Exception:
            pass

    return False

# ---------------------------------------------------------------------------
# DC normalisation: surrogate-magnitude rescale back to a 1000 mg baseline.
#
# Enabled with dataProcessing.dcNormalisation=true.  Applied to the rows of
# ONE event before they are written:
#
# Early seizure-detector firmware reported a surrogate magnitude (|x|+|y|+|z|)
# instead of the true vector magnitude.  Its still level depends on device
# orientation (1000..~1732 mg rather than ~1000 mg) and some recordings use
# different unit scalings entirely.  When the event still level - the lower
# quartile (Q1) of per-datapoint M means - exceeds
# dcNormalisationSurrogateThreshold (default 1100 mg) AND even the quietest
# datapoint sits above 1050 mg, the whole event is scaled by 1000/Q1.  Q1 is
# used instead of the median because movement raises datapoint means, so the
# median lands on active datapoints in mostly-active true-magnitude events
# and false-triggers; the still level lives at the low end.  Q1 is robust to
# brief high-g spikes (e.g. fall events are NOT rescaled or excluded - only
# sustained elevated baselines are, and those are linear scalings that the
# rescale corrects).  X/Y/Z are checked independently via their own vector
# magnitude and only scaled if they are themselves anomalous (surrogate events
# typically carry no 3D data at all).
#
# Newer true-magnitude data (≈1000 mg still level) is left byte-identical, so
# on-device inference needs NO runtime transform: the production app already
# supplies ~1000 mg-offset data, matching training.  (An earlier revision also
# subtracted each datapoint's mean to make all data zero-mean; that was
# dropped as over-complicated because it forced a matching on-device
# per-datapoint mean subtraction for zero benefit on modern data.)
# ---------------------------------------------------------------------------

_ROW_META_COLS = 10   # eventId..o2sat columns before M000 in dp2row() output
_DP_SAMPLES = 125     # accelerometer samples per datapoint (5s @ 25Hz)


def _dc_normalisation_params(config):
    """Return (enabled, surrogateThresholdMg) from config['dataProcessing']."""
    if not isinstance(config, dict):
        return (False, 1100.0)
    dpCfg = config.get('dataProcessing')
    if not isinstance(dpCfg, dict):
        return (False, 1100.0)
    enabled = bool(dpCfg.get('dcNormalisation', False))
    thr = dpCfg.get('dcNormalisationSurrogateThreshold', 1100.0)
    try:
        thr = float(thr)
    except (TypeError, ValueError):
        thr = 1100.0
    if not (thr > 0):
        thr = 1100.0
    return (enabled, thr)


def _segment_floats(row, start, count):
    """Return (indices, values) for the finite numeric samples in row[start:start+count].

    Fast path when the whole segment is numeric (the common case); falls back
    to per-value scanning so None/missing samples are skipped safely.
    """
    end = min(start + count, len(row))
    seg = row[start:end]
    try:
        vals = list(map(float, seg))
        idxs = list(range(start, start + len(vals)))
        # filter non-finite (nan/inf) while keeping index alignment
        pairs = [(i, v) for i, v in zip(idxs, vals) if v == v and v not in (float('inf'), float('-inf'))]
        if len(pairs) == len(vals):
            return idxs, vals
        return [p[0] for p in pairs], [p[1] for p in pairs]
    except (TypeError, ValueError):
        idxs, vals = [], []
        for i, v in zip(range(start, end), seg):
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv == fv and fv not in (float('inf'), float('-inf')):
                idxs.append(i)
                vals.append(fv)
        return idxs, vals


def _still_baseline_level(candidates):
    """Lower-quartile (Q1) of per-datapoint means: the event's still level.

    Movement only ever RAISES a datapoint's mean magnitude (magnitude is
    non-negative; oscillation energy adds), so the still level lives at the
    low end of the distribution.  The median instead lands on active
    datapoints in mostly-active events and false-triggers on modern
    true-magnitude data.  Surrogate still levels are elevated in EVERY
    datapoint (including quiet ones), so Q1 still detects them.
    """
    if len(candidates) == 1:
        return candidates[0]
    return statistics.quantiles(sorted(candidates), n=4)[0]


# A datapoint mean at/below this level proves ~1 g data is present in the
# event, so no rescale is needed however elevated the rest looks. True still
# is 1000 mg +/- observed device bias (~tens of mg - see dcOffsetAugmentation
# dcOffsetAugmentationMax); surrogate/unit-scaling still levels never sit
# this low (and orientations that read ~1000 need no correction anyway).
_BASELINE_CONFIRM_MG = 1050.0


def _normalise_event_rows(rows, config=None, eventObj=None, debug=False):
    """Rescale surrogate-magnitude events back to a ~1000 mg still level.

    Rows are modified in place and returned.  No-op unless
    config['dataProcessing']['dcNormalisation'] is true, and no-op for events
    whose still level (lower-quartile of per-datapoint means) is already
    at/below the surrogate threshold (i.e. all modern true-magnitude data
    passes through untouched, preserving the 1000 mg 1 g offset the on-device
    code supplies).
    """
    enabled, thr = _dc_normalisation_params(config)
    if not enabled or not rows:
        return rows

    nMeta = _ROW_META_COLS
    n = _DP_SAMPLES
    eventId = eventObj.get('id', '?') if isinstance(eventObj, dict) else '?'

    # --- Pass 1: gather per-datapoint M means and XYZ vector-magnitude means
    mSegs = []      # per row: (indices, values) for M segment
    xyzSegs = []    # per row: [(idxs, vals)] for X, Y, Z segments
    mMeans = []
    vecMeans = []
    for r in rows:
        mIdx, mVal = _segment_floats(r, nMeta, n)
        mSegs.append((mIdx, mVal))
        mMeans.append(sum(mVal) / len(mVal) if mVal else None)
        rowVec = None
        segs3 = []
        for ax in range(3):
            i3, v3 = _segment_floats(r, nMeta + n + ax * n, n)
            segs3.append((i3, v3))
        xyzSegs.append(segs3)
        if all(len(s[1]) == n for s in segs3):
            xs, ys, zs = segs3[0][1], segs3[1][1], segs3[2][1]
            vm = sum((xs[j] * xs[j] + ys[j] * ys[j] + zs[j] * zs[j]) ** 0.5
                     for j in range(n)) / n
            if vm > 0:
                rowVec = vm
        if rowVec is not None:
            vecMeans.append(rowVec)

    # --- Pass 2: surrogate-magnitude rescale factors (per representation).
    # Trigger: the still level (Q1 of per-datapoint means) exceeds the
    # threshold AND even the quietest datapoint sits above _BASELINE_CONFIRM_MG
    # (a datapoint near 1 g proves true-magnitude data is present, however
    # active the rest of the event is). The rescale maps the still level
    # itself (Q1) back to 1000 mg.
    mScale = 1.0
    baseCandidates = [m for m in mMeans if m is not None and m > 100.0]
    if baseCandidates:
        q1M = _still_baseline_level(baseCandidates)
        if q1M > thr and min(baseCandidates) > _BASELINE_CONFIRM_MG:
            mScale = 1000.0 / q1M
            print("[flattenData] DC normalisation: event %s Q1 M baseline %.0f mg > %.0f mg"
                  " (min %.0f mg) - rescaling M by %.4f (surrogate magnitude / unit scaling)"
                  % (eventId, q1M, thr, min(baseCandidates), mScale))
    scale3D = 1.0
    vecCandidates = [v for v in vecMeans if v > 100.0]
    if vecCandidates:
        q1Vec = _still_baseline_level(vecCandidates)
        if q1Vec > thr and min(vecCandidates) > _BASELINE_CONFIRM_MG:
            scale3D = 1000.0 / q1Vec
            print("[flattenData] DC normalisation: event %s Q1 3D vector baseline %.0f mg > %.0f mg"
                  " (min %.0f mg) - rescaling X/Y/Z by %.4f"
                  % (eventId, q1Vec, thr, min(vecCandidates), scale3D))

    # --- Pass 3: apply scale only (no centring - the 1000 mg DC offset is
    # preserved so training data matches what the on-device code supplies).
    if mScale == 1.0 and scale3D == 1.0:
        return rows
    for rowI, r in enumerate(rows):
        mIdx, mVal = mSegs[rowI]
        if mVal and mScale != 1.0:
            for j, i in enumerate(mIdx):
                r[i] = round(mVal[j] * mScale, 3)
        if scale3D != 1.0:
            for ax in range(3):
                aIdx, aVal = xyzSegs[rowI][ax]
                if not aVal:
                    continue
                for j, i in enumerate(aIdx):
                    r[i] = round(aVal[j] * scale3D, 3)

    if debug:
        print("[flattenData] DC normalisation: event %s - %d datapoints rescaled"
              " to 1000 mg baseline (mScale=%.4f, scale3D=%.4f)"
              % (eventId, len(rows), mScale, scale3D))
    return rows


def writeRowToFile(rowLst, f):
    f.write(",".join([str(x) for x in rowLst]) + "\n")

import multiprocessing

def process_event(eventId, osd):
    eventObj = osd.getEvent(eventId, includeDatapoints=True)
    rows = []
    if not eventObj or 'datapoints' not in eventObj:
        return rows
    skipped_no_acc = 0
    for dp in eventObj['datapoints']:
        if dp is not None:
            if not _has_accelerometer_data(dp):
                skipped_no_acc += 1
                continue
            rowLst = dp2row(eventObj, dp)
            rows.append(rowLst)
    if skipped_no_acc > 0:
        print(f"[WARNING] flattenData: Skipped {skipped_no_acc} datapoints without accelerometer data for event {eventObj.get('id')} (user {eventObj.get('userId')})")
    return rows


def process_event_obj(eventObj, debug=False, validate=False, config=None):
    """
    Process an event object (dict) and return list of CSV rows.
    Optionally validates datapoint temporal continuity, drops overlapping
    datapoints, and applies seizureTimes constraints. Time gaps between
    datapoints are NOT filled: the rows keep their true dataTime values so the
    gap is visible as a time discontinuity, and downstream rolling buffers
    concatenate across it (no restart - matching the production device, whose
    windows likewise span stalls).
    
    Args:
       eventObj (dict): event object
       debug (bool): if True, print warnings about gaps/overlaps/constraints
       validate (bool): if True, perform temporal validation (gap/overlap
           detection). Gaps are reported and left as time discontinuities;
           no synthetic rows are inserted.
       config (dict): Configuration dict. May contain:
           - useSeizureTimesConstraint (bool): Enable seizureTimes filtering
           - seizureTimeMarginSeconds (float): Margin around seizureTimes window

    Returns:
       rows (list): List of lists, each of which is a CSV row.
    
    Note on seizureTimes constraint:
        When enabled, only datapoints that overlap with the seizureTimes window
        are included. This is useful for focusing training on actual seizure movement
        rather than the full event duration. The margin parameter extends the window
        before/after to provide temporal context for LSTM models.
        
        Rows always contain full 125 rawData + 375 rawData3D samples. When a datapoint
        is partially outside the seizureTimes window, it is still included completely
        (no truncation), as long as it overlaps with the window.
    """
    rows = []
    # If a list of events is passed, process each element
    if isinstance(eventObj, list):
        if debug: print(f"flattenData.process_event_obj: Received list of {len(eventObj)} events")
        for ev in eventObj:
            rows.extend(process_event_obj(ev, debug=debug, validate=validate, config=config))
        return rows

    if not eventObj or 'datapoints' not in eventObj:
        return rows
    
    if eventObj['datapoints'] is None or len(eventObj['datapoints']) == 0:
        return rows
    
    # If validation is disabled, use simple processing
    if not validate:
        skipped_no_acc = 0
        skipped_constraint = 0
        
        # First pass: Determine earliest datapoint time if constraint is enabled
        earliest_dt_end = None
        constraint_active, _, seizure_start_s, seizure_end_s, margin_s = \
            _prepare_seizure_time_constraint(eventObj, None, config, debug=debug)
        
        if constraint_active:
            # Find earliest datapoint time
            for dp in eventObj['datapoints']:
                if dp is not None and 'dataTime' in dp:
                    dt = parse_datatime(dp['dataTime'])
                    if dt is not None:
                        if earliest_dt_end is None or dt < earliest_dt_end:
                            earliest_dt_end = dt
            
            if earliest_dt_end is not None and debug:
                print(f"    Earliest datapoint end time: {earliest_dt_end.isoformat()}")
        
        # Second pass: Process datapoints
        for dp in eventObj['datapoints']:
            if dp is None:
                continue
            if not _has_accelerometer_data(dp):
                skipped_no_acc += 1
                continue
            
            # Apply seizureTimes constraint if active
            if constraint_active and earliest_dt_end is not None:
                if 'dataTime' in dp:
                    dt = parse_datatime(dp['dataTime'])
                    if dt and not _datapoint_in_seizure_window(dt, earliest_dt_end, seizure_start_s, seizure_end_s, margin_s):
                        skipped_constraint += 1
                        continue
            
            rowLst = dp2row(eventObj, dp)
            rows.append(rowLst)
        
        if skipped_no_acc > 0:
            print(f"[WARNING] flattenData: Skipped {skipped_no_acc} datapoints without accelerometer data for event {eventObj.get('id')} (user {eventObj.get('userId')})")
        if skipped_constraint > 0 and debug:
            print(f"[DEBUG] flattenData: Skipped {skipped_constraint} datapoints due to seizureTimes constraint for event {eventObj.get('id')}")
        return _normalise_event_rows(rows, config=config, eventObj=eventObj, debug=debug)
    
    # Validation enabled - perform temporal checks (gap/overlap detection)
    # Constants
    SAMPLE_FREQ = 25  # Hz
    SAMPLES_PER_DATAPOINT = 125
    SAMPLE_INTERVAL_MS = 1000.0 / SAMPLE_FREQ  # 40ms
    DATAPOINT_DURATION_MS = SAMPLES_PER_DATAPOINT * SAMPLE_INTERVAL_MS  # 5000ms
    GAP_TOLERANCE_MS = 2000  # Allow 2000ms jitter before considering it a gap (because we only record times to 1 second precision, so 2 sec error in difference)
    
    # Parse and sort datapoints by dataTime
    valid_datapoints = []
    for dp in eventObj['datapoints']:
        if dp is not None and 'dataTime' in dp:
            dt = parse_datatime(dp['dataTime'])
            if dt is not None:
                valid_datapoints.append((dt, dp))
            elif debug:
                print(f"Warning: Event {eventObj['id']} - Skipping datapoint with invalid dataTime: {dp.get('dataTime')}")
    
    if not valid_datapoints:
        return rows
    
    # Sort by dataTime (end time of datapoint)
    valid_datapoints.sort(key=lambda x: x[0])
    
    # Get earliest datapoint end time for constraint checking
    earliest_dt_end = valid_datapoints[0][0]
    
    # Prepare seizureTimes constraint if configured
    constraint_active, _, seizure_start_s, seizure_end_s, margin_s = \
        _prepare_seizure_time_constraint(eventObj, valid_datapoints, config, debug=debug)
    
    if constraint_active and debug:
        print(f"    Earliest datapoint end time: {earliest_dt_end.isoformat()}")
    
    # Track the end time of the last processed datapoint
    last_end_time = None
    event_has_issues = False
    gap_count = 0
    total_missing_datapoints = 0
    overlap_count = 0
    
    skipped_no_acc = 0
    skipped_constraint = 0

    for dt_end, dp in valid_datapoints:
        # Skip datapoints that have no accelerometer samples
        if not _has_accelerometer_data(dp):
            skipped_no_acc += 1
            continue
        
        # Apply seizureTimes constraint if active
        if constraint_active:
            if not _datapoint_in_seizure_window(dt_end, earliest_dt_end, seizure_start_s, seizure_end_s, margin_s):
                skipped_constraint += 1
                continue

        # Calculate start time of this datapoint
        # dataTime is the time of the LAST sample, so subtract (SAMPLES) intervals to get start time
        dt_start = dt_end - timedelta(milliseconds=(SAMPLES_PER_DATAPOINT) * SAMPLE_INTERVAL_MS)
        
        if last_end_time is None:
            # First datapoint - always include
            rowLst = dp2row(eventObj, dp)
            rows.append(rowLst)
            last_end_time = dt_end
        else:
            # Check for gap or overlap
            time_gap_ms = (dt_start - last_end_time).total_seconds() * 1000
            
            # We expect a gap of 40 ms (1/25Hz) between end of last and start of current,
            #   but because we only measure dataTime to 1 second precision, allow for some tolerance.
            if time_gap_ms > GAP_TOLERANCE_MS:
                # GAP DETECTED - do NOT fabricate data. Record the gap so it
                # can be reported, and leave a time discontinuity in the rows.
                # Downstream rolling buffers concatenate across the gap (no
                # restart - the production device likewise keeps appending
                # after a stall, so its windows span gaps too).
                # (Previously this inserted zero-filled datapoints, which
                # taught/scored the model on synthetic flat segments.)
                gap_duration_ms = time_gap_ms
                num_gap_datapoints = int(gap_duration_ms / DATAPOINT_DURATION_MS)
                
                # Only print the warning once.
                if (not event_has_issues) and debug:
                    print(f"\nEvent {eventObj['id']} (user {eventObj['userId']}) has data issues:")
                event_has_issues = True
                gap_count += 1
                total_missing_datapoints += num_gap_datapoints
                
                if debug:
                    print(f"  Gap #{gap_count}: {gap_duration_ms:.0f}ms ({num_gap_datapoints} missing datapoints) - rows omitted, buffer concatenates across gap")

                # No synthetic rows are inserted for the missing span: the next
                # real datapoint keeps its true dataTime, producing a time jump
                # that downstream buffer handling ignores (concatenation, not a
                # segment boundary).  last_end_time is left at the last real
                # datapoint so the jump is visible (do NOT advance it).
            
            elif time_gap_ms < -GAP_TOLERANCE_MS:
                # OVERLAP DETECTED - skip this datapoint
                overlap_ms = -time_gap_ms
                
                if not event_has_issues and debug:
                    print(f"\nEvent {eventObj['id']} (user {eventObj['userId']}) has data issues:")
                event_has_issues = True
                overlap_count += 1
                
                if debug:
                    print(f"  Overlap #{overlap_count}: {overlap_ms:.0f}ms - skipping datapoint")
                continue
            
            # Normal case or within tolerance - add datapoint
            rowLst = dp2row(eventObj, dp)
            rows.append(rowLst)
            last_end_time = dt_end

    if skipped_no_acc > 0:
        print(f"[WARNING] flattenData: Skipped {skipped_no_acc} datapoints without accelerometer data for event {eventObj.get('id')} (user {eventObj.get('userId')})")
    if gap_count > 0 and debug:
        print(f"[DEBUG] flattenData: Event {eventObj.get('id')} has {gap_count} time gap(s), "
              f"{total_missing_datapoints} missing datapoint(s) omitted "
              f"(no synthetic rows inserted; buffer concatenates across each gap)")
    if skipped_constraint > 0 and debug:
        print(f"[DEBUG] flattenData: Skipped {skipped_constraint} datapoints due to seizureTimes constraint for event {eventObj.get('id')}")
    
    return _normalise_event_rows(rows, config=config, eventObj=eventObj, debug=debug)


def iter_events_from_file(fname, debug=False):
    """Yield event objects from a JSON file.

    Supports two formats:
      - A JSON array of objects: [ {...}, {...}, ... ]
      - Newline-delimited JSON (NDJSON): one JSON object per line
    This function streams the file to avoid loading the entire file into memory.
    """
    with open(fname, 'r') as fh:
        # Quick check for NDJSON (one JSON object per line)
        if (debug): print(f"flattenData.iter_events_from_file: Reading from {fname}")
        first = fh.readline()
        if (debug): print(f"flattenData.iter_events_from_file: First 100 chars of first line: {first[:100]}")
        if not first:
            print("[WARNING] flattenData.iter_events_from_file: Input file %s is empty" % fname)
            return
        first_strip = first.lstrip()
        if first_strip.startswith('{') or first_strip.startswith('['):
            # Could be NDJSON (each line an object) or a JSON array.
            # If NDJSON, try to parse the first line as a standalone JSON object.
            try:
                    obj = json.loads(first)
                    # If the parsed object is a dict, this is likely NDJSON where each
                    # line is a JSON object. If it's a list, the file contains a
                    # JSON array (possibly contained entirely on one line). Handle both.
                    if isinstance(obj, dict):
                        # NDJSON: yield first object and continue parsing each subsequent line
                        yield obj
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                yield json.loads(line)
                            except Exception:
                                # fall through to array parser below if line isn't a JSON object
                                break
                        else:
                            return
                    elif isinstance(obj, list):
                        # First line contains a JSON array (or entire file). Yield each item.
                        for item in obj:
                            yield item
                        # We've consumed the first-line array; there may be no more data.
                        return
                    else:
                        # Some other JSON (unlikely) - skip to array parser
                        pass
            except Exception:
                # Not NDJSON - fall through to array parser
                pass
            # Rewind to start and stream-parse a JSON array of objects
            fh.seek(0)
        # Stream parse a JSON array using incremental decoding
        decoder = json.JSONDecoder()
        buffer = ''
        # Read in chunks
        for chunk in iter(lambda: fh.read(1024*64), ''):
            buffer += chunk
            pos = 0
            buflen = len(buffer)
            while True:
                # Skip whitespace and leading commas/brackets
                while pos < buflen and buffer[pos].isspace():
                    pos += 1
                if pos < buflen and buffer[pos] in '[,':
                    pos += 1
                    continue
                if pos < buflen and buffer[pos] == ']':
                    return
                try:
                    obj, offset = decoder.raw_decode(buffer[pos:])
                    yield obj
                    pos += offset
                except ValueError:
                    # Need more data
                    break
            # keep the remaining unread part in buffer
            buffer = buffer[pos:]

def flattenOsdb(inFname, outFname, debug=False, validate_datapoints=False, config=None):
    outFile = open(outFname, 'w') if outFname else sys.stdout
    if (debug): print("flattenData.flattenOsdb: Writing to %s" % (outFname if outFname else "stdout"))
    writeRowToFile(dp2row(None, None, header=True), outFile)

    # If an input filename or list of dataFiles is provided in config, stream
    # events from the files one-by-one to avoid loading everything into memory.
    if inFname is not None:
        if (os.path.exists(inFname)):
            if (debug): print("flattenData.flattenOsdb: Reading from %s" % inFname)
            for ev in iter_events_from_file(inFname, debug=debug):
                for rowLst in process_event_obj(ev, debug=validate_datapoints, validate=validate_datapoints, config=config):
                    writeRowToFile(rowLst, outFile)
        else:
            print("[ERROR] flattenData.flattenOsdb: Input file %s does not exist" % inFname)
            exit(-1)
    else:
        print("[ERROR] flattenData.flattenOsdb: No input file provided")
        exit(-1)
    if (debug): print("flattenData.flattenOsdb: Finished writing data")
    if outFname:
        outFile.close()
    if (debug): print("flattenData.flattenOsdb: Closed output file")

def main():
    parser = argparse.ArgumentParser(description='Flatten OSDB JSON to CSV')
    parser.add_argument('-i', default=None)
    parser.add_argument('-o', default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--validate-datapoints', action='store_true',
                        help='Validate datapoint temporal continuity and report gaps/overlaps (gaps are left as time discontinuities; no synthetic rows are inserted)')
    args = parser.parse_args()
    flattenOsdb(args.i, args.o, debug=args.debug, validate_datapoints=args.validate_datapoints)

if __name__ == "__main__":
    main()

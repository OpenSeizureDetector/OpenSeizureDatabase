#!/usr/bin/env python3

import argparse
import sys
import os
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


def create_zero_datapoint(end_time):
    """
    Create a zero-filled datapoint for gap filling.
    
    Args:
        end_time: datetime object representing the dataTime (end of datapoint)
    
    Returns:
        Dictionary with zero-filled rawData and rawData3D
    """
    return {
        'id': -1,
        'dataTime': end_time.strftime("%d-%m-%Y %H:%M:%S"),
        'hr': None,
        'rawData': [0] * 125,
        'rawData3D': [[0, 0, 0]] * 125,
        'maxVal': 0,
        'minVal': 0,
        'maxFreq': 0,
        'specPower': 0,
        'roiPower': 0,
        'alarmState': 0,
        'alarmPhrase': ""
    }


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
    rowLst.append(ev.get('id', ''))
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

def writeRowToFile(rowLst, f):
    f.write(",".join([str(x) for x in rowLst]) + "\n")

import multiprocessing

def process_event(eventId, osd):
    eventObj = osd.getEvent(eventId, includeDatapoints=True)
    rows = []
    if not eventObj or 'datapoints' not in eventObj:
        return rows
    for dp in eventObj['datapoints']:
        if dp is not None:
            rowLst = dp2row(eventObj, dp)
            rows.append(rowLst)
    return rows


def process_event_obj(eventObj, debug=False, validate=False):
    """
    Process an event object (dict) and return list of CSV rows.
    Optionally validates datapoint temporal continuity, fills gaps with zeros,
    and omits overlapping data.
    
    Args:
       eventObj (dict): event object
       debug (bool): if True, print warnings about gaps/overlaps
       validate (bool): if True, perform temporal validation and gap filling

    Returns:
       rows (list): List of lists, each of which is a CSV row.
    """
    rows = []
    # If a list of events is passed, process each element
    if isinstance(eventObj, list):
        if debug: print(f"flattenData.process_event_obj: Received list of {len(eventObj)} events")
        for ev in eventObj:
            rows.extend(process_event_obj(ev, debug=debug, validate=validate))
        return rows

    if not eventObj or 'datapoints' not in eventObj:
        return rows
    
    if eventObj['datapoints'] is None or len(eventObj['datapoints']) == 0:
        return rows
    
    # If validation is disabled, use simple processing
    if not validate:
        for dp in eventObj['datapoints']:
            if dp is not None:
                rowLst = dp2row(eventObj, dp)
                rows.append(rowLst)
        return rows
    
    # Validation enabled - perform temporal checks and gap filling
    # Constants
    SAMPLE_FREQ = 25  # Hz
    SAMPLES_PER_DATAPOINT = 125
    SAMPLE_INTERVAL_MS = 1000.0 / SAMPLE_FREQ  # 40ms
    DATAPOINT_DURATION_MS = SAMPLES_PER_DATAPOINT * SAMPLE_INTERVAL_MS  # 5000ms
    GAP_TOLERANCE_MS = 100  # Allow 100ms jitter before considering it a gap
    
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
    
    # Track the end time of the last processed datapoint
    last_end_time = None
    event_has_issues = False
    gap_count = 0
    overlap_count = 0
    
    for dt_end, dp in valid_datapoints:
        # Calculate start time of this datapoint
        # dataTime is the time of the LAST sample, so subtract (SAMPLES-1) intervals
        dt_start = dt_end - timedelta(milliseconds=(SAMPLES_PER_DATAPOINT - 1) * SAMPLE_INTERVAL_MS)
        
        if last_end_time is None:
            # First datapoint - always include
            rowLst = dp2row(eventObj, dp)
            rows.append(rowLst)
            last_end_time = dt_end
        else:
            # Check for gap or overlap
            time_gap_ms = (dt_start - last_end_time).total_seconds() * 1000
            
            if time_gap_ms > GAP_TOLERANCE_MS:
                # GAP DETECTED - fill with zero-filled datapoints
                gap_duration_ms = time_gap_ms
                num_gap_datapoints = int(gap_duration_ms / DATAPOINT_DURATION_MS)
                
                if not event_has_issues and debug:
                    print(f"\nEvent {eventObj['id']} (user {eventObj['userId']}) has data issues:")
                event_has_issues = True
                gap_count += 1
                
                if debug:
                    print(f"  Gap #{gap_count}: {gap_duration_ms:.0f}ms ({num_gap_datapoints} missing datapoints)")
                
                # Create zero-filled datapoints to fill the gap
                for i in range(num_gap_datapoints):
                    gap_end_time = last_end_time + timedelta(
                        milliseconds=DATAPOINT_DURATION_MS * (i + 1)
                    )
                    zero_dp = create_zero_datapoint(gap_end_time)
                    rowLst = dp2row(eventObj, zero_dp)
                    rows.append(rowLst)
                
                # Update last_end_time to account for filled gap
                last_end_time = last_end_time + timedelta(
                    milliseconds=DATAPOINT_DURATION_MS * num_gap_datapoints
                )
            
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
    
    return rows


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

def flattenOsdb(inFname, outFname, debug=False, validate_datapoints=False):
    outFile = open(outFname, 'w') if outFname else sys.stdout
    if (debug): print("flattenData.flattenOsdb: Writing to %s" % (outFname if outFname else "stdout"))
    writeRowToFile(dp2row(None, None, header=True), outFile)

    # If an input filename or list of dataFiles is provided in config, stream
    # events from the files one-by-one to avoid loading everything into memory.
    if inFname is not None:
        if (os.path.exists(inFname)):
            if (debug): print("flattenData.flattenOsdb: Reading from %s" % inFname)
            for ev in iter_events_from_file(inFname, debug=debug):
                for rowLst in process_event_obj(ev, debug=validate_datapoints, validate=validate_datapoints):
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
                        help='Validate datapoint temporal continuity, fill gaps with zeros, and report issues')
    args = parser.parse_args()
    flattenOsdb(args.i, args.o, debug=args.debug, validate_datapoints=args.validate_datapoints)

if __name__ == "__main__":
    main()

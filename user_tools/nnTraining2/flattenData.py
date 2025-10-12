#!/usr/bin/env python3

import argparse
import sys
import os

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
    rawData = dp.get('rawData', [None]*125)
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


def process_event_obj(eventObj):
    """Process an event object (dict) and return list of CSV rows."""
    rows = []
    if not eventObj or 'datapoints' not in eventObj:
        return rows
    for dp in eventObj['datapoints']:
        if dp is not None:
            rowLst = dp2row(eventObj, dp)
            rows.append(rowLst)
    return rows


def iter_events_from_file(fname):
    """Yield event objects from a JSON file.

    Supports two formats:
      - A JSON array of objects: [ {...}, {...}, ... ]
      - Newline-delimited JSON (NDJSON): one JSON object per line
    This function streams the file to avoid loading the entire file into memory.
    """
    with open(fname, 'r') as fh:
        # Quick check for NDJSON (one JSON object per line)
        first = fh.readline()
        if not first:
            return
        first_strip = first.lstrip()
        if first_strip.startswith('{') or first_strip.startswith('['):
            # Could be NDJSON (each line an object) or a JSON array.
            # If NDJSON, try to parse the first line as a standalone JSON object.
            try:
                obj = json.loads(first)
                # Looks like NDJSON; yield first and then subsequent lines
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

def flattenOsdb(inFname, outFname, configObj, debug=False):
    dbDir = libosd.configUtils.getConfigParam("cacheDir", configObj)
    outFile = open(outFname, 'w') if outFname else sys.stdout
    if (debug): print("flattenData.flattenOsdb: Writing to %s" % (outFname if outFname else "stdout"))
    writeRowToFile(dp2row(None, None, header=True), outFile)

    # If an input filename or list of dataFiles is provided in config, stream
    # events from the files one-by-one to avoid loading everything into memory.
    if inFname is not None:
        if (debug): print("flattenData.flattenOsdb: Reading from %s" % inFname)
        for ev in iter_events_from_file(inFname):
            for rowLst in process_event_obj(ev):
                writeRowToFile(rowLst, outFile)
    else:
        if (debug): print("flattenData.flattenOsdb: No input file specified, checking config for dataFiles")
        dataFilesLst = libosd.configUtils.getConfigParam("dataFiles", configObj)
        if dataFilesLst:
            for fname in dataFilesLst:
                fpath = fname
                # If the config uses cacheDir paths, resolve relative to cacheDir
                if os.path.exists(os.path.join(dbDir, fname)):
                    fpath = os.path.join(dbDir, fname)
                if (debug): print("flattenData.flattenOsdb: Reading from dataFile %s" % fpath)
                for ev in iter_events_from_file(fpath):
                    for rowLst in process_event_obj(ev):
                        writeRowToFile(rowLst, outFile)
        else:
            # FIXME - I'm not sure this will even work - it is AI generated!    
            if (debug): print("flattenData.flattenOsdb: No dataFiles in config, reading from in-memory DB")
            # No input file or dataFiles list - read from OSDB database in cacheDir
            # Fall back to OsdDbConnection in-memory behaviour if no files are
            # configured (preserve backward compatibility).
            osd = libosd.osdDbConnection.OsdDbConnection(cacheDir=dbDir, debug=False)
            eventIdsLst = osd.getEventIds()
            for eventId in eventIdsLst:
                rows = process_event(eventId, osd)
                for rowLst in rows:
                    writeRowToFile(rowLst, outFile)
    if (debug): print("flattenData.flattenOsdb: Finished writing data")
    if outFname:
        outFile.close()
    if (debug): print("flattenData.flattenOsdb: Closed output file")

def main():
    parser = argparse.ArgumentParser(description='Flatten OSDB JSON to CSV')
    parser.add_argument('--config', default="flattenConfig.json")
    parser.add_argument('-i', default=None)
    parser.add_argument('-o', default=None)
    args = parser.parse_args()
    configObj = libosd.configUtils.loadConfig(args.config)
    flattenOsdb(args.i, args.o, configObj)

if __name__ == "__main__":
    main()

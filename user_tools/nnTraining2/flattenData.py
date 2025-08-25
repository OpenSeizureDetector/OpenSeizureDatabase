#!/usr/bin/env python3

import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libosd.osdDbConnection
import libosd.configUtils

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

def flattenOsdb(inFname, outFname, configObj):
    dbDir = libosd.configUtils.getConfigParam("cacheDir", configObj)
    osd = libosd.osdDbConnection.OsdDbConnection(cacheDir=dbDir, debug=False)
    if inFname is not None:
        osd.loadDbFile(inFname, useCacheDir=False)
    else:
        dataFilesLst = libosd.configUtils.getConfigParam("dataFiles", configObj)
        for fname in dataFilesLst:
            osd.loadDbFile(fname, useCacheDir=False)

    outFile = open(outFname, 'w') if outFname else sys.stdout
    writeRowToFile(dp2row(None, None, header=True), outFile)

    eventIdsLst = osd.getEventIds()
    # Use multiprocessing to process events in parallel
    with multiprocessing.Pool() as pool:
        # Pass osd as a global variable to child processes
        from functools import partial
        results = pool.map(partial(process_event, osd=osd), eventIdsLst)
    for rows in results:
        for rowLst in rows:
            writeRowToFile(rowLst, outFile)
    if outFname:
        outFile.close()

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

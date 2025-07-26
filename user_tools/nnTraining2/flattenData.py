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

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libosd.osdDbConnection
import libosd.dpTools
import libosd.osdAlgTools
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

def dp2accVector(dpObj):
    '''Convert a datapoint object into an input vector to be fed into the neural network.   Note that if dp is not a dict, it is assumed to be a json string
    representation instead.
    '''
    dpInputData = []
    if (type(dpObj) is dict):
        rawDataStr = libosd.dpTools.dp2rawData(dpObj)
    else:
        rawDataStr = dpObj
    accData, hr = libosd.dpTools.getAccelDataFromJson(rawDataStr)
    #print(accData, hr)

    if (accData is not None):
        for n in range(0,len(accData)):
            dpInputData.append(accData[n])
    else:
        print("*** Error in Datapoint: ", dpObj)
        print("*** No acceleration data found with datapoint.")
        exit(-1)

    return dpInputData
    

def dp2row(ev, dp, header=False):
    ''' convert event Object ev and Datapoint object dp into a list of data to be output to a csv file.
    If header=True, returns a .csv header which will represent the columns in the data.
    '''
    rowLst = []
    if (header):
        rowLst.append("id")
        rowLst.append("userId")
        rowLst.append("typeStr")
        rowLst.append("type")
        rowLst.append("dataTime")
        rowLst.append("hr")
        rowLst.append("o2sat")
        # FIXME Hard Coded Array Length
        for n in range(0,125):
            rowLst.append("M%03d" % n)
    else:
        rowLst.append(ev['id'])
        rowLst.append(ev['userId'])
        rowLst.append('"%s/%s"' % (ev['type'], ev['subType']))
        rowLst.append(type2id(ev['type']))
        rowLst.append(libosd.dpTools.getParamFromDp('dataTime',dp))
        rowLst.append(libosd.dpTools.getParamFromDp('hr',dp))
        rowLst.append(libosd.dpTools.getParamFromDp('o2sat',dp))
        rawData = libosd.dpTools.getParamFromDp('rawData', dp)
        if rawData is not None:
            rowLst.extend(rawData)
        else:
            print("flattenOsdb.dp2row - ignoring Missing raw Data: ",dp)
            rowLst = None
    return(rowLst)

def writeRowToFile(rowLst, f):
    first = True
    for item in rowLst:
        if not first:
            f.write(",")
        f.write(str(item))
        first = False
    f.write("\n")



def flattenOsdb(inFname, outFname, configObj, debug=False):
    '''
    flatten the osdb data file inFname into a csv file named outFname, using configuration
    data in configObj.
    If inFname is None, uses the osdb data files listed in configuration entry 'dataFiles'
    if outFname is None, sends output to stdout.
    '''
    if ("cacheDir" in configObj['osdbConfig']):
        print("flattenOsdb - using cacheDir from configObj: %s" % configObj['osdbConfig']['cacheDir'])
        dbDir = configObj['osdbConfig']['cacheDir']
    else:
        dbDir = None
    dbDir = libosd.configUtils.getConfigParam("cacheDir", configObj)

    if ("seizureTimeRange" in configObj['dataProcessing']):
        print("flattenOsdb - using seizureTimeRange from configObj: %s" % configObj['dataProcessing']['seizureTimeRange'])
        seizureTimeRangeDefault = configObj['dataProcessing']['seizureTimeRange']
    seizureTimeRangeDefault = libosd.configUtils.getConfigParam("seizureTimeRange", configObj)


    # initialise the osdb connection which we use to load the files.
    osd = libosd.osdDbConnection.OsdDbConnection(cacheDir=dbDir, debug=debug)

    if inFname is not None:
        print("flattenOsdb - loading file %s" % inFname)
        eventsObjLen = osd.loadDbFile(inFname, useCacheDir=False)
    else:
        dataFilesLst = libosd.configUtils.getConfigParam("dataFiles", configObj)
        for fname in dataFilesLst:
            eventsObjLen = osd.loadDbFile(fname, useCacheDir=False)
            print("loaded %d events from file %s" % (eventsObjLen, fname))


    print("Events Loaded")
    try:
        if outFname is not None:
            outPath = os.path.join(".", outFname)
            print("sending output to file %s" % outPath)
            outFile = open(outPath,'w')
        else:
            print("sending output to stdout")
            outFile = sys.stdout
        writeRowToFile(dp2row(None, None, header=True), outFile)

        eventIdsLst = osd.getEventIds()
        nEvents = len(eventIdsLst)

        for eventNo in range(0,nEvents):
            eventId = eventIdsLst[eventNo]
            eventObj = osd.getEvent(eventId, includeDatapoints=True)
            if (not 'datapoints' in eventObj or eventObj['datapoints'] is None):
                print("Event %s: No datapoints - skipping" % eventId)
            else:
                #print("nDp=%d" % len(eventObj['datapoints']))
                for dp in eventObj['datapoints']:
                    if dp is not None:
                        # Filter seizure data to only include data within specified time range of event time
                        # and which contains significant movement.
                        eventTime = eventObj['dataTime']
                        dpTime = dp['dataTime']
                        eventTimeSec = libosd.dpTools.dateStr2secs(eventTime)
                        dpTimeSec = libosd.dpTools.dateStr2secs(dpTime)
                        timeDiffSec = dpTimeSec - eventTimeSec

                        # The valid time range for datapoints is determined for seizure events either by a range
                        # included in the seizure event object, or a default in the configuration file.
                        # If it is not specified, or the event is not a seizure, all datapoints are included.
                        includeDp = True
                        if (eventObj['type'].lower() == 'seizure'):
                            # Check that this datapoint is within the specified time range.
                            eventSeizureTimeRange = libosd.osdDbConnection.extractJsonVal(eventObj,"seizureTimes")
                            if (eventSeizureTimeRange is not None):
                                seizureTimeRange = eventSeizureTimeRange
                            else:
                                seizureTimeRange = seizureTimeRangeDefault
                            if (seizureTimeRange is not None):
                                if (timeDiffSec < seizureTimeRange[0]):
                                    includeDp=False
                                if (timeDiffSec > seizureTimeRange[1]):
                                    includeDp=False
                            # Check we have real movement to analyse, otherwise reject the datapoint from seizure training data to avoid false alarms when no movement.
                            accArr = np.array(dp['rawData'])
                            accStd = 100. * np.std(accArr) / np.average(accArr)
                            if (eventObj['type'].lower() == 'seizure'):
                                if (accStd <configObj['dataProcessing']['accSdThreshold']):
                                    print("Warning: Ignoring Low SD Seizure Datapoint: Event ID=%s: %s, %s - diff=%.1f, accStd=%.1f%%" % (eventId, eventTime, dpTime, timeDiffSec, accStd))
                                    includeDp = False


                        if (includeDp):
                            rowLst = dp2row(eventObj, dp, header=False)
                            if (rowLst is not None):
                                writeRowToFile(rowLst, outFile)
    finally:
        if (outFname is not None):
            outFile.close()
            print("Output written to file %s" % outFname)


    return True

 

def main():
    print("flattenOsdb.main()")
    parser = argparse.ArgumentParser(description='Produce a flattened version of OpenSeizureDatabase data')
    parser.add_argument('--config', default="flattenConfig.json",
                        help='name of json file containing configuration data')
    parser.add_argument('-i', default=None,
                        help='Input filename (uses configuration datafiles list if not specified)')
    parser.add_argument('-o', default=None,
                        help='Output filename (uses stout if not specified)')
    parser.add_argument('--debug', action="store_true",
                        help='Write debugging information to screen')
    argsNamespace = parser.parse_args()
    args = vars(argsNamespace)
    print(args)



    configObj = libosd.configUtils.loadConfig(args['config'])
    # print("configObj=",configObj.keys())
    # Load a separate OSDB Configuration file if it is included.
    if configObj is not None:
        if ("osdbCfg" in configObj):
            osdbCfgFname = libosd.configUtils.getConfigParam("osdbCfg",configObj)
            print("Loading separate OSDB Configuration File %s." % osdbCfgFname)
            osdbCfgObj = libosd.configUtils.loadConfig(osdbCfgFname)
            # Merge the contents of the OSDB Configuration file into configObj
            configObj = configObj | osdbCfgObj

        print("configObj=",configObj.keys())


    flattenOsdb(args['i'], args['o'], configObj)

if __name__ == "__main__":
    main()

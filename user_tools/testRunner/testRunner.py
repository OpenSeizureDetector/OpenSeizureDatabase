#!/usr/bin/env python3

import argparse
import json
import sys
import os
import importlib
import dateutil.parser
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
#import libosd.analyse_event
import libosd.webApiConnection
import libosd.osdDbConnection
import libosd.osdAppConnection
import libosd.dpTools
import libosd.configUtils


OTHERS_INDEX = 0
ALL_INDEX = 1
FALSE_INDEX = 2
NDA_INDEX = 3

def runTest(configObj, debug=False):
    print("runTest - configObj="+json.dumps(configObj))
    if ('dbDir' in configObj.keys()):
        dbDir = configObj['dbDir']
    else:
        dbDir = None

    invalidEvents = configObj['invalidEvents']
    print("invalid events", invalidEvents)

    # Load each of the three events files (tonic clonic seizures,
    #all seizures and false alarms).
    osd = libosd.osdDbConnection.OsdDbConnection(cacheDir=dbDir, debug=debug)
    for fname in configObj['dataFiles']:
        eventsObjLen = osd.loadDbFile(fname)
        print("loaded %d events from file %s" % (eventsObjLen, fname))
    osd.removeEvents(invalidEvents)
    osd.listEvents()

    requireHrData = configObj['requireHrData']
    require3dData = configObj['require3dData']

    eventIdsLst = filterEvents(osd, requireHrData, require3dData, debug=False)
    
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

            settingsStr = json.dumps(algObj['settings'])
            print("settingsStr=%s (%s)" % (settingsStr, type(settingsStr)))
            algs.append(eval("module.%s(settingsStr, debug)" % (classId)))
            algNames.append(algObj['name'])
        else:
            print("Algorithm %s is disabled in configuration file - ignoring"
                  % algObj['name'])

    

    # Run each event through each algorithm
    tcResults, tcResultsStrArr = testEachEvent(eventIdsLst, osd, algs, algNames, requireHrData=requireHrData, require3dData=require3dData, debug=debug)
    saveResults2("output", tcResults, tcResultsStrArr, eventIdsLst, osd, algs, algNames)
    
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
                 

def filterEvents(osd, requireHrData, require3dData, debug=False):
    ''' returns a list of event ids with events that do not contain Hr data removed (if requireHrData is true) 
    and with events that do not contain 3d data removed (if require 3dData is true)
    '''
    eventIdsLst = []
    eventIdsLstAll = osd.getEventIds()
    # Filter out events that do not have HR data if reqested
    print("testEachEvent - starting with %d events" % len(eventIdsLstAll))
    if (requireHrData):
        print("testEachEvent - filtering for events that contain HR data")
        for eventNo in range(0, len(eventIdsLstAll)):
            eventId = eventIdsLstAll[eventNo]
            eventObj = osd.getEvent(eventId, includeDatapoints=False)
            if getEventVal(eventObj,'hasHrData'):
                eventIdsLst.append(eventId)
            else:
                print("Rejecting Event %s for lack of HR data" % eventId)
    else:
        eventIdsLst = eventIdsLstAll
    print("After HR data filter, we have %d events" % len(eventIdsLst))
    # Filter out events that do not have 3d data if reqested
    if (require3dData):
        eventIdsLstAll = eventIdsLst
        eventIdsLst = []
        print("testEachEvent - filtering for events that contain 3d data")
        for eventNo in range(0, len(eventIdsLstAll)):
            eventId = str(eventIdsLstAll[eventNo])
            eventObj = osd.getEvent(eventId, includeDatapoints=False)
            if getEventVal(eventObj,'has3dData'):
                #print("getEventVal(eventObj,'has3dData') = ",getEventVal(eventObj,'has3dData'))
                eventIdsLst.append(eventId)
            else:
                print("Rejecting Event %s for lack of 3d data" % eventId)
    print("After 3d Data filter, we have %d events" % len(eventIdsLst))
    eventIdsLst.sort()
    #print(eventIdsLst)
    return eventIdsLst



def testEachEvent(eventIdsLst, osd, algs, algNames, requireHrData = False, require3dData = False, debug=False):
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
    nStatus =5 # The number of possible OSD statuses 0=OK, 1=WARNING, 2=ALARM etc.
    results = np.zeros((nEvents, nAlgs, nStatus))
    resultsStrArr = []
    for eventNo in range(0,nEvents):
        eventId = eventIdsLst[eventNo]
        #print("Analysing event %s" % eventId)
        eventObj = osd.getEvent(eventId, includeDatapoints=True)
        print("Analysing event %s (%s, userId=%s, desc=%s)" % (eventId, eventObj['type'], eventObj['userId'], eventObj['desc']))
        eventResultsStrArr = []
        for algNo in range(0, nAlgs):
            alg = algs[algNo]
            print("Processing Algorithm %d: %s (%s): " % (algNo, algNames[algNo], alg.__class__.__name__))
            alg.resetAlg()
            sys.stdout.write("Looping through Datapoints: ")
            sys.stdout.flush()
            statusStr = "_"
            lastDpTimeSecs = 0
            lastDpTimeStr = ''
            if ('datapoints' in eventObj):
                for dp in eventObj['datapoints']:
                    #if (debug): print(dp)
                    dpTimeStr = dp['dataTime']
                    dpTimeSecs = libosd.dpTools.dateStr2secs(dpTimeStr)
                    alarmState = libosd.dpTools.getParamFromDp('alarmState',dp)
                    if (debug): print("%s, %.1fs, alarmState=%d" % (dpTimeStr, dpTimeSecs-lastDpTimeSecs, alarmState))
                    
                    # FIXME - hard coded constant!
                    #if (dpTimeSecs - lastDpTimeSecs >= 3.):
                    if (alarmState == 5):
                        if (debug): print("Skipping Manual Alarm datapoint (duplicate)")
                        if (debug): print("alarmStatus=%s  %s, %s, %d" %\
                                        (alarmState, dpTimeStr, lastDpTimeStr, (dpTimeSecs-lastDpTimeSecs)))
                    else:
                        rawDataStr = libosd.dpTools.dp2rawData(dp, debug=False)
                        if (rawDataStr is not None):
                            retVal = alg.processDp(rawDataStr)
                            #print(alg.__class__.__name__, retVal)
                            retObj = json.loads(retVal)
                            statusVal = retObj['alarmState']
                            results[eventNo][algNo][statusVal] += 1
                            statusStr = "%s%d" % (statusStr, statusVal)
                            sys.stdout.write("%d" % statusVal)
                            # Write out debugging information (only works for OSD algorithm!)  
                            if alg.__class__.__name__ == 'OsdAlg' and debug:
                                sys.stdout.write(" - specPower=%.0f (%.0f), roiPower=%.0f (%.0f), roiRatio=%.0f (%.0f), alarmState=%.0f (%.0f)\n" % (
                                    retObj['specPower'], dp['specPower'],
                                    retObj['roiPower'], dp['roiPower'],
                                    retObj['roiRatio'], dp['roiRatio'],
                                    retObj['alarmState'], dp['alarmState']
                                ))
                            lastDpTimeSecs = dpTimeSecs
                            lastDpTimeStr = dpTimeStr
                        else:
                            print("Invalid datapoint in event %s" % eventId)
                            print("Aborting because of invalid data")
                            exit(-1)
                    sys.stdout.flush()
            else:
                print("Skipping Event with no datapoints")
                exit(-1)
            sys.stdout.write("\n")
            sys.stdout.flush()
            #print(statusStr)
            eventResultsStrArr.append(statusStr)
            print("Finished Algorithm %d (%s): " % (algNo, alg.__class__.__name__))
            sys.stdout.write("\n")
            sys.stdout.flush()
        resultsStrArr.append(eventResultsStrArr)
        if str(eventId)=="86657" and debug:
            exit(-1)
    #print(results)
    return(results, resultsStrArr)
    

def type2index(typeStr, subTypeStr=None):
    retVal = OTHERS_INDEX
    if (typeStr.lower() == "nda"):
        retVal = NDA_INDEX
    elif (typeStr.lower() == "false alarm"):
        retVal = FALSE_INDEX
    elif (typeStr.lower() == "seizure"):
        retVal = ALL_INDEX
    return(retVal)

def saveResults2(outFileRoot, results, resultsStrArr, eventIdsLst, osd, algs, algNames):
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
    nAlgs = len(algs)
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
        lineStr = "%s, %s, %s, %s, %s, %s" % (
            eventId, 
            eventObj['dataTime'], 
            eventObj['type'], 
            eventObj['subType'], 
            eventObj['userId'],
            eventObj['dataSourceName'])
        for algNo in range(0,nAlgs):
            # Increment count of correct results
            # If the correct result is to alarm
            if (results[eventNo][algNo][2]>0 and expectAlarm):
                correctCount[outputIndex, algNo] += 1
            # If correct result is NOT to alarm
            if (results[eventNo][algNo][2]==0 and not expectAlarm):
                correctCount[outputIndex, algNo] += 1

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
        if (reportedAlarmState==2 and expectAlarm):
            correctCount[outputIndex, nAlgs] += 1
        if (reportedAlarmState!=2 and not expectAlarm):
            correctCount[outputIndex, nAlgs] += 1

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
                lineStr = "%s, %.2f" % (lineStr,1.*correctCount[outputIndex, algNo]/totalCount[outputIndex])
            print(lineStr)
            outf.write(lineStr)
            outf.write("\n")
            
            outf.close()
            print("Output written to file %s" % outputs[outputIndex])


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

    runTest(configObj, args['debug'])
    


if __name__ == "__main__":
    main()

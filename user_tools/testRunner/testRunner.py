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
    eventsObjLen = osd.loadDbFile(configObj['tcSeizuresFname'])
    print("tcSeizures  eventsObjLen=%d" % eventsObjLen)
    osd.removeEvents(invalidEvents)
    osd.listEvents()

    osdAll = libosd.osdDbConnection.OsdDbConnection(debug=debug)
    eventsObjLen = osdAll.loadDbFile(configObj['allSeizuresFname'])
    osdAll.removeEvents(invalidEvents)
    print("all Seizures eventsObjLen=%d" % eventsObjLen)

    osdFalse = libosd.osdDbConnection.OsdDbConnection(debug=debug)
    eventsObjLen = osdFalse.loadDbFile(configObj['falseAlarmsFname'])
    osdFalse.removeEvents(invalidEvents)
    print("false alarms eventsObjLen=%d" % eventsObjLen)

    
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
    tcResults, tcResultsStrArr = testEachEvent(osd, algs, debug)
    saveResults("tcResults.csv", tcResults, tcResultsStrArr, osd, algs, algNames, True)
    
    allSeizureResults, allSeizureResultsStrArr = testEachEvent(osdAll, algs, debug)
    saveResults("allSeizureResults.csv", allSeizureResults, allSeizureResultsStrArr, osdAll, algs, algNames, True)
    
    falseAlarmResults, falseAlarmResultsStrArr = testEachEvent(osdFalse, algs, debug)
    saveResults("falseAlarmResults.csv", falseAlarmResults, falseAlarmResultsStrArr, osdFalse, algs, algNames, False)

    summariseResults(tcResults, allSeizureResults, falseAlarmResults, algNames)

def testEachEvent(osd, algs, debug=False):
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
    eventIdsLst = osd.getEventIds()
    nEvents = len(eventIdsLst)
    nAlgs = len(algs)
    nStatus =5 # The number of possible OSD statuses 0=OK, 1=WARNING, 2=ALARM etc.
    results = np.zeros((nEvents, nAlgs, nStatus))
    resultsStrArr = []
    for eventNo in range(0,nEvents):
        eventId = eventIdsLst[eventNo]
        print("Analysing event %s" % eventId)
        eventObj = osd.getEvent(eventId, includeDatapoints=True)
        print("Analysing event %s (%s, userId=%s)" % (eventId, eventObj['type'], eventObj['userId']))
        eventResultsStrArr = []
        for algNo in range(0, nAlgs):
            alg = algs[algNo]
            print("Processing Algorithm %d (%s): " % (algNo, alg.__class__.__name__))
            alg.resetAlg()
            sys.stdout.write("Looping through Datapoints: ")
            sys.stdout.flush()
            statusStr = "_"
            lastDpTimeSecs = 0
            lastDpTimeStr = ''
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
                    rawDataStr = libosd.dpTools.dp2rawData(dp, debug)
                    retVal = alg.processDp(rawDataStr)
                    #print(alg.__class__.__name__, retVal)
                    retObj = json.loads(retVal)
                    statusVal = retObj['alarmState']
                    results[eventNo][algNo][statusVal] += 1
                    statusStr = "%s%d" % (statusStr, statusVal)
                    sys.stdout.write("%d" % statusVal)
                    lastDpTimeSecs = dpTimeSecs
                    lastDpTimeStr = dpTimeStr
                sys.stdout.flush()
            sys.stdout.write("\n")
            sys.stdout.flush()
            #print(statusStr)
            eventResultsStrArr.append(statusStr)
            print("Finished Algorithm %d (%s): " % (algNo, alg.__class__.__name__))
            sys.stdout.write("\n")
            sys.stdout.flush()
        resultsStrArr.append(eventResultsStrArr)
    #print(results)
    return(results, resultsStrArr)
    

def saveResults(outFile, results, resultsStrArr, osd, algs, algNames,
                expectAlarm=True):
    print("Displaying Results")
    eventIdsLst = osd.getEventIds()
    nEvents = len(eventIdsLst)
    print("Displaying %d Events" % nEvents)

    outf = open(outFile,"w")
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
        alarmPhrases = ['OK','WARN','ALARM','FALL','unused','MAN_ALARM']
        lineStr = "%s, %s" % (lineStr, alarmPhrases[eventObj['osdAlarmState']])
        if (eventObj['osdAlarmState']==2 and expectAlarm):
            correctCount[nAlgs] += 1
        if (eventObj['osdAlarmState']!=2 and not expectAlarm):
            correctCount[nAlgs] += 1

        for algNo in range(0,nAlgs):
            lineStr = "%s, %s" % (lineStr, resultsStrArr[eventNo][algNo])

        lineStr = "%s, %s" % (lineStr, eventObj['desc'])
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
    

    outf.close()
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
    #parser.add_argument('--out', default="trOutput.csv",
    #                    help='name of output CSV file')
    parser.add_argument('--debug', action="store_true",
                        help='Write debugging information to screen')
    argsNamespace = parser.parse_args()
    args = vars(argsNamespace)
    print(args)


    inFile = open(args['config'],'r')
    configObj = json.load(inFile)
    inFile.close()
    runTest(configObj, args['debug'])
    


if __name__ == "__main__":
    main()

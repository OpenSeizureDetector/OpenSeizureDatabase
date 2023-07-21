#!/usr/bin/env python3
# Test the HR Algorithm running on the connected phone
# Uses simulated HR data.

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

    eventData = configObj['eventHrData']
    #print("EventData=",eventData)
    
    # Run each event through each algorithm
    tcResults, tcResultsStrArr, eventObjLst = testEachEvent(eventData, algs, debug)
    saveResults2("output", tcResults, tcResultsStrArr, eventObjLst, algs, algNames)
    

def testEachEvent(eventData, algs, debug=False):
    """
    for each event in the eventData object, run each algorithm in the
    list 'algs', where each item in the algs list is an instance of an SdAlg
    class.
    Returns a numpy array of the outcome of each event for each algorithm.
    """
    # Now we loop through each event in the eventsList and run the event
    # through each of the specified algorithms.
    # we collect statistics of the number of alarms and warnings generated
    # for each event for each algorithm.
    # result[e][a][s] is the count of the number of datapoints in event e giving status s using algorithm a.
    
    nEvents = len(eventData)
    nAlgs = len(algs)
    nStatus =5 # The number of possible OSD statuses 0=OK, 1=WARNING, 2=ALARM etc.
    results = np.zeros((nEvents, nAlgs, nStatus))
    resultsStrArr = []
    eventObjLst = []
    for eventNo in range(0,nEvents):
        eventObj = makeEventObj(eventData[eventNo])
        eventObjLst.append(eventObj)
        #print("eventObj=",eventObj)
        
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
                        rawDataStr = libosd.dpTools.dp2rawData(dp, debug)
                        retVal = alg.processDp(rawDataStr)
                        print(alg.__class__.__name__, retVal)
                        retObj = json.loads(retVal)
                        statusVal = retObj['alarmState']
                        results[eventNo][algNo][statusVal] += 1
                        statusStr = "%s%d" % (statusStr, statusVal)
                        sys.stdout.write("%d" % statusVal)
                        lastDpTimeSecs = dpTimeSecs
                        lastDpTimeStr = dpTimeStr
                    sys.stdout.flush()
            else:
                print("Skipping Event with no datapoints")
            sys.stdout.write("\n")
            sys.stdout.flush()
            #print(statusStr)
            eventResultsStrArr.append(statusStr)
            print("Finished Algorithm %d (%s): " % (algNo, alg.__class__.__name__))
            sys.stdout.write("\n")
            sys.stdout.flush()
        resultsStrArr.append(eventResultsStrArr)
    #print(results)
    return(results, resultsStrArr, eventObjLst)
    

def makeEventObj(eventData):
    """ 
    Create an osdb compatible event object from the event HR data object eventData
    """
    hrVals = eventData['hrVals']
    nDp = len(hrVals)
    #print("makeEventObj = nDp=%d" % nDp)
    #print("hrVals=",hrVals,type(hrVals))
    eventObj = {}
    eventObj['id'] = eventData['id']
    eventObj['userId'] = 1
    eventObj['type'] = 'seizure'
    eventObj['subType'] = 'other'
    eventObj['datapoints'] = []
    eventObj['osdAlarmState'] = 0
    eventObj['desc'] = eventData['name']
    for hrVal in hrVals:
        dpObj = {}
        rawData = []
        for n in range(0,125):
            rawData.append(1000)
        dpObj['dataTime'] = "2023-04-30T00:00:00Z"
        dpObj['rawData'] = rawData
        dpObj['hr'] = hrVal
        eventObj['datapoints'].append(dpObj)
    

    return (eventObj)




def type2index(typeStr, subTypeStr=None):
    retVal = OTHERS_INDEX
    if (typeStr.lower() == "nda"):
        retVal = NDA_INDEX
    elif (typeStr.lower() == "false alarm"):
        retVal = FALSE_INDEX
    elif (typeStr.lower() == "seizure"):
        retVal = ALL_INDEX
    return(retVal)

def saveResults2(outFileRoot, results, resultsStrArr, eventsLst, algs, algNames):
    print("saveResults2")
    eventIdsLst = []
    nEvents = len(eventsLst)
    for event in eventsLst:
        eventIdsLst.append(event['id'])

 
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
    lineStr = "eventId, type, subType, userId"
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
        eventObj = eventsLst[eventNo]
        print(eventObj)
        outputIndex = type2index(eventObj['type'])
        if (eventObj['type'].lower()=="seizure"):
            expectAlarm=True
        else:
            expectAlarm=False
        totalCount[outputIndex] += 1
        lineStr = "%s, %s, %s, %s" % (eventId, eventObj['type'], eventObj['subType'], eventObj['userId'])
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
        alarmPhrases = ['OK','WARN','ALARM','FALL','unused','MAN_ALARM',"NDA"]
        lineStr = "%s, %s" % (lineStr, alarmPhrases[eventObj['osdAlarmState']])
        if (eventObj['osdAlarmState']==2 and expectAlarm):
            correctCount[outputIndex, nAlgs] += 1
        if (eventObj['osdAlarmState']!=2 and not expectAlarm):
            correctCount[outputIndex, nAlgs] += 1

        for algNo in range(0,nAlgs):
            lineStr = "%s, %s" % (lineStr, resultsStrArr[eventNo][algNo])

        lineStr = "%s, %s" % (lineStr, eventObj['desc'])
        print(lineStr)

        if outfLst[outputIndex] is not None:
            outfLst[outputIndex].write(lineStr)
            outfLst[outputIndex].write("\n")
  

    for outputIndex in range(0,len(outfLst)):
        outf = outfLst[outputIndex]
        if outf is not None:
            lineStr = "#Total, , ,"
            for algNo in range(0,nAlgs+1):
                lineStr = "%s, %d" % (lineStr, totalCount[outputIndex])
            print(lineStr)
            outf.write(lineStr)
            outf.write("\n")
            
            lineStr = "#Correct Count, , ,"
            for algNo in range(0,nAlgs+1):
                lineStr = "%s, %d" % (lineStr,correctCount[outputIndex, algNo])
            print(lineStr)
            outf.write(lineStr)
            outf.write("\n")

            lineStr = "#Correct Prop, , ,"
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
    print("testDeviceHrAlg.main()")
    parser = argparse.ArgumentParser(description='Device HR Algorithm Tester')
    parser.add_argument('--config', default="testDeviceHrAlg.json",
                        help='name of json file containing test configuration')
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

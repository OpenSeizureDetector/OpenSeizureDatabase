#!/usr/bin/python3

import argparse
import json
import sys
import os
import importlib
import dateutil.parser
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#import libosd.analyse_event
import libosd.webApiConnection
import libosd.osdDbConnection

def dateStr2secs(dateStr):
    ''' Convert a string representation of date/time into seconds from
    the start of 1970 (standard unix timestamp)
    '''
    parsed_t = dateutil.parser.parse(dateStr)
    return parsed_t.timestamp()


def dp2rawData(dp):
    '''Accepts a dataPoint object from the osd Database and converts it into
    a 'raw data' JSON string that would have been received by the phone to
    create it.
    This is useful to reproduce the exact phone behaviour from datapoints
    stored in the database.
    '''
    if ('dataTime' in dp):
        currTs = dateStr2secs(dp['dataTime'])
    else:
        currTs = None
        
    if ('dataJSON' in dp):
        dpObj = json.loads(dp['dataJSON'])
    else:
        dpObj = None
    if ('dataJSON' in dpObj):
        try:
            dataObj = json.loads(dpObj['dataJSON'])

            # Create raw data list
            accelLst = []
            accelLst3d = []
            # FIXME:  It is not good to hard code the length of an array!
            for n in range(0,125):
                accelLst.append(dataObj['rawData'][n])
                if ("data3D" in dataObj.keys()):
                    print("3dData present")
                    accelLst3d.append(dataObj['rawData3D'][n*3])
                    accelLst3d.append(dataObj['rawData3D'][n*3 + 1])
                    accelLst3d.append(dataObj['rawData3D'][n*3 + 2])

            rawDataObj = {"dataType": "raw", "Mute": 0}
            rawDataObj['HR'] = dataObj['hr']
            rawDataObj['data'] = accelLst
            rawDataObj['data3D'] = accelLst3d
            # FIXME - add o2sat
            dataJSON = json.dumps(rawDataObj)
        except (json.decoder.JSONDecodeError, TypeError):
            dataJSON = None
    else:
        dataJSON = None
    return dataJSON


def runTest(configObj, outFile="trOutput.csv", debug=False):
    print("runTest - configObj="+json.dumps(configObj))

    invalidEvents = configObj['invalidEvents']
    print("invalid events", invalidEvents)

    # Load each of the three events files (tonic clonic seizures,
    #all seizures and false alarms).
    osd = libosd.osdDbConnection.OsdDbConnection(debug=debug)
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
        print(algObj['name'])
        moduleId = algObj['alg'].split('.')[0]
        classId = algObj['alg'].split('.')[1]
        print("Importing Module %s" % moduleId)
        module = importlib.import_module(moduleId)

        settingsStr = json.dumps(algObj['settings'])
        print("settingsStr=%s (%s)" % (settingsStr, type(settingsStr)))
        algs.append(eval("module.%s(settingsStr)" % (classId)))
        algNames.append(algObj['name'])



    # Run each event through each algorithm
    tcResults = testEachEvent(osd, algs)
    allSeizureResults = testEachEvent(osdAll, algs)
    falseAlarmResults = testEachEvent(osdFalse, algs)
    results = falseAlarmResults

    saveResults("tcResults.csv", tcResults, osd, algs, algNames, True)
    saveResults("allSeizureResults.csv", allSeizureResults, osdAll, algs, algNames, True)
    saveResults("falseAlarmResults.csv", falseAlarmResults, osdFalse, algs, algNames, False)

def testEachEvent(osd, algs):
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
    for eventNo in range(0,nEvents):
        eventId = eventIdsLst[eventNo]
        print("Analysing event %s" % eventId)
        eventObj = osd.getEvent(eventId, includeDatapoints=True)
        for algNo in range(0, nAlgs):
            alg = algs[algNo]
            print("Processing Algorithm %d (%s): " % (algNo, alg.__class__.__name__))
            alg.resetAlg()
            sys.stdout.write("Looping through Datapoints: ")
            for dp in eventObj['datapoints']:
                retVal = alg.processDp(dp2rawData(dp))
                #print(alg.__class__.__name__, retVal)
                retObj = json.loads(retVal)
                statusVal = retObj['alarmState']
                results[eventNo][algNo][statusVal] += 1
                sys.stdout.write("%d" % statusVal)
                sys.stdout.flush()
            sys.stdout.write("\n")
            sys.stdout.flush()
    #print(results)
    return(results)
    

def saveResults(outFile, results, osd, algs, algNames, expectAlarm=True):
    print("Displaying Results")
    eventIdsLst = osd.getEventIds()
    nEvents = len(eventIdsLst)
    print("Displaying %d Events" % nEvents)

    outf = open(outFile,"w")
    lineStr = "eventId, type, subType, userId"
    nAlgs = len(algs)
    for algNo in range(0,nAlgs):
        lineStr = "%s, %s" % (lineStr, algNames[algNo])
    lineStr = "%s, reported, desc" % lineStr
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
        lineStr = "%s, %s" % (lineStr, eventObj['desc'])
        print(lineStr)
        outf.write(lineStr)
        outf.write("\n")

    lineStr = "#Correct Count, , , ,"
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



def main():
    print("testRunner.main()")
    parser = argparse.ArgumentParser(description='Seizure Detection Test Runner')
    parser.add_argument('--config', default="testConfig.json",
                        help='name of json file containing test configuration')
    parser.add_argument('--out', default="trOutput.csv",
                        help='name of output CSV file')
    parser.add_argument('--debug', action="store_true",
                        help='Write debugging information to screen')
    argsNamespace = parser.parse_args()
    args = vars(argsNamespace)
    print(args)


    inFile = open(args['config'],'r')
    configObj = json.load(inFile)
    inFile.close()
    runTest(configObj, args['out'], args['debug'])
    


if __name__ == "__main__":
    main()

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
import deviceAlg

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

    
    # Run each event through each algorithm
    results = testHrFrozenFault(algs, debug)

    print(results)
    

def testHrFrozenFault(algs, debug=False):
    """
    Check that the system enters a fault condition if we send the same HR value for more than
    60 seconds, but does not if the HR value changes within a 1 minute period. 
    """
    nEvents = int(70 / 5 ) # 70 seconds
    nAlgs = len(algs)
    nStatus =5 # The number of possible OSD statuses 0=OK, 1=WARNING, 2=ALARM etc.
    results = np.zeros((nEvents, nAlgs, nStatus))
    resultsStrArr = []
    eventObjLst = []
    eventData = [65] * 30

    for eventNo in range(0,nEvents):
        eventObj = makeEventObj(eventData)
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
            print("Finished Algorithm %d (%s): " % (algNo, alg.__class__.__name__))
            sys.stdout.write("\n")
            sys.stdout.flush()
    #print(results)
    return(results, resultsStrArr, eventObjLst)
    

def makeEventObj(hrVals):
    """ 
    Create an osdb compatible event object from the event HR data object eventData
    """
    nDp = len(hrVals)
    #print("makeEventObj = nDp=%d" % nDp)
    #print("hrVals=",hrVals,type(hrVals))
    eventObj = {}
    eventObj['id'] = 1
    eventObj['userId'] = 1
    eventObj['type'] = 'seizure'
    eventObj['subType'] = 'other'
    eventObj['datapoints'] = []
    eventObj['osdAlarmState'] = 0
    eventObj['desc'] = "Dummy Event to test Hr frozen warning"
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






def main():
    print("testDeviceHrFrozenFault.main()")
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

#!/usr/bin/env python3
"""
Scan through the events in the database and check that the top level event alarm state is the most severe of the alarm states
for the datapoints associated with the event.
(Written because I think there is something wrong with the data produced by the dashboard analysis)

Graham Jones 2023

Licence: GPL v3 or later.

"""
import sys
import os
import argparse
import pandas as pd
import json
import tabulate

# Make the libosd folder accessible in the search path.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import libosd.osdDbConnection
import libosd.configUtils

import libosd.tidy_db







if (__name__=="__main__"):
    print("checkAlarmStates.py.main()")
    parser = argparse.ArgumentParser(description='Check that the OSDAlarmState parameter in each event is correct given the associated datapoints')
    parser.add_argument('--config', default="osdb.cfg",
                        help='name of json file containing configuration information and login credientials - see osdb.cfg.template')
    parser.add_argument('--fix', action='store_true',
                        help="Correct any errors and write back to the OSDB Database files.")
    parser.add_argument('--debug', action='store_true',
                        help="Write debugging information to screen")
    
    argsNamespace = parser.parse_args()
    args = vars(argsNamespace)
    print(args)

    debug = args['debug']
    cfgObj = libosd.configUtils.loadConfig(args['config'])
    print(cfgObj)


    dbDir = libosd.configUtils.getConfigParam("cacheDir", cfgObj)


    invalidEvents = libosd.configUtils.getConfigParam("invalidEvents", cfgObj)

    # Load each of the three events files (tonic clonic seizures,
    #all seizures and false alarms).
    osd = libosd.osdDbConnection.OsdDbConnection(cacheDir=dbDir, debug=debug)


    dataFilesLst = libosd.configUtils.getConfigParam("dataFiles", cfgObj)
    for fname in dataFilesLst:
        eventsObjLen = osd.loadDbFile(fname)
        print("loaded %d events from file %s" % (eventsObjLen, fname))
    osd.removeEvents(invalidEvents)


    for event in osd.getAllEvents():
        evtAlarmState = event['osdAlarmState']
        alarmCounts = [0,0,0,0,0,0,0]
        if ('datapoints' in event.keys()):
            for dp in event['datapoints']:
                #print(dp.keys())
                if ('alarmState' in dp.keys()):
                    dpAlarmState = dp['alarmState']
                    alarmCounts[dpAlarmState] += 1
                else:
                    print("Error - event Id %s datapoint does not contain alarmState" % event['id'])
            correctAlarmState = 0   # OK
            if (alarmCounts[1]>0):
                correctAlarmState = 1  # WARNING
            if (alarmCounts[2]>0): 
                correctAlarmState = 2   # ALARM
            if (alarmCounts[3]>0):
                correctAlarmState = 3   # FALL
            #if (alarmCounts[5]>0):
            #    correctAlarmState = 5   # MANUAL ALARM
            if (evtAlarmState != correctAlarmState):
                print(event['id'], evtAlarmState, correctAlarmState, alarmCounts)
        else:
            print("ERROR - Event %s does not contain any datapoints" % event['id'])


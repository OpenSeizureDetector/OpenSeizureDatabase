#!/usr/bin/env python3

# Tidy up the raw data files that have been downloaded from the server.
# It does the following:
#    - Add manually specified seizure start and end times (read from a file)
#    - expand the dataJSON strings in the raw data so that each event
#         and each datapoint is a single object without embedded JSON strings.
#


import argparse
import sys
import os
import json
import importlib
import csv

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import libosd.osdDbConnection
import libosd.dpTools
import libosd.configUtils

filenamesLst = [
    'tcSeizures',
    'allSeizures',
    'fallEvents',
    'falseAlarms'
 #   'unknownEvents'
]

def readSeizureTimesObj(cfgObj, debug=False):
    seizureTimesObj = {}
    seizureTimesFname = libosd.configUtils.getConfigParam("seizureTimesFname", cfgObj)
    if os.path.exists(seizureTimesFname):
        fp = open(seizureTimesFname,'r')
        lines = csv.reader(fp)
        firstLine = True
        for line in lines:
            if firstLine:
                firstLine = False
            else:
                if (debug): print(line)
                seizureTimesObj[str(line[0])] = [ float(line[1]), float(line[2])]
    else:
        print("seizureTimesFname %s not found - not adding seizure times" % seizureTimesFname)
    if (debug): print(seizureTimesObj)
    return(seizureTimesObj)

def tidyDatapoint(cfgObj, dp, debug=False):
    eventId = dp['eventId']
    for dpParam in list(dp):
        # Loop through each element in the datapoint.
        if dpParam == "dataJSON":
            # Expand the dataJSON elements
            dpDataObj = json.loads(dp['dataJSON'])
            if ('dataJSON' in dpDataObj.keys()):
                if (dpDataObj['dataJSON'] is not None and dpDataObj['dataJSON']!=''):
                    try:
                        dpDataObj2 = json.loads(dpDataObj['dataJSON'])
                        for dpDataParam in dpDataObj2.keys():
                            if (dpDataParam=="dataTime" and 
                                "dataTime" in dp.keys()):
                                # Avoid overwriting existing dataTime element if it already exists(because NDA events do not have dataTime set in dataJSON)
                                if (debug): print("not overwriting existing dataTime element")
                            else:
                                if not dpDataParam in cfgObj['skipElements']:
                                    if (dpDataParam == "rawData"):
                                        # Truncate rawData to 125 elements (=5 seconds at 25 Hz)
                                        # FIXME - if we have a different sample frequency this will fail....
                                        dp['rawData'] = dpDataObj2['rawData'][:125]
                                    elif (dpDataParam == "rawData3D"):
                                        # Trunate rawData£d to 3*125 elements (=5 seconds at 25 Hz)
                                        # FIXME - if we have a different sample frequency this will fail....
                                        dp['rawData3D'] = dpDataObj2['rawData3D'][:3*125]
                                    else:
                                        dp[dpDataParam] = dpDataObj2[dpDataParam]
                    except json.JSONDecodeError as e:
                        print("Event ID %s: Error Decoding datapoint: %s" % \
                            (eventId, dpDataObj['dataJSON']))
            del dp['dataJSON']
    for dpParam in list(dp):
        # Copy 'normal' elements to output datapoint object.
        if dpParam in cfgObj['skipElements']:
            del dp[dpParam]
    return


def tidyEventObj(cfgObj, eventObj, debug=False):
    ''' Tidy eventObj in place, modifying the original object.'''
    if (debug): print("eventId=%s" % eventObj['id'])
    for param in list(eventObj):   # the list() is to get a copy of the keys at the start to avoid an error when the object is modified.
        if param == "dataJSON":
            # expand event dataJSON string into separate elements.
            if (eventObj['dataJSON'] is not None and eventObj['dataJSON']!=''):
                dataObj = json.loads(eventObj['dataJSON'])
                for dataParam in list(dataObj):
                    if (dataParam=="dataTime" and 
                        "dataTime" in eventObj.keys()):
                        # Avoid overwriting existing dataTime element if it already exists(because NDA events do not have dataTime set in dataJSON)
                        if (debug): print("not overwriting existing dataTime element")
                    else:
                        if not dataParam in cfgObj['skipElements']:
                            eventObj[dataParam] = dataObj[dataParam]
            del eventObj['dataJSON']
        elif (param == "datapoints"):
            # Expand each datapoints 'dataJSON' string to create an output datapoints list.
            dpLst = eventObj['datapoints']
            outDpLst = []
            for dp in dpLst:
                #Loop through each datapoint
                tidyDatapoint(cfgObj, dp, debug=debug)
    # Now prune out the elements to be skipped.
    for param in list(eventObj):
        if param in cfgObj['skipElements']:
            del eventObj[param]
    return

def loadSeizureTimes(cfgObj, debug=False):
    seizureTimesObj = readSeizureTimesObj(cfgObj, debug)
    if (debug): print("seizureTimesObj=",seizureTimesObj)
    if (debug): print("seizureTimesOBj.keys() ",list(seizureTimesObj))
    return seizureTimesObj

def updateEventSeizureTimes(eventObj, seizureTimesObj, debug=False):
    if seizureTimesObj is not None:
        eventId = str(eventObj['id'])
        if eventId in list(seizureTimesObj):
            if (debug): print("Adding seizure times to event id %s" % eventId)
            eventObj['seizureTimes'] = seizureTimesObj[eventId]
        else:
            # print("event '%s' not in seizureTimesObj '%s'" % (eventId, list(seizureTimesObj)[0]), type(eventId), type(list(seizureTimesObj)[0]))
            pass

def updateEventAlarmState(event, debug=False):
    ''' Scan through the datapoints associated with event Object event, and update the event alarm state parameter
    to be the 'worst' of the datapoints alarm states that make up the event.
    This is necessary because a but in makeOsdb.py has resulted in the event alarm state being incorrect for some false alarm events.
    GJ 17feb2023
    '''
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
            if (debug): print(event['id'], evtAlarmState, correctAlarmState, alarmCounts)
            event['osdAlarmState'] = correctAlarmState
    else:
        print("updateEventAlarmStates(): ERROR - Event %s does not contain any datapoints" % event['id'])


def updateDBSeizureTimes(cfgObj, inObj, debug=False):
    """
    Loop through each event in inObj and update the manually derived seizure times using the file specified in cfgObj.
    """
    seizureTimesObj = loadSeizureTimes(cfgObj, debug)
    for eventObj in inObj:
        updateEventSeizureTimes(eventObj, seizureTimesObj, debug)
    return

def updateDBAlarmStates(cfgObj, inObj, debug=False):
    """
    Loop through each event in inObj and update the alarm state of the event to be consistent with the 'worst' alarm state in the associated datapoints.
    """
    for eventObj in inObj:
        updateEventAlarmState(eventObj, debug)
    return


def tidyDbObj(cfgObj, inObj, debug=False):
    """
    Loop through each event in inObj and copy it to the returned outObj, expanding any dataJSON strings into
    parameters in the returned object
    """
    for eventObj in inObj:
        tidyEventObj(cfgObj, eventObj, debug)
    return


def tidyDbFile(cfgObj, inFname, outFname, debug=False):
    print("tidyDbFile(%s, %s, %d)" % (inFname, outFname, debug))
    osdIn = libosd.osdDbConnection.OsdDbConnection(debug=debug)
    osdOut = osdIn
    eventsObjLen = osdIn.loadDbFile(inFname)
    tidyDbObj(cfgObj, osdIn.getAllEvents(), debug)
    #osdOut.addEvents(outObj)
    osdOut.saveDbFile(outFname)
    print("Tidied data saved to file %s" % outFname)

def updateDbFileSeizureTimes(cfgObj, inFname, debug=False):
    print("updateDbFileSeizureTimes(%s,  %d)" % (inFname, debug))
    osdIn = libosd.osdDbConnection.OsdDbConnection(debug=debug)
    eventsObjLen = osdIn.loadDbFile(inFname)
    if (debug): print("Updating DB Object - starting with %d events" % len(osdIn.getAllEvents()))
    updateDBSeizureTimes(cfgObj, osdIn.getAllEvents(), debug)
    if (debug): print("Finished Updating DB Object - ending with %d events" % len(osdIn.getAllEvents()))
    if (debug): print("Saving DB Object")
    osdIn.saveDbFile(inFname)
    print("DB data saved to file %s" % inFname)

def updateDbFileAlarmStates(cfgObj, inFname, debug=False):
    print("updateDbFileAlarmStates(%s,  %d)" % (inFname, debug))
    osdIn = libosd.osdDbConnection.OsdDbConnection(debug=debug)
    eventsObjLen = osdIn.loadDbFile(inFname)
    if (debug): print("Updating DB Object")
    if (debug): print("Updating DB Object - starting with %d events" % len(osdIn.getAllEvents()))
    updateDBAlarmStates(cfgObj, osdIn.getAllEvents(), debug)
    if (debug): print("Finished Updating DB Object - ending with %d events" % len(osdIn.getAllEvents()))
    if (debug): print("Saving DB Object")
    osdIn.saveDbFile(inFname)
    print("DB data saved to file %s" % inFname)


def updateDb(cfgObj, inStr, outStr, times=False, alarmStates=False, debug=False):
    print("outStr=%s, debug=%d, cfgObj=" % (outStr, debug),cfgObj)

    for fnameBase in filenamesLst:
        inFname = "%s_%s.json" % (inStr, fnameBase)
        outFname = "%s_%s.json" % (outStr, fnameBase)
        print("inFname=%s" % inFname)
        if (times):
            print("Updating Seizure Event Times")
            updateDbFileSeizureTimes(cfgObj, inFname, debug)
        elif (alarmStates):
            print("Updating Alarm States")
            updateDbFileAlarmStates(cfgObj, inFname, debug)
        else:
            print("Tidying database")
            tidyDbFile(cfgObj,inFname, outFname, debug)


if (__name__=="__main__"):
    print("tidyDb.py.main()")
    parser = argparse.ArgumentParser(description='Tidy the database')
    parser.add_argument('--config', default="osdb.cfg",
                        help='name of json file containing configuration information and login credientials - see osdb.cfg.template')
    parser.add_argument('--debug', action='store_true',
                        help="Write debugging information to screen")
    parser.add_argument('--in', default="osdb_3min",
                        help='root of input filenames')
    parser.add_argument('--out', default="public",
                        help='root of output filenames')
    parser.add_argument('--times', action='store_true',
                        help="Update seizure times in input database")
    parser.add_argument('--alarmStates', action='store_true',
                        help="Update event alarm states in input database")
   
    argsNamespace = parser.parse_args()
    args = vars(argsNamespace)
    print(args)

    cfgObj = libosd.configUtils.loadConfig(args['config'])

    updateDb(cfgObj, args['in'], args['out'], times=args['times'], alarmStates=args['alarmStates'], debug=args['debug'])


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
    print("seizureTimesObj=",seizureTimesObj)
    print("seizureTimesOBj.keys() ",list(seizureTimesObj))
    return seizureTimesObj

def updateEventSeizureTimes(eventObj, seizureTimesObj, debug=False):
    if seizureTimesObj is not None:
        eventId = str(eventObj['id'])
        if eventId in list(seizureTimesObj):
            print("Adding seizure times to event id %s" % eventId)
            eventObj['seizureTimes'] = seizureTimesObj[eventId]
        else:
            # print("event '%s' not in seizureTimesObj '%s'" % (eventId, list(seizureTimesObj)[0]), type(eventId), type(list(seizureTimesObj)[0]))
            pass

def updateDBSeizureTimes(cfgObj, inObj, debug=False):
    """
    Loop through each event in inObj and update the manually derived seizure times using the file specified in cfgObj.
    """
    seizureTimesObj = loadSeizureTimes(cfgObj, debug)
    for eventObj in inObj:
        updateEventSeizureTimes(eventObj, seizureTimesObj, debug)
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
    inObj = osdIn.getAllEvents()
    tidyDbObj(cfgObj, inObj, debug)
    #osdOut.addEvents(outObj)
    osdOut.saveDbFile(outFname)
    print("Tidied data saved to file %s" % outFname)

def updateDbFileSeizureTimes(cfgObj, inFname, outFname, debug=False):
    print("updateDbFileSeizureTimes(%s,  %d)" % (inFname, debug))
    osdIn = libosd.osdDbConnection.OsdDbConnection(debug=debug)
    eventsObjLen = osdIn.loadDbFile(inFname)
    inObj = osdIn.getAllEvents()
    print("Updating DB Object")
    updateDBSeizureTimes(cfgObj, inObj, debug)
    print("Saving DB Object")
    osdIn.saveDbFile(inFname)
    print("DB data saved to file %s" % inFname)


def updateDb(cfgObj, inStr, outStr, times=False, debug=False):
    print("outStr=%s, debug=%d, cfgObj=" % (outStr, debug),cfgObj)

    for fnameBase in filenamesLst:
        inFname = "%s_%s.json" % (inStr, fnameBase)
        outFname = "%s_%s.json" % (outStr, fnameBase)
        print("inFname=%s" % inFname)
        if (times):
            updateDbFileSeizureTimes(cfgObj, inFname, debug)
        else:
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
   
    argsNamespace = parser.parse_args()
    args = vars(argsNamespace)
    print(args)

    cfgObj = libosd.configUtils.loadConfig(args['config'])

    updateDb(cfgObj, args['in'], args['out'], args['times'], args['debug'])


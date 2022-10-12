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
    'fallEvents'
 #   'falseAlarms',
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
                seizureTimesObj[line[0]] = [ float(line[1]), float(line[2])]
    else:
        print("seizureTimesFname %s not found - not adding seizure times" % seizureTimesFname)
    if (debug): print(seizureTimesObj)
    return(seizureTimesObj)

def tidyDatapoint(cfgObj, dp, debug=False):
    outDpObj = {}
    for dpParam in dp.keys():
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
                                    outDpObj['rawData'] = dpDataObj2['rawData'][:125]
                                elif (dpDataParam == "rawData3D"):
                                    outDpObj['rawData3D'] = dpDataObj2['rawData3D'][:3*125]
                                else:
                                    outDpObj[dpDataParam] = dpDataObj2[dpDataParam]
                    except json.JSONDecodeError as e:
                        print("Event ID %s: Error Decoding datapoint: %s" % \
                            (dpDataObj['eventId'], dpDataObj['dataJSON']))
        else:
            # Copy 'normal' elements to output datapoint object.
            if not dpParam in cfgObj['skipElements']:
                outDpObj[dpParam] = dp[dpParam]
    return(outDpObj)

def tidyDbObj(cfgObj, inObj, debug=False):
    """
    Loop through each event in inObj and copy it to the returned outObj, expanding any dataJSON strings into
    parameters in the returned object
    """
    outObj = []
    for eventObj in inObj:
        if (debug): print("eventId=%s" % eventObj['id'])
        tidyObj = {}
        for param in eventObj.keys():
            if param == "dataJSON":
                # expand event dataJSON string into separate elements.
                if (eventObj['dataJSON'] is not None and eventObj['dataJSON']!=''):
                    dataObj = json.loads(eventObj['dataJSON'])
                    for dataParam in dataObj.keys():
                        if not dataParam in cfgObj['skipElements']:
                            tidyObj[dataParam] = dataObj[dataParam]
            elif (param == "datapoints"):
                # Expand each datapoints 'dataJSON' string to create an output datapoints list.
                dpLst = eventObj['datapoints']
                outDpLst = []
                for dp in dpLst:
                    #Loop through each datapoint
                    outDpObj = tidyDatapoint(cfgObj, dp, debug=debug)
                    outDpLst.append(outDpObj)
                tidyObj['datapoints'] = outDpLst
            else:
                if not param in cfgObj['skipElements']:
                    tidyObj[param] = eventObj[param]
        outObj.append(tidyObj)
    return outObj


def tidyDbFile(cfgObj, inFname, outFname, debug=False):
    print("tidyDbFile(%s, %s, %d)" % (inFname, outFname, debug))
    osdIn = libosd.osdDbConnection.OsdDbConnection(debug=debug)
    osdOut = libosd.osdDbConnection.OsdDbConnection(debug=debug)
    eventsObjLen = osdIn.loadDbFile(inFname)
    inObj = osdIn.getAllEvents()
    outObj = tidyDbObj(cfgObj, inObj, debug)
    osdOut.addEvents(outObj)
    osdOut.saveDbFile(outFname)
    print("Tidied data saved to file %s" % outFname)



def tidyDb(cfgObj, inStr, outStr, debug):
    print("outStr=%s, debug=%d, cfgObj=" % (outStr, debug),cfgObj)

    seizureTimesObj = readSeizureTimesObj(cfgObj, debug)
    print("seizureTimesObj=",seizureTimesObj)

    for fnameBase in filenamesLst:
        inFname = "%s_%s.json" % (inStr, fnameBase)
        outFname = "%s_%s.json" % (outStr, fnameBase)
        print("inFname=%s" % inFname)
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
    
    argsNamespace = parser.parse_args()
    args = vars(argsNamespace)
    print(args)

    cfgObj = libosd.configUtils.loadConfig(args['config'])

    tidyDb(cfgObj, args['in'], args['out'],args['debug'])


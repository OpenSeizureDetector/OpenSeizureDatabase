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





def summariseOsdbFile(inFname, outFname, debug=False):
    '''
    Load the osdb data file inFname and output a text summary of the data contained in it.
    if outFname is None, sends output to stdout.
    '''

    osd = libosd.osdDbConnection.OsdDbConnection(debug=debug)

    if inFname is not None:
        print("flattenOsdb - loading file %s" % inFname)
        eventsObjLen = osd.loadDbFile(inFname, useCacheDir=False)
    else:
        print("No input file specified")
        return False    
    
    print("%d Events Loaded" % eventsObjLen)
    try:
        if outFname is not None:
            outPath = os.path.join(".", outFname)
            print("sending output to file %s" % outPath)
            outFile = open(outPath,'w')
        else:
            print("sending output to stdout")
            outFile = sys.stdout

        eventIdsLst = osd.getEventIds()
        nEvents = len(eventIdsLst)

        nNoDatapoints = 0
        nDatapoints = 0
        nSeizures = 0
        nNonSeizures = 0
        userIdCounts = {}
        userIdSeizureCounts = {}
        userIdNonSeizureCounts = {}
        seizureTypesCounts = {}
        nonSeizureTypesCounts = {}
        for eventNo in range(0,nEvents):
            eventId = eventIdsLst[eventNo]
            eventObj = osd.getEvent(eventId, includeDatapoints=True)
            userId = eventObj['userId']
            if (not 'datapoints' in eventObj or eventObj['datapoints'] is None):
                print("Event %s: No datapoints - skipping" % eventId)
                nNoDatapoints += 1
            else:
                #print("nDp=%d" % len(eventObj['datapoints']))
                nDatapoints += 1
                if (eventObj['type'].lower() == 'seizure'):
                    nSeizures += 1
                    if userId in userIdSeizureCounts:
                        userIdSeizureCounts[userId] += 1
                    else:
                        userIdSeizureCounts[userId] = 1
                    if 'subType' in eventObj and eventObj['subType'] is not None and len(eventObj['subType']) > 0:
                        subType = eventObj['subType'].lower()
                    else:
                        subType = "null"
                    if subType in seizureTypesCounts:
                        seizureTypesCounts[subType] += 1
                    else:
                        seizureTypesCounts[subType] = 1
                else:
                    nNonSeizures += 1
                    if userId in userIdNonSeizureCounts:
                        userIdNonSeizureCounts[userId] += 1
                    else:
                        userIdNonSeizureCounts[userId] = 1
                    if 'subType' in eventObj and eventObj['subType'] is not None and len(eventObj['subType']) > 0:
                        subType = eventObj['subType'].lower()
                    else:
                        subType = "null"
                    if subType in nonSeizureTypesCounts:
                        nonSeizureTypesCounts[subType] += 1
                    else:
                        nonSeizureTypesCounts[subType] = 1
                if userId in userIdCounts:
                    userIdCounts[userId] += 1
                else:
                    userIdCounts[userId] = 1
        outFile.write("Summary of %d events:\n" % nEvents)
        #outFile.write("  %d events with no datapoints\n" % nNoDatapoints)
        #outFile.write("  %d events with datapoints\n" % (nDatapoints))
        outFile.write("  %d seizures\n" % nSeizures)
        outFile.write("  %d non-seizures\n" % nNonSeizures)
        outFile.write("  %d unique users\n" % len(userIdCounts))
        outFile.write("  %d unique users with seizures\n" % len(userIdSeizureCounts))
        outFile.write("  %d unique users with non-seizures\n" % len(userIdNonSeizureCounts))

        outFile.write("\n")
        outFile.write("  Seizure types:\n")
        for seizureType in seizureTypesCounts.keys():
            outFile.write("    %s: %d (%.1f%%)\n" % (seizureType, seizureTypesCounts[seizureType], 100.*seizureTypesCounts[seizureType]/nSeizures))

        outFile.write("\n")
        outFile.write("  Non-seizure types:\n")
        for nonSeizureType in nonSeizureTypesCounts.keys():
            outFile.write("    %s: %d (%.1f%%)\n" % (nonSeizureType, nonSeizureTypesCounts[nonSeizureType], 100.*nonSeizureTypesCounts[nonSeizureType]/nNonSeizures))

        outFile.write("\n")
        outFile.write("  Users with seizures:\n")
        for userId in userIdSeizureCounts.keys():
            outFile.write("    %s: %d (%.1f%%)\n" % (userId, userIdSeizureCounts[userId], 100.*userIdSeizureCounts[userId]/nSeizures))
        #outFile.write("  Users with non-seizures:\n")
        #for userId in userIdNonSeizureCounts.keys():
        #    outFile.write("    %s: %d (%.1f%%)\n" % (userId, userIdNonSeizureCounts[userId], 100.*userIdNonSeizureCounts[userId]/nNonSeizures))
    finally:
        if (outFname is not None):
            outFile.close()
            print("Output written to file %s" % outFname)


    return True

 

def main():
    print("summariseOsdbFile.main()")
    parser = argparse.ArgumentParser(description='Produce a summary of the data stored in an OSDB JSON format file')
    parser.add_argument('-i', default=None,
                        help='Input filename (uses configuration datafiles list if not specified)')
    parser.add_argument('-o', default=None,
                        help='Output filename (uses stout if not specified)')
    parser.add_argument('--debug', action="store_true",
                        help='Write debugging information to screen')
    argsNamespace = parser.parse_args()
    args = vars(argsNamespace)
    print(args)

    summariseOsdbFile(args['i'], args['o'], args['debug'])

if __name__ == "__main__":
    main()

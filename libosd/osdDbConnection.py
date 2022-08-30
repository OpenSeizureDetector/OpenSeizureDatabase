#!/usr/bin/env python
"""
Python interface to the published static OSD seizure database
"""

import os
import json
import dateutil.parser

def dateStr2secs(dateStr):
    parsed_t = dateutil.parser.parse(dateStr)
    return parsed_t.timestamp()

def extractJsonVal(row, elem, debug=False):
    if (debug): print("extractJsonVal(): row=",row)
    dataJSON = row['dataJSON']
    if (dataJSON is not None):
        if (debug): print("extractJsonVal(): dataJSON=",dataJSON)
        dataObj = json.loads(dataJSON)
        if (elem in dataObj.keys()):
            elemVal = dataObj[elem]
        else:
            elemVal = None
    else:
        elemVal = None
    return(elemVal)





class OsdDbConnection:
    DEBUG = False
    cacheDir = os.path.join(os.path.expanduser("~"),"osd/osdb")
    cacheFname = "osdb"  # base of filename
    download = True
    maxEvents = 10000
    

    def __init__(self, cacheDir = None, debug=False):
        self.DEBUG = debug
        if (self.DEBUG): print("libosd.OsdDbConnection.__init__()")
        if (cacheDir is not None):
            self.cacheDir = cacheDir

        if (self.DEBUG): print("cacheDir=%s, debug=%s" %
              (self.cacheDir, self.DEBUG))
        self.eventsLst = []

    def loadDbFile(self, fname):
        ''' Retrieve a list of events data from a json file
        '''
        fpath = os.path.join(self.cacheDir, fname)
        if (self.DEBUG): print("OsdDbConnection.loadDbFile - fpath=%s" % fpath)
        fp = open(fpath,"r")
        self.eventsLst.extend(json.load(fp))
        fp.close()
        return(len(self.eventsLst))

            
    def getEvent(self, eventId, includeDatapoints=False):
        for event in self.eventsLst:
            if (event['id']==eventId):
                return event
        print("Event not found in cache")
        return None

    def getEventIds(self):
        """ Returns a list of all the eventIds in the database.
        """
        eventIdsLst = []
        for event in self.eventsLst:
            eventIdsLst.append(event['id'])
        return eventIdsLst
            

    def removeEvents(self, invalidLst):
        for evId in invalidLst:
            print("removeEvents: evId=%s" % evId)
            for event in self.eventsLst:
                if (event['id'] == evId):
                    print("Removing event Id %s" % evId)
                    self.eventsLst.remove(event)

    
    def listEvents(self):
        for event in self.eventsLst:
            phoneAppVersion = extractJsonVal(event,"phoneAppVersion",False)
            dataSource = extractJsonVal(event,"dataSourceName",False)
            watchSdName =extractJsonVal(event,"watchSdName",False)
            watchSdVersion =extractJsonVal(event,"watchSdVersion",False)
            print("%d, %s, %d, %s, %s, %s, %s, %s, %s, %s" %
                  (event['id'],
                   event['dataTime'],
                   event['userId'],
                   event['type'],
                   event['subType'],
                   phoneAppVersion,dataSource, watchSdName, watchSdVersion,
                   event['desc']
                   ))

    
if (__name__ == "__main__"):
    print("libosd.osdDbConnection.main()")
    osd = OsdDbConnection(debug=True)
    #eventsObjLen = osd.loadDbFile("osdb_tcSeizures.json")
    #eventsObjLen = osd.loadDbFile("osdb_allSeizures.json")
    eventsObjLen = osd.loadDbFile("osdb_falseAlarms.json")
    #eventsObjLen = osd.loadDbFile("osdb_unknownEvents.json")
    osd.listEvents()
    print("eventsObjLen=%d" % eventsObjLen)
    #eventsObjLen = osd.getEvents()
    #print("eventsObj = ", eventsObj)
    #print(eventsObj['results'])


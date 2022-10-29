#!/usr/bin/env python3
"""
Python interface to the published static OSD seizure database
"""

import os
import json
from xml.etree.ElementInclude import include
import jsbeautifier
import dateutil.parser
import sklearn.model_selection



def dateStr2secs(dateStr):
    parsed_t = dateutil.parser.parse(dateStr)
    return parsed_t.timestamp()


def extractJsonVal(row, elem, debug=False):
    if (debug):
        print("extractJsonVal(): row=", row)
    if (elem in row.keys()):
        elemVal = row[elem]
    elif 'dataJSON' in row.keys():
        dataJSON = row['dataJSON']
        if (dataJSON is not None):
            if (debug):
                print("extractJsonVal(): dataJSON=", dataJSON)
            dataObj = json.loads(dataJSON)
            if (elem in dataObj.keys()):
                elemVal = dataObj[elem]
            else:
                elemVal = None
        else:
            elemVal = None
    else:
        elemVal = None
    return (elemVal)


class OsdDbConnection:
    '''
    OsdDbConnection is a class that provides an interface to an
    OpenSeizureDatabase JSON file distribution.
    The files are assumed to be stored  in ~/osd/osdb unless
    cacheDir is provided to the contstructor to specify an alternative
    location.
    '''

    def __init__(self, cacheDir=None, debug=False):
        '''
        __init__: Creates an instance of an OsdDbConnection

        Parameters
        ----------
        cacheDir : String, optional
            Director path that contains the OSDB files. If None, defaults to ~/osd/osdb, by default None
        debug : bool, optional
            Write debugging information to console, by default False
        '''
        self.debug = debug
        if (self.debug):
            print("libosd.OsdDbConnection.__init__()")
        self.cacheDir = os.path.join(os.path.expanduser("~"), "osd/osdb")
        if (cacheDir is not None):
            self.cacheDir = cacheDir

        if (self.debug):
            print("cacheDir=%s, debug=%s" %
                  (self.cacheDir, self.debug))
        self.eventsLst = []

    def loadDbFile(self, fname):
        '''
        loadDbFile:  Retrieve a list of events data from a json file

        Parameters
        ----------
        fname : String
            file name of database file to load.
        '''
        fpath = os.path.join(self.cacheDir, fname)
        if (os.path.exists(fpath)):
            if (self.debug):  print("OsdDbConnection.loadDbFile - fpath=%s" % fpath)
            fp = open(fpath, "r")
            self.eventsLst.extend(json.load(fp))
            fp.close()
            return (len(self.eventsLst))
        else:
            print("ERROR: OsdDbConnection.loadDbFile - fpath %s does not exist" % fpath)
            return(0)

    def saveDbFile(self, fname, pretty=False):
        '''
        saveDbFile :  Save the loaded list of events data to a json file in the cache directory.

        Parameters
        ----------
        fname : String
            Filename of file to be written
        pretty: Boolean
            If true, use jsbeautifier to prettify output.

        Returns
        -------
        boolean
            True on success or False on error.
        '''
        fpath = os.path.join(self.cacheDir, fname)
        if (self.debug):
            print("OsdDbConnection.saveDbFile - fpath=%s" % fpath)
        try:
            fp = open(fpath, "w")
            jsonStr = json.dumps(self.eventsLst)
            if (pretty):
                options = jsbeautifier.default_options()
                options.indent_size = 2
                fp.write(jsbeautifier.beautify(jsonStr, options))
            else:
                fp.write(jsonStr)
            fp.close()
            if (self.debug):
                print("OsdDbConnection.saveDbFile - fpath=%s closed." % fpath)
            return True
        except Exception as e:
            print(type(e))    # the exception instance
            print(e.args)     # arguments stored in .args
            print(e)      
            print("OsdDbConnection.saveDbFile - Error Saving file %s" % fpath)
            return False

    def getAllEvents(self):
        """
        Return an object containing all the events in the database
        """
        return self.eventsLst

    def getEvent(self, eventId, includeDatapoints=False):
        '''
        getEvent : Retrieve a single event from the database by event ID.

        Parameters
        ----------
        eventId : String
            ID of event to be retrieived (type is String in case we switch to 
            firebase in the future, which uses string identifiers.)
        includeDatapoints : bool, optional
            If true includes an array of datapoint objects in the returned data, by default False

        Returns
        -------
        Dictionary
            A dictionary containing the event data, or None if event not found
        '''
        for event in self.eventsLst:
            #print("getEvent",type(eventId), type(self.eventsLst[0]['id']))
            if (event['id'] == eventId):
                return event
        print("osdDbConnection.getEvent(): Event %s not found in cache" % eventId)
        return None

    def getEvents(self, eventIdLst, includeDatapoints=False):
        '''
        getEvents : Retrieve a list of event objects based on the IDs in eventIdLst

        Parameters
        ----------
        eventIdLst : List of Strings
            Each element is the ID of an event to be retrieived (type is String in case we switch to 
            firebase in the future, which uses string identifiers.)
        includeDatapoints : bool, optional
            If true includes an array of datapoint objects in the returned data, by default False

        Returns
        -------
        List of Dictionaries
            A list of dictionaries containing the event data, or an empty list if events not found
        '''
        retLst = []
        for event in self.eventsLst:
            #print("getEvent",type(eventId), type(self.eventsLst[0]['id']))
            if (event['id'] in eventIdLst):
                retLst.append(event)
        return retLst

    def addEvent(self, eventObj):
        '''
        addEvent : Appends an event object to the stored list of events.

        Parameters
        ----------
        eventObj : Dictionary
            A dictionary describing an event.
        '''
        self.eventsLst.append(eventObj)

    def addEvents(self, newEventsLst):
        '''
        addEvents : Append a new list of event objects to the stored list of events.

        Parameters
        ----------
        newEventsLst : List of dictionaries
            A list of event objects
        '''
        self.eventsLst.extend(newEventsLst)

    def getEventIds(self, start=None, end=None):
        '''
        getEventIds : Returns a list of all the eventIds in the database.

        Parameters
        ----------
        start : String, optional
            Date/time of start of range, by default None
        end : String, optional
            Dae/time of end of range, by default None

        Returns
        -------
        list of strings
            list of eventIds that are stored in the database.

        FIXME:  date/time range does not do anything! 
        '''
        eventIdsLst = []
        for event in self.eventsLst:
            # Filter by date - FIXME - make this work!
            # if (start is not None):
            #    startDateTime = pd.to_datetime(start, utc=True)
            #    dateQueryStr = 'dataTime >= "%s"' % startDateTime
            #    print("Applying Date Query: %s" % dateQueryStr)
            #    df = df.query(dateQueryStr)

            # Filter by end date
            #    if (end is not None):
            #        endDateTime = pd.to_datetime(end, utc=True)
            #        dateQueryStr = 'dataTime <= "%s"' % endDateTime
            #        print("Applying Date Query: %s" % dateQueryStr)
            #        df = df.query(dateQueryStr)

            eventIdsLst.append(event['id'])
        return eventIdsLst

    def removeEvents(self, invalidLst):
        '''
        removeEvents : Remove the specified events from the loaded database.   

        Parameters
        ----------
        invalidLst : List of strings
            list of eventIds to be removed from the loaded database
        '''
        for evId in invalidLst:
            if (self.debug): print("removeEvents: evId=%s" % evId)
            for event in self.eventsLst:
                if (event['id'] == evId):
                    if (self.debug): print("Removing event Id %s" % evId)
                    self.eventsLst.remove(event)

    def listEvents(self):
        '''
        listEvents Write a summary list of the events stored in the database to the console.
        '''
        for event in self.eventsLst:
            phoneAppVersion = extractJsonVal(event, "phoneAppVersion", False)
            dataSource = extractJsonVal(event, "dataSourceName", False)
            watchSdName = extractJsonVal(event, "watchSdName", False)
            watchSdVersion = extractJsonVal(event, "watchSdVersion", False)
            print("%d, %s, %d, %s, %s, %s, %s, %s, %s, %s" %
                  (event['id'],
                   event['dataTime'],
                   event['userId'],
                   event['type'],
                   event['subType'],
                   phoneAppVersion, dataSource, watchSdName, watchSdVersion,
                   event['desc']
                   ))

    def getTestTrainEvents(self,
                           testProp=0.2):
        """
        Split the events in the current database into a set of test and a
        set of train data, and save them to new JSON files.
        """
        print("getTestTrainEvents - testProp=%.2f" %
              (testProp))
        print("Total Events=%d" % len(self.eventsLst))
        eventIdLst = self.getEventIds()
        print("Total Events=%d" % len(eventIdLst))

            # Split into test and train data sets.
        trainIdLst, testIdLst =\
            sklearn.model_selection.train_test_split(eventIdLst,
                                                    test_size=testProp,
                                                    random_state=4)
        print("len(train)=%d, len(test)=%d" % (len(trainIdLst), len(testIdLst)))
        print("test=",testIdLst)

        return(trainIdLst, testIdLst)

    def saveEventsToFile(self, eventIdLst, fname, includeDatapoints = False):
        eventsLst = self.getEvents(eventIdLst, includeDatapoints)
        outFile = open(fname,"w")
        outFile.write(json.dumps(eventsLst))
        outFile.close()

if (__name__ == "__main__"):
    print("libosd.osdDbConnection.main()")
    osd = OsdDbConnection(debug=True)
    eventsObjLen = osd.loadDbFile("osdb_3min_tcSeizures.json")
    osd.listEvents()
    print("eventsObjLen=%d" % eventsObjLen)
    trainEvents, testEvents = osd.getTestTrainEvents()

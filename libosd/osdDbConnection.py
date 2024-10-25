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

import pandas as pd
import csv

import libosd.osdUtils

def dateStr2secs(dateStr):
    """
    dateStr2secs Convert a string formatted date/time to unix timestamp (seconds from start of 1970)

    Args:
        dateStr (String): date/time as a string

    Returns:
        long: timestamp as seconds since the start of 1970
    """
    parsed_t = dateutil.parser.parse(dateStr)
    return parsed_t.timestamp()


def extractJsonVal(row, elem, debug=False):
    """
    extractJsonVal returns the value of element 'elem' in the object (dict) 'row'
    If 'elem' is not found in 'row', it looks for an element called 'dataJSON' and looks for 'elem' in that.

    Args:
        row (dict): an object representation of a database row
        elem (String): the name of the element to be extracted from row.
        debug (bool, optional): print debug information to console. Defaults to False.

    Returns:
        undefined: the value of the element 'elem'
    """
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

    def loadDbFile(self, fname, useCacheDir=True):
        '''
        loadDbFile:  Retrieve a list of events data from a json file.
        Uses the default cache directory unless useCacheDir is False

        Parameters
        ----------
        fname : String
            file name of database file to load.
        useCacheDir : Boolean
            If true (default) uses the default cache director to load files, otherwise uses the
            unmodified file name fname as the full path.
        '''
        if (useCacheDir):
            fpath = os.path.join(self.cacheDir, fname)
        else:
            fpath = fname
            
        if os.path.exists(fpath):
            if self.debug:  print("OsdDbConnection.loadDbFile - fpath=%s" % fpath)
            with open(fpath, "r") as fp:
                self.eventsLst.extend(json.load(fp))
            return len(self.eventsLst)
        else:
            print("ERROR: OsdDbConnection.loadDbFile - fpath %s does not exist" % fpath)
            return(0)



    def saveEventsToFile(self, eventIdLst, fname, pretty=False, useCacheDir = False):
        '''
        saveDbFile :  Save the list of events data to a json file in the cache directory.

        Parameters
        ----------
        eventIdLst: List of Object IDs to save
            List of event object ids to save.
        fname : String
            Filename of file to be written
        pretty: Boolean
            If true, use jsbeautifier to prettify output.
        useCacheDir: Boolean
            If true, save data to cache directory, other wise save to current working directory.

        Returns
        -------
        boolean
            True on success or False on error.
        '''
        if (useCacheDir):
            fpath = os.path.join(self.cacheDir,fname)
        else:
            fpath = fname
        if (self.debug):
            print("OsdDbConnection.saveEventsToFile_2 - fpath=%s" % fpath)

        eventsLst = self.getEvents(eventIdLst, includeDatapoints=True)

        if (self.debug): print("OsdDbConnection() - got events list")

        try:
            with open(fpath, "w") as fp:
                if (pretty):
                    if (self.debug): print("OsdDbConnection.saveDbFile() - pretty output selected - saving prettified file")
                    jsonStr = json.dumps(eventsLst)
                    if (self.debug): print("OsdDbConnection.saveDbFile() - created JSON string")
                    options = jsbeautifier.default_options()
                    options.indent_size = 2
                    fp.write(jsbeautifier.beautify(jsonStr, options))
                else:
                    if (self.debug): print("OsdDbConnection.saveDbFile() - saving unformatted file")
                    json.dump(eventsLst,fp)
            if (self.debug):
                print("OsdDbConnection.saveEventsToFile - fpath=%s closed." % fpath)
            return True
        except Exception as e:
            print(type(e))    # the exception instance
            print(e.args)     # arguments stored in .args
            print(e)      
            print("OsdDbConnection.saveDbFile - Error Saving file %s" % fpath)
            return False


    def saveDbFile(self, fname, pretty=False, useCacheDir=False):
        '''
        saveDbFile :  Save the loaded list of events data to a json file (in the cache directory if useCacheDir is True, 
            otherwise to current working directory).

        Parameters
        ----------
        fname : String
            Filename of file to be written
        pretty: Boolean
            If true, use jsbeautifier to prettify output.
        useCacheDir: Boolean
            If true, data is written to the cache directory, otherwise it is written to the current working directory.

        Returns
        -------
        boolean
            True on success or False on error.
        '''
        return self.saveEventsToFile(self.getEventIds(), fname, pretty=pretty, useCacheDir=useCacheDir)


    def saveIndexFile(self, fname, useCacheDir=False):
        '''
        Save a csv index of the events stored in the current database - if useCacheDir is true, saves to the cache directory, 
        otherwise saves to current working directory.'''
        if (self.debug): print("osdDbConnection: saveIndexFile - fname=%s, useCacheDir=%d" % (fname, useCacheDir))

        if (useCacheDir):
            fpath = os.path.join(self.cacheDir, fname)
        else:
            fpath = fname
        if (self.debug): print("osdbConnection.saveIndexFile: fpath=%s" % fpath)

        # Read the event list into a pandas data frame.
        df = pd.read_json(json.dumps(self.getAllEvents(includeDatapoints=False, debug=False)))
        #df['dataTime'] = pd.to_datetime(df['dataTime'], errors='raise', utc=True, format="%d-%m-%Y %H:%M:%S")
        df['dataTime'] = pd.to_datetime(df['dataTime'], errors='raise', utc=True, format="mixed", dayfirst=True)

        # Force the dataTime objects to be local time without tz offsets (avoids error about offset naive and offset aware datatime comparisons)
        #   (from https://stackoverflow.com/questions/46295355/pandas-cant-compare-offset-naive-and-offset-aware-datetimes)
        df['dataTime']=df['dataTime'].dt.tz_localize(None)

        df.sort_values(by='dataTime', ascending = True, inplace = True) 

        columnsLst = ['id', 'dataTime', 'userId', 'type', 'subType', 'osdAlarmState',
                      'dataSourceName', 'phoneAppVersion', 'watchSdVersion',
                      'has3dData', 'hasHrData', 'hasO2SatData', 'alarmFreqMin', 'alarmFreqMax', 'alarmThresh', 'alarmRatioThresh',
                      'desc']
        df.to_csv(fpath, columns = columnsLst, quoting=csv.QUOTE_NONNUMERIC, index=False)
        if (self.debug): print("osdDbConnection: saveIndexFile - data saved to file: %s" % fpath)


    def getAllEvents(self, includeDatapoints=True, debug=False):
        """
        Return an object containing all the events in the database
        """
        if (includeDatapoints):
            return self.eventsLst

        outLst = []
        for event in self.eventsLst:
            outEvent = event.copy()
            if ('datapoints' in outEvent.keys()):
                del outEvent['datapoints']
            else:
                if (debug): print("Event does not contain datapoints?", outEvent)
            outLst.append(outEvent)
        return (outLst)

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
            # force conversion to string so we can work with both numeric and alphanumeric event IDs
            if (str(event['id']) == str(eventId)):
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
        #print("getEvents - self.eventsLst = ", self.eventsLst)
        #print("getEvents - eventIdLst=",eventIdLst)
        for event in self.eventsLst:
            #print("getEvents",event['id'], type(event['id']), type(self.eventsLst[0]['id']))
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

    def getFilteredEventsLst(self, 
                             includeUserIds = None, 
                             excludeUserIds = None,
                             includeTypes = None,
                             excludeTypes = None,
                             includeSubTypes = None,
                             excludeSubTypes = None,
                             includeDataSources = None,
                             excludeDataSources = None,
                             includeText = None,
                             excludeText = None,
                             requireHrData = False,
                             requireO2SatData = False,
                             require3dData = False,
                             debug = False
                             ):
        """
        getFilteredEventsLst: returns a list of event IDs which satisfies the specified filters.

        Each of the 'include' filters is applied in turn to build up a list of events which satisfy the 'include'
        filters.   If all of the 'include' filters are None, it matches all results.
        Each of the 'exclude' filters is then applied in turn and events removed from the list which satisfy the 'exclude'
        filters.
        The resulting list of event ID strings is returned.

        Args:
            includeUserIds ([String], optional): Include only this list of user IDs. Defaults to None, in which case all user IDs are included
            excludeUserIds ([String], optional): Exclude this list of user IDs. Defaults to None, in which case no user IDs are excluded.
            includeTypes ([String], optional): Include only this list of event types. Defaults to None, in which case all event types are included.
            excludeTypes ([String], optional): Exclude this list of event types. Defaults to None, in which case no event types are excluded.
            includeSubTypes ([String], optional): Include only this list of event subTypes. Defaults to None, in which case all event subTypes are included.
            excludeSubTypes ([String], optional): Exclude this list of event subTypes. Defaults to None, in which case no event subTypes are excluded.
            includeDataSources ([String], optional): Include only this list of data sources. Defaults to None, in which case all event types are included.
            excludeDataSources ([String], optional): Exclude this list of data sources. Defaults to None, in which case no event types are excluded.
            includeText ([String], optional): Include only events containing this list of string descriptions. Defaults to None, in which case all descriptions are included.
            excludeText ([String], optional): Exclude events containing this list of string descriptions. Defaults to None, in which case no events are excluded.
        """
        eventsLst = []

        # if all of the include filters is None, we match all of the events.
        if ((includeUserIds is None or len(includeUserIds) == 0) and
            (includeDataSources is None or len(includeDataSources) == 0) and
            (includeTypes is None or len(includeTypes) == 0) and
            (includeSubTypes is None or len(includeSubTypes) == 0) and
            (includeText is None or len(includeText) == 0)):
            print("getFilteredEventsLst - all include filters are none, so returning all events")
            matchingUserIdLst = self.getMatchingElementsLst('userId', None, debug)
            nAdded = libosd.osdUtils.appendUniqueEntriesToLst(eventsLst, matchingUserIdLst)
            print("Added %d events" % nAdded)
        else:
            if (includeUserIds is not None):
                # Include User IDs
                matchingUserIdLst = self.getMatchingElementsLst('userId', includeUserIds, debug)
                nAdded = libosd.osdUtils.appendUniqueEntriesToLst(eventsLst, matchingUserIdLst)
                print("Added %d events matching the specified user Ids" % nAdded)

            if (includeTypes is not None):
                # Include Event Types
                matchingTypesLst = self.getMatchingElementsLst('type', includeTypes, debug)
                nAdded = libosd.osdUtils.appendUniqueEntriesToLst(eventsLst, matchingTypesLst)
                print("Added %d events matching the specified types" % nAdded)

            if (includeSubTypes is not None):
                # Include Event Subtypes
                matchingSubTypesLst = self.getMatchingElementsLst('subType', includeSubTypes, debug)
                nAdded = libosd.osdUtils.appendUniqueEntriesToLst(eventsLst, matchingSubTypesLst)
                print("Added %d events matching the specified subtypes" % nAdded)

            if (includeDataSources is not None):
                # Include Data Sources
                matchingDataSourcesLst = self.getMatchingElementsLst('dataSourceName', includeDataSources, debug)
                nAdded = libosd.osdUtils.appendUniqueEntriesToLst(eventsLst, matchingDataSourcesLst)
                print("Added %d events matching the specified data sources" % nAdded)

            if (includeText is not None):
                # Include Text
                matchingTextLst = self.getMatchingElementsLst('desc', includeText, debug)
                nAdded = libosd.osdUtils.appendUniqueEntriesToLst(eventsLst, matchingTextLst)
                print("Added %d events matching the specified text description" % nAdded)


        ## Now remove excluded events
        # Exclude User IDs
        if (excludeUserIds is not None):
            matchingUserIdLst = self.getMatchingElementsLst('userId', excludeUserIds, debug)
            nRemoved = libosd.osdUtils.removeEntriesFromLst(eventsLst, matchingUserIdLst)
            print("Removed %d events matching the specified user Ids" % nRemoved)

        if (excludeTypes is not None):
            # Exclude Event Types
            matchingTypesLst = self.getMatchingElementsLst('type', excludeTypes, debug)
            nAdded = libosd.osdUtils.removeEntriesFromLst(eventsLst, matchingTypesLst)
            print("Removed %d events matching the specified types" % nAdded)

        if (excludeSubTypes is not None):
            # Exclude Event subTypes
            matchingSubTypesLst = self.getMatchingElementsLst('subType', excludeSubTypes, debug)
            nAdded = libosd.osdUtils.removeEntriesFromLst(eventsLst, matchingSubTypesLst)
            print("Removed %d events matching the specified subTypes" % nAdded)

        if (excludeDataSources is not None):
            # Exclude Event subTypes
            matchingDataSourcesLst = self.getMatchingElementsLst('dataSourceName', excludeDataSources, debug)
            nAdded = libosd.osdUtils.removeEntriesFromLst(eventsLst, matchingDataSourcesLst)
            print("Removed %d events matching the specified datasources" % nAdded)

        if (excludeText is not None):
            # Exclude Event subTypes
            matchingTextLst = self.getMatchingElementsLst('desc', excludeText, debug)
            nAdded = libosd.osdUtils.removeEntriesFromLst(eventsLst, matchingTextLst)
            print("Removed %d events matching the specified text" % nAdded)

        # Filter out events which do not have 3d data
        if (require3dData):
            matching3dLst = self.getMatchingElementsLst('has3dData',[False], stringVals=False, debug=debug)
            nAdded = libosd.osdUtils.removeEntriesFromLst(eventsLst, matching3dLst)
            print("Removed %d events which do not contain 3d data" % nAdded)
            
        # Filter out events which do not have Hr data
        if (requireHrData):
            matchingHrLst = self.getMatchingElementsLst('hasHrData',[False], stringVals=False, debug=debug)
            nAdded = libosd.osdUtils.removeEntriesFromLst(eventsLst, matchingHrLst)
            print("Removed %d events which do not contain Hr data" % nAdded)
            
        # Filter out events which do not have 3d data
        if (requireO2SatData):
            matchingO2SatLst = self.getMatchingElementsLst('hasO2SatData',[False], stringVals=False, debug=debug)
            nAdded = libosd.osdUtils.removeEntriesFromLst(eventsLst, matchingO2SatLst)
            print("Removed %d events which do not contain O2Sat data" % nAdded)
            


        return eventsLst


    def getMatchingElementsLst(self, elementName = None, valsLst = None, stringVals = True, debug = False):
        """
        getMatchingElementsLst _summary_

        returns a list of event ID strings with where the event 'elementName' value is in valsLst.   
        Returns all events if valsLst is None.

        Args:
            elmentName ([String]): element name to match
            valsLst ([String], optional): List of element values to match. Defaults to None.
            stringVals [String]: compare values as substring comparision rather than simple value comparison..
        """

        if (elementName is None):
            if (debug): print("getMatchingElementsLst(): elementName is None - returning all events")
            return(self.getEventIds())

        if (valsLst is None):
            if (debug): print("getMatchingElementsLst(): valsLst is None - returning all events")
            return(self.getEventIds())
        
        matchingEventsLst = []
        for event in self.eventsLst:
            if (elementName in event):
                if (stringVals):
                    if any(val in str(event[elementName]) for val in valsLst):  # if str(event[elementName]) in valsLst:
                        if (debug): print("Matching event %s for element Name %s as substring comparison" % (event['id'], elementName))
                        matchingEventsLst.append(event['id'])
                else:
                    if event[elementName] in valsLst:
                        if (debug): print("Matching event %s for element %s" % (event['id'], elementName))
                        matchingEventsLst.append(event['id'])
        if (debug): print("getMatchingTypesLst() - matched %d events" % len(matchingEventsLst))
        return matchingEventsLst







    def listEvents(self):
        '''
        listEvents Write a summary list of the events stored in the database to the console.
        '''
        for event in self.eventsLst:
            phoneAppVersion = extractJsonVal(event, "phoneAppVersion", False)
            dataSource = extractJsonVal(event, "dataSourceName", False)
            watchSdName = extractJsonVal(event, "watchSdName", False)
            watchSdVersion = extractJsonVal(event, "watchSdVersion", False)
            print("%s, %s, %d, %s, %s, %s, %s, %s, %s, %s" %
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


if (__name__ == "__main__"):
    print("libosd.osdDbConnection.main()")
    osd = OsdDbConnection(debug=True)
    eventsObjLen = osd.loadDbFile("osdb_3min_tcSeizures.json")
    osd.listEvents()
    print("eventsObjLen=%d" % eventsObjLen)
    trainEvents, testEvents = osd.getTestTrainEvents()

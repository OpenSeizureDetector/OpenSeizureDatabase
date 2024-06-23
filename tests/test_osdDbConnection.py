#!/usr/bin/env python
'''
     _summary_ : Tests the libosd.osdDbConnection functions - such as including and excluding userIds.
'''
import random
import sys
import os
import unittest
import pandas as pd
#sys.path.append('..')


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import libosd.osdDbConnection
import libosd.configUtils


class TestOsdDbConnection(unittest.TestCase):
    def setUp(self):
        print("setUp Start")
        cacheDir = os.path.join(os.path.dirname(__file__), 'testData')
        self.configObj = {
            "cacheDir" : cacheDir,
            "nEvents": 100,
            "nDp": 5,
            "dataFiles": [
                "testAllData.json"
            ],
            "eventFilters": {
                "includeUserIds" : None, 
                "excludeUserIds" : None,
                "includeTypes" : None,
                "excludeTypes" : None,
                "includeSubTypes" : None,
                "excludeSubTypes" : None,
                "includeDataSources" : None,
                "excludeDataSources" : None,
                "includeText" : None,
                "excludeText" : None
            },
            "testProp" : 0.30,
            "trainDataFile": "trainData.json",
            "testDataFile": "testData.json",
 	        "invalidEvents": []
        }

        # Make a simplified dummy database object.
        allData = []
        dummyAcc = []
        dataTimeStr = "09-05-2022 02:37:25"
        userIdsLst = ["1", "2", "3", "4", "5"]

        for n in range(0,125):
            dummyAcc.append(n)
        dummyDp = {
            'id' : -1,
            'dataTime': dataTimeStr,
            'hr': 75,
            'rawData': dummyAcc
        }

        dummyDpArr = []
        for n in range(0,int(self.configObj['nDp'])):
            dummyDpArr.append(dummyDp)

        # Create dummy seizure data
        for n in range(0,int(self.configObj['nEvents']/2)):
            userId = userIdsLst[random.randint(0,len(userIdsLst)-1)]
            allData.append(
                { 'id': n,
                'userId': userId,
                'dataTime': dataTimeStr,
                'type': 'seizure',
                'subType': 'test seizure',
                'desc': 'test seizure',
                'dataSourceName':'test datasource 1',
                'has3dData': False,
                'hasHrData': True,
                'hasO2SatData': True,
                'datapoints': dummyDpArr
                }
            )

        # Create dummy false alarm data
        for n in range(int(self.configObj['nEvents']/2),int(self.configObj['nEvents'])):
            userId = userIdsLst[random.randint(0,len(userIdsLst)-1)]
            allData.append(
                { 'id': n+int(self.configObj['nEvents']/2),
                'userId': userId,
                'dataTime': dataTimeStr,
                'type': 'False Alarm',
                'subType': 'testing',
                'desc': 'testing false alarm',
                'dataSourceName':'test datasource 2',
                'has3dData': True,
                'hasHrData': False,
                'hasO2SatData': False,
                'datapoints': dummyDpArr
                }
            )
        self.osd = libosd.osdDbConnection.OsdDbConnection(cacheDir = self.configObj['cacheDir'], debug=False)
        self.osd.addEvents(allData)
        self.osd.saveDbFile("testAllData.json",True)
        pass
        print("setUp Finish")

    def tearDown(self):
        pass



    def calculateUserIdsCount(self):
        userIdsCount = {}
        for event in self.osd.getAllEvents(includeDatapoints=False, debug=False):
            if (event['userId'] in userIdsCount.keys()):
                userIdsCount[event['userId']] += 1
            else:
                userIdsCount[event['userId']] = 1
        #print("userIds Count = ",userIdsCount)
        return userIdsCount


    def calculateTypesCount(self):
        typesCount = {}
        for event in self.osd.getAllEvents(includeDatapoints=False, debug=False):
            if (event['type'] in typesCount.keys()):
                typesCount[event['type']] += 1
            else:
                typesCount[event['type']] = 1
        #print("types Count = ",typesCount)
        return typesCount


    def test_removeEntriesFromLst(self):
        print("test_removeEntriesFromLst Start")
        inLst = [ "1", "2", "3", "4", "5"]
        nRem = libosd.osdUtils.removeEntriesFromLst(inLst,["1", "6"])
        self.assertEqual(nRem, 1, "Incorrect number of entries removed")
        self.assertEqual(len(inLst),4,"Incorrect length of inLst")
        print("test_removeEntriesFromLst Finish")


    def test_getFilteredEventsLst(self):
        print("test_getFilteredEventsLst")

        print("test_setup - checking length of all data");
        self.assertEqual(len(self.osd.eventsLst),self.configObj['nEvents'],"Confirm length of all data is 100")
        print("test_setup finished")

        initialEventsCount = len(self.osd.eventsLst)
        userIdsCount = self.calculateUserIdsCount()

        filteredEventsLst = self.osd.getFilteredEventsLst(
            includeUserIds=None,
            excludeUserIds = None,
            includeTypes = None,
            excludeTypes = None,
            includeSubTypes = None,
            excludeSubTypes = None,
            includeDataSources = None,
            excludeDataSources = None,
            includeText = None,
            excludeText = None,
            debug = True
        )

        # With all filters set to None we should get all the events in the database
        dbLength = len(self.osd.getEventIds())
        self.assertEqual(len(filteredEventsLst), dbLength)

        filteredEventsLst = self.osd.getFilteredEventsLst(
            includeUserIds=[],
            excludeUserIds = [],
            includeTypes = [],
            excludeTypes = [],
            includeSubTypes = [],
            excludeSubTypes = [],
            includeDataSources = [],
            excludeDataSources = [],
            includeText = [],
            excludeText = [],
            debug = True
        )

        # With all filters set to None we should get all the events in the database
        dbLength = len(self.osd.getEventIds())
        self.assertEqual(len(filteredEventsLst), dbLength, "Using Empty Arrays rather than None")



        # Check Include User Ids
        print("Check Include User Ids")
        filteredEventsLst = self.osd.getFilteredEventsLst(
            includeUserIds=["2", "3"],
            excludeUserIds = None,
            includeTypes = None,
            excludeTypes = None,
            includeSubTypes = None,
            excludeSubTypes = None,
            includeDataSources = None,
            excludeDataSources = None,
            includeText = None,
            excludeText = None,
            debug = True
        )
        self.assertEqual(len(filteredEventsLst), userIdsCount["2"] + userIdsCount["3"],"Included userIds 2 and 3 incorrectly")

        # Check Exclude User Ids
        filteredEventsLst = self.osd.getFilteredEventsLst(
            includeUserIds=None,
            excludeUserIds = ["1"],
            includeTypes = None,
            excludeTypes = None,
            includeSubTypes = None,
            excludeSubTypes = None,
            includeDataSources = None,
            excludeDataSources = None,
            includeText = None,
            excludeText = None,
            debug = True
        )
        self.assertEqual(len(filteredEventsLst), initialEventsCount - userIdsCount["1"],"Excluded userid 1 incorrectly")

        # Check Include Event Types
        print("Check Include Event Types")
        filteredEventsLst = self.osd.getFilteredEventsLst(
            includeUserIds= None,
            excludeUserIds = None,
            includeTypes = ["seizure"],
            excludeTypes = None,
            includeSubTypes = None,
            excludeSubTypes = None,
            includeDataSources = None,
            excludeDataSources = None,
            includeText = None,
            excludeText = None,
            debug = True
        )
        self.assertEqual(len(filteredEventsLst), initialEventsCount/2,"Included seizure events incorrectly")

        # Check Include Event subTypes
        print("Check Include Event subTypes")
        filteredEventsLst = self.osd.getFilteredEventsLst(
            includeUserIds= None,
            excludeUserIds = None,
            includeTypes = ["seizure"],
            excludeTypes = None,
            includeSubTypes = ["testing"],
            excludeSubTypes = None,
            includeDataSources = None,
            excludeDataSources = None,
            includeText = None,
            excludeText = None,
            debug = True
        )
        # We are matching type seizure and the sub-type form false alarms, so we should return all data.
        self.assertEqual(len(filteredEventsLst), initialEventsCount,"Included type and subtype incorrectly")


        # Check Include DataSource
        print("Check Include DataSource")
        filteredEventsLst = self.osd.getFilteredEventsLst(
            includeUserIds= None,
            excludeUserIds = None,
            includeTypes = None,
            excludeTypes = None,
            includeSubTypes = None,
            excludeSubTypes = None,
            includeDataSources = ["test datasource 2"],
            excludeDataSources = None,
            includeText = None,
            excludeText = None,
            debug = True
        )
        # Half the events should be test datasource 2
        self.assertEqual(len(filteredEventsLst), initialEventsCount/2,"Included datasource incorrectly")

        # Check Include Full Text
        print("Check Include Text")
        filteredEventsLst = self.osd.getFilteredEventsLst(
            includeUserIds= None,
            excludeUserIds = None,
            includeTypes = None,
            excludeTypes = None,
            includeSubTypes = None,
            excludeSubTypes = None,
            includeDataSources = None,
            excludeDataSources = None,
            includeText = ["testing false alarm"],
            excludeText = None,
            debug = True
        )
        # Half the events should be matched
        self.assertEqual(len(filteredEventsLst), initialEventsCount/2,"Included full text incorrectly")

        # Check Include substring in text
        print("Check Include substring inText")
        filteredEventsLst = self.osd.getFilteredEventsLst(
            includeUserIds= None,
            excludeUserIds = None,
            includeTypes = None,
            excludeTypes = None,
            includeSubTypes = None,
            excludeSubTypes = None,
            includeDataSources = None,
            excludeDataSources = None,
            includeText = ["test"],
            excludeText = None,
            debug = True
        )
        # All events should be matched
        self.assertEqual(len(filteredEventsLst), initialEventsCount,"Included substring text incorrectly")

        # Check Exclude User Ids
        filteredEventsLst = self.osd.getFilteredEventsLst(
            includeUserIds=None,
            excludeUserIds = ["1"],
            includeTypes = None,
            excludeTypes = None,
            includeSubTypes = None,
            excludeSubTypes = None,
            includeDataSources = None,
            excludeDataSources = None,
            includeText = None,
            excludeText = None,
            debug = True
        )
        self.assertEqual(len(filteredEventsLst), initialEventsCount - userIdsCount["1"],"Excluded userid 1 incorrectly")

        # Check Exclude Types
        filteredEventsLst = self.osd.getFilteredEventsLst(
            includeUserIds=None,
            excludeUserIds = None,
            includeTypes = None,
            excludeTypes = ["seizure"],
            includeSubTypes = None,
            excludeSubTypes = None,
            includeDataSources = None,
            excludeDataSources = None,
            includeText = None,
            excludeText = None,
            debug = True
        )
        self.assertEqual(len(filteredEventsLst), initialEventsCount/2,"Excluded event type incorrectly")


        # Check Exclude Types
        filteredEventsLst = self.osd.getFilteredEventsLst(
            includeUserIds=None,
            excludeUserIds = None,
            includeTypes = None,
            excludeTypes = None,
            includeSubTypes = None,
            excludeSubTypes = ["test seizure", "testing"],
            includeDataSources = None,
            excludeDataSources = None,
            includeText = None,
            excludeText = None,
            debug = True
        )
        self.assertEqual(len(filteredEventsLst), 0 ,"Excluded event subType incorrectly")


        # Check Exclude Datasources
        filteredEventsLst = self.osd.getFilteredEventsLst(
            includeUserIds=None,
            excludeUserIds = None,
            includeTypes = None,
            excludeTypes = None,
            includeSubTypes = None,
            excludeSubTypes = None,
            includeDataSources = None,
            excludeDataSources = ["test datasource 2"],
            includeText = None,
            excludeText = None,
            debug = True
        )
        self.assertEqual(len(filteredEventsLst), initialEventsCount/2 ,"Excluded datasources incorrectly")

        # Check Exclude Text
        filteredEventsLst = self.osd.getFilteredEventsLst(
            includeUserIds=None,
            excludeUserIds = None,
            includeTypes = None,
            excludeTypes = None,
            includeSubTypes = None,
            excludeSubTypes = None,
            includeDataSources = None,
            excludeDataSources = None,
            includeText = None,
            excludeText = ["test"],
            debug = True
        )
        self.assertEqual(len(filteredEventsLst), 0 ,"Excluded text incorrectly")

        # Check Require3dData
        filteredEventsLst = self.osd.getFilteredEventsLst(
            includeUserIds=None,
            excludeUserIds = None,
            includeTypes = None,
            excludeTypes = None,
            includeSubTypes = None,
            excludeSubTypes = None,
            includeDataSources = None,
            excludeDataSources = None,
            includeText = None,
            excludeText = None,
            require3dData=True,
            requireHrData=False,
            requireO2SatData=False,
            debug = True
        )
        self.assertEqual(len(filteredEventsLst), 50 ,"Require 3dData incorrect")

        # Check RequireHrData
        filteredEventsLst = self.osd.getFilteredEventsLst(
            includeUserIds=None,
            excludeUserIds = None,
            includeTypes = None,
            excludeTypes = None,
            includeSubTypes = None,
            excludeSubTypes = None,
            includeDataSources = None,
            excludeDataSources = None,
            includeText = None,
            excludeText = None,
            require3dData=False,
            requireHrData=True,
            requireO2SatData=False,
            debug = True
        )
        self.assertEqual(len(filteredEventsLst), 50 ,"Require Hr incorrect")

        # Check RequireO2SatData
        filteredEventsLst = self.osd.getFilteredEventsLst(
            includeUserIds=None,
            excludeUserIds = None,
            includeTypes = None,
            excludeTypes = None,
            includeSubTypes = None,
            excludeSubTypes = None,
            includeDataSources = None,
            excludeDataSources = None,
            includeText = None,
            excludeText = None,
            require3dData=True,
            requireHrData=False,
            requireO2SatData=True,
            debug = True
        )
        self.assertEqual(len(filteredEventsLst), 0 ,"Require O2Sat Data incorrect")


if __name__ == "__main__":
    unittest.main()
    #    print("Everything passed")
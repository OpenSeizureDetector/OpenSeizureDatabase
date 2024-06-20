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
                'subType': 'test',
                'desc': 'test',
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
                'subType': 'test',
                'desc': 'test',
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
        # FIXME - the code is working as it is supposed to - the later 'includeXXX = None' statements tell it to include
        #         all events.   Need to think what this is really supposed to do!
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


    '''
    def test_excludeUserIds(self):
        print("test_excludeUserIds Start")
        initialEventsCount = len(self.osd.eventsLst)
        userIdsCount = self.calculateUserIdsCount()

        # Exclude user "1"
        self.osd.excludeUserIds(["1"])

        # Check the number of events in the database has decreased by the correct amount.
        self.assertEqual(len(self.osd.eventsLst), initialEventsCount - userIdsCount["1"],"Excluded userid 1 incorrectly")

        newUserIdsCount = self.calculateUserIdsCount()
        newInitialEventsCount = len(self.osd.eventsLst)

        self.osd.excludeUserIds(None)
        self.assertEqual(len(self.osd.eventsLst), newInitialEventsCount,"Null parameter should not change anything")
        print("test_excludeUserIds Finished")


    def test_getMatchingUserIdsLst(self):
        print("test_getMatchingUserIdsLst()")
        initialEventsCount = len(self.osd.eventsLst)
        userIdsCount = self.calculateUserIdsCount()

        # Include users "2" and "5"
        filteredLst = self.osd.getMatchingUserIdsLst(["2", "5"])

        # Check the number of events in the database has decreased by the correct amount.
        self.assertEqual(len(filteredLst), userIdsCount["2"] + userIdsCount["5"],"Included userIds 2 and 5 incorrectly")
        print("test_getMatchingUserIdsLst() - Finished")

    def test_getMatchingTypesLst(self):
        print("test_getMatchingTypesLst()")
        initialEventsCount = len(self.osd.eventsLst)
        userIdsCount = self.calculateUserIdsCount()

        # Include users "2" and "5"
        filteredLst = self.osd.getMatchingTypesLst(["seizure"])

        # Check the number of events in the database has decreased by the correct amount.
        self.assertEqual(len(filteredLst), 50,"Incorrect number of seizure events")
        print("test_getMatchingTypesLst() - Finished")



    def test_includeUserIds(self):
        print("test_includeUserIds Start")
        initialEventsCount = len(self.osd.eventsLst)
        userIdsCount = self.calculateUserIdsCount()

        # Include users "2" and "5"
        self.osd.includeUserIds(["2", "5"])

        # Check the number of events in the database has decreased by the correct amount.
        self.assertEqual(len(self.osd.eventsLst), userIdsCount["2"] + userIdsCount["5"],"Included userIds 2 and 5 incorrectly")

        newUserIdsCount = self.calculateUserIdsCount()
        newInitialEventsCount = len(self.osd.eventsLst)

        self.osd.includeUserIds(None)
        self.assertEqual(len(self.osd.eventsLst), newInitialEventsCount,"Null parameter should not change anything")
        print("test_includeUserIds Finished")


    #def test_includeTypeSubTypes(self):
    #    initialEventsCount = len(self.osd.eventsLst)
    #    typesCount = self.calculateTypesCount()#
    #
    #    # Include type "seizure"
    #    #self.osd.includeTypeSubTypes([ ["seizure", None]])

    #    # Check the number of events in the database has decreased by the correct amount.
    #    self.assertEqual(len(self.osd.eventsLst), typesCount["seizure"] ,"Included seizure events incorrectly")

    #    newTypesCount = self.calculateTypesCount()
    #    newInitialEventsCount = len(self.osd.eventsLst)

    #    #self.osd.includeTypeSubTypes(None)
    #    self.assertEqual(len(self.osd.eventsLst), newInitialEventsCount,"Null parameter should not change anything")
    '''

if __name__ == "__main__":
    unittest.main()
    #    print("Everything passed")
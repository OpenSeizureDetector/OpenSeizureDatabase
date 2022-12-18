#!/usr/bin/env python
'''
     _summary_ : Tests the various data augmentation functions to make sure they produce the expected results.
'''
import sys
import os
import datetime
import unittest
import pandas as pd
#sys.path.append('..')


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import libosd.osdDbConnection
import libosd.configUtils
import user_tools.nnTraining.makeTestTrainData
import user_tools.nnTraining.flattenOsdb


class TestSplit(unittest.TestCase):
    def setUp(self):
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

        for n in range(0,int(self.configObj['nEvents']/2)):
            if (n/2 == int(n/2)):
                userId = 1
            else:
                userId = 2
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
        for n in range(int(self.configObj['nEvents']/2),int(self.configObj['nEvents'])):
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

    def tearDown(self):
        pass


    def test_setup(self):
        print("test_setup - checking length of all data");
        self.assertEqual(len(self.osd.eventsLst),self.configObj['nEvents'],"Confirm length of all data is 100")

    def test_splitData(self):
        user_tools.nnTraining.makeTestTrainData.makeTestTrainData(self.configObj)      
        osdTrain = libosd.osdDbConnection.OsdDbConnection(cacheDir = self.configObj['cacheDir'], debug=False)
        osdTrain.loadDbFile(self.configObj['trainDataFile'])
        osdTest = libosd.osdDbConnection.OsdDbConnection(cacheDir = self.configObj['cacheDir'], debug=False)
        osdTest.loadDbFile(self.configObj['testDataFile'])

        lenTrain = len(osdTrain.getAllEvents())
        lenTest = len(osdTest.getAllEvents())

        print("Checking Total Number of Events")
        self.assertEqual(lenTrain+lenTest,self.configObj['nEvents'], "Checking Total Number of Events")

        print("Checking Train/Test Ratio")
        testFrac = lenTest / (lenTrain+lenTest)
        self.assertAlmostEqual(testFrac,self.configObj['testProp'])


class TestFlatten(unittest.TestCase):
    ''' Test the flattenOsdb.py module which converts a json formatted osdb file into a simple csv formated file.'''
    def setUp(self):
        cacheDir = os.path.join(os.path.dirname(__file__), 'testData')
        rawFilePath = os.path.join(cacheDir,"trainDataRaw.csv")
        self.configObj = {
            "cacheDir" : cacheDir,
            "nEvents": 100,
            "dataFiles": [
                "testAllData.json"
            ],
            "testProp" : 0.30,
            "trainDataFile": "trainData.json",
            "testDataFile": "testData.json",
            "trainCsvFileRaw": rawFilePath,

 	        "invalidEvents": []
        }


    def tearDown(self):
        pass


    def test_runFlattenOsdb(self):
        print("test runFlattenOsdb - checking length of all data");

        # Check we can load the training data file and it is the correct length.
        self.osdTrain = libosd.osdDbConnection.OsdDbConnection(cacheDir = self.configObj['cacheDir'], debug=False)
        self.osdTrain.loadDbFile(self.configObj['trainDataFile'])
        self.assertEqual(len(self.osdTrain.getAllEvents()),self.configObj['nEvents']*(1-self.configObj['testProp']))

        # Check that flattenOsdb runs
        user_tools.nnTraining.flattenOsdb.flattenOsdb(
            self.configObj['trainDataFile'], 
            self.configObj['trainCsvFileRaw'], 
            self.configObj, debug=False)

        # Read the csv file in, and check it has the same number of elements as the original train file.
        self.dfRaw = pd.read_csv(self.configObj['trainCsvFileRaw'])
        #print(len(self.dfRaw), self.dfRaw)

        # Calculate total number of datapoints in raw data
        nDp = 0
        for evObj in self.osdTrain.getAllEvents():
            nDp = nDp + len(evObj['datapoints'])
        self.assertEqual(nDp, len(self.dfRaw))



    def test_validateFlattenedData(self):
        osdTrain = libosd.osdDbConnection.OsdDbConnection(cacheDir = self.configObj['cacheDir'], debug=False)
        osdTrain.loadDbFile(self.configObj['trainDataFile'])
        evLst = osdTrain.getAllEvents()
        dfRaw = pd.read_csv(self.configObj['trainCsvFileRaw'])

        #print(dfRaw)
        rowNo = 0
        for evNo in range(0, len(evLst)):
            evObj = evLst[evNo]
            for dpNo in range(0,len(evObj['datapoints'])):
                # Check that the eventId has been transferred to the flattened data correctly.
                self.assertEqual(evObj['id'], dfRaw['id'][rowNo])

                # Check that the userid has been transferred to the flattened data correctly
                #print(evObj['userId'])
                #print(dfRaw.columns)
                self.assertEqual(evObj['userId'], dfRaw['userId'][rowNo])

                # Check that the hr has been transferred to the flattened data correctly
                self.assertEqual(evObj['datapoints'][dpNo]['hr'], dfRaw['hr'][rowNo])

                # Check that the first accelerationd datapoint has been transferred to the flattened data correctly
                self.assertEqual(evObj['datapoints'][0]['rawData'][0], dfRaw['M000'][rowNo])


                rowNo += 1
            

  
 
if __name__ == "__main__":
    unittest.main()
    #    print("Everything passed")
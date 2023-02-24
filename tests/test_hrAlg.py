#!/usr/bin/env python
'''
     _summary_ : Tests the hrAlg heart rate algorithm implementation (part of testRunner)
'''
import os
import sys
import unittest
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..','user_tools'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..','user_tools','testRunner'))
import user_tools.testRunner.hrAlg as hrAlg

class TestHrAlg(unittest.TestCase):
    def setUp(self):
        #[[4, 9]] * 3, columns=['A', 'B']
        rowLst=[]
        rowLst.append(self.makeRow(1,1,1,"dataTime1",-1, None))
        rowLst.append(self.makeRow(1,1,1,"dataTime1",-1, None))
        rowLst.append(self.makeRow(1,1,1,"dataTime1",70, None))
        rowLst.append(self.makeRow(2,1,1,"dataTime1",140, None))
        rowLst.append(self.makeRow(3,1,1,"dataTime1",70, None))
        rowLst.append(self.makeRow(4,2,1,"dataTime1",70, None))
        rowLst.append(self.makeRow(5,2,1,"dataTime1",70, None))
        rowLst.append(self.makeRow(6,3,1,"dataTime1",70, None))
        rowLst.append(self.makeRow(7,3,0,"dataTime1",70, None))
        rowLst.append(self.makeRow(8,3,0,"dataTime1",70, None))
        rowLst.append(self.makeRow(9,3,0,"dataTime1",70, None))
        rowLst.append(self.makeRow(10,3,0,"dataTime1",70, None))

        self.rowLst = rowLst

        algCfg = {
            'mode': "MODE_SIMPLE",
            'thresh_high': 110,
            'thresh_low': 50,
            'thresh_offset_high': 30,
            'thresh_offset_low': 30,
            'moving_average_time_window': 30
        }
        self.expectedWindowDps = 6  # 30 seconds x 5 sec per datapoint.

        algCfgStr = json.dumps(algCfg)
        print(algCfg, type(algCfg))
        print(algCfgStr, type(algCfgStr))
        self.hrAlg = hrAlg.HrAlg(algCfgStr, debug=True)
        for n in range(0,10):
            self.hrAlg.addToHist(self.rowLst[n]['hr'])



    def tearDown(self):
        pass

    def makeRow(self, id, userId, type, dataTime, hr, o2sat):
        '''
        parameters are:
        id - eventId
        userId - user ID
        type - seizure=1, non-seizure=0
        dataTime - data time
        hr - heart rate (bpm)
        o2sat - o2 saturation (%)
        adds dummy acceleration data.
        '''
        row = {'eventId': id,
               'userId': userId,
               'dataTime': dataTime,
               'hr': hr,
               'o2sat': o2sat}
        rawData=[]
        for n in range(0,125):
            rawData.append(1000.)
        row['rawData'] = rawData
        return(row)

        
    def test_setup(self):
        print('test_setup(): ', self.rowLst)
        self.assertEqual(len(self.rowLst),12)
        # 30 sec = 6 datapoints
        self.assertEqual(self.hrAlg.mMovingAverageTimeWindowDps,self.expectedWindowDps)

    def test_getHrDataFromJson(self):
        hrVal = self.hrAlg.getHrDataFromJson(json.dumps(self.rowLst[0]))
        self.assertEqual(hrVal,-1)
        hrVal = self.hrAlg.getHrDataFromJson(json.dumps(self.rowLst[3]))
        self.assertEqual(hrVal,140)

    def test_addToHist(self):
        for n in range(0,10):
            self.hrAlg.addToHist(self.rowLst[n]['hr'])
        self.assertEqual(len(self.hrAlg.mHRHist),self.expectedWindowDps)
        avHr = self.hrAlg.calcAvgHr()
        self.assertEqual(avHr,70)  # The last 6 data points should all be 70 bpm.

        # Now add in a bad data point and check it does not affect the average.
        self.hrAlg.addToHist(-1)
        avHr = self.hrAlg.calcAvgHr()
        self.assertEqual(avHr,70)  # The last data point should  be ignored as invalid.

    def test_checkAlarmSimple(self):
        self.assertEqual(self.hrAlg.checkAlarmSimple(150),1)
        self.assertEqual(self.hrAlg.checkAlarmSimple(101),0)
        self.assertEqual(self.hrAlg.checkAlarmSimple(40),1)

    def test_checkAlarmAdaptiveThreshold(self):
        print('test_checkAlarmAdaptiveThreshold: ', self.hrAlg.mHRHist)
        # Check the average is what we expect first.    
        self.assertEqual(self.hrAlg.calcAvgHr(), 70)
        # We expect average HR to be 70, so thresholds are 100 and 40
        self.assertEqual(self.hrAlg.checkAlarmAdaptiveThreshold(99),0)
        self.assertEqual(self.hrAlg.checkAlarmAdaptiveThreshold(100),0)
        self.assertEqual(self.hrAlg.checkAlarmAdaptiveThreshold(101),1)
        self.assertEqual(self.hrAlg.checkAlarmAdaptiveThreshold(40),0)
        self.assertEqual(self.hrAlg.checkAlarmAdaptiveThreshold(39),1)

    def test_checkAlarmAverageHR(self):
        print("test_checkAlarmAverageHR()")
        self.assertEqual(self.hrAlg.calcAvgHr(), 70)

        # Force average HR to 110
        for n in range(0,6):
            self.hrAlg.addToHist(110)
        self.assertEqual(self.hrAlg.checkAlarmAverageHR(),0)
        # Force average HR to 111
        for n in range(0,6):
            self.hrAlg.addToHist(111)
        self.assertEqual(self.hrAlg.checkAlarmAverageHR(),1)
        # Force average HR to 49
        for n in range(0,6):
            self.hrAlg.addToHist(49)
        self.assertEqual(self.hrAlg.checkAlarmAverageHR(),1)




if __name__ == "__main__":
    unittest.main()
    print("Everything passed")
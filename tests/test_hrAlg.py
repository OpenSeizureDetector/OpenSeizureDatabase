#!/usr/bin/env python
'''
     _summary_ : Tests the hrAlg heart rate algorithm implementation (part of testRunner)
'''
import os
import sys
import unittest
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import user_tools.testRunner.hrAlg as hrAlg

class TestHrAlg(unittest.TestCase):
    def setUp(self):
        #[[4, 9]] * 3, columns=['A', 'B']
        rowLst=[]
        rowLst.append(self.makeRow(1,1,1,"dataTime1","70", None))
        rowLst.append(self.makeRow(1,1,1,"dataTime1","70", None))
        rowLst.append(self.makeRow(1,1,1,"dataTime1","70", None))
        rowLst.append(self.makeRow(2,1,1,"dataTime1","70", None))
        rowLst.append(self.makeRow(3,1,1,"dataTime1","70", None))
        rowLst.append(self.makeRow(4,2,1,"dataTime1","70", None))
        rowLst.append(self.makeRow(5,2,1,"dataTime1","70", None))
        rowLst.append(self.makeRow(6,3,1,"dataTime1","70", None))
        rowLst.append(self.makeRow(7,3,0,"dataTime1","70", None))
        rowLst.append(self.makeRow(8,3,0,"dataTime1","70", None))
        rowLst.append(self.makeRow(9,3,0,"dataTime1","70", None))
        rowLst.append(self.makeRow(10,3,0,"dataTime1","70", None))

        self.rowLst = rowLst

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
        row = [id, userId, type, dataTime, hr, o2sat]
        for n in range(0,125):
            row.append(1000.)
        return(row)

        
    def test_userAug(self):
        '''Check that after applying user Augmentation that seizure events are equally balanced between users.'''
        augDf = user_tools.nnTraining.augmentData.userAug(self.df)
        seizuresDf, nonSeizureDf = user_tools.nnTraining.augmentData.getSeizureNonSeizureDfs(augDf)
        props = user_tools.nnTraining.augmentData.getUserCounts(seizuresDf)
        #print("test_userAug():\n", props)
        self.assertAlmostEqual(props[3],0.3333333)


if __name__ == "__main__":
    unittest.main()
    print("Everything passed")
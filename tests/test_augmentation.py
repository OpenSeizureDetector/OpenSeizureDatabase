#!/usr/bin/env python
'''
     _summary_ : Tests the various data augmentation functions to make sure they produce the expected results.
'''
import os
import sys
import unittest
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import user_tools.nnTraining.augmentData

class TestAug(unittest.TestCase):
    def setUp(self):
        #[[4, 9]] * 3, columns=['A', 'B']
        rowLst=[]
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

        columnsLst = ["id","userId","type","dataTime","hr","o2sat"]
        for n in range(0,125):
            columnsLst.append("M%03d" % n)
        self.df = pd.DataFrame(rowLst,
        columns=columnsLst)

        #print(self.df)

    def tearDown(self):
        pass

    def makeRow(self, id, userId, type, dataTime, hr, o2sat):
        row = [id, userId, type, dataTime, hr, o2sat]
        for n in range(0,125):
            row.append(1000.)
        return(row)

    def test_analyseDf(self):
        #augmentData.analyseDf(self.df)
        props = user_tools.nnTraining.augmentData.getUserCounts(self.df)
        print("props=")
        print(props)
        print("props[3]=")
        print(props[3])
        self.assertAlmostEqual(props[3],0.5)
        
    def test_userAug(self):
        '''Check that after applyig user Augmentation that seizure events are equally balanced between users.'''
        augDf = user_tools.nnTraining.augmentData.userAug(self.df)
        seizuresDf, nonSeizureDf = user_tools.nnTraining.augmentData.getSeizureNonSeizureDfs(augDf)
        props = user_tools.nnTraining.augmentData.getUserCounts(seizuresDf)
        #print("test_userAug():\n", props)
        self.assertAlmostEqual(props[3],0.3333333)

    def test_noiseAug(self):
        ''' Test the noise augmentation function. '''
        noiseVal = 10.
        noiseFac = 5
        seizuresDf, nonSeizureDf = user_tools.nnTraining.augmentData.getSeizureNonSeizureDfs(self.df)
        nSeizureEvents = len(seizuresDf)
        augDf = user_tools.nnTraining.augmentData.noiseAug(self.df, noiseVal, noiseFac, debug=True)
        seizuresDf, nonSeizureDf = user_tools.nnTraining.augmentData.getSeizureNonSeizureDfs(augDf)

        # Check we have created the correct number of new rows of augmented data.
        nAugSeizuesDf = len(seizuresDf)
        expectedNAugSeizuresDf = nSeizureEvents * (1+noiseFac)
        print("test_noiseAug")
        self.assertEqual(nAugSeizuesDf, expectedNAugSeizuresDf, "Number of rows in seizuresDf incorrect after noise augmentation")

        # Check the augmented data mean and standard deviation are correct.
        # Get the first row of augmented data (it follows after the measured seizure events)
        accStartCol = seizuresDf.columns.get_loc('M001')-1
        accEndCol = seizuresDf.columns.get_loc('M124')+1
        #print("accStartCol=%d, accEndCol=%d" % (accStartCol, accEndCol))
        rowArr = seizuresDf.iloc[nSeizureEvents+1]
        #print("rowArrLen=%d" % len(rowArr), type(rowArr), rowArr)
        accArr = rowArr.iloc[accStartCol:accEndCol]
        #print("accArrLen=%d" % len(accArr), type(accArr), accArr)
        inArr =np.array(accArr)
        meanVal = inArr.mean()
        stdVal =inArr.std()
        stdErr = (noiseVal - stdVal)/noiseVal # Fractional error in standard deviation.
        self.assertAlmostEqual(stdErr, 0. , places=0, msg="Noise Augmentation Standard Deviation")

    def test_phaseAug(self):
        '''Check that after applyig phase Augmentation that we have the correct number of seizure events.'''
        seizuresDf, nonSeizureDf = user_tools.nnTraining.augmentData.getSeizureNonSeizureDfs(self.df)
        nSeizureEvents = len(seizuresDf)
        
        augDf = user_tools.nnTraining.augmentData.phaseAug(self.df)
        seizuresDf, nonSeizureDf = user_tools.nnTraining.augmentData.getSeizureNonSeizureDfs(augDf)

        # Check we have created the correct number of new rows of augmented data.
        nAugSeizuesDf = len(seizuresDf)
        expectedNAugSeizuresDf = nSeizureEvents * (1+125)
        print("test_phaseAug")
        self.assertEqual(nAugSeizuesDf, expectedNAugSeizuresDf, "Number of rows in seizuresDf incorrect after phase augmentation")


        self.assertEqual(True, False, "FIXME- get phase augmentation testing working")

if __name__ == "__main__":
    unittest.main()
    print("Everything passed")
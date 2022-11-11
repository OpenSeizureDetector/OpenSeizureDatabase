#!/usr/bin/env python
'''
     _summary_ : Tests the various data augmentation functions to make sure they produce the expected results.
'''

import unittest
import pandas as pd

import augmentData

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
        props = augmentData.getUserCounts(self.df)
        print("props=")
        print(props)
        print("props[3]=")
        print(props[3])
        self.assertAlmostEqual(props[3],0.5)
        
    def test_userAug(self):
        augDf = augmentData.userAug(self.df)
        props = augmentData.getUserCounts(augDf)
        print(props)
        self.assertAlmostEqual(props[3],0.3333)


if __name__ == "__main__":
    unittest.main()
    print("Everything passed")
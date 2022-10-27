#!/usr/bin/env python

import numpy as np
import nnTrainer


testInLst=[]
for n in range(0,125):
    testInLst.append(1000)

augLst = nnTrainer.generateNoiseAugmentedData(testInLst,10,5,True)

origArr = np.array(testInLst)
origMean= np.mean(origArr)
origStd = 100. * np.std(origArr)/origMean
for inLst in augLst:
    #print(len(testInLst), len(inLst))
    inArr = np.array(inLst)
    inMean= np.mean(inArr)
    inStd = 100. * np.std(inArr)/inMean
    print("Original length=%d, mean=%.2f, std=%.1f%% : Aug length=%d, mean=%.2f, std=%.2f%%" %
        (len(testInLst), origMean, origStd, len(inLst), inMean, inStd))
    #print(inLst)
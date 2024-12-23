#!/usr/bin/env python

'''
'''
import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libosd.dpTools

import nnModel

class SvmModel(nnModel.NnModel):
    def __init__(self):
        print("SvmModel Constructor")

    def makeModel(self):
        # Fixme - initialise the SVM model.
        return None

    def dp2vector(self, dpObj):
        '''Convert a datapoint object into an input vector to be fed into the model.   
        Note that if dp is not a dict, it is assumed to be a json string
        representation instead.
        '''
        dpInputData = []
        if (type(dpObj) is dict):
            rawDataStr = libosd.dpTools.dp2rawData(dpObj)
        else:
            rawDataStr = dpObj
        accData, hr = libosd.dpTools.getAccelDataFromJson(rawDataStr)
        #print(accData, hr)

        if (accData is not None):
            accArr = np.array(accData)
        else:
            print("*** Error in Datapoint: ", dpObj)
            print("*** No acceleration data found with datapoint.")
            print("*** I recommend adding event %s to the invalidEvents list in the configuration file" % dp['eventId'])
            exit(-1)

        std = np.std(accArr)
        mean = np.mean(accArr)
        acc25pc = np.percentile(accArr, 25)
        acc50pc = np.percentile(accArr, 50)
        acc75pc = np.percentile(accArr, 75)
        iqr = acc75pc - acc25pc
        skew = pd.Series(accArr).skew()
        kurtosis = pd.Series(accArr).kurtosis()
        variance = np.var(accArr)
        min = np.min(accArr)
        max = np.max(accArr)
        range = max-min
        median = np.median(accArr)

        dpInputData = [
            std,
            mean,
            acc25pc,
            acc50pc,
            acc75pc,
            iqr,
            skew,
            kurtosis,
            variance,
            min,
            max,
            range,
            median
        ]


        return dpInputData
        
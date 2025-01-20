#!/usr/bin/env python

'''
AmberModel implements Jamie Pordoy's AMBER LSTM based model.
It uses Jamie's code in the 'amber' directory.
'''
import sys
import os
import math
import numpy as np
import keras

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import amber.model

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libosd.dpTools
import libosd.osdAlgTools as oat

import nnModel

class AmberModel(nnModel.NnModel):
    def __init__(self, configObj, debug=False):
        super().__init__(configObj, debug)
        print("AmberModel Constructor - debug=%d" % (self.debug))
        self.rowHidden = configObj['amberRowHidden']    # no. neurons in conv layers
        self.colHidden = configObj['amberColHidden']        # no. neurons in LSTM layers
        self.numClasses = 2

        self.modelInstance = amber.model.Amber(self.rowHidden, self.colHidden, self.numClasses)




    def makeModel(self, input_shape=None, num_classes=3, nLayers=3):
        ''' Create the keras/tensorflow model based on https://keras.io/examples/vision/mnist_convnet/
        note that this function ignotes the input_shape parameter and uses the shape derived within this class.
        '''
        self.modelInstance.build_model(num_features=1, input_shape=(125, 1))
        return self.modelInstance.model



    def accData2vector(self, accData, normalise=False):
        if (self.debug): print("accData2Vector(): ")
        return accData

    def dp2vector(self, dpObj, normalise=False):
        '''Convert a datapoint object into an input vector to be fed into the neural network.   Note that if dp is not a dict, it is assumed to be a json string
        representation instead.
        if normalise is True, applies Z normalisation to accelerometer data
        to give a mean of zero and standard deviation of unity.
        https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-normalize-or-standardize-a-dataset-in-python.md
        '''
        if (type(dpObj) is dict):
            rawDataStr = libosd.dpTools.dp2rawData(dpObj)
        else:
            rawDataStr = dpObj
        accData, hr = libosd.dpTools.getAccelDataFromJson(rawDataStr)
        #print(accData, hr)
        if (accData is not None):
            dpInputData = self.accData2vector(accData, normalise)
        else:
            print("*** Error in Datapoint: ", dpObj)
            print("*** No acceleration data found with datapoint.")
            print("*** I recommend adding event %s to the invalidEvents list in the configuration file" % dpObj['eventId'])
            dpInputData = None

        return dpInputData
        

def main():
    '''
    NOTE:  This main() function is only used for testing - this class
    would normally be used via the nnTrainer.py script.
    '''
    print("AmberModel.main() - Testing")
    nSamp = 125  # Number of samples per analysis datapoint.
    configObj={ "analysisSamp": 250, "specSamp": 50, "specStep":5}
    model = AmberModel(configObj, True)

    model.makeModel(num_classes=2, nLayers=3)

    # First Test data slicing using simple sequences of numbers

    rawData1 = [ x for x in range(0,nSamp) ]
    rawData2 = [ x for x in range(nSamp, 2*nSamp) ]
    rawData3 = [ x for x in range(2*nSamp, 3*nSamp) ]

    print(rawData1)
    print(rawData2)

    retVal = model.accData2vector(rawData1, False)
    retVal = model.accData2vector(rawData2, False)
    retVal = model.accData2vector(rawData3, False)

    model.resetAccBuf()
    # Now create some sine waves so we can test the spectral analysis.
    # Note that because we are calculating vector magnitude we see double the frequency generated
    #   in the spectrograph.
    # We get strange effects as we have signals go higher than 1g so suddenly go negative, then inverted by
    #   magnitude calculation.

    freq1 = 2   # Hz
    ampl1 = 4000  # mg
    phase1 = 0    # deg
    sampleFreq = 25.0  # Hz

    freq2 = 6
    ampl2 = 200
    phase2 = 0

    dc = 1000  #mg

    seqVals = [ x for x in range(0,3*nSamp) ]
    sinData = []
    for n in seqVals:
        timeSecs = n / sampleFreq
        a1 = ampl1*math.cos(2*math.pi * freq1*timeSecs + phase1*2*math.pi/360.)
        a2 = ampl2*math.cos(2*math.pi * freq2*timeSecs + phase2*2*math.pi/360.)
        sinData.append(int(dc + a1 + a2))

    sinData1 = sinData[0:nSamp]
    sinData2 = sinData[nSamp:2*nSamp]
    sinData3 = sinData[2*nSamp:3*nSamp]

    retVal = model.accData2vector(sinData1, False)
    retVal = model.accData2vector(sinData2, False)


if __name__ == "__main__":
    main()
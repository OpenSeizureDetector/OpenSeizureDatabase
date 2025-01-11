#!/usr/bin/env python

'''
SpecCnnModel converts the raw input data into a spectrogram, which is passed to an image processing style
CNN.
The key parameters that are expected in configObj are:
   - analysisSamp - total length of time sequence data (in sample counts) to analyse in the model (data is delivered in 5 second chunks, of 125 samples, so multiples of 125 are best)
   - specSamp - the length of time sequence data (in sample counts) to analyse for each vertical strip of the spectrogram 50 samples is 2 seconds, which gives a 0.5 Hz resolution.
'''
import sys
import os
import numpy as np
import keras

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libosd.dpTools

import nnModel

class SpecCnnModel(nnModel.NnModel):
    def __init__(self, configObj, debug=False):
        super().__init__(configObj, debug)
        print("SpecCnnModel Constructor - debug=%d" % (self.debug))
        self.analysisSamp = configObj['analysisSamp']    # The number of samples included in each analysis
        self.specSamp = configObj['specSamp']            # The number of samples used to calculate each spectrum
        self.specStep = configObj['specStep']            # The amount the spectrum 'window' moves between spectra.  If specStep==specSamp there is no overlap.
        self.accBuf = []

        self.sampleFreq = 25.0   # Note Hard Coded sample frequency!!!
        self.analysisTime = self.analysisSamp * self.sampleFreq   # The time period covered by each anaysis in seconds.
        self.specTime = self.specSamp / self.sampleFreq           # The time period covered by each spectrum in seconds.
        self.freqRes = 1.0/self.specTime   # The frequency resolution of the output spectrogram.
        #self.nSpec = int(self.analysisSamp / self.specSamp)   # The number of spectra in the spectrogram.
        self.nSpec = int(self.analysisSamp / self.specStep)   # The number of spectra in the spectrogram.
        
        self.nFreq = int(self.specSamp/2)                     # The number of frequency bins in each spectrum.

        self.inputShape = (self.nFreq, self.nSpec)

        if (self.debug): print("SpecCnnModel Constructor - analysisSamp=%d, specSamp=%d, nSpec=%d" % (self.analysisSamp, self.specSamp, self.nSpec))
        if (self.debug): print("SpecCnnModel Constructor - specTime = %.1f sec, freqRes=%.2f Hz, nFreq=%d" % (self.specTime, self.freqRes, self.nFreq))
        if (self.debug): print("SpecCnnModel Constructor:  inputShape=", self.inputShape)


    def makeModel(self, num_classes, nLayers=3):
        ''' Create the keras/tensorflow model'''
        input_layer = keras.layers.Input(self.inputShape)

        prevLayer = input_layer
        for n in range(0,nLayers):
            print("Adding convolution layer number %d" % (n+1))
            conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(prevLayer)
            conv1 = keras.layers.BatchNormalization()(conv1)
            conv1 = keras.layers.ReLU()(conv1)
            prevLayer=conv1

        gap = keras.layers.GlobalAveragePooling1D()(prevLayer)

        output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

        self.model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        return self.model

    def appendToAccBuf(self, accData):
        if (self.debug): print("appendToAccBuf(): len(accData)=%d, len self.accBuf=%d" % (len(accData), len(self.accBuf)))
    
        # Add accData to the end of accBuf
        self.accBuf.extend(accData)
        if (self.debug): print("appendToAccBuf(): after extend, len self.accBuf=%d" % (len(self.accBuf)))

        # if accBuf is now longer than required, trim it
        if len(self.accBuf) > self.analysisSamp:
            self.accBuf = self.accBuf[-self.analysisSamp:]

        if (self.debug): print("appendToAccBuf: after trim, len self.accBuf=%d" % (len(self.accBuf)))
        if (self.debug): print("appendToAccBuf: accBuf=",self.accBuf)


    def accData2vector(self, accData, normalise=False):
        if (self.debug): print("accData2Vector(): ")
        self.appendToAccBuf(accData)

        # Check if we have enough data in our accelerometer data buffer, return if not.
        if (len(self.accBuf)<self.analysisSamp):
            if (self.debug): print("accData2vector(): Insufficient data in buffer, returning None")
            return None

        i = 0
        while (i<=(len(self.accBuf)-self.specSamp)):  #
            specBuf = self.accBuf[i:i+self.specSamp]
            if (self.debug): print("accData2vector(): i=%d, len(specBuf)=%d, specBuf=" % (i, len(specBuf)), specBuf)
            if (len(specBuf)!=self.specSamp):
                print("ERROR - specBuf incorrect length????")
                exit(-1)
            i+= self.specStep


        dpInputData = []        
        for n in range(0,len(accData)):
            dpInputData.append(accData[n])
        
        return dpInputData

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
    print("SpecCnnModel.main() - Testing")
    nSamp = 125  # Number of samples per analysis datapoint.
    configObj={ "analysisSamp": 250, "specSamp": 50, "specStep":25}
    model = SpecCnnModel(configObj, True)

    model.makeModel(num_classes=2, nLayers=3)


    rawData1 = [ x for x in range(0,nSamp) ]
    rawData2 = [ x for x in range(nSamp, 2*nSamp) ]
    rawData3 = [ x for x in range(2*nSamp, 3*nSamp) ]

    print(rawData1)
    print(rawData2)

    retVal = model.accData2vector(rawData1, False)
    retVal = model.accData2vector(rawData2, False)
    retVal = model.accData2vector(rawData3, False)

if __name__ == "__main__":
    main()
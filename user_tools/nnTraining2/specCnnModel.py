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
import math
import numpy as np
import keras

import matplotlib.pyplot as plt
import matplotlib.cm as cm


sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libosd.dpTools
import libosd.osdAlgTools as oat

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
        if (self.specStep < self.specSamp):
            self.nSpec = int(self.analysisSamp / self.specStep) - int(self.specSamp/self.specStep) +1   # The number of spectra in the spectrogram.
        else:
            self.nSpec = int(self.analysisSamp / self.specStep)      # The number of spectra in the spectrogram.
        
        self.nFreq = int(self.specSamp/2)                     # The number of frequency bins in each spectrum.
        self.imgSeq = 0
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

    def resetAccBuf(self):
        print("resetAccBuf()")
        self.accBuf = []

    def generateSpectralHistoryFromAccelLst(self, accLst, windowLen=125, stepLen=125, normalise=False, zeroTol=0.001, sdThresh=10):
        '''
        Returns a numpy array representing the spectral history.   
        Any values where |value|<tol are set to zero
        if the standard deviation (mili-g) of the acceleration in a time slice is less than sdThresh,
        the output is set to zero so we do not see noise in the image for very low movement levels.
        windowLen is the number of samples to analyse - 125 samples at 25 Hz is a 5 second window.
        Note:  This is based on the version in ../dataSummariser/eventAnalyser.py
        '''
        if (self.debug): print("generateSpectralHistoryFromAccelLst():  len(accLst)=%d, windowLen=%d, stepLen=%d" % (len(accLst), windowLen, stepLen))

        specLst = []
        fftLen = int(windowLen/2)

        rawArr = np.array(accLst, dtype="float")    
        endPosn = windowLen
        while endPosn<=len(accLst):
            sliceRaw = rawArr[endPosn-windowLen:endPosn]
            # remove DC component.
            sliceAvg = np.mean(sliceRaw)
            print(sliceAvg, type(sliceAvg), type(sliceRaw))
            slice = sliceRaw - sliceAvg
            if (self.debug): print("generateSpectralHistoryFromAccelLst(): sliceAvg=%.1f, sliceRaw=" % (sliceAvg), sliceRaw)
            if (self.debug): print("generateSpectralHistoryFromAccelLst(): slice   =", slice)

            sliceStd = slice.std()    #/  slice.mean()
            if (self.debug): print("generateSpectralHistoryFromAccelLst():  endPosn=%d, slice.mean=%.3f mg, slice..std=%.3f mg" % (endPosn, slice.mean(), slice.std()))
            if (sliceStd >=sdThresh):
                fft, fftFreq = oat.getFFT(slice, sampleFreq=25)
                fftMag = np.absolute(fft)
                #print(fftMag)
                # Clip small values to zero to reduce normalisation artefacts.
                fftMag[abs(fftMag) < zeroTol] = 0
                if (normalise):
                    if np.max(fftMag[0:fftLen]) != 0:
                        specLst.append(fftMag[0:fftLen]/np.max(fftMag[0:fftLen]))   
                    else:
                        specLst.append(np.zeros(fftLen))   # Force output to zero if all values are zero
                else:
                    specLst.append(fftMag[0:fftLen])   
            else:
                specLst.append(np.zeros(fftLen))   # Zero the output if there is very low movement.
            endPosn += stepLen
        specImg = np.stack(specLst, axis=1)

        return specImg


    def plotSpectralHistory(self, specImg, outFname="specHist.png"):
        fig, ax = plt.subplots(2,1)
        fig.set_figheight(200/25.4)
        fig.suptitle("Spectral History",
                     fontsize=11)

        # Bilinear interpolation - this will look blurry, 'nearest' is blocky
        ax[0].imshow(specImg, interpolation='nearest', origin='lower', aspect=5, cmap="Oranges",
                   extent=[0,len(self.accBuf)/25.,0,12.5])
        #ax[0].set_yticks(np.arange(0, 12.5, 1))
        ax[0].set_yticks([3,8])
        ax[0].grid(which='major', color='k', linestyle='-', linewidth=0.3)
        ax[0].title.set_text("Vector Magnitude")
        ax[0].set_xlabel('Time (sec)')
        ax[0].set_ylabel('Freq (Hz)')

        rawTimestampLst = [ x / self.sampleFreq for x in range(0,len(self.accBuf))]
        ax[1].plot(rawTimestampLst,self.accBuf)

        plt.tight_layout()
        fig.savefig(outFname)
        print("image written to %s" % outFname)
        plt.close(fig)


    def accData2vector(self, accData, normalise=False):
        if (self.debug): print("accData2Vector(): ")
        self.appendToAccBuf(accData)

        # Check if we have enough data in our accelerometer data buffer, return if not.
        if (len(self.accBuf)<self.analysisSamp):
            if (self.debug): print("accData2vector(): Insufficient data in buffer, returning None")
            return None

        specImg = self.generateSpectralHistoryFromAccelLst(self.accBuf, 
                                                            windowLen=self.specSamp, stepLen=self.specStep,
                                                            normalise=False, zeroTol=0.001, sdThresh=10)
        self.plotSpectralHistory(specImg, "specHist_%03d.png" % self.imgSeq)
        self.imgSeq += 1

        if (self.debug): print("accData2Vector() - return shape is ",specImg.shape)

        if (specImg.shape != self.inputShape):
            print("ERROR:  Expected Input shape is ",self.inputShape,", but image shape is ", specImg.shape)
            exit(-1)

        #i = 0
        #while (i<=(len(self.accBuf)-self.specSamp)):  #
        #    specBuf = self.accBuf[i:i+self.specSamp]
        #    if (self.debug): print("accData2vector(): i=%d, len(specBuf)=%d, specBuf=" % (i, len(specBuf)), specBuf)
        #    if (len(specBuf)!=self.specSamp):
        #        print("ERROR - specBuf incorrect length????")
        #        exit(-1)
        #    i+= self.specStep


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
    '''
    NOTE:  This main() function is only used for testing - this class
    would normally be used via the nnTrainer.py script.
    '''
    print("SpecCnnModel.main() - Testing")
    nSamp = 125  # Number of samples per analysis datapoint.
    configObj={ "analysisSamp": 250, "specSamp": 50, "specStep":5}
    model = SpecCnnModel(configObj, True)

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
    ampl1 = 500  # mg
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
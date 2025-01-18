#!usr/bin/env python3

import sys
import os
import keras
import keras.saving
from keras.models import Sequential
from keras.layers import GRU, LSTM
import numpy as np
import joblib

import json
import numpy as np
import sdAlg

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../nnTraining2'))
import nnTraining2.specCnnModel
import libosd

class SpecAlg(sdAlg.SdAlg):
    '''
    A testRunner representation of the proposed spectragram model - see the nnTraining2 folder for
    details of the model.
    '''
    def __init__(self, settingsStr, debug=True):
        print("SpecAlg.__init__() - settingsStr=%s" % settingsStr)
        print("SpecAlg.__init__(): settingsStr=%s (%s)"
                               % (settingsStr, type(settingsStr)))
        super().__init__(settingsStr, debug)

        self.mModelFname = self.settingsObj['modelFname']
        self.mModeStr = self.settingsObj['mode']
        self.mSamplePeriod = self.settingsObj['samplePeriod']
        self.mWarnTime = self.settingsObj['warnTime']
        self.mAlarmTime = self.settingsObj['alarmTime']
        if ('sdThresh') in self.settingsObj:
            self.mSdThresh = self.settingsObj['sdThresh']
            print("SpecAlg.__init__:  Set mSdThresh to %.2f" % self.mSdThresh)
        else:
            self.mSdThresh = 5.0    # Default to 5% stdev threshold


        self.alarmState = 0
        self.alarmCount = 0

        self.modelTools = nnTraining2.specCnnModel.SpecCnnModel(self.settingsObj, debug=False)
        
        #Load Model From Yout URL path
        self.model = keras.models.load_model(self.mModelFname)
        #Model Summary
        self.model.summary()
        
    def dp2vector(self, dpObj, inputFormat=1, normalise=False):
        '''Convert a datapoint object into an input vector to be fed into the neural network.   Note that if dp is not a dict, it is assumed to be a json string
            representation instead.
        '''
        inputArry = self.modelTools.dp2vector(dpObj,False)


        return inputArry


        
    def processDp(self, dpStr, eventId):
        #print(dpStr)
        #inputLst = nnTraining.nnTrainer.dp2vector(dpStr, normalise=False)
        inputArry = self.dp2vector(dpStr, inputFormat=3, normalise=False)

        pSeizure = 0.0
        if (inputArry is None):
            # dp2vector returns none for invalid data or low standard deviation data
            inAlarm = False
        else:
            inputArry = inputArry.reshape(1,inputArry.shape[0], inputArry.shape[1], 1)
            # we use predict_on_batch() rather than predict() because it is much, much faster.
            #retVal = self.model.predict(inputArry, verbose=0)
            #print("processDp - inputArry=",inputArry, inputArry.shape)
            retVal = self.model.predict_on_batch(inputArry)
            #print(retVal)
            pSeizure = retVal[0][1]
            if (pSeizure>0.5):
                #print("ALARM - pSeizure=%f" % pSeizure)
                inAlarm=True
            else:
                inAlarm=False

        if (inAlarm):
            #print("inAlarm - roiPower=%f, roiRatio=%f" % (roiPower, roiRatio))
            self.alarmCount += self.mSamplePeriod
            #print("alarmCount=%d" % self.alarmCount)

            if (self.alarmCount > self.mAlarmTime):
                self.alarmState = 2
            elif (self.alarmCount > self.mWarnTime):
                self.alarmState = 1
        else:
            # if we are not in alarm state revert back to warning or ok.
            if (self.alarmState == 2):
                self.alarmState = 1
                self.alarmCount = self.mWarnTime # + 1 // to give agreement with phone version
            else:
                self.alarmState = 0
                self.alarmCount = 0

        # If we are in 'single' mode, just report the alarm state
        # based on this current datapoint - otherwise we report the
        # result based on this and previous datapoints derived above.
        if (self.mModeStr == 'single'):
            if (inAlarm):
                self.alarmState = 2
            else:
                self.alarmState = 0
            
        extraData = {
            'alarmState': self.alarmState,
            'pSeizure': float(pSeizure)
            }
        return json.dumps(extraData)
        
    def resetAlg(self):
        self.alarmState = 0
        self.alarmCount = 0




if __name__ == "__main__":
    print("SpecAlg.main()")
 
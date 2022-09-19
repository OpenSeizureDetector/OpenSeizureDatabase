#!usr/bin/python3

import sys
import os
import keras
from keras.models import Sequential
from keras.layers import GRU, LSTM
import numpy as np
import joblib

import json
import numpy as np
import sdAlg

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import nnTraining.nnTrainer

class NnAlg(sdAlg.SdAlg):
    def __init__(self, settingsStr, debug=True):
        print("nnAlg.__init__() - settingsStr=%s" % settingsStr)
        print("nnAlg.__init__(): settingsStr=%s (%s)"
                               % (settingsStr, type(settingsStr)))
        super().__init__(settingsStr, debug)

        self.mModelFname = self.settingsObj['modelFname']
        self.mModeStr = self.settingsObj['mode']
        self.mSamplePeriod = self.settingsObj['samplePeriod']
        self.mWarnTime = self.settingsObj['warnTime']
        self.mAlarmTime = self.settingsObj['alarmTime']

        self.alarmState = 0
        self.alarmCount = 0

        
        #Load Model From Yout URL path
        self.model = keras.models.load_model(self.mModelFname)
        #Model Summary
        self.model.summary()
        


        
    def processDp(self, dpStr):
        #print(dpStr)
        inputLst = nnTraining.nnTrainer.dp2vector(dpStr, normalise=False)
        #print("inputLst=",inputLst)
        inputArry = np.array(inputLst).reshape((1,125,1))

        retVal = self.model.predict(inputArry, verbose=0)
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
#            'specPower': specPower,
#            'roiPower': roiPower,
#            'roiRatio': roiRatio,
            'alarmState': self.alarmState,
            #'fftArr': fftArr,
            #'fftFreq': fftFreq,
            }
        return json.dumps(extraData)
        #retVal = {"alarmState": 0}
        #return(json.dumps(retVal))
        
    def resetAlg(self):
        self.alarmState = 0
        self.alarmCount = 0

                  
if __name__ == "__main__":
    print("osdAlg.Jamie1Alg.main()")
    settingsObj = {
        "alarmFreqMin" : 3,
        "alarmFreqMax" : 8,
        "alarmThreshold" : 100,
        "alarmRatioThreshold" : 57
        }
    alg = Jamie1Alg(json.dumps(settingsObj),True)

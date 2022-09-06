#!/usr/bin/env python

"""
A seizure detection algorithm based on a simple example neural network.

Jamie Pordoy, 2022
"""

import json
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import pickle
import joblib

import sdAlg

class SimpleNNAlg(sdAlg.SdAlg):
    def __init__(self, settingsStr, debug=False):
        print("SimpleNNAlg.__init__() - settingsStr=%s" % settingsStr)
        print("simpleNNA.__init__(): settingsStr=%s (%s)"
                               % (settingsStr, type(settingsStr)))
        super().__init__(settingsStr, debug)

        modelFname = self.settingsObj['modelFile']
        print("Loading Model %s..." % modelFname)
        #self.model = pickle.load(open(modelFname,'rb'))  #,'rb'
        #self.model = tf.keras.models.load_model(modelFname)
        self.model = joblib.load(modelFname)
        self.model.summary()

        

    def getAlarmState(self, rawData):
        ''' getAlarmState(rawData) - determines the alarm state associated with the snapshot
        of raw seizure detector data rawData[]
         @return the alarm state (0=ok, 1 = alarm)
        '''
        if (self.DEBUG): print("FIXME - implement getAlarmState by evaluating neural network for given data")
        inputVector = []
        print(rawData)
        #inputVector.append(rawData['roiPower'])
        for n in range(0,256):
            inputVector.append(n)
        print("inputVector=",inputVector)
        alarmState = 0
        alarmState = self.model.predict(inputVector)
        print("alarmState=",alarmState)
        return(alarmState)
        
    def processDp(self, dpStr):
        print(dpStr)
        inAlarm = self.getAlarmState(json.loads(dpStr))
        
        extraData = {
            'alarmCount': 0,
            'alarmState': inAlarm,
            }
        return json.dumps(extraData)
        
    def resetAlg(self):
        self.alarmState = 0
        self.alarmCount = 0

                  
if __name__ == "__main__":
    print("osdAlg.OsdAlg.main()")
    settingsObj = {
        "alarmFreqMin" : 3,
        "alarmFreqMax" : 8,
        "alarmThreshold" : 100,
        "alarmRatioThreshold" : 57
        }
    alg = LstmAlg(json.dumps(settingsObj),True)

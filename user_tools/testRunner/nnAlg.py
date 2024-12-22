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
sys.path.append(os.path.join(os.path.dirname(__file__), '../nnTraining'))
import nnTraining.nnTrainer
import nnTraining.cnnDeepModel
import libosd

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

        # inputFormat is used by dp2vector to create the input array required for the specific network.
        # inputFormat = 1:  A simple 125 point array of accelerometer readings
        # inputFormat = 2:  A two dimensional array (2, 125)  The first row is acceleromater readings as per inputFormat 1, the other is HR values.
        #
        if not "inputFormat" in self.settingsObj.keys():
            self.inputFormat = 1
        else:
            self.inputFormat = self.settingsObj['inputFormat']

        self.alarmState = 0
        self.alarmCount = 0

        # self.cnn = nnTraining.cnnDeepModel.CnnModel()
        
        #Load Model From Yout URL path
        self.model = keras.models.load_model(self.mModelFname)
        #Model Summary
        self.model.summary()
        
    def dp2vector(self, dpObj, inputFormat=1, normalise=False):
        '''Convert a datapoint object into an input vector to be fed into the neural network.   Note that if dp is not a dict, it is assumed to be a json string
            representation instead.
            if normalise is True, applies Z normalisation to accelerometer data
            to give a mean of zero and standard deviation of unity.
            https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-normalize-or-standardize-a-dataset-in-python.md
            # inputFormat is used by dp2vector to create the input array required for the specific network.
              # inputFormat = 1:  A simple 125 point array of accelerometer readings
              # inputFormat = 2:  A two dimensional array (2, 125)  The first row is acceleromater readings as per inputFormat 1, the other is HR values.

                '''
        dpInputData = []
        if (type(dpObj) is dict):
            rawDataStr = libosd.dpTools.dp2rawData(dpObj)
        else:
            rawDataStr = dpObj
        accData, hr = libosd.dpTools.getAccelDataFromJson(rawDataStr)
        #print(accData, hr)

        hrLst = [ hr for i in range(0,len(accData)) ]    
        if (accData is not None):
            if (normalise):
                accArr = np.array(accData)
                accArrNorm = (accArr - np.average(accArr)) / (np.std(accArr))
                accData = accArrNorm.tolist()
        else:
            print("*** Error in Datapoint: ", dpObj)
            print("*** No acceleration data found with datapoint.")
            print("*** I recommend adding event %s to the invalidEvents list in the configuration file" % dpObj['eventId'])
            exit(-1)

        if len(accData) != 125:
            print("*** ERROR:  accData is not 125 points in event Id %s***" % dpObj['eventId'])
            exit(-1)

        if (inputFormat == 1):
            # Simple list of 125 acceleration data points
            dpInputData = accData
            inputArry = np.array(dpInputData).reshape((1,125,1))
        elif (inputFormat == 2):
            # Two rows of 125 readings - the first is acceleration, the second is heart rate.
            dpInputData = [accData, hrLst]
            print("dpInputData=",dpInputData)
            dpInputDataArry = np.array(dpInputData)
            print("dpInputDataArry=",dpInputDataArry, dpInputDataArry.shape)
            inputArry = dpInputDataArry.reshape((1,125,2))
        else:
            print("*** ERROR - Unrecognised inputFormat: %s" % inputFormat)
            exit(-1)



        return inputArry


        
    def processDp(self, dpStr, eventId):
        #print(dpStr)
        #inputLst = nnTraining.nnTrainer.dp2vector(dpStr, normalise=False)
        inputArry = self.dp2vector(dpStr, inputFormat=self.inputFormat, normalise=False)

        # we use predict_on_batch() rather than predict() because it is much, much faster.
        #retVal = self.model.predict(inputArry, verbose=0)
        print("processDp - inputArry=",inputArry, inputArry.shape)
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


# FIXME - this is a fiddle - copied from AMBER.py to see if we can load the model.
#         without it here we get errors about not being able to find EnhancedFusionLayer, so the class must not be being
#         serialised into the model properly.
@keras.saving.register_keras_serializable()
class EnhancedFusionLayer(keras.Layer):
    def __init__(self, num_heads, key_dim, **kwargs):
        super(EnhancedFusionLayer, self).__init__(**kwargs)
        self.attention = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
    
    def call(self, inputs):
        concatenated_inputs = keras.layers.Concatenate()(inputs)
        attention_output = self.attention(concatenated_inputs, concatenated_inputs)
        return keras.layers.Add()([concatenated_inputs, attention_output])
    
    def get_config(self):
        config = super(EnhancedFusionLayer, self).get_config()
        config.update({
            "num_heads": self.attention.num_heads,
            "key_dim": self.attention.key_dim
        })
        return config

    def build(self, input_shape):
        ''' Implement a build method to avoid a keras warning'''
        super().build(input_shape)

    @classmethod
    def from_config(cls, config):
        ''' This seems to be needed if we are trying to load the saved model somewhere that does not have access to this class definition...'''
        #config["num_heads"] = keras.layers.deserialize(config["num_heads"])
        #config["key_dim"] = keras.layers.deserialize(config["key_dim"])
        print("from_config - config=",config)
        return cls(config["num_heads"], config["key_dim"])  #,**config)



if __name__ == "__main__":
    print("osdAlg.Jamie1Alg.main()")
    settingsObj = {
        "alarmFreqMin" : 3,
        "alarmFreqMax" : 8,
        "alarmThreshold" : 100,
        "alarmRatioThreshold" : 57
        }
    alg = Jamie1Alg(json.dumps(settingsObj),True)

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

class AmberAlg(sdAlg.SdAlg):
    '''
    A testRunner representation of the AMBER model by Jamie Pordoy (see
    https://d197for5662m48.cloudfront.net/documents/publicationstatus/210950/preprint_pdf/ca55d8e1a98cbc0c85bcc91217c7d27c.pdf)
    '''
    def __init__(self, settingsStr, debug=True):
        print("AmberAlg.__init__() - settingsStr=%s" % settingsStr)
        print("AmberAlg.__init__(): settingsStr=%s (%s)"
                               % (settingsStr, type(settingsStr)))
        super().__init__(settingsStr, debug)

        self.mModelFname = self.settingsObj['modelFname']
        self.mModeStr = self.settingsObj['mode']
        self.mSamplePeriod = self.settingsObj['samplePeriod']
        self.mWarnTime = self.settingsObj['warnTime']
        self.mAlarmTime = self.settingsObj['alarmTime']
        if ('sdThresh') in self.settingsObj:
            self.mSdThresh = self.settingsObj['sdThresh']
            print("AmberAlg.__init__:  Set mSdThresh to %.2f" % self.mSdThresh)
        else:
            self.mSdThresh = 5.0    # Default to 5% stdev threshold


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
            # Check we have real movement to analyse, otherwise reject the datapoint.
            accArr = np.array(accData)
            accAvg = np.average(accArr)
            if accAvg != 0:
                accStd = 100. * np.std(accArr) / accAvg
            else:
                accStd = 0.0
            if (accStd < self.mSdThresh):
                print("AmberAlg.dp2vector(): Rejecting Low Movement Datapoint")
                return None

        else:
            print("AmberAlg.dp2vector(): *** Error in Datapoint: ", dpObj)
            print("AmberAlg.dp2vector(): *** No acceleration data found with datapoint.")
            print("AmberAlg.dp2vector(): *** I recommend adding event %s to the invalidEvents list in the configuration file" % dpObj['eventId'])
            exit(-1)

        if len(accData) != 125:
            print("AmberAlg.dp2vector(): *** ERROR:  accData is not 125 points in event Id %s***" % dpObj['eventId'])
            exit(-1)

        # Two rows of 125 readings - the first is acceleration, the second is heart rate.
        dpInputData = [accData, hrLst]
        print("dpInputData=",dpInputData)
        dpInputDataArry = np.array(dpInputData)
        print("dpInputDataArry=",dpInputDataArry, dpInputDataArry.shape)
        inputArry = dpInputDataArry.reshape((1,125,2))
        return inputArry


        
    def processDp(self, dpStr, eventId):
        #print(dpStr)
        #inputLst = nnTraining.nnTrainer.dp2vector(dpStr, normalise=False)
        inputArry = self.dp2vector(dpStr, inputFormat=self.inputFormat, normalise=self.mNormalise)

        if (inputArry is None):
            # dp2vector returns none for invalid data or low standard deviation data
            inAlarm = False
        else:
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
            'pSeizure': pSeizure
            }
        return json.dumps(extraData)
        
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
    print("amberAlg.main()")
 
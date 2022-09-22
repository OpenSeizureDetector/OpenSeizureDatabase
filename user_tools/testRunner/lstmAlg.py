#!usr/bin/python3

"""
A seizure detection algorithm based on an LSTM neural network.

Jamie Pordoy, 2022
"""

import json
import numpy as np
import sdAlg

class LstmAlg(sdAlg.SdAlg):
    def __init__(self, settingsStr, debug=False):
        print("LstmAlg.__init__() - settingsStr=%s" % settingsStr)
        print("LstmA.__init__(): settingsStr=%s (%s)"
                               % (settingsStr, type(settingsStr)))
        super().__init__(settingsStr, debug)

        self.alarmState = 0
        self.alarmCount = 0
        self.mSampleFreq = self.settingsObj['sampleFreq'];
        self.mSamplePeriod = self.settingsObj['samplePeriod'];
        self.mWarnTime = self.settingsObj['warnTime'];
        self.mAlarmTime = self.settingsObj['alarmTime'];
        self.mNSamp = (int)(self.mSamplePeriod * self.mSampleFreq);


    def getAccelDataFromJson(self,jsonStr):
        if (jsonStr is not None):
            jsonObj = json.loads(jsonStr)
            #print(jsonObj.keys())
            if ("data3D" in jsonObj.keys()):
                #print("3dData present")
                accData = []
                for n in range(int(len(jsonObj['data3D'])/3)):
                    #print(n)
                    x = jsonObj['data3D'][3 * n]
                    y = jsonObj['data3D'][3 * n + 1]
                    z = jsonObj['data3D'][3 * n + 2]
                    #accData.append(math.sqrt(x*x + y*y + z*z))
                    accData.append(abs(x)+abs(y)+abs(z))


                if (len(accData)==0):
                    if (self.DEBUG): print("no 3d data, so using 'data' values")
                    accData = jsonObj['data']

            else:
                #print("no 3d data, so using 'data' values")
                accData = jsonObj['data']
            #print(accData)
        else:
            accData = None
        return(accData)
        

    def getAlarmState(self, rawData):
        ''' getAlarmState(rawData) - determines the alarm state associated with the snapshot
        of raw seizure detector data rawData[]
         @return the alarm state (0=ok, 1 = alarm)
        '''
        if (self.DEBUG): print("FIXME - implement getAlarmState by evaluating neural network for given data")
        alarmState = 0;
        return(alarmState)
        
    def processDp(self, dpStr):
        #print(dpStr)
        accData = self.getAccelDataFromJson(dpStr)
        if (accData is not None):
            inAlarm = self.getAlarmState(accData)
        else:
            inAlarm = 0

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

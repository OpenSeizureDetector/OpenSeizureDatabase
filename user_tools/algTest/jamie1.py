#!usr/bin/python3


import keras
from keras.models import Sequential
from keras.layers import GRU, LSTM
import numpy as np
import joblib

import json
import numpy as np
import sdAlg

class Jamie1Alg(sdAlg.SdAlg):
    def __init__(self, settingsStr, debug=True):
        print("Jamie1Alg.__init__() - settingsStr=%s" % settingsStr)
        print("Jamie1Alg.__init__(): settingsStr=%s (%s)"
                               % (settingsStr, type(settingsStr)))
        super().__init__(settingsStr, debug)

        self.mModelFname = self.settingsObj['modelFname']
        self.mModeStr = self.settingsObj['mode']
        self.mSampleFreq = self.settingsObj['sampleFreq']
        self.mAlarmFreqMin = self.settingsObj['alarmFreqMin']
        self.mAlarmFreqMax = self.settingsObj['alarmFreqMax']
        self.mSamplePeriod = self.settingsObj['samplePeriod']
        self.mWarnTime = self.settingsObj['warnTime']
        self.mAlarmTime = self.settingsObj['alarmTime']
        self.mAlarmThresh = self.settingsObj['alarmThresh']
        self.mAlarmRatioThresh = self.settingsObj['alarmRatioThresh']

        self.mFreqRes = 1.0 / self.mSamplePeriod
        self.mFreqCutoff = self.mSampleFreq / 2.0
        self.mNSamp = (int)(self.mSamplePeriod * self.mSampleFreq)

        self.alarmState = 0
        self.alarmCount = 0

        
        #Load Model From Yout URL path
        self.model_joblib = joblib.load(self.mModelFname)

        #Model Summary
        self.model_joblib.summary()



    def getMagnitude(self,cVal):
        """ Return the magnitude of complex variable cVal.
        Note actually returns magnitude ^2 for consistency with 
        pebble implementation!
        """
        #print(cVal)
        sumSq = cVal.real * cVal.real + cVal.imag * cVal.imag
        #print(cVal,"sumSq=%f, abs=%f" % (sumSq, np.abs(cVal)))
        return(sumSq)


    def getAccelDataFromJson(self,jsonStr):
        if (jsonStr is not None):
            jsonObj = json.loads(jsonStr)
            accData = jsonObj['data']
            hrVal = jsonObj['HR']
        else:
            accData = None
            hrVal = None
        return(accData,hrVal)
        
    def freq2fftBin(self,freq):
        n = int(freq / self.mFreqRes)
        return(n)

    def getSpecPower(self, accData, plotData = False):
        nFreqCutoff = self.freq2fftBin(self.mFreqCutoff)
        fftArr = np.fft.fft(accData)
        fftFreq = np.fft.fftfreq(fftArr.shape[-1], 1.0/self.mSampleFreq)

        specPower = 0.

        # range starts at 1 to avoid the DC component
        for i in range(1,self.mNSamp):
            if (i<=nFreqCutoff):
                specPower = specPower + self.getMagnitude(fftArr[i])
        # The following may not be correct
        specPower = specPower / self.mNSamp / 2
        return specPower

    def getRoiPower(self, accData, plotData = False):
        nMin = self.freq2fftBin(self.mAlarmFreqMin)
        nMax = self.freq2fftBin(self.mAlarmFreqMax)
        fftArr = np.fft.fft(accData)
        fftFreq = np.fft.fftfreq(fftArr.shape[-1], 1.0/self.mSampleFreq)

        roiPower = 0.

        for i in range(nMin, nMax):
            roiPower = roiPower + self.getMagnitude(fftArr[i])
        roiPower = roiPower / (nMax - nMin)
        return roiPower

    def getSpectrumRatio(self, accData):
        specPower = self.getSpecPower(accData);
        roiPower = self.getRoiPower(accData);

        if (specPower > self.mAlarmThresh):
            specRatio = 10.0 * roiPower / specPower;
        else:
            specRatio = 0.0;
        return(specRatio);


    def getAlarmState(self, accData, hrVal):
        ''' getAlarmState(rawData) - determines the alarm state associated with the snapshot of raw
         acceleration data rawData[]
         @return the alarm state (0=ok, 1 = alarm)
        '''
        specPower = self.getSpecPower(accData)
        roiPower = self.getRoiPower(accData)
        alarmRatio = self.getSpectrumRatio(accData);
        
        inputLst = []
        for n in range(0,len(accData)):
            rowLst = [specPower, roiPower, alarmRatio, hrVal, accData[n]]
            inputLst.append(rowLst)

        inputArry = np.array(inputLst).reshape((1,125,5))

        #print(inputLst)
        #print(inputArry)
            
        # make and show prediction
        retVal = self.model_joblib.predict(inputArry)
        print(retVal)
        pSeizure = retVal[0][1]
        if (retVal[0][1]>0.5):
            #print("ALARM - pSeizure=%f" % pSeizure)
            alarmState = 2
        else:
            alarmState = 0
        
        return(alarmState);
        
    def processDp(self, dpStr):
        #self.logD("Jamie1Alg.processDp: dpStr=%s." % dpStr)
        #print(dpStr)
        accData, hrVal = self.getAccelDataFromJson(dpStr)
        if (accData is not None):
            inAlarm = self.getAlarmState(accData, hrVal)
        else:
            inAlarm = 0

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
            self.alarmState = inAlarm
            
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

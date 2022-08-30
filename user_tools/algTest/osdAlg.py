#!usr/bin/python3

import json
import numpy as np
import sdAlg

class OsdAlg(sdAlg.SdAlg):
    alarmState = 0
    alarmCount = 0
    def __init__(self, settingsStr, debug=True):
        print("OsdAlg.__init__() - settingsStr=%s" % settingsStr)
        print("OsdAlg.__init__(): settingsStr=%s (%s)"
                               % (settingsStr, type(settingsStr)))
        super().__init__(settingsStr, debug)

        self.mSampleFreq = self.settingsObj['sampleFreq'];
        self.mAlarmFreqMin = self.settingsObj['alarmFreqMin'];
        self.mAlarmFreqMax = self.settingsObj['alarmFreqMax'];
        self.mSamplePeriod = self.settingsObj['samplePeriod'];
        self.mWarnTime = self.settingsObj['warnTime'];
        self.mAlarmTime = self.settingsObj['alarmTime'];
        self.mAlarmThresh = self.settingsObj['alarmThresh'];
        self.mAlarmRatioThresh = self.settingsObj['alarmRatioThresh'];

        self.mFreqRes = 1.0 / self.mSamplePeriod;
        self.mFreqCutoff = self.mSampleFreq / 2.0;
        self.mNSamp = (int)(self.mSamplePeriod * self.mSampleFreq);

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
        else:
            accData = None
        return(accData)
        
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


    def getAlarmState(self, accData):
        ''' getAlarmState(rawData) - determines the alarm state associated with the snapshot of raw
         acceleration data rawData[]
         @return the alarm state (0=ok, 1 = alarm)
        '''
        alarmRatio = self.getSpectrumRatio(accData);

        if (alarmRatio <= self.mAlarmRatioThresh):
            alarmState = 0;
        else:
            alarmState = 1;
        return(alarmState);
        
    def processDp(self, dpStr):
        #self.logD("OsdAlg.processDp: dpStr=%s." % dpStr)
        #print(dpStr)
        accData = self.getAccelDataFromJson(dpStr)
        if (accData is not None):
            inAlarm = self.getAlarmState(accData)
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

        extraData = {
#            'specPower': specPower,
#            'roiPower': roiPower,
#            'roiRatio': roiRatio,
            'alarmCount': self.alarmCount,
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
    print("osdAlg.OsdAlg.main()")
    settingsObj = {
        "alarmFreqMin" : 3,
        "alarmFreqMax" : 8,
        "alarmThreshold" : 100,
        "alarmRatioThreshold" : 57
        }
    alg = OsdAlg(json.dumps(settingsObj),True)

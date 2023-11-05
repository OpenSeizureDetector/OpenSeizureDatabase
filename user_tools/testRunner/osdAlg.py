#!usr/bin/python3

"""
A representation of the 'original' OpenSeizureDetector algorithm which developed over time in
terms of how it calculated the acceleration value for analysis:
    Default (V0) - use the vector magnitude that is stored in the database (do not re-calculate from 3d data)
    Version 1 (V1) used an approximation of the vector magnitude of acceleration
            val = |x| + |y| + |z|
    Version 2 (V2) used the true vector magnitude:
            val = sqrt(x^2 + y^2 + z^2)
    Version 3 (V3) adds a constant offset onto each acceleration value before calculating the magnitude
            val = sqrt((x+c)^2 + (y+c)^2 + (z+c)^2)
This uses the 'data3D' parameter which is sent by the Garmin app V1.2 and later
so does not work on earlier data and reverts to the acceleration magnitude value sent by the watch ('data' parameter).
"""
import math
import json
import numpy as np
import sdAlg

class OsdAlg(sdAlg.SdAlg):
    def __init__(self, settingsStr, debug=False):
        print("OsdAlg.__init__() - settingsStr=%s" % settingsStr)
        print("OsdAlg.__init__(): settingsStr=%s (%s)"
                               % (settingsStr, type(settingsStr)))
        super().__init__(settingsStr, debug)

        self.ACCEL_SCALE_FACTOR = 1000.  # Scale factor used in Android App
        self.mSampleFreq = self.settingsObj['sampleFreq'];
        self.mAlarmFreqMin = self.settingsObj['alarmFreqMin'];
        self.mAlarmFreqMax = self.settingsObj['alarmFreqMax'];
        self.mSamplePeriod = self.settingsObj['samplePeriod'];
        self.mWarnTime = self.settingsObj['warnTime'];
        self.mAlarmTime = self.settingsObj['alarmTime'];
        self.mAlarmThresh = self.settingsObj['alarmThresh'];
        self.mAlarmRatioThresh = self.settingsObj['alarmRatioThresh'];
        self.mMode = self.settingsObj['mode']
        self.mOffset = self.settingsObj['offset']

        self.mFreqRes = 1.0 / self.mSamplePeriod
        # FIXME - Frequency cutoff should really be mSampleFreq/2, but set to 12.0 for consistency with android app.
        #self.mFreqCutoff = self.mSampleFreq / 2.0
        self.mFreqCutoff = 12.0
        self.mNSamp = (int)(self.mSamplePeriod * self.mSampleFreq)
        self.alarmState = 0
        self.alarmCount = 0

    def getMagnitude(self,cVal):
        """ Return the magnitude of complex variable cVal.
        Note actually returns magnitude ^2 for consistency with 
        pebble implementation!
        FIXME - should really square root the value to give true magnitude, but this will 
                 require changes to the alarm thresholds.
        """
        #print(cVal)
        sumSq = cVal.real * cVal.real + cVal.imag * cVal.imag
        #print(cVal,"sumSq=%f, abs=%f" % (sumSq, np.abs(cVal)))
        return(sumSq)


    def getAccelDataFromJson(self,jsonStr):
        if (jsonStr is not None):
            jsonObj = json.loads(jsonStr)
            #print(jsonObj.keys())
            if self.mMode == "V0":
                accData = jsonObj['data']
                return(accData)
            if "data3D" in jsonObj.keys():
                #print("3dData present")
                accData = []
                for n in range(int(len(jsonObj['data3D'])/3)):
                    #print(n)
                    x = jsonObj['data3D'][3 * n]
                    y = jsonObj['data3D'][3 * n + 1]
                    z = jsonObj['data3D'][3 * n + 2]

                    if (self.mMode) == "V1":
                        accData.append(abs(x)+abs(y)+abs(z))
                    elif (self.mMode) == "V2":
                        accData.append(math.sqrt(x*x + y*y + z*z))
                    elif (self.mMode) == "V3":
                        #print("osdAlg v3 - adding offset of %.1f milli-g to each axis" % self.mOffset)
                        x = x + self.mOffset
                        y = y + self.mOffset
                        z = z + self.mOffset
                        accData.append(math.sqrt(x*x + y*y + z*z))
                    else:
                        print("OsdAlg.getAccelDataFromJson() - invalid mode specified - %s." % self.mMode)
                        exit(-1)

                if (len(accData)==0):
                    print("getAccelDataFromJson(): ERROR - 3d data array empty")
                    #accData = jsonObj['data']
                    exit(-1)

            else:
                print("getAccelDataFromJson(): ERROR - no 3d data, so using 'data' values", jsonObj.keys())
                print("getAccelDataFromJson() - jsonStr=%s" % jsonStr)
                #accData = jsonObj['data']
                exit(-1)
            #print(accData)
        else:
            print("getAccelDataFromJson(): ERROR - null JSON string received.")
            #accData = None
            exit(-1)
        #print("getAccelDataFromJson() - jsonStr=%s, accData=" % jsonStr,accData)

        #for n in range(0,len(accData)):
        #    print("%03d - %.0f   %.0f   Diff: %.2f%%" % (n, jsonObj['data'][n], accData[n], 100.*(jsonObj['data'][n]-accData[n])/accData[n]))
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
        for i in range(1,int(self.mNSamp/2)):
            if (i<=nFreqCutoff):
                specPower = specPower + self.getMagnitude(fftArr[i])
        # FIXME - The following may not be correct, but is consistent with android app
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
        self.specPower = self.getSpecPower(accData) / self.ACCEL_SCALE_FACTOR;
        self.roiPower = self.getRoiPower(accData) / self.ACCEL_SCALE_FACTOR;
        #print("mAlarmThresh = %f" % self.mAlarmThresh)
        if (self.roiPower > self.mAlarmThresh):
            self.specRatio = 10.0 * self.roiPower / self.specPower;
        else:
            self.specRatio = 0.0;
        return(self.specRatio);


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
        if (self.DEBUG): print ("OsdAlg.processDp: dpStr=%s." % dpStr)
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
            'specPower': self.specPower,
            'roiPower': self.roiPower,
            'roiRatio': self.specRatio,
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
    #settingsObj = {
    #    "alarmFreqMin" : 3,
    #    "alarmFreqMax" : 8,
    #    "alarmThreshold" : 100,
    #    "alarmRatioThreshold" : 57
    #    }
    #alg = OsdAlg(json.dumps(settingsObj),True)

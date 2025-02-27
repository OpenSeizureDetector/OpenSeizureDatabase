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
    Version 4 (V4) - process each axis (x, y, z) independently and return the highest alarm level of the three.
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

        flapSettings = self.settingsObj['flapSettings']
        if (flapSettings is not None):
            self.mFlapSettings = flapSettings
        else:
            self.mFlapSettings = None

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
            self.jsonObj = jsonObj
            #print(jsonObj.keys())
            if self.mMode == "V0":
                accData = jsonObj['data']
                return(accData)
            if self.mMode in ["V1", "V2", "V3"]:
                if "data3D" in jsonObj.keys() and len(jsonObj['data3D'])>0:
                    #print("3dData present")
                    accData = []
                    dataSum = 0
                    for n in range(int(len(jsonObj['data3D'])/3)):
                        #print(n)
                        # I thought the float typecast might be necessary, but it does not make any difference.
                        #  Why can't python have proper typecasting to avoid this uncertainty!
                        x = float(jsonObj['data3D'][3 * n])
                        y = float(jsonObj['data3D'][3 * n + 1])
                        z = float(jsonObj['data3D'][3 * n + 2])

                        dataSum = dataSum + x + y + z

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

                    if (len(accData)==0 or dataSum ==0):
                        if (self.DEBUG): print("getAccelDataFromJson(): 3d data array empty so using 'data' values")
                        accData = jsonObj['data']
                        #exit(-1)

                else:
                    if (self.DEBUG): print("getAccelDataFromJson(): No 3d data, so using 'data' values")
                    #print("getAccelDataFromJson() - jsonStr=%s" % jsonStr)
                    accData = jsonObj['data']
                    #exit(-1)
            #print(accData)
            if (self.mMode == "V4"):
                #print("V4 - 3d mode not implemented")
                if "data3D" in jsonObj.keys() and len(jsonObj['data3D'])>0:
                    #print("ok, we have 3d data")
                    accDataX = []
                    accDataY = []
                    accDataZ = []
                    dataSum = 0
                    for n in range(int(len(jsonObj['data3D'])/3)):
                        #print(n)
                        # I thought the float typecast might be necessary, but it does not make any difference.
                        #  Why can't python have proper typecasting to avoid this uncertainty!
                        x = float(jsonObj['data3D'][3 * n])
                        y = float(jsonObj['data3D'][3 * n + 1])
                        z = float(jsonObj['data3D'][3 * n + 2])

                        accDataX.append(x)
                        accDataY.append(y)
                        accDataZ.append(z)

                        dataSum = dataSum + x + y + z

                    if (dataSum == 0):
                        print("getAccelDataFromJson(): 3d data array is all zeros - giving up")
                        exit(-1)

                    accData = [ accDataX, accDataY, accDataZ]
                else:
                    print("ERROR OSDAlg V4 (3D) needs 3d data for all events")
                    exit(-1)
        else:
            print("getAccelDataFromJson(): ERROR - null JSON string received.")
            #accData = None
            exit(-1)
        #print("getAccelDataFromJson() - jsonStr=%s, accData=" % jsonStr,accData)

        #if self.mMode=="V2":
        #    for n in range(0,len(accData)):
        #        print("%03d - %.0f   %.0f   Diff: %.2f%%" % (n, jsonObj['data'][n], accData[n], 100.*(jsonObj['data'][n]-accData[n])/accData[n]))
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

    def getFlapRoiPower(self, accData, plotData = False):
        ''' getFlapRoiPower(accData) - calculate the power in the region of interest (flapAlarmFreqMin to flapAlarmFreqMax)
        @param accData - the acceleration data to be analysed
        @return the power in the region of interest
        '''
        flapAlarmFreqMin = self.mFlapSettings['flapAlarmFreqMin']
        flapAlarmFreqMax = self.mFlapSettings['flapAlarmFreqMax']
        nMin = self.freq2fftBin(flapAlarmFreqMin)
        nMax = self.freq2fftBin(flapAlarmFreqMax)
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
        if (self.DEBUG): print(self.specRatio)
        return(self.specRatio);

    def getFlapSpectrumRatio(self, accData):
        ''' getFlapSpectrumRatio(accData) - calculate the spectrum ratio for the region of interest
        @param accData - the acceleration data to be analysed
        @return the spectrum ratio for the region of interest
        '''
        self.specPower = self.getSpecPower(accData) / self.ACCEL_SCALE_FACTOR;
        self.flapRoiPower = self.getFlapRoiPower(accData) / self.ACCEL_SCALE_FACTOR;
        #print("mAlarmThresh = %f" % self.mAlarmThresh)
        if (self.flapRoiPower > self.mFlapSettings['flapAlarmThresh']):
            self.flapSpecRatio = 10.0 * self.flapRoiPower / self.specPower;
        else:
            self.flapSpecRatio = 0.0;
        return(self.flapSpecRatio);


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

    def getFlapAlarmState(self, accData):
        ''' getAlarmState(rawData) - determines the alarm state associated with the snapshot of raw
         acceleration data rawData[] using the flap detection algorithm.
         @return the alarm state (0=ok, 1 = alarm)
        '''
        alarmRatio = self.getFlapSpectrumRatio(accData);

        if (alarmRatio <= self.mFlapSettings['flapAlarmRatioThresh']):
            alarmState = 0;
        else:
            alarmState = 1;
        return(alarmState);



    def processDp(self, dpStr, eventId):
        #if (self.DEBUG): print ("OsdAlg.processDp: dpStr=%s." % dpStr)
        #print(dpStr)
        accData = self.getAccelDataFromJson(dpStr)
        if (accData is not None):
            if (self.mMode == "V4"):  # 3d mode
                inAlarmX = self.getAlarmState(accData[0])
                inAlarmY = self.getAlarmState(accData[1])
                inAlarmZ = self.getAlarmState(accData[2])
                inAlarm = max(inAlarmX, inAlarmY, inAlarmZ)

                if (self.mFlapSettings is not None and self.mFlapSettings['enabled']):
                    inAlarmFlapX = self.getFlapAlarmState(accData[0])
                    inAlarmFlapY = self.getFlapAlarmState(accData[1])
                    inAlarmFlapZ = self.getFlapAlarmState(accData[2])
                    inAlarmFlap = max(inAlarmFlapX, inAlarmFlapY, inAlarmFlapZ)
                else:
                    inAlarmFlap = 0

            else:
                inAlarm = self.getAlarmState(accData)
                if (self.mFlapSettings is not None and self.mFlapSettings['enabled']):
                    inAlarmFlap = self.getFlapAlarmState(accData)
                else:
                    inAlarmFlap = 0
        else:
            inAlarm = 0
            inAlarmFlap = 0

        ''' force an alarm state if we are in a flap alarm state '''
        if (inAlarmFlap):
            inAlarm = 1

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
            'specRatio': self.specRatio,
            'roiRatio': self.specRatio,
            'alarmCount': self.alarmCount,
            'alarmState': self.alarmState,
            #'fftArr': fftArr,
            #'fftFreq': fftFreq,
            }
        
        if self.mFlapSettings is not None and self.mFlapSettings['enabled']:
            flapData = {
                'flapRoiPower': self.flapRoiPower,
                'flapSpecRatio': self.flapSpecRatio,
                }        
            extraData.update(flapData)
        
        self.writeOutput([
            eventId,
            self.alarmState,
            self.specPower,
            self.roiPower,
            self.specRatio
        ])
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

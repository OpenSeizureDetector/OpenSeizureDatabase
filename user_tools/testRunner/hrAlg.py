#!usr/bin/python3

"""
Seizure detection algorithm based on heart rate.

It has 3 modes of operation
MODE_SIMPLE - a simple high / low heart rate threshold
MODE_ADAPTIVE_THRESHOLD - use the moving average of measured heart rate and an offset to calculate a threshold that varies based on baseline heart rate.
MODE_AVERAGE_HR - use the moving average heart rate to compare with the thresholds.

Parameters:
'mode' - the oeprating mode as given above.
'thresh_high' - high heart rate alarm threshold (used for MODE_SIMPLE and MODE_AVERAGE_HR)
'thresh_offset_high' - the offset between the average measured heart rate and the high alarm threshold.
'thresh_offset_low' - the offset between the average measured heart rate and the low alarm threshold.
'moving_average_time_window' - the time window used to calculate the moving average heart rate (in seconds) - must be a multiple of 5 seconds.

"""

import json
import sdAlg

class HrAlg(sdAlg.SdAlg):
    def __init__(self, settingsStr, debug=False):
        print("hrAlg.__init__(): settingsStr=%s (%s)"
                               % (settingsStr, type(settingsStr)))
        super().__init__(settingsStr, debug)

        self.mMode = self.settingsObj['mode']
        self.mThreshHigh = self.settingsObj['thresh_high']
        self.mThreshOffsetHigh = self.settingsObj['thresh_offset_high']
        self.mThreshOffsetLow = self.settingsObj['thresh_offset_low']
        self.mMovingAverageTimeWindowSecs = self.settingsObj['moving_average_time_window']
        self.mMovingAverageTimeWindowDps = int(self.mMovingAverageTimeWindowSecs / 5.0)
        print("hrAlg.__init__(): mMovingTimeAverageTimeWindowDps = %d" % self.mMovingAverageTimeWindowDps)

        self.mHRHist = []


    def getHrDataFromJson(self,jsonStr):
        hrVal = -1
        if (jsonStr is not None):
            jsonObj = json.loads(jsonStr)
            #print(jsonObj.keys())
            if 'hr' in jsonObj.keys():
                hrVal = jsonObj['hr']
        return(hrVal)
        
    def calcAvgHr(self):
        avHr = -1
        sumHr
        nAv = 0
        for n in range(0,len(self.mHRHist)):
            if (self.mHRHist[n]!=-1):
                sumHr += self.mHRHist[n]
                nAv += 1
        if (nAv>0):
            avHr = sumHr / nAv
            if(self.DEBUG): print("HrAlg.calcAvgHr - nAv = %d, sumHr = %f, avHr=%f" % (nAv, sumHr, avHr))
        return(avHr)

    def addToHist(self,hrVal):
        if (self.DEBUG): print("HrAlg.addToHist - length before=%d" % len(self.mHRHist))
        self.mHRHist.append(hrVal)
        if (len(self.mHRHist)>self.mMovingAverageTimeWindowDps):
            del self.mHRHist[0]
        if (self.DEBUG): print("HrAlg.addToHist - length after=%d" % len(self.mHRHist))


    def checkAlarmSimple(self):
        '''Checks the current state of Heart Rate and Heart Rate History to determine if we are in an
        alarm condition.  Returns 1 for alarm or 0 for OK.
        '''
        if(self.DEBUG): print("HrAlg.checkAlarmSimple()")
        return(0)

    def checkAlarmAdaptiveThreshold(self):
        '''Checks the current state of Heart Rate and Heart Rate History to determine if we are in an
        alarm condition.  Returns 1 for alarm or 0 for OK.
        '''
        if(self.DEBUG): print("HrAlg.checkAlarmAdaptiveThreshold()")
        return(0)

    def checkAlarmAverageHR(self):
        '''Checks the current state of Heart Rate and Heart Rate History to determine if we are in an
        alarm condition.  Returns 1 for alarm or 0 for OK.
        '''
        if(self.DEBUG): print("HrAlg.checkAlarmAverageHR()")
        return(0)


    def processDp(self, dpStr):
        if (self.DEBUG): print ("HrAlg.processDp: dpStr=%s." % dpStr)
        #print(dpStr)
        hrVal = self.getHrDataFromJson(dpStr)
        self.addToHist(hrVal)

        if (self.mMode == "MODE_SIMPLE"):
            if (self.DEBUG): print("HrAlg.processDp - MODE_SIMPLE")
            self.alarmState = self.checkAlarmSimple()
        elif (self.mMode == "MODE_ADAPTIVE_THRESHOLD"):
            if (self.DEBUG): print("HrAlg.processDp - MODE_ADAPTIVE_THRESHOLD")
            self.alarmState = self.checkAlarmAdaptiveThreshold()
        elif (self.mMode == "MODE_AVERAGE_HR"):
            if (self.DEBUG): print("HrAlg.processDp - MODE_AVERAGE_HR")
            self.alarmState = self.checkAlarmAverageHR()
        else:
            print("HrAlg.processDP - invalid mode: %s" % self.mMode)
            raise

        self.alarmCount = 0

        extraData = {
            'alarmCount': self.alarmCount,
            'alarmState': self.alarmState,
            }
        return json.dumps(extraData)
        
    def resetAlg(self):
        self.alarmState = 0
        self.alarmCount = 0
        self.mHRHist = []

                  
if __name__ == "__main__":
    print("hrAlg.HrAlg.main()")
    #settingsObj = {
    #    "alarmFreqMin" : 3,
    #    "alarmFreqMax" : 8,
    #    "alarmThreshold" : 100,
    #    "alarmRatioThreshold" : 57
    #    }
    #alg = OsdAlg(json.dumps(settingsObj),True)

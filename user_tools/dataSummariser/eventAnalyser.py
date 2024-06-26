#!/usr/bin/env python3

import argparse
import os
import sys
import dateutil.parser
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
#import libosd.webApiConnection
import libosd.dpTools as dpt
import libosd.osdAlgTools as oat


def dateStr2secs(dateStr):
    parsed_t = dateutil.parser.parse(dateStr)
    return parsed_t.timestamp()

def getDictVal(dict,key):
    if (dict is not None):
        if (key in dict.keys()):
            return dict[key]

    return None
            
class EventAnalyser:
    def __init__(self, debug=False):
        self.DEBUG = debug

    def getEventDataPoints(self, eventObj):
        if (self.DEBUG): print("eventObj=", eventObj)
        if (not 'datapoints' in eventObj.keys()):
            print("Event %s contans no datapoints" % (eventObj['id']))
            return(eventObj,None)
        dataPointsLst = eventObj['datapoints']
        # Sort datapoints into time order
        dataPointsLst.sort(key=lambda dp: dateStr2secs(dp['dataTime']))
        return(eventObj, dataPointsLst)

    def analyseEvent(self, eventObj):
        if (self.DEBUG): print("eventAnalyser.analyseEvent()")
        self.eventId = eventObj['id']
        self.eventObj, self.dataPointsLst = self.getEventDataPoints(eventObj)
        
        if (self.DEBUG): print("eventDataObj=",eventObj)
        alarmTime = dateStr2secs(self.eventObj['dataTime'])
        if (self.DEBUG): print("dataTime=",self.eventObj['dataTime'])
        self.dataTime = dateutil.parser.parse(self.eventObj['dataTime'])
        self.dataTimeStr = self.dataTime.strftime("%Y-%m-%d %H:%M:%S")

        # Populate the analysis parameter variables.
        if (self.DEBUG): print("analyseEvent: eventObj=",eventObj)
        if ('alarmRatioThresh' in eventObj):
            if (self.DEBUG): print("Reading parameters from event object")
            self.alarmThresh = eventObj['alarmThresh']
            self.alarmRatioThresh = eventObj['alarmRatioThresh']
            self.alarmFreqMin = eventObj['alarmFreqMin']
            self.alarmFreqMax = eventObj['alarmFreqMax']
        else:
            if (self.DEBUG): print("Reading parameters from first datapoint Object")
            if (self.dataPointsLst is not None):
                dp = self.dataPointsLst[0]
                #dpObj = json.loads(dp['dataJSON'])
                #dpDataObj = json.loads(dpObj['dataJSON'])
                self.alarmThresh = dpt.getParamFromDp('alarmThresh',dp)
                self.alarmRatioThresh = dpt.getParamFromDp('alarmRatioThresh',dp)
                self.alarmFreqMin = dpt.getParamFromDp('alarmFreqMin',dp)
                self.alarmFreqMax = dpt.getParamFromDp('alarmFreqMax',dp)
            else:
                self.alarmThresh = -999
                self.alarmRatioThresh = -999
                self.alarmFreqMin = -999
                self.alarmFreqMax = -999
        

        # Collect all the raw data into a single list with associated
        # time from the alarm (in seconds)
        self.rawTimestampLst = []
        self.accelLst = []
        self.xAccelLst = []
        self.yAccelLst = []
        self.zAccelLst = []
        self.analysisTimestampLst = []
        self.specPowerLst = []
        self.roiPowerLst = []
        self.roiRatioLst = []
        self.roiRatioThreshLst = []
        self.alarmRatioThreshLst = []
        self.alarmThreshLst = []
        self.alarmStateLst = []
        self.accMeanLst = []
        self.accSdLst = []
        self.hrLst = []
        self.adaptiveHrAvLst = []
        self.averageHrAvLst = []
        self.o2satLst = []
        self.pSeizureLst = []
        self.minRoiAlarmPower = 0
        self.dataPointsTdiff = []
        self.nDataPoints = 0
        self.nDpGaps = 0
        self.nDpExtras = 0
        if (self.dataPointsLst is not None):
            self.nDataPoints = len(self.dataPointsLst)
            #prevTs = dateStr2secs(self.dataPointsLst[0]['dataTime'])
            for nDp in range(len(self.dataPointsLst)):
                dp = self.dataPointsLst[nDp]
                currTs = dateStr2secs(dp['dataTime'])
                if (self.DEBUG): print(eventObj['dataTime'], dp['dataTime'], alarmTime, currTs)
                
                #print(dataObj)
                self.analysisTimestampLst.append(currTs - alarmTime)
                specPower = dpt.getParamFromDp('specPower',dp)
                if specPower is None: specPower = 0
                roiPower = dpt.getParamFromDp('roiPower',dp)
                if roiPower is None: roiPower = 0
                self.specPowerLst.append(specPower)
                self.roiPowerLst.append(roiPower)
                if (specPower is not None and specPower!=0):
                    roiRatio = roiPower / specPower
                else:
                    roiRatio = 999
                self.roiRatioLst.append(roiRatio)

                if (self.alarmThresh is not None):
                    if (roiPower >= self.alarmThresh):
                        self.roiRatioThreshLst.append(roiRatio)
                    else:
                        self.roiRatioThreshLst.append(0.)
                else:
                    self.roiRatioThreshLst.append(0.)
                alarmState = dpt.getParamFromDp('alarmState',dp)
                self.alarmStateLst.append(alarmState)
                # Record the minimum ROI Power that caused a WARNING or ALARM
                if (alarmState is not None and alarmState>0):
                    if (roiPower>self.minRoiAlarmPower):
                        self.minRoiAlarmPower = roiPower
                if self.alarmThresh is not None:
                    self.alarmThreshLst.append(self.alarmThresh)
                else:
                    self.alarmThreshLst.append(0)
                if (self.alarmRatioThresh is not None):
                    self.alarmRatioThreshLst.append(self.alarmRatioThresh/10.)
                else:
                    self.alarmRatioThreshLst.append(0)
                self.hrLst.append(dpt.getParamFromDp('hr',dp))
                self.adaptiveHrAvLst.append(dpt.getParamFromDp('adaptiveHrAv',dp))
                self.averageHrAvLst.append(dpt.getParamFromDp('averageHrAv',dp))
                self.o2satLst.append(dpt.getParamFromDp('o2Sat',dp))

                if ('pSeizure' in dp.keys()):
                    self.pSeizureLst.append(dp['pSeizure'])
                else:
                    self.pSeizureLst.append(-1)

                # Add to the raw data lists
                if 'rawData' in dp:
                    accLst = dp['rawData']
                else:
                    accLst = [0]*125
                accArr = np.array(accLst)
                self.accMeanLst.append(accArr.mean())
                self.accSdLst.append(100.*accArr.std()/accArr.mean())
                # FIXME:  IT is not good to hard code the length of an array!
                for n in range(0,125):
                    self.accelLst.append(accLst[n])
                    self.rawTimestampLst.append((currTs + n*1./25.)-alarmTime)
                    if 'rawData3D' in dp:
                        self.xAccelLst.append(dp['rawData3D'][3*n])
                        self.yAccelLst.append(dp['rawData3D'][3*n + 1])
                        self.zAccelLst.append(dp['rawData3D'][3*n + 2])
                    else:
                        self.xAccelLst.append(0)
                        self.yAccelLst.append(0)
                        self.zAccelLst.append(0)
                # Make a list of the time differences between datapoints.
                if (nDp > 0):
                    self.dataPointsTdiff.append(currTs-prevTs)
                prevTs = currTs

            # Count how man gaps we have in the data points list, and how many
            # 'extra' datapoints that would not have been expected.
            for tDiff in self.dataPointsTdiff:
                if (tDiff > 6):
                    self.nDpGaps += 1
                if (tDiff < 4):
                    self.nDpExtras += 1
        else:
            print("WARNING - Event %s does not contain any datapoints"
                  % self.eventId)
            self.nDataPoints = 0



    def generateSpectralHistoryFromAccelLst(self, accLst, normalise=False, zeroTol=0.001, sdThresh=10):
        '''
        Returns a numpy array representing the spectral history.   Any values where |value|<tol are set to zero
        if the standard deviation (mili-g) of the acceleration in a time slice is less than sdThresh,
        the output is set to zero so we do not see noise in the image for very low movement levels.
        '''
        specLst = []
        windowLen = 125   #  Samples to analyse - 125 samples at 25 Hz is a 5 second window

        rawArr = np.array(accLst)    
        endPosn = windowLen
        while endPosn<len(accLst):
            slice = rawArr[endPosn-windowLen:endPosn]
            sliceStd = slice.std()    #/  slice.mean()
            #print("generateSpectralHistoryFromAccelLst() - slice.mean=%.3f mg, slice..std=%.3f mg" % (slice.mean(), slice.std()))
            if (sliceStd >=sdThresh):
                fft, fftFreq = oat.getFFT(slice, sampleFreq=25)
                fftMag = np.absolute(fft)
                #print(fftMag)
                # Clip small values to zero to reduce normalisation artefacts.
                fftMag[abs(fftMag) < zeroTol] = 0
                if (normalise):
                    if np.max(fftMag[1:62]) != 0:
                        specLst.append(fftMag[1:62]/np.max(fftMag[1:62]))   # Ignore DC component in position 0
                    else:
                        specLst.append(np.zeros(61))   # Force output to zero if all values are zero
                else:
                    specLst.append(fftMag[1:62])   # Ignore DC component in position 0
            else:
                specLst.append(np.zeros(61))   # Zero the output if there is very low movement.
            endPosn += 1
        specImg = np.stack(specLst, axis=1)

        return specImg

        


    def plotSpectralHistory(self, outFname="spectralHistory.png", colImgFname="colImg.png"):
        '''Produce an image showing spectral intensity vs time.
        Must be called after analyseEvent()
        '''
        magSpecImg = self.generateSpectralHistoryFromAccelLst(self.accelLst, normalise=False)
        xSpecImg = self.generateSpectralHistoryFromAccelLst(self.xAccelLst, normalise=False)
        #exit(-1)
        print(xSpecImg)
        ySpecImg = self.generateSpectralHistoryFromAccelLst(self.yAccelLst, normalise=False)
        zSpecImg = self.generateSpectralHistoryFromAccelLst(self.zAccelLst, normalise=False)

        # Normalise the magnitude spectrum image as a single image.
        magSpecImg = magSpecImg/np.max(magSpecImg)

        # Normalise the individual axes spectrum images as a set, so different magnitudes of movement in different axes are visible.
        maxVal = np.max((xSpecImg, ySpecImg, zSpecImg))
        xSpecImg = xSpecImg/maxVal
        ySpecImg = ySpecImg/maxVal
        zSpecImg = zSpecImg/maxVal


        fig, ax = plt.subplots(4,1)
        fig.set_figheight(200/25.4)
        fig.suptitle('Event Number %s, %s\n%s, %s' % (
            self.eventId,
            self.dataTimeStr,
            self.eventObj['type'],
            self.eventObj['subType']),
                     fontsize=11)

        # Bilinear interpolation - this will look blurry, 'nearest' is blocky
        ax[0].imshow(magSpecImg, interpolation='nearest', origin='lower', aspect=5, cmap="Oranges",
                   extent=[0,len(self.accelLst)/25.,0,12.5])
        #ax[0].set_yticks(np.arange(0, 12.5, 1))
        ax[0].set_yticks([3,8])
        ax[0].grid(which='major', color='k', linestyle='-', linewidth=0.3)
        ax[0].title.set_text("Vector Magnitude")
        ax[0].set_xlabel('Time (sec)')
        ax[0].set_ylabel('Freq (Hz)')

        ax[1].imshow(xSpecImg, interpolation='nearest', origin='lower', aspect=5, cmap="Oranges",
                   extent=[0,len(self.xAccelLst)/25.,0,12.5])
        ax[1].set_yticks([3,8])
        ax[1].grid(which='major', color='k', linestyle='-', linewidth=0.3)
        ax[1].title.set_text("X Direction")
        ax[1].set_xlabel('Time (sec)')
        ax[1].set_ylabel('Freq (Hz)')

        ax[2].imshow(ySpecImg, interpolation='nearest', origin='lower', aspect=5, cmap="Oranges",
                   extent=[0,len(self.yAccelLst)/25.,0,12.5])
        ax[2].set_yticks([3,8])
        ax[2].grid(which='major', color='k', linestyle='-', linewidth=0.3)
        ax[2].title.set_text("Y Direction")
        ax[2].set_xlabel('Time (sec)')
        ax[2].set_ylabel('Freq (Hz)')

        ax[3].imshow(zSpecImg, interpolation='nearest', origin='lower', aspect=5, cmap="Oranges",
                   extent=[0,len(self.zAccelLst)/25.,0,12.5])
        ax[3].set_yticks([3,8])
        ax[3].grid(which='major', color='k', linestyle='-', linewidth=0.3)
        ax[3].title.set_text("Z Direction")
        ax[3].set_xlabel('Time (sec)')
        ax[3].set_ylabel('Freq (Hz)')

        plt.tight_layout()
        fig.savefig(outFname)
        print("image written to %s" % outFname)
        plt.close(fig)


        imgCol = np.stack((xSpecImg, ySpecImg, zSpecImg))
        print(imgCol.shape)
        imgCol = np.transpose(imgCol,(1,2,0))
        print(imgCol.shape)
        fig, ax = plt.subplots(1,1)
        ax.imshow(imgCol, origin='lower', aspect=5,
                  extent=[0,len(self.yAccelLst)/25.,0,12.5])
        fig.savefig(colImgFname)
        plt.close(fig)

    def saveAccelCsv(self,outFname="accelData.csv"):
        with open(outFname,"w") as fp:
            fp.write("time, magnitude, x, y,z\n")
            for n in range(0, len(self.accelLst)):
                fp.write("%.3f, " % self.rawTimestampLst[n])
                fp.write("%.1f, " % self.accelLst[n])
                fp.write("%.1f, " % self.xAccelLst[n])
                fp.write("%.1f, " % self.yAccelLst[n])
                fp.write("%.1f\n" % self.zAccelLst[n])

    def plotRawDataGraph(self,outFname="rawData.png"):
        if (self.DEBUG): print("plotRawDataGraph")
        #fig, ax = plt.subplots(1,1)
        fig, ax = plt.subplots(4,1)
        fig.set_figheight(200/25.4)

        fig.suptitle('Event Number %s, %s\n%s, %s' % (
            self.eventId,
            self.dataTimeStr,
            self.eventObj['type'],
            self.eventObj['subType']),
                     fontsize=11)
        ax[0].plot(self.rawTimestampLst,self.accelLst)
        if 'seizureTimes' in self.eventObj.keys():
            tStart = self.eventObj['seizureTimes'][0]
            tEnd = self.eventObj['seizureTimes'][1]
            ax[0].axvspan(tStart, tEnd, color='blue', alpha=0.2)
        ax[0].set_title("Raw Data (Vector Magnitude)")
        ax[0].set_ylabel("Acceleration (~milli-g)")
        ax[0].grid(True)

        ax2 = ax[0].twinx()
        ax2.plot(self.analysisTimestampLst, self.accSdLst, color='red')
        ax2.set_ylabel("Acceleration Standard Deviation (%)", color='red')

        ax[1].plot(self.rawTimestampLst,self.xAccelLst)
        if 'seizureTimes' in self.eventObj.keys():
            tStart = self.eventObj['seizureTimes'][0]
            tEnd = self.eventObj['seizureTimes'][1]
            ax[1].axvspan(tStart, tEnd, color='blue', alpha=0.2)
        ax[1].set_title("Raw Data (X-Direction Acceleration)")
        ax[1].set_ylabel("Acceleration (~milli-g)")
        ax[1].grid(True)

        ax[2].plot(self.rawTimestampLst,self.yAccelLst)
        if 'seizureTimes' in self.eventObj.keys():
            tStart = self.eventObj['seizureTimes'][0]
            tEnd = self.eventObj['seizureTimes'][1]
            ax[2].axvspan(tStart, tEnd, color='blue', alpha=0.2)
        ax[2].set_title("Raw Data (Y-Direction Acceleration)")
        ax[2].set_ylabel("Acceleration (~milli-g)")
        ax[2].grid(True)

        ax[3].plot(self.rawTimestampLst,self.zAccelLst)
        if 'seizureTimes' in self.eventObj.keys():
            tStart = self.eventObj['seizureTimes'][0]
            tEnd = self.eventObj['seizureTimes'][1]
            ax[3].axvspan(tStart, tEnd, color='blue', alpha=0.2)
        ax[3].set_title("Raw Data (Z-Direction Acceleration)")
        ax[3].set_ylabel("Acceleration (~milli-g)")
        ax[3].grid(True)


        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        fig.savefig(outFname)
        plt.close(fig)
        print("Graph written to %s" % outFname)



    def plotRawDataGraph_orig(self,outFname="rawData.png"):
        if (self.DEBUG): print("plotRawDataGraph")
        fig, ax = plt.subplots(1,1)
        fig.suptitle('Event Number %s, %s\n%s, %s' % (
            self.eventId,
            self.dataTimeStr,
            self.eventObj['type'],
            self.eventObj['subType']),
                     fontsize=11)
        ax.plot(self.rawTimestampLst,self.accelLst)
        if 'seizureTimes' in self.eventObj.keys():
            tStart = self.eventObj['seizureTimes'][0]
            tEnd = self.eventObj['seizureTimes'][1]
            ax.axvspan(tStart, tEnd, color='blue', alpha=0.2)
        ax.set_title("Raw Data")
        ax.set_ylabel("Acceleration (~milli-g)")
        ax.grid(True)

        ax2 = ax.twinx()
        ax2.plot(self.analysisTimestampLst, self.accSdLst, color='red')
        ax2.set_ylabel("Acceleration Standard Deviation (%)", color='red')
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        fig.savefig(outFname)
        plt.close(fig)
        print("Graph written to %s" % outFname)

    def plotHrGraph(self,outFname="hrData.png"):
        if (self.DEBUG): print("plotHrGraph")
        fig, ax = plt.subplots(1,1)
        fig.suptitle('Event Number %s, %s\n%s, %s' % (
            self.eventId,
            self.dataTimeStr,
            self.eventObj['type'],
            self.eventObj['subType']),
                     fontsize=11)
        ax.plot(self.analysisTimestampLst, self.hrLst, linestyle='solid')
        ax.plot(self.analysisTimestampLst, self.adaptiveHrAvLst, label='Adaptive Avg', linestyle='dashed')
        ax.plot(self.analysisTimestampLst, self.averageHrAvLst, label='average Avg', linestyle='dashdot')
        ax.plot(self.analysisTimestampLst, self.o2satLst, linestyle='dotted')
        ax.legend(['HR (bpm)','Adaptive Avg (bpm)', 'Average Avg (bpm)','O2 Sat (%)'])
        ax.set_title("Heart Rate / O2 Sat")
        ax.set_xlabel("Time (seconds)")
        ax.grid(True)
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        fig.savefig(outFname)
        plt.close(fig)
        print("Graph written to %s" % outFname)

        
    def plotAnalysisGraph(self,outFname="analysis.png"):
        if (self.DEBUG): print("plotAnalysisGraph")
        fig, ax = plt.subplots(3,1, figsize=(12,8))
        fig.suptitle('Event Number %s, %s\n%s, %s' % (
            self.eventId,
            self.dataTimeStr,
            self.eventObj['type'],
            self.eventObj['subType']),
                     fontsize=11)
        ax[0].plot(self.analysisTimestampLst, self.specPowerLst)
        ax[0].plot(self.analysisTimestampLst, self.roiPowerLst)
        ax[0].plot(self.analysisTimestampLst, self.alarmThreshLst)
        ax[0].legend(['Spectrum Power','ROI Power', 'ROI Power Threshold'])
        #if len(self.alarmThreshLst)>0:
        #    ax[0].set_ylim(0,max(self.alarmThreshLst)*10)
        ax[0].set_ylim(0,10000)
        ax[0].set_ylabel("Average Power per bin")
        ax[0].set_title("Spectrum / ROI Powers")
        ax[0].grid(True)
        ax[1].plot(self.analysisTimestampLst, self.roiRatioThreshLst)
        ax[1].plot(self.analysisTimestampLst, self.roiRatioLst, '.')
        ax[1].plot(self.analysisTimestampLst, self.alarmRatioThreshLst)
        ax[1].plot(self.analysisTimestampLst, self.alarmStateLst)
        ax[1].set_title("ROI Ratio & Alarm State")
        ax[1].legend(['ROI Ratio (thresholded)','ROI Ratio (raw)', 'Alarm Ratio Threshold','Alarm State'])
        ax[1].set_ylabel("Number")
        ax[1].set_xlabel("Time (seconds)")
        ax[1].grid(True)
        ax[2].plot(self.analysisTimestampLst, self.pSeizureLst)
        ax[2].set_title("CNN Calculated Seizure Probability")
        ax[2].set_ylabel("pSeizure")
        ax[2].set_ylim(0,1.0)
        ax[2].set_xlabel("Time (seconds)")
        ax[2].grid(True)
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        fig.savefig(outFname)
        plt.close(fig)
        print("Graph written to %s" % outFname)


    def plotSpectrumGraph(self,outFname="spectrum.png"):
        if (self.DEBUG): print("plotSpectrumGraph")
        fig, ax = plt.subplots(1,1, figsize=(8,5))
        if len(self.analysisTimestampLst)>0:
            # Find the datapoint at the event time.
            minTimeDiff = 9999
            zeroDpN = 0
            zeroDp = self.dataPointsLst[zeroDpN]
            for n in range (0,len(self.analysisTimestampLst)):
                #print(n,self.analysisTimestampLst[n])
                if (abs(self.analysisTimestampLst[n])<=minTimeDiff):
                    #print("Found datapoint close to zero")
                    zeroDpN = n
                    zeroDp = self.dataPointsLst[n]
                    minTimeDiff = abs(self.analysisTimestampLst[n])
            #print(zeroDp)
            specLst = []
            specTimesLst = []
            for n in range(zeroDpN-1,zeroDpN+2):
                if (n>0 and n<len(self.dataPointsLst)):
                    dp = self.dataPointsLst[n]
                    specVals = dp['simpleSpec']
                    # Normalise the spectrum so maximum value is always 1
                    if len(specVals)>0:
                        specMax = max(specVals)
                        if (specMax!=0):
                            specNorm = [float(v)/specMax for v in specVals]
                        else:
                            specNorm = [float(v) for v in specVals]
                    else:
                        specNorm = 1
                    specLst.append(specNorm)
                    specTimesLst.append(self.analysisTimestampLst[n])
                else:
                    if (self.DEBUG): print("skipping out of range datapoint")
            #print(specLst)
            fig.suptitle('Event Number %s, %s\n%s, %s' % (
                self.eventId,
                self.dataTimeStr,
                self.eventObj['type'],
                self.eventObj['subType']),
                         fontsize=11)
            for spec in specLst:
                ax.plot(range(1,11),spec)
        ax.set_ylabel("Power per bin (normalised)")
        ax.set_title("Datapoint Spectra")
        ax.grid(True)
        #ax.legend(specTimesLst)
        ax.set_xlabel("Frequency (Hz)")
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        fig.savefig(outFname)
        plt.close(fig)
        print("Graph written to %s" % outFname)


        
            
if (__name__=="__main__"):
    print("analyse_event.py.main()")
    parser = argparse.ArgumentParser(description='analyse event')
    parser.add_argument('--config', default="credentials.json",
                        help='name of json file containing api login token')
    parser.add_argument('--event', default=None,
                        help='ID Number of the event to analyse')
    parser.add_argument('--list', action="store_true",
                        help='List all events in the database')
    parser.add_argument('--test',
                        help='Address of Device running OpenSeizureDetector Ap for Testing')
    argsNamespace = parser.parse_args()
    args = vars(argsNamespace)
    print(args)

    #analyse_event(configFname=args['config'])

    analyser = EventAnalyser(configFname=args['config'])

    if (args['event'] is not None):
        if (args['test'] is not None):
            print("Running Event Number %d on test server %s" %
                  (int(args['event']), args['test']))
            analyser.testEvent(int(args['event']), args['test'])
        else:
            print("Analysing Event Number %d" % int(args['event']))
            analyser.analyseEvent(int(args['event']))
    elif (args['list']):
        analyser.listEvents()
    else:
        print("Not doing anything")


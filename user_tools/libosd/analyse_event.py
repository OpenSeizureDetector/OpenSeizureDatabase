#!/usr/bin/python3

import argparse
import json
import libosd.webApiConnection
import libosd.osdAppConnection
import sys
import dateutil.parser
import time

import matplotlib.pyplot as plt


def dateStr2secs(dateStr):
    parsed_t = dateutil.parser.parse(dateStr)
    return parsed_t.timestamp()


def runTest():
    # Note, in test DB graham4 is a staff (researcher) user
    #              and graham29 is a normal user

    for unameStr in ("graham29", "graham4"):
        print("**********************")
        print("Trying as user %s" % unameStr)
        print("**********************")
        osd = libosd.libosd(cfg="client.cfg", uname=unameStr,
                            passwd="testpwd1", debug=True)
        eventsObj = osd.getEvents(userId=38)
        if (eventsObj is not None):
            for event in eventsObj:
                print(event)
        else:
            print("ERROR - No Data Returned");

        datapointsObj = osd.getDataPointsByEvent(eventId=395)
        if (datapointsObj is not None):
            for dp in datapointsObj:
                #print("dp=",dp)
                dpObj = json.loads(dp['dataJSON'])
                dataObj = json.loads(dpObj['dataJSON'])
                #print(dataObj.keys())
                print("%s, %d, %d, %.0f, %.0f, %.2f" % (dataObj['dataTime'],
                                                  dp['id'], dp['eventId'],
                                                  dataObj['specPower'],dataObj['roiPower'],
                                                  dataObj['roiPower']/dataObj['specPower']                                      ))
        else:
            print("ERROR - No Data Returned")

            
class EventAnalyser:
    configObj = None
    DEBUG = False

    def __init__(self, configFname = "credentials.json", debug=False):
        self.DEBUG = debug
        if (debug): print("EventAnalyser.__init__: configFname=%s" % configFname)
        self.loadConfig(configFname)
        if (debug): print("self.configObj=",self.configObj)
        self.osd = libosd.webApiConnection.WebApiConnection(cfg="client.cfg",
                                                            uname=self.configObj["uname"],
                                                            passwd=self.configObj["passwd"],
                                                            debug=debug)
            
    def loadConfig(self,configFname):
        # Opening JSON file
        try:
            f = open(configFname)
            if (self.DEBUG): print("Opened File")
            self.configObj = json.load(f)
            f.close()
            if (self.DEBUG): print("configObj=",self.configObj)
        except BaseException as e:
            print("Error Opening File %s" % configFname)
            print("OS error: {0}".format(e))
            print(sys.exc_info())
            self.configObj = None

    def listEvents(self):
        eventsObj = self.osd.getEvents()
        if (eventsObj is not None):
            for event in eventsObj:
                print(event)
        else:
            print("ERROR - No Data Returned");


    def getEventDataPoints(self, eventId):
        eventObj = self.osd.getEvent(eventId)
        if (self.DEBUG): print("eventObj=", eventObj)
        dataPointsObj = self.osd.getDataPointsByEvent(eventId)
        # Make sure we are sorted into time order
        dataPointsObj.sort(key=lambda dp: dateStr2secs(dp['dataTime']))
        return(eventObj, dataPointsObj)

    def loadEvent(self, eventId):
        if (self.DEBUG): print("loadEvent: eventId=%d" % eventId)
        self.eventId = eventId
        self.eventObj, self.dataPointsObj = self.getEventDataPoints(eventId)
        if ('dataJSON' in self.eventObj):
            if (self.eventObj['dataJSON'] is None):
                eventDataObj = {}
            else:
                print("Creating eventDataObj",self.eventObj)
                eventDataObj = json.loads(self.eventObj['dataJSON'])
        else:
            eventDataObj = {}
        #print(self.eventObj, eventDataObj)
        alarmTime = dateStr2secs(self.eventObj['dataTime'])
        self.dataTime = dateutil.parser.parse(self.eventObj['dataTime'])
        self.dataTimeStr = self.dataTime.strftime("%Y-%m-%d %H:%M")

        # Populate the analysis parameter variables.
        if ('alarmRatioThresh' in eventDataObj):
            print("Reading parameters from event object")
            self.alarmThresh = eventDataObj['alarmThresh']
            self.alarmRatioThresh = eventDataObj['alarmRatioThresh']
            self.alarmFreqMin = eventDataObj['alarmFreqMin']
            self.alarmFreqMax = eventDataObj['alarmFreqMax']
        else:
            print("Reading parameters from first datapoint Object")
            dp = self.dataPointsObj[0]
            dpObj = json.loads(dp['dataJSON'])
            dpDataObj = json.loads(dpObj['dataJSON'])
            self.alarmThresh = dpDataObj['alarmThresh']
            self.alarmRatioThresh = dpDataObj['alarmRatioThresh']
            self.alarmFreqMin = dpDataObj['alarmFreqMin']
            self.alarmFreqMax = dpDataObj['alarmFreqMax']


        

        # Collect all the raw data into a single list with associated
        # time from the alarm (in seconds)
        self.rawTimestampLst = []
        self.accelLst = []
        self.analysisTimestampLst = []
        self.specPowerLst = []
        self.roiPowerLst = []
        self.roiRatioLst = []
        self.roiRatioThreshLst = []
        self.alarmRatioThreshLst = []
        self.alarmThreshLst = []
        self.alarmStateLst = []
        self.hrLst = []
        self.o2satLst = []
        self.minRoiAlarmPower = 0
        for dp in self.dataPointsObj:
            currTs = dateStr2secs(dp['dataTime'])
            #print(dp['dataTime'], currTs)
            dpObj = json.loads(dp['dataJSON'])
            dataObj = json.loads(dpObj['dataJSON'])
            #print(dataObj)
            self.analysisTimestampLst.append(currTs - alarmTime)
            self.specPowerLst.append(dataObj['specPower'])
            self.roiPowerLst.append(dataObj['roiPower'])
            if (dataObj['specPower']!=0):
                roiRatio = dataObj['roiPower']/dataObj['specPower']
            else:
                roiRatio = 999
            self.roiRatioLst.append(roiRatio)
                
            if (dataObj['roiPower'] >= self.alarmThresh):
                self.roiRatioThreshLst.append(roiRatio)
            else:
                self.roiRatioThreshLst.append(0.)
            self.alarmStateLst.append(dataObj['alarmState'])
            # Record the minimum ROI Power that caused a WARNING or ALARM
            if (dataObj['alarmState']>0):
                if (dataObj['roiPower']>self.minRoiAlarmPower):
                    self.minRoiAlarmPower = dataObj['roiPower']
            self.alarmThreshLst.append(self.alarmThresh)
            self.alarmRatioThreshLst.append(self.alarmRatioThresh/10.)
            self.hrLst.append(dataObj['hr'])
            self.o2satLst.append(dataObj['o2Sat'])

            # Add to the raw data lists
            accLst = dataObj['rawData']
            # FIXME:  IT is not good to hard code the length of an array!
            for n in range(0,125):
                self.accelLst.append(accLst[n])
                self.rawTimestampLst.append((currTs + n*1./25.)-alarmTime)
        
    def plotRawDataGraph(self,outFname="rawData.png"):
        if (self.DEBUG): print("plotRawDataGraph")
        fig, ax = plt.subplots(2,1, figsize=(8,8))
        fig.suptitle('Event Number %d, %s\n%s, %s' % (
            self.eventId,
            self.dataTimeStr,
            self.eventObj['type'],
            self.eventObj['subType']),
                     fontsize=11)
        ax[0].plot(self.rawTimestampLst,self.accelLst)
        ax[0].set_title("Raw Data")
        ax[0].set_ylabel("Acceleration (~milli-g)")
        ax[0].grid(True)
        ax[1].plot(self.analysisTimestampLst, self.hrLst)
        ax[1].plot(self.analysisTimestampLst, self.o2satLst)
        ax[1].legend(['HR (bpm)','O2 Sat (%)'])
        ax[1].set_title("Heart Rate / O2 Sat")
        ax[1].set_xlabel("Time (seconds)")
        ax[1].grid(True)
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        fig.savefig(outFname)
        print("Graph written to %s" % outFname)

    def plotAnalysisGraph(self,outFname="analysis.png"):
        if (self.DEBUG): print("plotAnalysisGraph")
        fig, ax = plt.subplots(2,1, figsize=(8,8))
        fig.suptitle('Event Number %d, %s\n%s, %s' % (
            self.eventId,
            self.dataTimeStr,
            self.eventObj['type'],
            self.eventObj['subType']),
                     fontsize=11)
        ax[0].plot(self.analysisTimestampLst, self.specPowerLst)
        ax[0].plot(self.analysisTimestampLst, self.roiPowerLst)
        ax[0].plot(self.analysisTimestampLst, self.alarmThreshLst)
        ax[0].legend(['Spectrum Power','ROI Power', 'ROI Power Threshold'])
        ax[0].set_ylim(0,max(self.alarmThreshLst)*10)
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
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        fig.savefig(outFname)
        print("Graph written to %s" % outFname)


    def plotSpectrumGraph(self,outFname="spectrum.png"):
        if (self.DEBUG): print("plotSpectrumGraph")
        fig, ax = plt.subplots(1,1, figsize=(8,5))
        if len(self.analysisTimestampLst)>0:
            # Find the datapoint at the event time.
            for n in range (0,len(self.analysisTimestampLst)):
                #print(n,self.analysisTimestampLst[n])
                if (abs(self.analysisTimestampLst[n])<=1.):
                    #print("Found datapoint close to zero")
                    zeroDpN = n
                    zeroDp = self.dataPointsObj[n]
            #print(zeroDp)
            specLst = []
            specTimesLst = []
            for n in range(zeroDpN-1,zeroDpN+2):
                dp = self.dataPointsObj[n]
                dpObj = json.loads(dp['dataJSON'])
                dataObj = json.loads(dpObj['dataJSON'])
                specVals = dataObj['simpleSpec']
                # Normalise the spectrum so maximum value is always 1
                specMax = max(specVals)
                specNorm = [float(v)/specMax for v in specVals]
                specLst.append(specNorm)
                specTimesLst.append(self.analysisTimestampLst[n])
            #print(specLst)
            fig.suptitle('Event Number %d, %s\n%s, %s' % (
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
        ax.legend(specTimesLst)
        ax.set_xlabel("Frequency (Hz)")
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        fig.savefig(outFname)
        print("Graph written to %s" % outFname)


        
    def analyseEvent(self, eventId):
        if (self.DEBUG): print("analyseEvent: eventId=%d" % eventId)
        eventObj, dataPointsObj = self.getEventDataPoints(eventId)
        alarmTime = dateStr2secs(eventObj['dataTime'])
        # FIXME - plan ahead for when we pass 3 direction values,
        #  not magnitude!
        rawDataType = 0  # 0 = magnitude, 1=3 directions.
        
        # Collect all the raw data into a single list with associated
        # time from the alarm (in seconds)
        rawTimestampLst = []
        accelLst = []
        analysisTimestampLst = []
        specPowerLst = []
        roiPowerLst = []
        roiRatioLst = []
        alarmRatioThreshLst = []
        alarmStateLst = []
        hrLst = []
        o2satLst = []
        for dp in dataPointsObj:
            currTs = dateStr2secs(dp['dataTime'])
            if (self.DEBUG): print(dp['dataTime'], currTs)
            dpObj = json.loads(dp['dataJSON'])
            dataObj = json.loads(dpObj['dataJSON'])
            if (self.DEBUG): print(dataObj)
            analysisTimestampLst.append(currTs - alarmTime)
            specPowerLst.append(dataObj['specPower'])
            roiPowerLst.append(dataObj['roiPower'])
            roiRatioLst.append(dataObj['roiPower']/dataObj['specPower'])
            alarmStateLst.append(dataObj['alarmState'])
            alarmRatioThreshLst.append(dataObj['alarmRatioThresh']/10.)
            hrLst.append(dataObj['hr'])
            o2satLst.append(dataObj['o2Sat'])

            # Add to the raw data lists
            accLst = dataObj['rawData']
            # FIXME:  IT is not good to hard code the length of an array!
            for n in range(0,125):
                accelLst.append(accLst[n])
                rawTimestampLst.append((currTs + n*1./25.)-alarmTime)

        #for n in range(0,len(rawTimestampLst)):
        #    print(n,rawTimestampLst[n],accelLst[n])
        fig, ax = plt.subplots()
        ax.plot(rawTimestampLst,accelLst)
        fig.savefig("plot.png")

        fig, ax = plt.subplots(4,1, figsize=(5,9))
        fig.suptitle('Event Number %d, %s\n%s, %s\n%s' % (
            eventId,
            eventObj['dataTime'],
            eventObj['type'],
            eventObj['subType'],
            eventObj['desc']),
                     fontsize=11)
        ax[0].plot(rawTimestampLst,accelLst)
        ax[0].set_title("Raw Data")
        ax[1].plot(analysisTimestampLst, specPowerLst)
        ax[1].plot(analysisTimestampLst, roiPowerLst)
        ax[1].set_title("Spectrum / ROI Powers")
        ax[2].plot(analysisTimestampLst, hrLst)
        ax[2].plot(analysisTimestampLst, o2satLst)
        ax[2].set_title("Heart Rate / O2 Sat")
        ax[3].plot(analysisTimestampLst, roiRatioLst)
        ax[3].plot(analysisTimestampLst, alarmRatioThreshLst)
        ax[3].plot(analysisTimestampLst, alarmStateLst)
        ax[3].set_title("ROI Ratio & Alarm State")
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        fig.savefig("plot2.png")

    def testEvent(self, eventId, addr):
        ''' Retrieve event Number eventId from the remote database
        and feed its data to an OSD App at address addr.
        '''
        print("test_event: eventId=%d" % eventId)
        eventObj, dataPointsObj = self.getEventDataPoints(eventId)

        f = open("Event_%d_data.json" % eventId,"w")
        f.write(json.dumps(eventObj))
        f.write(json.dumps(dataPointsObj))
        f.close

        osd = libosd.osdAppConnection.OsdAppConnection(addr)

        for dp in dataPointsObj:
            currTs = dateStr2secs(dp['dataTime'])
            #print(dp['dataTime'], currTs)
            dpObj = json.loads(dp['dataJSON'])
            dataObj = json.loads(dpObj['dataJSON'])
            #print(dataObj)
            
            # Create raw data list
            accelLst = []
            accelLst3D = []
            # FIXME:  IT is not good to hard code the length of an array!
            for n in range(0,125):
                accelLst.append(dataObj['rawData'][n])
                if ('rawData3D' in dataObj.keys()):
                    accelLst3D.append(dataObj['rawData3D'][3*n])
                    accelLst3D.append(dataObj['rawData3D'][3*n + 1])
                    accelLst3D.append(dataObj['rawData3D'][3*n + 2])
                else:
                    # No 3D data available so do not send it.
                    pass

            rawDataObj = {"dataType": "raw", "Mute": 0}
            rawDataObj['HR'] = dataObj['hr']
            rawDataObj['data'] = accelLst
            rawDataObj['data3D'] = accelLst3D
            # FIXME - add o2sat
            dataJSON = json.dumps(rawDataObj)
            print(dp['dataTime'],dataJSON)
            osd.sendData(dataJSON)
            time.sleep(5)
        
            
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


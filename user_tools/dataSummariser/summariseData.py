#!/usr/bin/python3
"""
Produces a summary of events stored in the OpenSeizureDatabase.
The summary includes plots of the accelerometer traces and analysis
of the OpenSeizureDetector performance during the event.
"""

import argparse
import json
import sys
import os
import importlib
import dateutil.parser
import datetime
import numpy as np
import jinja2
import distutils

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import libosd.osdDbConnection
import eventAnalyser

def dateStr2secs(dateStr):
    ''' Convert a string representation of date/time into seconds from
    the start of 1970 (standard unix timestamp)
    '''
    parsed_t = dateutil.parser.parse(dateStr)
    return parsed_t.timestamp()


def makeIndex(configObj, evensLst=None, debug=False):
    """Make an html index to the summary files"""

def makeSummaries(configObj, eventsLst=None, outDir="output",
                  index=False, debug=False):
    """
    Make a summary of each event with ID in eventsLst.
    If eventsLst is None, a summary of each event in the database
    is produced.
    """
    # Load each of the three events files (tonic clonic seizures,
    #all seizures and false alarms).
    osd = libosd.osdDbConnection.OsdDbConnection(debug=debug)
    eventsObjLen = osd.loadDbFile(configObj['tcSeizuresFname'])
    print("tcSeizures  eventsObjLen=%d" % eventsObjLen)

    eventsObjLen = osd.loadDbFile(configObj['allSeizuresFname'])
    print("all Seizures eventsObjLen=%d" % eventsObjLen)

    eventsObjLen = osd.loadDbFile(configObj['falseAlarmsFname'])
    print("false alarms eventsObjLen=%d" % eventsObjLen)

    # Remove invalid events    
    invalidEvents = configObj['invalidEvents']
    print("Removing invalid events from database: ", invalidEvents)
    osd.removeEvents(invalidEvents)

    if eventsLst is None:
        eventsLst = osd.getEventIds()

    # Copy css and js into output directory.
    templateDir = os.path.join(os.path.dirname(__file__), 'templates/')
    distutils.dir_util.copy_tree(os.path.join(templateDir,"js"),
                                 os.path.join(outDir,"js"))
    distutils.dir_util.copy_tree(os.path.join(templateDir,"css"),
                                 os.path.join(outDir,"css"))

    tcSeizuresLst = []
    allSeizuresLst = []
    falseAlarmLst = []
    otherEventsLst = []
    for eventId in eventsLst:
        print("Producing Summary for Event %s" % eventId)
        eventObj = osd.getEvent(eventId, includeDatapoints=True)
        #print(eventObj)
        if not index:
            summariseEvent(eventObj)

        summaryObj = {}
        #summaryObj[''] = eventObj['']
        summaryObj['id'] = eventObj['id']
        summaryObj['dataTime'] = eventObj['dataTime']
        summaryObj['userId'] = eventObj['userId']
        summaryObj['type'] = eventObj['type']
        summaryObj['subType'] = eventObj['subType']
        summaryObj['desc'] = eventObj['desc']
        summaryObj['dataSourceName'] = getEventValue('dataSourceName',eventObj)
        summaryObj['phoneAppVersion'] = getEventValue('phoneAppVersion',eventObj)
        summaryObj['watchAppVersion'] = getEventValue('watchSdVersion', eventObj)
        summaryObj['url'] = "Event_%d_summary/index.html" % eventId

    
        if (eventObj['type']=='Seizure'):
            allSeizuresLst.append(summaryObj)
            if (eventObj['subType']=='Tonic-Clonic'):
                tcSeizuresLst.append(summaryObj)
        elif (eventObj['type']=='False Alarm'):
            falseAlarmLst.append(summaryObj)
        else:
            otherEventsLst.append(summaryObj)

    print("tcSeizures",tcSeizuresLst)

    # Render page
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(
            templateDir
        ))
    template = env.get_template('summary_index.html.template')
    os.makedirs(outDir, exist_ok=True)
    outFilePath = os.path.join(outDir,'index.html')
    outfile = open(outFilePath, 'w')
    #dataTime = dateutil.parser.parse(analyser.eventObj['dataTime'])
    pageData={
        'tcSeizures': tcSeizuresLst,
        'allSeizures': allSeizuresLst,
        'falseAlarms': falseAlarmLst,
        'otherEvents': otherEventsLst,
        }
    #print(pageData)
    outfile.write(template.render(data=pageData))
    outfile.close()



def getEventValue(param, eventObj):
    if ('dataJSON' in eventObj.keys()):
        #print(param, eventObj['dataJSON'])
        if (eventObj['dataJSON'] is not None):
            eventDataObj = json.loads(eventObj['dataJSON'])
            if (param in eventDataObj.keys()):
                return(eventDataObj[param])
            else:
                return('-')
        else:
            print("Error parsing Event %s" % eventObj['id'])
            return('-')
    else:
        return('-')
    
def summariseEvent(eventObj, outDirParent="output"):
    eventId = eventObj['id']
    #print("summariseEvent - EventId=%s" % eventId)
    outDir = os.path.join(outDirParent,"Event_%d_summary" % eventId)
    os.makedirs(outDir, exist_ok=True)
    #print("makeEventSummary - outDir=%s" % outDir)
    

    outFile = open(os.path.join(outDir,"rawData.json"),"w")
    json.dump(eventObj, outFile,sort_keys=True, indent=4)
    outFile.close()
    

    analyser = eventAnalyser.EventAnalyser(debug=False)
    analyser.analyseEvent(eventObj)
    #print("event analysis complete...")
    #print(analyser.eventObj)
    # Extract data from first datapoint to get OSD settings at time of event.
    dp=analyser.dataPointsLst[0]
    dpObj = json.loads(dp['dataJSON'])
    dataObj = json.loads(dpObj['dataJSON'])
    #print(dataObj)

    templateDir = os.path.join(os.path.dirname(__file__), 'templates/')
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(
            templateDir
        ))

    # Render page
    template = env.get_template('index.html.template')
    outFilePath = os.path.join(outDir,'index.html')
    outfile = open(outFilePath, 'w')
    dataTime = dateutil.parser.parse(analyser.eventObj['dataTime'])
    pageData={
        'eventId': analyser.eventObj['id'],
        'userId': analyser.eventObj['userId'],
        'eventDate': dataTime.strftime("%Y-%m-%d %H:%M"),
        'osdAlarmState': analyser.eventObj['osdAlarmState'],
        'eventType': analyser.eventObj['type'],
        'eventSubType': analyser.eventObj['subType'],
        'eventDesc': analyser.eventObj['desc'],
        'alarmFreqMin': analyser.alarmFreqMin,
        'alarmFreqMax': analyser.alarmFreqMax,
        'alarmThreshold': analyser.alarmThresh,
        'alarmRatioThreshold': analyser.alarmRatioThresh,
        'roiRatioMax': np.max(analyser.roiRatioLst),
        'roiRatioMaxThresholded': np.max(analyser.roiRatioThreshLst),
        'minRoiAlarmPower' : analyser.minRoiAlarmPower,
        'pageDateStr': (datetime.datetime.now()).strftime("%Y-%m-%d %H:%M"),
        }
    #print(pageData)
    outfile.write(template.render(data=pageData))
    outfile.close()

    # Plot Raw data graph
    analyser.plotRawDataGraph(os.path.join(outDir,'rawData.png'))
    analyser.plotHrGraph(os.path.join(outDir,'hrData.png'))

    # Plot Analysis data graph
    analyser.plotAnalysisGraph(os.path.join(outDir,'analysis.png'))

    # Plot Spectrum graph
    analyser.plotSpectrumGraph(os.path.join(outDir,'spectrum.png'))

       
    print("Data written to %s" % outFilePath)


def main():
    print("summariseData.main()")
    parser = argparse.ArgumentParser(description='Summarise Data in OpenSeizureDatabase')
    parser.add_argument('--config', default="osdbCfg.json",
                        help='name of json configuration file')
    parser.add_argument('--event',
                        help='event to summarise (or comma separated list of event IDs)')
    parser.add_argument('--outDir', default="output",
                        help='output directory')
    parser.add_argument('--index', action="store_true",
                        help='Re-build index, not all summaries')
    parser.add_argument('--debug', action="store_true",
                        help='Write debugging information to screen')
    
    argsNamespace = parser.parse_args()
    args = vars(argsNamespace)
    print(args)


    inFile = open(args['config'],'r')
    configObj = json.load(inFile)
    inFile.close()
    if args['event'] is not None:
        eventsLst = args['event'].split(',')
        eventsLst2 = []
        for eventId in eventsLst:
            eventsLst2.append(eventId.strip())
    else:
        eventsLst2 = None    
    makeSummaries(configObj, eventsLst2,
                  outDir=args['outDir'], index=args['index'], debug=args['debug'])
    


if __name__ == "__main__":
    main()
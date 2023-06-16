#!/usr/bin/env python3
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
import distutils.dir_util

import jsbeautifier

sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
import libosd.osdDbConnection
import libosd.webApiConnection
import libosd.tidy_db
import eventAnalyser

def dateStr2secs(dateStr):
    ''' Convert a string representation of date/time into seconds from
    the start of 1970 (standard unix timestamp)
    '''
    parsed_t = dateutil.parser.parse(dateStr)
    return parsed_t.timestamp()


def makeIndex(configObj, evensLst=None, debug=False):
    """Make an html index to the summary files"""

def makeSummaries(configObj, eventsLst=None, remoteDb=False, outDir="output",
                  index=False, debug=False):
    """
    Make a summary of each event with ID in eventsLst.
    If eventsLst is None, a summary of each event in the database
    is produced.
    """

    # If we are not using the remote database, load data from local database files.
    if remoteDb:
        osd = libosd.webApiConnection.WebApiConnection(cfg=configObj['credentialsFname'],
                                               download=True,
                                               debug=debug)
    else:
        # Load each of the three events files (tonic clonic seizures,
        #all seizures and false alarms).
        osd = libosd.osdDbConnection.OsdDbConnection(debug=debug)

        for fname in configObj['dataFiles']:
            print("Loading OSDB File %s." % fname)
            eventsObjLen = osd.loadDbFile(fname)
            print("......eventsObjLen=%d" % eventsObjLen)

        # Remove invalid events    
        invalidEvents = configObj['invalidEvents']
        print("Removing invalid events from database: ", invalidEvents)
        osd.removeEvents(invalidEvents)

        if eventsLst is None:
            eventsLst = osd.getEventIds()

        print("eventsLst=",eventsLst)

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
        os.makedirs(outDir, exist_ok=True)
        eventObj = osd.getEvent(eventId, includeDatapoints=True)
        libosd.tidy_db.tidyEventObj(configObj, eventObj, debug)
        print(eventObj.keys())
        if not index:
            makeOutDir(eventObj,outDir)
        analyser = eventAnalyser.EventAnalyser(debug=debug)
        analyser.analyseEvent(eventObj)
        print("returned from eventAnalyser.analyseEvent")
        #print(analyser.dataPointsTdiff)
        #print(eventObj)
        if not index:
            # Make detailed summary of event as a separate web page
            summariseEvent(eventObj)

        print("summariseData: Making summaryObj")
        # Build the index of the events in the database.
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
        summaryObj['nDataPoints'] = analyser.nDataPoints
        summaryObj['nDpGaps'] = analyser.nDpGaps
        summaryObj['nDpExtras'] = analyser.nDpExtras
        # print("eventID=", eventId, type(eventId))
        summaryObj['url'] = "Event_%s_summary/index.html" % eventId

    
        if (eventObj['type']=='Seizure'):
            allSeizuresLst.append(summaryObj)
            if (eventObj['subType']=='Tonic-Clonic'):
                tcSeizuresLst.append(summaryObj)
        elif (eventObj['type']=='False Alarm'):
            falseAlarmLst.append(summaryObj)
        else:
            otherEventsLst.append(summaryObj)

    #print("tcSeizures",tcSeizuresLst)

    # Render page
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(
            templateDir
        ))
    template = env.get_template('summary_index.html.template')
    outFilePath = os.path.join(outDir,'index.html')
    outfile = open(outFilePath, 'w')
    #dataTime = dateutil.parser.parse(analyser.eventObj['dataTime'])
    pageData={
        'events': {
            'tcSeizures': tcSeizuresLst,
            'allSeizures': allSeizuresLst,
            'falseAlarms': [],
            'otherEvents': []
        }
    }
    #print(pageData)
    print("Rendering index page")
    outfile.write(template.render(data=pageData))
    outfile.close()



def getEventValue(param, eventObj):
    if (param in eventObj.keys()):
        return(eventObj[param])
    else:
        return('-')

def makeOutDir(eventObj, outDirParent="output"):
    eventId = eventObj['id']
    #print("summariseEvent - EventId=%s" % eventId)
    outDir = os.path.join(outDirParent,"Event_%s_summary" % eventId)
    os.makedirs(outDir, exist_ok=True)
    #print("makeEventSummary - outDir=%s" % outDir)

    outFile = open(os.path.join(outDir,"rawData.json"),"w")
    options = jsbeautifier.default_options()
    options.indent_size = 2
    jsonStr = json.dumps(eventObj,sort_keys=True)
    outFile.write(jsbeautifier.beautify(jsonStr, options))
    
    outFile.close()
    return outDir
   

def summariseEvent(eventObj, outDirParent="output"):
    eventId = eventObj['id']
    outDir = makeOutDir(eventObj, outDirParent)
    
    analyser = eventAnalyser.EventAnalyser(debug=False)
    analyser.analyseEvent(eventObj)
    #print("event analysis complete...")
 
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

    if len(analyser.roiRatioLst)>0:
        roiRatioMax = np.max(analyser.roiRatioLst)
    else:
        roiRatioMax = -1
    if len(analyser.roiRatioThreshLst)>0:
        roiRatioMaxThresh = np.max(analyser.roiRatioThreshLst)
    else:
        roiRatioMaxThresh = -1

    eventDataObj = None
    if 'dataJSON' in analyser.eventObj:
        if (analyser.eventObj['dataJSON'] is not None and
            analyser.eventObj['dataJSON'] != ''):
            eventDataObj = json.loads(analyser.eventObj['dataJSON'])
            
    pageData={
        'eventId': analyser.eventObj['id'],
        'userId': analyser.eventObj['userId'],
        'eventDate': dataTime.strftime("%Y-%m-%d %H:%M"),
        'osdAlarmState': analyser.eventObj['osdAlarmState'],
        'eventType': analyser.eventObj['type'],
        'eventSubType': analyser.eventObj['subType'],
        'eventDesc': analyser.eventObj['desc'],
        'nDatapoints': analyser.nDataPoints,
        'phoneAppVersion': analyser.eventObj['phoneAppVersion'], 
        'watchAppVersion': analyser.eventObj['watchSdVersion'],
        'dataSourceName': analyser.eventObj['dataSourceName'],
        'osdAlarmActive': analyser.eventObj['osdAlarmActive'],
        'alarmFreqMin': analyser.alarmFreqMin,
        'alarmFreqMax': analyser.alarmFreqMax,
        'alarmThreshold': analyser.alarmThresh,
        'alarmRatioThreshold': analyser.alarmRatioThresh,
        'roiRatioMax': roiRatioMax,
        'roiRatioMaxThresholded': roiRatioMaxThresh,
        'minRoiAlarmPower' : analyser.minRoiAlarmPower,
        'hrAlarmActive': analyser.eventObj['hrAlarmActive'],
        'hrThreshMin': analyser.eventObj['hrThreshMin'],
        'hrThreshMax': analyser.eventObj['hrThreshMax'],
        'pageDateStr': (datetime.datetime.now()).strftime("%Y-%m-%d %H:%M"),
        }
    print(pageData)
    outfile.write(template.render(data=pageData))
    outfile.close()

    # Plot spectral history image
    analyser.plotSpectralHistory(os.path.join(outDir,'spectralHistory.png'))


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
    parser.add_argument('--remote', action="store_true",
                        help="Load events data from remote database, not locally cached OSDB")
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
            eventsLst2.append(int(eventId.strip()))
    else:
        eventsLst2 = None    
    makeSummaries(configObj, eventsLst2, remoteDb=args['remote'],
                  outDir=args['outDir'], index=args['index'], debug=args['debug'])
    


if __name__ == "__main__":
    main()

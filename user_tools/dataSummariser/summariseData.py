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
#import importlib
import dateutil.parser
import datetime
import numpy as np
import jinja2
#import distutils.dir_util

import matplotlib.pyplot as plt

import pandas as pd

import jsbeautifier

sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
import libosd.osdDbConnection
import libosd.webApiConnection
import libosd.tidy_db
import libosd.configUtils
import eventAnalyser

def dateStr2secs(dateStr):
    ''' Convert a string representation of date/time into seconds from
    the start of 1970 (standard unix timestamp)
    '''
    parsed_t = dateutil.parser.parse(dateStr)
    return parsed_t.timestamp()


def loadOsdbData(configObj, remoteDb=False, debug=False):
    ''' Load the OSDB data from either the local databaes or the remote on-line db, and return an osd object containing the data
    '''
    # If we are not using the remote database, load data from local database files.
    if remoteDb:
        osd = libosd.webApiConnection.WebApiConnection(cfg=configObj['credentialsFname'],
                                               download=True,
                                               debug=debug)
    else:
        # Load each of the three events files (tonic clonic seizures,
        #all seizures and false alarms).
        cacheDir = libosd.configUtils.getConfigParam('cacheDir', configObj)
        osd = libosd.osdDbConnection.OsdDbConnection(cacheDir=cacheDir, debug=debug)

        for fname in configObj['dataFiles']:
            print("Loading OSDB File %s." % fname)
            eventsObjLen = osd.loadDbFile(fname)
            print("......eventsObjLen=%d" % eventsObjLen)

    # Remove invalid events    
    invalidEvents = configObj['invalidEvents']
    if (debug): print("Removing invalid events from database: ", invalidEvents)
    osd.removeEvents(invalidEvents)
    if (debug): print("loadOsdbData() - returning %d events" % len(osd.getAllEvents(includeDatapoints=False)))

    return osd



def loadOsdDf(configObj, remoteDb=False, debug=False):
    '''
    Load the OSDB events data (not the raw data points) into a pandas dataframe
    '''
    osd = loadOsdbData(configObj, remoteDb=remoteDb, debug=debug)
    # Create empty dataframes for the different classes of events
    allEventsDf = pd.DataFrame()
    print(osd, dir(osd))
    eventLst = osd.getAllEvents(includeDatapoints=False)
    print("Loaded %d events" % len(eventLst))

    # Read the event list into a pandas data frame.
    df = pd.read_json(json.dumps(eventLst))
    # Convert dataTime strings to dateTime objects - note that without the errors= parameter, it fails silently!
    df['dataTime'] = pd.to_datetime(df['dataTime'], errors='raise', utc=True, format="mixed")
    #print(df.dtypes)
    #print(df['dataTime'])
    # Force the dataTime objects to be local time without tz offsets (avoids error about offset naive and offset aware datatime comparisons)
    #   (from https://stackoverflow.com/questions/46295355/pandas-cant-compare-offset-naive-and-offset-aware-datetimes)
    df['dataTime']=df['dataTime'].dt.tz_localize(None)
    #df['userId']=str(df['userId'])

    if debug: print(df.columns, df)

    return(df)
   


def makeIndex(configObj, evensLst=None, debug=False):
    """Make an html index to the summary files"""


def makeUserSummary(configObj, userId, remoteDb=False, outDir='output', debug=False):
    print("makeUserSummary, userId=%s" % userId)

    # Read all the OSDB event data into a data frame.
    osdDf = loadOsdDf(configObj, remoteDb, debug)

    #print(osdDf, osdDf.columns)
    #print(osdDf['userId'])

    # Select data only for the required user
    userDf = osdDf[osdDf['userId']==int(userId)]
                   
    #print(userDf)

    # Group the data by event type and time period
    groupingPeriod = "ME"  # = Month End
    print("Grouping into periods of %s" % groupingPeriod)
    groupedDf=userDf.groupby(['type', 'subType',pd.Grouper(
        key='dataTime',
        freq=groupingPeriod)], observed=False)['id'].count()

    # Force months with zero data to be included as zero rather than excluded from the result:
    #   From https://stackoverflow.com/questions/54355740/how-to-get-pd-grouper-to-include-empty-groups
    m = pd.MultiIndex.from_product([userDf.type.unique(), userDf.subType.unique(), 
                                    pd.date_range(userDf.dataTime.min() , userDf.dataTime.max() + pd.offsets.MonthEnd(1), freq='ME', normalize=True)])

    if (debug): print("New Index, m=",m)

    if (debug): print("Before Reindexing:", groupedDf)
    groupedDf = groupedDf.reindex(m)
    groupedDf = groupedDf.fillna(0)
    if (debug): print("After Reindexing:", groupedDf)

    # Loop through the grouped data
    #for groupParts, group in groupedDf:
    #    eventType, subType, dataTime = groupParts
    #    if (debug): print()
    #    if (debug): print("Starting New Group....")
    #    if (debug): print("type=%s, subType=%s, dataTime=%s" % (eventType, subType,
    #                                               dataTime.strftime('%Y-%m-%d %H:%M:%S')))
    print("Counts...")
    #pd.set_option('display.max_rows', None)
    
    #print(groupedDf['id'].count())
    #pd.reset_option('display.max_rows')

    print("groupedDf=", groupedDf, type(groupedDf))


    # Calculate monthly totals for all seizures.
    allSeizureDf = pd.concat([groupedDf['Seizure'].unstack().sum()], axis=1)
    allSeizureDf.columns = ["allSeizures"]
    allSeizureDf['avg'] = allSeizureDf.rolling(window=3).mean()
    allSeizureDf['avg'] = allSeizureDf['avg'].fillna(0)
    print(allSeizureDf, type(allSeizureDf), allSeizureDf.columns)

    # Plot Seizure Graph.
    fig, ax = plt.subplots(figsize=(12, 4))
    allSeizureDf['allSeizures'].plot( ax=ax, marker='x', linestyle='none')
    allSeizureDf['avg'].plot(ax=ax)
    ax.set_ylabel("Seizures per Month")
    ax.set_xlabel("Month Ending (date)")

    fig.savefig("userdata1.png")

def makeSummaries(configObj, eventsLst=None, remoteDb=False, outDir="output",
                  index=False, debug=False):
    """
    Make a summary of each event with ID in eventsLst.
    If eventsLst is None, a summary of each event in the database
    is produced.
    """
    print("makeSummaries()")
    osd = loadOsdbData(configObj, remoteDb, debug)

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
        if eventObj is None:
            print("*****ERROR - Event %s not found in database ****" % eventId)
            exit(-1)
        libosd.tidy_db.tidyEventObj(configObj, eventObj, debug)
        print(eventObj.keys())

        analyser = eventAnalyser.EventAnalyser(debug=debug)
        analyser.analyseEvent(eventObj)
        print("returned from eventAnalyser.analyseEvent")
        #print(analyser.dataPointsTdiff)
        #print(eventObj)
        if not index:
            # Make detailed summary of event as a separate web page
            summariseEvent(eventObj, outDir)

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
    with open(outFilePath, "w") as outFile:
        outFile.write(template.render(data=pageData))



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

    options = jsbeautifier.default_options()
    options.indent_size = 2
    jsonStr = json.dumps(eventObj,sort_keys=True)
    with open(os.path.join(outDir,"rawData.json"),"w") as outFile:
        outFile.write(jsbeautifier.beautify(jsonStr, options))

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
        'eventType': getEventVal(analyser.eventObj,'type'),
        'eventSubType': getEventVal(analyser.eventObj,'subType'),
        'eventDesc': getEventVal(analyser.eventObj,'desc'),
        'nDatapoints': analyser.nDataPoints,
        'phoneAppVersion': getEventVal(analyser.eventObj,'phoneAppVersion'), 
        'watchAppVersion': getEventVal(analyser.eventObj,'watchSdVersion'),
        'dataSourceName': getEventVal(analyser.eventObj,'dataSourceName'),
        'osdAlarmActive': getEventVal(analyser.eventObj,'osdAlarmActive'),
        'alarmFreqMin': analyser.alarmFreqMin,
        'alarmFreqMax': analyser.alarmFreqMax,
        'alarmThreshold': analyser.alarmThresh,
        'alarmRatioThreshold': analyser.alarmRatioThresh,
        'roiRatioMax': roiRatioMax,
        'roiRatioMaxThresholded': roiRatioMaxThresh,
        'minRoiAlarmPower' : analyser.minRoiAlarmPower,
        'hrAlarmActive': getEventVal(analyser.eventObj,'hrAlarmActive'),
        'hrThreshMin': getEventVal(analyser.eventObj,'hrThreshMin'),
        'hrThreshMax': getEventVal(analyser.eventObj,'hrThreshMax'),
        'pageDateStr': (datetime.datetime.now()).strftime("%Y-%m-%d %H:%M"),
        }
    print(pageData)

    with open(outFilePath, "w") as outFile:
        outFile.write(template.render(data=pageData))

    # Plot spectral history image
    analyser.plotSpectralHistory(os.path.join(outDir,'spectralHistory.png'), 
                                 colImgFname=os.path.join(outDir,'spectralHistoryColour.png'))


    # Plot Raw data graph
    analyser.plotRawDataGraph(os.path.join(outDir,'rawData.png'))
    analyser.plotHrGraph(os.path.join(outDir,'hrData.png'))

    # Plot Analysis data graph
    analyser.plotAnalysisGraph(os.path.join(outDir,'analysis.png'))

    # Plot Spectrum graph
    analyser.plotSpectrumGraph(os.path.join(outDir,'spectrum.png'))

    analyser.saveAccelCsv(os.path.join(outDir,'accelData.csv'))
       
    print("Data written to %s" % outFilePath)


def getEventVal(eventObj, elemId):
    if (elemId in eventObj.keys()):
        return eventObj[elemId]
    else:
        return None

def main():
    print("summariseData.main()")
    parser = argparse.ArgumentParser(description='Summarise Data in OpenSeizureDatabase')
    parser.add_argument('--config', default="osdbCfg.json",
                        help='name of json configuration file')
    parser.add_argument('--remote', action="store_true",
                        help="Load events data from remote database, not locally cached OSDB")
    parser.add_argument('--event',
                        help='event to summarise (or comma separated list of event IDs) - specify ALL to produce summary of all events in database')
    parser.add_argument('--user',
                        help='produce summary of contributions by a specific user id.')
    parser.add_argument('--outDir', default="output",
                        help='output directory')
    parser.add_argument('--index', action="store_true",
                        help='Re-build index, not all summaries')
    parser.add_argument('--debug', action="store_true",
                        help='Write debugging information to screen')
    
    argsNamespace = parser.parse_args()
    args = vars(argsNamespace)
    print(args)

    with open(args['config'], 'r') as inFile:
        configObj = json.load(inFile)

    if args['user'] is not None:
        print("Producing summary for user %s" % args['user'])
        makeUserSummary(configObj, userId=args['user'], remoteDb=args['remote'],
                  outDir=args['outDir'], debug=args['debug'])
        exit(0)

    if args['event'] is not None:
        eventsLst = args['event'].split(',')
        eventsLst2 = []
        for eventId in eventsLst:
            eventsLst2.append(str(eventId.strip()))
        if (len(eventsLst2) == 1 and eventsLst2[0]=='ALL'):
            print("Selecting all events")
            eventsLst2 = None
        makeSummaries(configObj, eventsLst2, remoteDb=args['remote'],
                      outDir=args['outDir'], index=args['index'], debug=args['debug'])
    


if __name__ == "__main__":
    main()

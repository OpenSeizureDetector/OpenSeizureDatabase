#!/usr/bin/env python3
"""
false_alarm_analysis.py
This script loads OSDB data from the locally stored JSON files
and creates a summary of the false alarm rate for a given user.

Graham Jones 2023


Licence: GPL v3 or later.

"""
import sys
import os
import argparse
import pandas as pd
import json
import tabulate

# Make the libosd folder accessible in the search path.
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
import libosd.osdDbConnection
import libosd.configUtils


            
def getEventsDf(osd,
                         start=None,
                         end=None,
                         userIds=None,
                         eventTypes=None,
                         debug=False):
    """
    Returns a Pandas Dataframe containing a summary of all the events in the database between start and end times.

    Parameters:
    osd - instance of OpenSeizureDatabase, with data loaded.
    start - None or date from which to extract data in dd/mm/yyyy format.
    end - None or date
    userIds - list of userIds to be included in the output.
    debug - print debugging information to console.
    """
    print("getEventsDf - debug=%d" % debug)
    cfgObj = {'groupingPeriod': '1d'}
    # Create empty dataframes for the different classes of events
    allEventsDf = pd.DataFrame()
    print(osd, dir(osd))
    eventLst = osd.getAllEvents(includeDatapoints=False)
    print("Loaded %d events" % len(eventLst))

    # Read the event list into a pandas data frame.
    df = pd.read_json(json.dumps(eventLst))
    # Convert dataTime strings to dateTime objects - note that without the errors= parameter, it fails silently!
    df['dataTime'] = pd.to_datetime(df['dataTime'], errors='raise', utc=True)
    print(df.dtypes)
    print(df['dataTime'])
    # Force the dataTime objects to be local time without tz offsets (avoids error about offset naive and offset aware datatime comparisons)
    #   (from https://stackoverflow.com/questions/46295355/pandas-cant-compare-offset-naive-and-offset-aware-datetimes)
    df['dataTime']=df['dataTime'].dt.tz_localize(None)


    if debug: print(df)

    # Filter out warnings (unless they are tagged as a seizure) and tests.
    print("Filtering out warnings (unless they are associated with a seizure or a fall event)")
    df=df.query("type=='Seizure' or type=='Fall' or osdAlarmState!=1")

    # Filter by date
    if (start is not None):
        startDateTime = pd.to_datetime(start, utc=True)
        dateQueryStr = 'dataTime >= "%s"' % startDateTime
        print("Applying Date Query: %s" % dateQueryStr)
        df = df.query(dateQueryStr)

    # Filter by end date
    if (end is not None):
        endDateTime = pd.to_datetime(end, utc=True)
        dateQueryStr = 'dataTime <= "%s"' % endDateTime
        print("Applying Date Query: %s" % dateQueryStr)
        df = df.query(dateQueryStr)


    # Filter by userId
    if (userIds is not None):
        print("Applying userId Query")
        df = df[df['userId'].isin(userIds)]
        print(df)

    # Filter by event Type
    if (eventTypes is not None):
        print("Applying event type Query")
        df = df[df['type'].isin(eventTypes)]
        print(df)


    return(df)

def getGroupedData(df, groupingPeriod='1D', debug=False):
    #
    # This is to set the print order when we print the data frames
    columnList = ['id', 'userId',
                  'dataTime', 'type',
                  'subType', 'osdAlarmState',
                  'desc']

    # Group the data by userID and time period
    print("Grouping into periods of %s" % groupingPeriod)
    groupedDf=df.groupby(['type', 'userId',pd.Grouper(
        key='dataTime',
        freq=groupingPeriod)])

    # Loop through the grouped data
    for groupParts, group in groupedDf:
        eventType, userId, dataTime = groupParts
        if (debug): print()
        if (debug): print("Starting New Group....")
        print("UserId=%d, type=%s, dataTime=%s" % (userId, eventType,
                                                   dataTime.strftime('%Y-%m-%d %H:%M:%S')))
    print("Counts...")
    pd.set_option('display.max_rows', None)
    print(groupedDf['id'].count())
    pd.reset_option('display.max_rows')
    return(groupedDf)



if (__name__=="__main__"):
    print("makeOsdDb.py.main()")
    parser = argparse.ArgumentParser(description='Produce a data dashboard from the OpenSeizureDatabase.')
    parser.add_argument('--config', default="osdbCfg.json",
                        help='name of json file containing configuration information and login credientials - see osdb.cfg.template')
    parser.add_argument('--start', default=None,
                        help="Start date for saving data (yyyy-mm-dd format).  Data before this date is not extracted from the database")
    parser.add_argument('--end', default=None,
                        help="End date for saving data (yyyy-mm-dd format).  Data after this date is not extracted from the database")
    parser.add_argument('--debug', action='store_true',
                        help="Write debugging information to screen")
    parser.add_argument('--outDir', default="osdb_dashboard",
                        help='output directory')
    
    argsNamespace = parser.parse_args()
    args = vars(argsNamespace)
    print(args)

    configObj = libosd.configUtils.loadConfig(args['config'])
    print(configObj)

    dbDir = libosd.configUtils.getConfigParam("cacheDir", configObj)
    debug = args['debug']
    invalidEvents = libosd.configUtils.getConfigParam("invalidEvents", configObj)

    # Load each of the three events files (tonic clonic seizures,
    #all seizures and false alarms).
    osd = libosd.osdDbConnection.OsdDbConnection(cacheDir=dbDir, debug=debug)

    outDir = args['outDir']

    dataFilesLst = libosd.configUtils.getConfigParam("dataFiles", configObj)
    for fname in dataFilesLst:
        eventsObjLen = osd.loadDbFile(fname)
        print("loaded %d events from file %s" % (eventsObjLen, fname))
    osd.removeEvents(invalidEvents)
    #osd.listEvents()
    print("Events Loaded - converting to Pandas Dataframe")

    print("debug=",debug)
    df = getEventsDf(osd, start=None, end=None, userIds=[39], eventTypes=['False Alarm'],debug=debug)


    columnList = ['id', 'userId',
                  'dataTime', 'type',
                  'subType', 'osdAlarmState', 'alarmThresh', 'alarmRatioThresh',
                  'desc']

    print(df[columnList])
    print("Analysing...")

    dfGrouped = getGroupedData(df, groupingPeriod='1W')
    print(dfGrouped)

    df.to_csv("FalseAlarmsList.csv", columns=columnList)
    df2 = dfGrouped['id'].count()

    print(df2)
    df2.to_csv("FalseAlarmCounts.csv")

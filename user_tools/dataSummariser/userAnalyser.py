#!/bin/env python
#
# Functions to generate user based summaries of OpenSeizureDatabase data - called from summariseData.py

import sys, os
import matplotlib.pyplot as plt

import pandas as pd
import jinja2
import datetime

import summariseData


def makeUserSummary(configObj, userId, remoteDb=False, outDirParent='output', debug=False):
    print("makeUserSummary, userId=%s" % userId)
    outDir = os.path.join(outDirParent,"User_%s_summary" % userId)
    os.makedirs(outDir, exist_ok=True)

    # Read all the OSDB event data into a data frame.
    osdDf = summariseData.loadOsdDf(configObj, remoteDb, debug)

    #print(osdDf, osdDf.columns)
    #print(osdDf['userId'])

    # Select data only for the required user (this will contain both seizure and false alarm data)
    userAllDataDf = osdDf[osdDf['userId']==int(userId)].sort_values('dataTime')
    
    userSeizureDf = userAllDataDf[userAllDataDf['type']=='Seizure']
    # Calculate the time between subsequent seizures, and the rolling average time.
    userSeizureDf['spacing'] = userSeizureDf['dataTime'].diff().dt.days   #.astype('timedelta64[d]')
    userSeizureDf['spacing_avg'] = userSeizureDf['spacing'].rolling(window=3).mean()
    print("userDf=",userSeizureDf)

    userSeizureDf['correct'] = userSeizureDf['alarmState']==2

    allSeizuresCount = len(userSeizureDf)
    allSeizuresCorrectCount = len(userSeizureDf[userSeizureDf['correct']])
    if (allSeizuresCount>0):
        allSeizuresReliability = 1.0*allSeizuresCorrectCount/allSeizuresCount
    else:
        allSeizuresReliability = 0.0

    tcSeizuresCount = len(userSeizureDf[userSeizureDf['subType']=='Tonic-Clonic'])
    tcSeizuresCorrectCount = len(userSeizureDf[((userSeizureDf['subType']=='Tonic-Clonic') & (userSeizureDf['correct']))])
    if (tcSeizuresCount > 0):
        tcSeizuresReliability = 1.0*tcSeizuresCorrectCount/tcSeizuresCount
    else:
        tcSeizuresReliability = 0.0

    # Sometimes we have two seizures recorded in rapid succession, which are really part of the same seizure 'event' 
    #   and contain duplicate data.  These mess up the reliability statistics, so we group into periods of 10 minutes
    #   to make sure we are only counting one actual seizure (assumes you don't have 2 seizures within 10 minutes..)
    tenMinUserSeizureDf = pd.concat(
        [
            userSeizureDf.groupby([
            pd.Grouper( key='dataTime', freq='10Min'),
            ]
            )['alarmState'].max()
        ], axis=1)
    tenMinUserSeizureDf['alarmState'] = tenMinUserSeizureDf['alarmState'].fillna(0)
    tenMinUserSeizureDf = tenMinUserSeizureDf[tenMinUserSeizureDf['alarmState']>0]

    print(tenMinUserSeizureDf)


    # Sometimes we have two seizures recorded in rapid succession, which are really part of the same seizure 'event' which mess up the calculation
    #  of seizure spacing time.
    #  To avoid this we just identify which days have seizure events and calculate the time between seizure days.
    dailyUserSeizureDf = pd.concat(
        [
            userSeizureDf.groupby([
            pd.Grouper( key='dataTime', freq='d'),
            ]
            )['id'].count()
        ], axis=1)
    
    #print("dailyUserSeizureDf=\n", dailyUserSeizureDf, dailyUserSeizureDf.columns)
    dailyUserSeizureDf = dailyUserSeizureDf[dailyUserSeizureDf['id']>0]
    #print("dailyUserSeizureDf=\n", dailyUserSeizureDf, dailyUserSeizureDf.columns)
    dailyUserSeizureDf['dataTime'] = dailyUserSeizureDf.index
    dailyUserSeizureDf['spacing'] = dailyUserSeizureDf['dataTime'].diff().dt.days   #.astype('timedelta64[d]')
    dailyUserSeizureDf['spacing_avg'] = dailyUserSeizureDf['spacing'].rolling(window=3).mean()


    # Group the data by event type and time period
    groupingPeriod = "ME"  # = Month End
    print("Grouping into periods of %s" % groupingPeriod)
    groupedDf=userSeizureDf.groupby(['type', 'subType',pd.Grouper(
        key='dataTime',
        freq=groupingPeriod)], observed=False)['id'].count()

    # Force months with zero data to be included as zero rather than excluded from the result:
    #   From https://stackoverflow.com/questions/54355740/how-to-get-pd-grouper-to-include-empty-groups
    m = pd.MultiIndex.from_product([userSeizureDf.type.unique(), userSeizureDf.subType.unique(), 
                                    pd.date_range(userSeizureDf.dataTime.min() , userSeizureDf.dataTime.max() + pd.offsets.MonthEnd(1), freq='ME', normalize=True)])

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

    #print("groupedDf=", groupedDf, type(groupedDf))


    # Calculate monthly totals for all seizures.
    allSeizureDf = pd.concat([groupedDf['Seizure'].unstack().sum()], axis=1)
    allSeizureDf.columns = ["allSeizures"]
    allSeizureDf['avg'] = allSeizureDf.rolling(window=3).mean()
    allSeizureDf['avg'] = allSeizureDf['avg'].fillna(0)
    #print(allSeizureDf, type(allSeizureDf), allSeizureDf.columns)

    
    pageData={
        'userId': userId,
        'nSeizures': len(userSeizureDf),
        'nAllSeizures': allSeizuresCount,
        'nAllSeizuresCorrect': allSeizuresCorrectCount,
        'allSeizuresReliability': allSeizuresReliability,
        'nTcSeizures': tcSeizuresCount,
        'nTcSeizuresCorrect': tcSeizuresCorrectCount,
        'tcSeizuresReliability': tcSeizuresReliability,
        'tcSeizureLst': userSeizureDf[userSeizureDf['subType']=='Tonic-Clonic'].to_dict('records'),
        'seizureLst': userSeizureDf.to_dict('records'),
        'pageDateStr': (datetime.datetime.now()).strftime("%Y-%m-%d %H:%M"),
        }
    #print(pageData)


    # Render page
    templateDir = os.path.join(os.path.dirname(__file__), 'templates/')
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(
            templateDir
        ))
    template = env.get_template('userSummary.template')
    outFilePath = os.path.join(outDir,'index.html')

    with open(outFilePath, "w") as outFile:
        outFile.write(template.render(data=pageData))



    # Plot Seizure Time Spacings Graph
    fig, ax = plt.subplots(figsize=(12, 4))
    userSeizureDf.plot( ax=ax, y='spacing', x='dataTime', marker='x', linestyle='none')
    userSeizureDf.plot(ax=ax, y='spacing_avg', x='dataTime')
    ax.set_ylabel("Time between Seizures (days)")
    ax.set_xlabel("Date")
    ax.grid(True)
    fig.savefig(os.path.join(outDir,"seizureSpacing.png"))

    # Plot Daily Seizure Time Spacings Graph
    fig, ax = plt.subplots(figsize=(12, 4))
    dailyUserSeizureDf.plot( ax=ax, y='spacing', x='dataTime', marker='x', linestyle='none')
    dailyUserSeizureDf.plot(ax=ax, y='spacing_avg', x='dataTime')
    ax.set_ylabel("Time between Seizures (days)")
    ax.set_xlabel("Date")
    ax.grid(True)
    fig.savefig(os.path.join(outDir,"daily_user_seizure_spacing.png"))


    # Plot Seizure Graph.
    fig, ax = plt.subplots(figsize=(12, 4))
    allSeizureDf['allSeizures'].plot( ax=ax, marker='x', linestyle='none')
    allSeizureDf['avg'].plot(ax=ax)
    ax.set_ylabel("Seizures per Month")
    ax.set_xlabel("Month Ending (date)")
    ax.grid(True)

    fig.savefig(os.path.join(outDir,"monthly_user_seizures.png"))

#!/usr/bin/env python3

import argparse
from re import X
import sys
import os
import json
import importlib
from urllib.parse import _NetlocResultMixinStr
#from tkinter import Y
import sklearn.model_selection
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime as dt

import gc

import pandas as pd
import imblearn

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libosd.osdDbConnection
import libosd.dpTools
import libosd.osdAlgTools
import libosd.configUtils


def calcProcessTime(starttime, cur_iter, max_iter):
    # From https://stackoverflow.com/questions/44926127/calculating-the-amount-of-time-left-until-completion
    telapsed = time.time() - starttime
    testimated = (telapsed/(cur_iter+1))*(max_iter)
    finishtime = starttime + testimated
    finishtime = dt.datetime.fromtimestamp(finishtime).strftime("%H:%M:%S")  # in time
    lefttime = testimated-telapsed  # in seconds
    iterTime = (telapsed/(cur_iter+1))
    return (int(telapsed), int(lefttime), finishtime, iterTime)

def type2id(typeStr):
    if typeStr.lower() == "seizure":
        id = 1
    elif typeStr.lower() == "false alarm":
        id = 0
    elif typeStr.lower() == "nda":
        id = 0
    else:
        id = 2
    return id


def getUserCounts(df):
    counts = df['userId'].value_counts()
    total = counts.sum()
    props = counts/total
    return(props)

def analyseDf(df):
    props=getUserCounts(df)
    print("analyseDf(): Distribution by user for all data")
    print(props)

    props=getUserCounts(df[df['type']==1])
    print("analyseDf(): Distribution by user for seizure data")
    print(props)



def loadCsv(inFname, debug=False):
    '''
    loadCsv - read osdb csv file into a pandas dataframe.
    if inFname is None, reads from stdin.
    '''
    TAG = "augmentData.loadCsv()"
    if inFname is not None:
        print("%s: reading from file %s" % (TAG, inFname))
        inFile = inFname
    else:
        inFile = sys.stdin

    df = pd.read_csv(inFile)

    #print(df)
    if (debug): print("%s: returning %d datapoints" % (TAG, len(df)))

    return(df)


def getSeizureNonSeizureDfs(df):
    ''' returns two datasets, seizure, non-seizure from dataframe df
    '''
    seizuresDf = df[df['type']==1]
    nonSeizureDf = df[df['type']!=1]
    return (seizuresDf, nonSeizureDf)



def userAug(df):
    ''' implement user augmentation to oversample data to balance the contributions of different users
    It expects df to be a pandas dataframe representation of a flattened osdb dataset.
    '''
    seizuresDf, nonSeizureDf = getSeizureNonSeizureDfs(df)
    y = seizuresDf['userId']
    ros = imblearn.over_sampling.RandomOverSampler(random_state=0)
    xResamp, yResamp = ros.fit_resample(seizuresDf, y)
    print("userAug(): Distribution of seizure data after user augmentation")
    analyseDf(xResamp)
    # Combine seizure and non-seizure data back into single dataframe to return
    df = pd.concat([nonSeizureDf,xResamp])
    return(df)


def noiseAug(df, noiseAugVal, noiseAugFac, debug=False):
    ''' Implement noise augmentation of the seizue datapoints in dataframe df
     It expects df to be a pandas dataframe representation of a flattened osdb dataset.
    '''
    tStart = time.time()
    seizuresDf, nonSeizureDf = getSeizureNonSeizureDfs(df)
    augDf = seizuresDf
    if(debug): print(seizuresDf.columns)
    accStartCol = seizuresDf.columns.get_loc('M001')-1
    accEndCol = seizuresDf.columns.get_loc('M124')+1
    print("noiseAug(): Augmenting %d seizure datpoints.  accStartCol=%d, accEndCol=%d" % (len(seizuresDf), accStartCol, accEndCol))
    outLst = []
    for n in range(0,len(seizuresDf)):
        if (debug): print("n=%d" % n)
        rowArr = seizuresDf.iloc[n]
        if (debug): print("rowArrLen=%d" % len(rowArr), type(rowArr), rowArr)
        accArr = rowArr.iloc[accStartCol:accEndCol]
        if (debug): print("accArrLen=%d" % len(accArr), type(accArr), accArr)
        inArr =np.array(accArr)
        if(debug): print(inArr.shape)
        for j in range(0,noiseAugFac):
            noiseArr = np.random.normal(0,noiseAugVal,inArr.shape)
            outArr = inArr + noiseArr
            noiseArr = None
            outRow = []
            for i in range(0,accStartCol):
                outRow.append(rowArr.iloc[i])
            outRow.extend(outArr.tolist())
            outLst.append(outRow)
            outRow = None
            noiseArr = None
            # gc.collect()
        inArr = None
        rowArr = None
        accArr = None
        #gc.collect()
        # Update every 1000 datapoints and run garbage collector to try to save memory.
        if (n % 1000 == 0):
            tElapsed, tRem, tCompletion, tIter = calcProcessTime(tStart,n,len(seizuresDf))
            sys.stdout.write("n=%d, tIter=%.1f ms, elapsed: %s(s), time left: %s(s), estimated finish time: %s\r" % (n, tIter*1000., tElapsed,tRem,tCompletion))
            sys.stdout.flush()
            gc.collect()
    print("noiseAug() - Creating dataframe")
    sys.stdout.flush()
    gc.collect()
    augDf = pd.DataFrame(outLst, columns=nonSeizureDf.columns)
    outLst = None
    if (debug): print("noiseAug() - augDf=", augDf)
    if (debug): print("noiseAug() nonSeizureDf=", nonSeizureDf)

    print("noiseAug() - Concatenating dataframe")
    sys.stdout.flush()
    df = pd.concat([seizuresDf, augDf, nonSeizureDf])
    if (debug): print("df=",df)
    return(df)

def phaseAug(df, debug=False):
    ''' Implement phase augmentation of the seizue datapoints in dataframe df
     It expects df to be a pandas dataframe representation of a flattened osdb dataset.
    '''
    seizuresDf, nonSeizureDf = getSeizureNonSeizureDfs(df)

    accStartCol = seizuresDf.columns.get_loc('M001')-1
    accEndCol = seizuresDf.columns.get_loc('M124')+1
    eventIdCol = seizuresDf.columns.get_loc('id')
    #print("accStartCol=%d, accEndCol=%d" % (accStartCol, accEndCol))
    outLst = []
    lastAccArr = None
    lastEventId = seizuresDf.iloc[0].iloc[eventIdCol]
    for n in range(0,len(seizuresDf)):
        rowArr = seizuresDf.iloc[n]
        eventId = rowArr.iloc[eventIdCol]
        # the Dataframe is a list of datapoints, so we have to look for events changing
        if (eventId != lastEventId):
            lastEventId = eventId
            lastAccArr = None
        accArr = rowArr.iloc[accStartCol:accEndCol]
        if (lastAccArr is not None):
            # Make one long list from two consecutive rows.
            combArr = lastAccArr.tolist().copy()
            combArr.extend(accArr)
            for n in range(0,len(accArr)):
                outArr = combArr[n:n+len(accArr)]
                outRow = []
                for i in range(0,accStartCol):
                    outRow.append(rowArr.iloc[i])
                outRow.extend(outArr)
                outLst.append(outRow)
                outArr = None,
                outRow = None
        lastAccArr = accArr.copy()
        rowArr = None
        accArr = None
    augDf = pd.DataFrame(outLst, columns=nonSeizureDf.columns)
    if (debug): print("phaseAug() - augDf=", augDf)

    df = pd.concat([seizuresDf, augDf, nonSeizureDf])
    if (debug): print("df=",df)
    return(df)




def main():
    print("flattenOsdb.main()")
    parser = argparse.ArgumentParser(description='Perform data augmentation on a flattened (.csv) version of the OpenSeizureDatabase data')
    parser.add_argument('-i', default=None,
                        help='Input filename (uses stdin if not specified)')
    parser.add_argument('-o', default='dfAug.csv',
                        help='Output filename (default dfAug.csv)')
    parser.add_argument('--debug', action="store_true",
                        help='Write debugging information to screen')
    parser.add_argument('-u', action="store_true",
                        help='Apply User Augmentation')
    parser.add_argument('-n', action="store_true",
                        help='Apply Noise Augmentation')
    parser.add_argument('-p', action="store_true",
                        help='Apply Phase Augmentation')
    argsNamespace = parser.parse_args()
    args = vars(argsNamespace)
    print(args)



    df = loadCsv(args['i'], args['debug'])
    if (args['u']):
        df = userAug(df)
        print("userAug returned df")
        analyseDf(df)
    if (args['n']):
        df = noiseAug(df)
        print("noiseAug returned df")
        analyseDf(df)
    if (args['p']):
        df = phaseAug(df)
        print("phaseAug returned df")
        analyseDf(df)

if __name__ == "__main__":
    main()

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
     noiseAugVal is the amplitude (in mg) of the noise applied.
     noiseAugFac is the number of augmented datapoints generated for each input datapoint.
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


def augmentSeizureData(configObj, dataDir=".", debug=False):
    '''
    Given a pandas dataframe of osdb data,
    Apply data augmentation to the seizure data and return a new, extended data frame.
    It uses augmentation functions defined in the  augmentData module.

    The following configuration object values are used:
      - useNoiseAugmentation:  boolean - if True, noise Augmentation is applied to seizure rows.
      - noiseAugmentationFactor: int - number of copies of each seizure row to create with random noise applied
      - noiseAugmentationValue: int - amplitude of random noise to apply with noise augmentation.
      - usePhaseAugmentation:  boolean - if True, phase augmentation is applied to seizure rows.
      - useUserAugmentation:  boolean - if True, user augmentation is applied to seizure rows.
      - oversample: boolean - if not 'None', applies oversampling to balance the seizure and non-seizure rows.
                        Valid values are 'none', 'random' and 'smote'
      - undersample: boolean - if not 'None', applies undersampling to balance the seizure and non-seizure rows.
                        Valid values are 'none' and 'random'

    '''
    TAG = "augmentData.augmentSeizureData()"
    trainCsvFname = configObj['dataFileNames']['trainDataFileCsv']
    trainAugCsvFname = configObj['dataFileNames']['trainAugmentedFileCsv']
    useNoiseAugmentation = configObj['dataProcessing']['noiseAugmentation']
    noiseAugmentationFactor = configObj['dataProcessing']['noiseAugmentationFactor']
    noiseAugmentationValue = configObj['dataProcessing']['noiseAugmentationValue']
    usePhaseAugmentation = configObj['dataProcessing']['phaseAugmentation']
    useUserAugmentation = configObj['dataProcessing']['userAugmentation']
    oversample = configObj['dataProcessing']['oversample']
    undersample = configObj['dataProcessing']['undersample']   

    trainCsvFnamePath = os.path.join(dataDir, trainCsvFname)
    trainAugCsvFnamePath = os.path.join(dataDir, trainAugCsvFname)
    if (debug): print("%s: trainCsvFnamePath=%s" % (TAG, trainCsvFnamePath))
    if (debug): print("%s: trainAugCsvFnamePath=%s" % (TAG, trainAugCsvFnamePath))
    if (trainCsvFname is None):
        print("%s: No input file specified.  Exiting." % TAG)
        sys.exit(1)

    print("%s: Loading data from file %s." % (TAG, trainCsvFnamePath))
    df = loadCsv(trainCsvFnamePath,debug)
    print("augmentData:  Loaded training data - Columns are:", df.columns)
    #df.to_csv("before_aug.csv")
    print("Applying Augmentation....")

    if usePhaseAugmentation:
        print("Phase Augmentation...")
        if (debug): print("%s: %d datapoints. Applying Phase Augmentation to Seizure data" % (TAG, len(df)))
        augDf = phaseAug(df)
        df = augDf
        df.to_csv("after_phaseAug.csv")

    if useUserAugmentation:
        print("User Augmentation...")
        if (debug): print("%s: %d datapoints. Applying User Augmentation to Seizure data" % (TAG, len(df)))
        augDf = userAug(df)
        df = augDf
        df.to_csv("after_userAug.csv")

    if useNoiseAugmentation: 
        print("Noise Augmentation...")
        if (debug): print("%s: %d datapoints.  Applying Noise Augmentation - factor=%d, value=%.2f%%" % (TAG, len(df), noiseAugmentationFactor, noiseAugmentationValue))
        augDf = noiseAug(df, 
                                    noiseAugmentationValue, 
                                    noiseAugmentationFactor, 
                                    debug=False)
        df = augDf
        df.to_csv("after_noiseAug.csv")

    print("After applying augmentation, columns are:",df.columns)


    # Oversample Data to balance positive and negative data
    if (oversample is not None and oversample.lower()!="none"):
        print("Oversampling...")
        # Oversample data to balance the number of datapoints in each of
        #    the seizure and false alarm classes.
        if (oversample.lower() == "random"):
            print("%s: %d datapoints: Using Random Oversampling" % (TAG, len(df)))
            oversampler = imblearn.over_sampling.RandomOverSampler(random_state=0)
        elif (oversample.lower() == "smote"):
            print("%s: %d datapoints: Using SMOTE Oversampling" % (TAG, len(df)))
            oversampler = imblearn.over_sampling.SMOTE()
        else:
            print("%s: Not Using Oversampling" % TAG)
            oversampler = None

        if oversampler != None:
            # Oversample training data
            if (debug): print("%s: Oversampling %d datapoints" % (TAG, len(df)))
            resampDf, resampTarg = oversampler.fit_resample(df, df['type'])
            #print(".....After:", x_resampled.shape, y_resampled.shape)
            df = resampDf
            if (debug): print("%s: %d datapoints after oversampling" % (TAG, len(df)))
        else:
            print("%s: Not using Oversampling" % TAG)
        df.to_csv("after_oversample.csv")


    # Undersample data to balance positive and negative data
    if (undersample is not None and undersample.lower() != "none"):
        print("Under Sampling...")
        # Undersample data to balance the number of datapoints in each of
        #    the seizure and false alarm classes.
        if (undersample.lower() == "random"):
            print("Using Random Undersampling")
            undersampler = imblearn.under_sampling.RandomUnderSampler(random_state=0)
        else:
            print("%s: Not using undersampling" % TAG)
            undersampler = None

        if undersampler != None:
            # Undersample training data
            if (debug): print("%s: Resampling.  %d datapoints" % (TAG,len(df)))
            resampDf, resampTarg = undersampler.fit_resample(df, df['type'])
            #print(".....After:", x_resampled.shape, y_resampled.shape)
            df = resampDf
        else:
            print("%s: Not using Undersampling" % TAG)
        df.to_csv("after_underample.csv")

    print("after undersampling, columns are:",df.columns)
                
    print("Saving augmented data file to %s" % trainAugCsvFnamePath)
    df.to_csv(trainAugCsvFnamePath)
    print("%s: saved %d datapoints to file %s" % (TAG, len(df), trainAugCsvFnamePath))

    return


def balanceTestData(configObj, debug=False):
    '''
    Produce a balanced test data file using random over or undersampling as specified
    in the config file.

    The following configuration object values are used:
      - oversample: boolean - if not 'None', applies oversampling to balance the seizure and non-seizure rows.
                        Valid values are 'none', 'random' and 'smote'
      - undersample: boolean - if not 'None', applies undersampling to balance the seizure and non-seizure rows.
                        Valid values are 'none' and 'random'
    '''
    TAG = "augmentData.balanceTestData()"
    testCsvFname = configObj['testDataFileCsv']
    testBalCsvFname = configObj['testBalancedFileCsv']
    oversample = libosd.configUtils.getConfigParam("oversample", configObj)
    undersample = libosd.configUtils.getConfigParam("undersample", configObj)

    print("%s: Loading data from file %s." % (TAG, testCsvFname))
    df = loadCsv(testCsvFname,debug)
    # Oversample Data to balance positive and negative data
    if (oversample is not None and oversample.lower()!="none"):
        print("Oversampling...")
        # Oversample data to balance the number of datapoints in each of
        #    the seizure and false alarm classes.
        if (oversample.lower() == "random"):
            print("%s: %d datapoints: Using Random Oversampling" % (TAG, len(df)))
            oversampler = imblearn.over_sampling.RandomOverSampler(random_state=0)
        elif (oversample.lower() == "smote"):
            print("%s: %d datapoints: Using SMOTE Oversampling" % (TAG, len(df)))
            oversampler = imblearn.over_sampling.SMOTE()
        else:
            print("%s: Not Using Oversampling" % TAG)
            oversampler = None

        if oversampler != None:
            # Oversample training data
            if (debug): print("%s: Oversampling %d datapoints" % (TAG, len(df)))
            resampDf, resampTarg = oversampler.fit_resample(df, df['type'])
            #print(".....After:", x_resampled.shape, y_resampled.shape)
            df = resampDf
            if (debug): print("%s: %d datapoints after oversampling" % (TAG, len(df)))
        else:
            print("%s: Not using Oversampling" % TAG)

    # Undersample data to balance positive and negative data
    if (undersample is not None and undersample.lower() != "none"):
        print("Under Sampling...")
        # Undersample data to balance the number of datapoints in each of
        #    the seizure and false alarm classes.
        if (undersample.lower() == "random"):
            print("Using Random Undersampling")
            undersampler = imblearn.under_sampling.RandomUnderSampler(random_state=0)
        else:
            print("%s: Not using undersampling" % TAG)
            undersampler = None

        if undersampler != None:
            # Undersample training data
            if (debug): print("%s: Resampling.  %d datapoints" % (TAG,len(df)))
            resampDf, resampTarg = undersampler.fit_resample(df, df['type'])
            #print(".....After:", x_resampled.shape, y_resampled.shape)
            df = resampDf
        else:
            print("%s: Not using Undersampling" % TAG)
                
    print("Saving augmented data file")
    df.to_csv(testBalCsvFname)
    print("%s: saved %d datapoints to file %s" % (TAG, len(df), testBalCsvFname))

    return




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

#!/usr/bin/env python3

import argparse
from re import X
import sys
import os
import json
import importlib
from urllib.parse import _NetlocResultMixinStr
#from tkinter import Y
import pandas as pd
import sklearn.model_selection
import sklearn.metrics
import imblearn.over_sampling
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libosd.osdDbConnection
import libosd.dpTools
import libosd.osdAlgTools
import libosd.configUtils

#import cnnModel

from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, precision_score, roc_auc_score, roc_curve, accuracy_score
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
#import tensorflow as tf
#import pandas as pd


def getTestTrainData(configObj, debug=False):
    """
    for each event in the OsdDbConnection 'osd', create a set of rows 
    of training data for the model - one row per datapoint.
    Returns the data as (test, train) split by the trainProp proportions.
    if seizureTimeRange is not None, it should be an array [min, max]
    which is the time range in seconds from the event time to include datapoints.
    The idea of this is that a seizure event may include datapoints before or
    after the seizure, which we do not want to include in seizure training data.
    So specifying seizureTimeRange as say [-20, 40] will only include datapoints
    that occur less than 20 seconds before the seizure event time and up to 
    40 seconds after the seizure event time.
    If osd is None, it assumes that we have pre-prepared test/train data as defined
    in configObj and loads that instead of preparing new test/train data.
    """
    if (debug): print("getTestTrainData: configObj=",configObj)
    invalidEvents = libosd.configUtils.getConfigParam("invalidEvents", configObj)
    testProp = libosd.configUtils.getConfigParam("testProp", configObj)
    randomSeed = libosd.configUtils.getConfigParam("randomSeed", configObj)
    dbDir = libosd.configUtils.getConfigParam("cacheDir", configObj)

 
    print("Loading all seizures data")
    osd = libosd.osdDbConnection.OsdDbConnection(debug=debug)
    for fname in configObj['dataFiles']:
        print("Loading OSDB File: %s" % fname)
        eventsObjLen = osd.loadDbFile(fname)
    print("Removing invalid events...")
    osd.removeEvents(invalidEvents)
 
    print("Preparing new test/train dataset")
    eventIdsLst = osd.getEventIds()
    print("getTestTrainData(): Splitting data by Event")
    # Split into test and train data sets.
    if (debug): print("Total Events=%d" % len(eventIdsLst))

    # Split events list into test and train data sets.
    trainIdLst, testIdLst =\
        sklearn.model_selection.train_test_split(eventIdsLst,
                                                test_size=testProp,
                                                random_state=randomSeed)
    if (debug): print("len(train)=%d, len(test)=%d" % (len(trainIdLst), len(testIdLst)))
    #print("test=",testIdLst)

    fname = os.path.join(dbDir,configObj['trainDataFile'])
    osd.saveEventsToFile(trainIdLst, fname, True)
    print("Training Data written to file %s" % fname)
    fname = os.path.join(dbDir,configObj['testDataFile'])

    osd.saveEventsToFile(testIdLst, fname, True)
    print("Test Data written to file %s" % fname)

 

def main():
    print("createTestTrainDataFiles.main()")
    parser = argparse.ArgumentParser(description='Split OSDB Data into Test and Train Data sets')
    parser.add_argument('--config', default="nnConfig.json",
                        help='name of json file containing configuration')
    parser.add_argument('--debug', action="store_true",
                        help='Write debugging information to screen')
    argsNamespace = parser.parse_args()
    args = vars(argsNamespace)
    print(args)



    configObj = libosd.configUtils.loadConfig(args['config'])
    print("configObj=",configObj)
    # Load a separate OSDB Configuration file if it is included.
    if ("osdbCfg" in configObj):
        osdbCfgFname = libosd.configUtils.getConfigParam("osdbCfg",configObj)
        print("Loading separate OSDB Configuration File %s." % osdbCfgFname)
        osdbCfgObj = libosd.configUtils.loadConfig(osdbCfgFname)
        # Merge the contents of the OSDB Configuration file into configObj
        configObj = configObj | osdbCfgObj

    print("configObj=",configObj)

    
    getTestTrainData(configObj, args['debug'])
        
    


if __name__ == "__main__":
    main()

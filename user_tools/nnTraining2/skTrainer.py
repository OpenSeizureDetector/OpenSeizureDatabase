#!/usr/bin/env python3

import argparse
from re import X
import sys
import os
import importlib
#from tkinter import Y
import sklearn.metrics
import imblearn.over_sampling

import sklearn.ensemble 
import sklearn.metrics 
#from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libosd.configUtils

try:
    from user_tools.nnTraining2 import augmentData
except ImportError:
    import augmentData
import skTester  # Add this import

# fpr from https://scikit-learn.org/stable/auto_examples/model_selection/plot_cost_sensitive_learning.html#sphx-glr-auto-examples-model-selection-plot-cost-sensitive-learning-py
def fpr_score(y, y_pred, pos_label=1, neg_label=0):
    """Calculate the false positive rate (FPR) and true positive rate (TPR) for binary classification."""
    cm = sklearn.metrics.confusion_matrix(y, y_pred, labels=[neg_label, pos_label])
    tn, fp, fn, tp = cm.ravel()
    tnr = tn / (tn + fp)
    tpr = tp / (tp + fn)
    fpr = 1 - tnr
    return (tpr, fpr)



def trainModel(configObj, dataDir='.', debug=False):
    ''' Create and train a new scikit-learn model, saving it with filename starting 
    with the modelFnameRoot parameter.
    '''
    TAG = "skTrainer.trainmodel()"
    print("%s" % (TAG))
    # Prefer history files if they exist, otherwise fall back to features files
    trainFeaturesHistoryCsvFname = libosd.configUtils.getConfigParam('trainFeaturesHistoryFileCsv', configObj['dataFileNames'])
    testFeaturesHistoryCsvFname = libosd.configUtils.getConfigParam('testFeaturesHistoryFileCsv', configObj['dataFileNames'])
    trainFeaturesCsvFname = libosd.configUtils.getConfigParam('trainFeaturesFileCsv', configObj['dataFileNames'])
    testCsvFname = libosd.configUtils.getConfigParam("testFeaturesFileCsv", configObj['dataFileNames'])
    valCsvFname = libosd.configUtils.getConfigParam('valDataFileCsv', configObj['dataFileNames'])

    trainFeaturesHistoryCsvFnamePath = os.path.join(dataDir, trainFeaturesHistoryCsvFname)
    testFeaturesHistoryCsvFnamePath = os.path.join(dataDir, testFeaturesHistoryCsvFname)
    trainFeaturesCsvFnamePath = os.path.join(dataDir, trainFeaturesCsvFname)
    testCsvFnamePath = os.path.join(dataDir, testCsvFname)

    # Use history files if they exist
    if os.path.exists(trainFeaturesHistoryCsvFnamePath):
        trainFeaturesCsvFnamePath = trainFeaturesHistoryCsvFnamePath
        print(f"{TAG}: Using history file for training: {trainFeaturesCsvFnamePath}")
    else:
        print(f"{TAG}: Using features file for training: {trainFeaturesCsvFnamePath}")
    if os.path.exists(testFeaturesHistoryCsvFnamePath):
        testCsvFnamePath = testFeaturesHistoryCsvFnamePath
        print(f"{TAG}: Using history file for testing: {testCsvFnamePath}")
    else:
        print(f"{TAG}: Using features file for testing: {testCsvFnamePath}")

    modelFnameRoot = libosd.configUtils.getConfigParam("modelFname", configObj['modelConfig'])
    modelClassName = libosd.configUtils.getConfigParam("modelClass", configObj['modelConfig'])
    n_estimators = libosd.configUtils.getConfigParam("n_estimators", configObj['modelConfig'])
    max_depth = libosd.configUtils.getConfigParam("max_depth", configObj['modelConfig'])

    foldResults = []


    # Load Model class from nnModelClassName
    modelFname = "%s.sklearn" % modelFnameRoot
    modelFnamePath = os.path.join(dataDir, modelFname)
    moduleId = modelClassName.split('.')[0]
    modelClassId = modelClassName.split('.')[1]

    print("%s: Importing Module %s" % (TAG, moduleId))
    module = importlib.import_module(moduleId)
    model = eval("module.%s(configObj=configObj['modelConfig'], debug=debug)" % modelClassId)

    # Load the training data from file
    trainFeaturesCsvFnamePath = os.path.join(dataDir, trainFeaturesCsvFname)
    testCsvFnamePath = os.path.join(dataDir, testCsvFname)

    print("%s: Loading training data from file %s" % (TAG, trainFeaturesCsvFnamePath))
    if not os.path.exists(trainFeaturesCsvFnamePath):
        print("ERROR: File %s does not exist" % trainFeaturesCsvFnamePath)
        exit(-1)

    df = augmentData.loadCsv(trainFeaturesCsvFnamePath, debug=debug)
    print("%s: Loaded %d datapoints from file %s" % (TAG, len(df), trainFeaturesCsvFnamePath))
    if (debug): print(df.head())
    #augmentData.analyseDf(df)

    # Determine feature columns
    features = configObj['dataProcessing']['features']
    n_history = configObj.get('dataProcessing', {}).get('nHistory', 1)
    # If using a history file, build feature column names with suffixes
    if any(f'_t-' in col for col in df.columns):
        feature_cols = []
        for feat in features:
            for h in range(n_history):
                feature_cols.append(f'{feat}_t-{n_history-1-h}')
    else:
        feature_cols = features
    xTrain = df[feature_cols]
    yTrain = df['type']

    if (debug): print(xTrain)
    if (debug): print(yTrain)

    print("\n%s: Training using %d seizure datapoints and %d false alarm datapoints"
        % (TAG, np.count_nonzero(yTrain == 1),
        np.count_nonzero(yTrain == 0)))


    # FIXME: The idea is to use rfModel to hide this detail and make skTrainer generic.
    #print("%s: Training using n_estimators=%d, max_depth=%d" % (TAG, n_estimators, max_depth))
    #model = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight=classWeights, random_state=42)

    # Train the model
    model.fit(xTrain, yTrain)

    print("%s: Model trained - saving to file" % (TAG))
    model.save(dataDir=dataDir, modelFname=modelFname)



    ############################################################
    # Test the model on the test data set
    print("%s: Testing model on test data" % TAG)
    testDf = augmentData.loadCsv(testCsvFnamePath, debug=debug)
    print("%s: Loaded %d datapoints from file %s" % (TAG, len(testDf), testCsvFnamePath))
    # Use same feature_cols for test set
    xTest = testDf[feature_cols]
    yTest = testDf['type']

    # Make predictions on the test set
    yPred = model.predict(xTest)

    tpr, fpr = fpr_score(yTest, yPred)
    print(f"{TAG}: True Positive Rate (TPR): {tpr:.4f}, False Positive Rate (FPR): {fpr:.4f}")

    # Calculate the accuracy of the model
    accuracy = sklearn.metrics.accuracy_score(yTest, yPred)
    print(f'{TAG}: Model Accuracy: {accuracy:.2f}')

    #print(sklearn.metrics.classification_report(yTest, yPred))

    print(sklearn.metrics.confusion_matrix(yTest, yPred))

    # Calculate accuracy statistics for the real world OSD algorithm reported in the test data
    yPredOsd = testDf['osdAlarmState'].apply(lambda x: 1 if x >= 2 else 0)
    yTestOsd = testDf['type']  
    #print("yPredOsd=", yPredOsd)
    #print("yTestOsd=", yTestOsd)
    tprOsd, fprOsd = fpr_score(yTestOsd, yPredOsd)
    print(f"OSD Algorithm: True Positive Rate (TPR): {tprOsd:.4f}, False Positive Rate (FPR): {fprOsd:.4f}")

    print("\nOSD Algorithm Predictions:")
    print("OSD Alarm State Accuracy: %.2f" % sklearn.metrics.accuracy_score(yTestOsd, yPredOsd))
    #print(sklearn.metrics.classification_report(yTestOsd, yPredOsd))
    print(sklearn.metrics.confusion_matrix(yTestOsd, yPredOsd))


    cm = sklearn.metrics.confusion_matrix(yTest, yPred, labels=[0, 1])
    cmOsd = sklearn.metrics.confusion_matrix(yTestOsd, yPredOsd, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tnOsd, fpOsd, fnOsd, tpOsd = cmOsd.ravel()
    accuracyOsd = sklearn.metrics.accuracy_score(yTestOsd, yPredOsd)

    foldResults = {
        'accuracy': accuracy,
        'accuracyOsd': accuracyOsd,
        'tpr': tpr,
        'fpr': fpr,
        'tprOsd': tprOsd,
        'fprOsd': fprOsd,
        'tn': tn, 
        'fp': fp, 
        'fn': fn, 
        'tp': tp,
        'tnOsd': tnOsd, 
        'fpOsd': fpOsd, 
        'fnOsd': fnOsd, 
        'tpOsd': tpOsd
    }

    print("skTrainer: Training Complete")
    return foldResults

def main():
    print("skTrainer.main()")
    parser = argparse.ArgumentParser(description='Seizure Detection SciKit Learn Model Trainer')
    parser.add_argument('--config', default="nnConfig.json",
                        help='name of json file containing test configuration')
    parser.add_argument('--debug', action="store_true",
                        help='Write debugging information to screen')
    parser.add_argument('--test', action="store_true",
                        help='Test existing model, do not re-train.')
    argsNamespace = parser.parse_args()
    args = vars(argsNamespace)
    print(args)

    configObj = libosd.configUtils.loadConfig(args['config'])
    print("configObj=",configObj)
    if ("osdbCfg" in configObj):
        osdbCfgFname = libosd.configUtils.getConfigParam("osdbCfg",configObj)
        print("Loading separate OSDB Configuration File %s." % osdbCfgFname)
        osdbCfgObj = libosd.configUtils.loadConfig(osdbCfgFname)
        configObj = configObj | osdbCfgObj

    print("configObj=",configObj.keys())

    debug = configObj['debug']
    if args['debug']: debug=True

    if not args['test']:
        trainModel(configObj, debug)
        skTester.testModel(configObj, debug=debug)
    else:
        skTester.testModel(configObj, debug=debug)

if __name__ == "__main__":
    main()

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
    trainFeaturesCsvFname = libosd.configUtils.getConfigParam('trainFeaturesFileCsv', configObj['dataFileNames'])
    valCsvFname = libosd.configUtils.getConfigParam('valDataFileCsv', configObj['dataFileNames'])
    testCsvFname = libosd.configUtils.getConfigParam("testFeaturesFileCsv", configObj['dataFileNames'])

    modelFnameRoot = libosd.configUtils.getConfigParam("modelFname", configObj['modelConfig'])
    modelClassName = libosd.configUtils.getConfigParam("modelClass", configObj['modelConfig'])
    n_estimators = libosd.configUtils.getConfigParam("n_estimators", configObj['modelConfig'])
    max_depth = libosd.configUtils.getConfigParam("max_depth", configObj['modelConfig'])

    foldResults = []


    # Load Model class from nnModelClassName
    modelFname = "%s.sklearn" % modelFnameRoot
    modelFnamePath = os.path.join(dataDir, modelFname)
    #moduleId = modelClassName.split('.')[0]
    #modelClassId = modelClassName.split('.')[1]

    #print("%s: Importing Module %s" % (TAG, moduleId))
    #module = importlib.import_module(moduleId)
    #model = eval("module.%s(configObj['modelConfig'])" % modelClassId)

    # Load the training data from file
    trainFeaturesCsvFnamePath = os.path.join(dataDir, trainFeaturesCsvFname)
    testCsvFnamePath = os.path.join(dataDir, testCsvFname)

    print("%s: Loading training data from file %s" % (TAG, trainFeaturesCsvFnamePath))
    if not os.path.exists(trainFeaturesCsvFnamePath):
        print("ERROR: File %s does not exist" % trainFeaturesCsvFnamePath)
        exit(-1)

    df = augmentData.loadCsv(trainFeaturesCsvFnamePath, debug=debug)
    print("%s: Loaded %d datapoints from file %s" % (TAG, len(df), trainFeaturesCsvFname))
    if (debug): print(df.head())
    #augmentData.analyseDf(df)

    xTrain = df[configObj['dataProcessing']['features']]
    yTrain = df['type']

    if (debug): print(xTrain)
    if (debug): print(yTrain)

    classWeights = None
    if 'classWeights' in configObj['modelConfig']:
        classWeightsStr = configObj['modelConfig']['classWeights']
        classWeights = {int(k): v for k, v in classWeightsStr.items()}  
    else:
        print("%s: No class weights defined in configObj['modelConfig'] - using default weights" % TAG)
        classWeights = sklearn.utils.class_weight.compute_class_weight(
            'balanced', np.unique(yTrain), yTrain)

    print("%s: Using class weights: %s" % (TAG, classWeights))


    print("\n%s: Training using %d seizure datapoints and %d false alarm datapoints"
        % (TAG, np.count_nonzero(yTrain == 1),
        np.count_nonzero(yTrain == 0)))


    # FIXME: The idea is to use rfModel to hide this detail and make skTrainer generic.
    print("%s: Training using n_estimators=%d, max_depth=%d" % (TAG, n_estimators, max_depth))
    model = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight=classWeights, random_state=42)

    # Train the model
    model.fit(xTrain, yTrain)

    print("%s: Model trained - saving to file %s" % (TAG, modelFnamePath))
    # Save the model to a file
    import joblib
    joblib.dump(model, modelFnamePath)

    ###############################################
    # Training Complete - now evaluate the model

    # Calculate feature importances
    print("skTrainer: Calculating feature importances")
    feature_importances = model.feature_importances_
    feature_names = xTrain.columns
    feature_importance_dict = dict(zip(feature_names, feature_importances))
    sorted_feature_importances = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    #print("Feature Importances:")
    #for feature, importance in sorted_feature_importances:
    #    print(f"{feature}: {importance:.4f}")

    # Save feature importances to a file
    feature_importance_fpath = os.path.join(dataDir, "%s_feature_importances.txt" % modelFnameRoot)
    with open(feature_importance_fpath, 'w') as f:
        f.write("Feature Importances:\n")
        for feature, importance in sorted_feature_importances:
            f.write(f"{feature}: {importance:.4f}\n")
    print("%s: Feature importances saved to %s" % (TAG, feature_importance_fpath))

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_importances)), feature_importances, align='center')
    plt.yticks(range(len(feature_importances)), feature_names)
    plt.xlabel('Importance')
    plt.title('Feature Importances')
    plt.tight_layout()
    fpath = os.path.join(dataDir, "%s_feature_importances.png" % modelFnameRoot)
    print("%s: Saving feature importances plot to %s" % (TAG, fpath))
    plt.savefig(fpath)
    plt.close()


    ############################################################
    # Test the model on the test data set
    print("%s: Testing model on test data" % TAG)
    testDf = augmentData.loadCsv(testCsvFnamePath, debug=debug)
    print("%s: Loaded %d datapoints from file %s" % (TAG, len(testDf), testCsvFname))
    xTest = testDf[configObj['dataProcessing']['features']]
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

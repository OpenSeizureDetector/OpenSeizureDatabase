#!/usr/bin/env python3

import argparse
from re import X
import sys
import os
import json
import importlib
from urllib.parse import _NetlocResultMixinStr
import joblib
#from tkinter import Y
import sklearn.model_selection
import sklearn.metrics
import imblearn.over_sampling
import imblearn.under_sampling
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


def generateNoiseAugmentedData(dpInputData, noiseAugVal, noiseAugFac, debug=False):
    '''
    Generate noiseAugFac copies of dpInputData with normally distributed random noise
    with standard deviation noiseAugVal added to each copy.
    '''
    inArr =np.array(dpInputData)
    if(debug): print(inArr.shape)
    outLst = []
    for n in range(0,noiseAugFac):
        noiseArr = np.random.normal(0,noiseAugVal,inArr.shape)
        outArr = dpInputData + noiseArr
        outLst.append(outArr.tolist())
    return(outLst)

def generatePhaseAugmentedData(dpInputData, lastDpInputData, debug=False):
    '''
    generate multiple copies of dpInputData and lastDpInputData, offset by one
    sample time each - returns a list of dpInputData arrays.
    '''
    # On first run we will have None for the last dpInputData, so we do not have enough
    # data to offset.
    if (lastDpInputData is None):
        outLst = [dpInputData]
        if (debug): print("generatePhaseAugmentedData: lastDpInputData is none - returning dpInputData (len=%d)" % len(outLst[0]))
        if len(outLst[0])!= 125:
            print("Error - outLst!=125")
            exit(-1)
            raise
    else:
        if (len(dpInputData)!= 125):
            print("ERROR: dpInptuData!=125")
            exit(-1)
            raise
        if (len(lastDpInputData)!= 125):
            print("ERROR: lastInptuData!=125")
            exit(-1)
            raise
        # Combine the two 125 data point lists into one 250 data point list.
        combinedDataLst = lastDpInputData.copy()
        combinedDataLst.extend(dpInputData)
        #if (debug): print("generatePhaseAugmentedData: lastDpInputData", len(lastDpInputData))
        #if (debug): print("generatePhaseAugmentedData:     dpInputData", len(dpInputData))
        #if (debug): print("generatePhaseAugmentedData: combinedDataLst", len(combinedDataLst))
        outLst = []
        for n in range(0,len(dpInputData)):
            offsetLst = combinedDataLst[n:n+len(dpInputData)]
            outLst.append(offsetLst)
            #if (debug): print("generatePhaseAugmentationData - len(dpInputData)=%d, len(offsetLst)=%d" % (len(dpInputData), len(offsetLst)))
    return(outLst)

def getDataFromEventIds(eventIdsLst, nnModel, osd, configObj, debug=False):
    '''
    getDataFromEventIds() - takes a list of event IDs to be used, an instance of OsdDbConnection to access the
    OSDB data, and a configuration object, and returns a tuple (outArr, classArr) which is a list of datapoints
    and a list of classes (0=OK, 1=seizure) for each datapoint.
    FIXME - this is where we need to implement Phase Augmentation.
    '''
    seizureTimeRangeDefault = libosd.configUtils.getConfigParam("seizureTimeRange", configObj)
    useNoiseAugmentation = libosd.configUtils.getConfigParam("noiseAugmentation", configObj)
    noiseAugmentationFactor = libosd.configUtils.getConfigParam("noiseAugmentationFactor", configObj)
    noiseAugmentationValue = libosd.configUtils.getConfigParam("noiseAugmentationValue", configObj)
    usePhaseAugmentation = libosd.configUtils.getConfigParam("phaseAugmentation", configObj)
    if(debug): print(useNoiseAugmentation, noiseAugmentationFactor, noiseAugmentationValue)
    nEvents = len(eventIdsLst)
    outArr = []
    classArr = []
    for eventNo in range(0,nEvents):
        eventId = eventIdsLst[eventNo]
        eventObj = osd.getEvent(eventId, includeDatapoints=True)
        eventType = eventObj['type']
        if (debug): print("Processing event %s" % eventId)
        if (debug): print("Processing event ID %s, type=%s (id=%s)" % (eventId, eventType, type2id(eventType)))
        if (debug): sys.stdout.flush()
        if (not 'datapoints' in eventObj or eventObj['datapoints'] is None):
            print("No datapoints - skipping")
        else:
            #print("nDp=%d" % len(eventObj['datapoints']))
            lastDpInputData = None
            for dp in eventObj['datapoints']:
                dpInputData = nnModel.dp2vector(dp, normalise=False)
                if (dpInputData is None):
                    print("***Skipping Invalid Datapoint")
                    break
                eventTime = eventObj['dataTime']
                dpTime = dp['dataTime']
                eventTimeSec = libosd.dpTools.dateStr2secs(eventTime)
                dpTimeSec = libosd.dpTools.dateStr2secs(dpTime)
                timeDiffSec = dpTimeSec - eventTimeSec

                # The valid time range for datapoints is determined for seizure events either by a range
                # included in the seizure event object, or a default in the configuration file.
                # If it is not specified, or the event is not a seizure, all datapoints are included.
                includeDp = True
                if (eventObj['type'].lower() == 'seizure'):
                    # Check that this datapoint is within the specified time range.
                    eventSeizureTimeRange = libosd.osdDbConnection.extractJsonVal(eventObj,"seizureTimes")
                    if (eventSeizureTimeRange is not None):
                        seizureTimeRange = eventSeizureTimeRange
                    else:
                        seizureTimeRange = seizureTimeRangeDefault
                    if (seizureTimeRange is not None):
                        if (timeDiffSec < seizureTimeRange[0]):
                            includeDp=False
                        if (timeDiffSec > seizureTimeRange[1]):
                            includeDp=False
                    # Check we have real movement to analyse, otherwise reject the datapoint from seizure training data to avoid false alarms when no movement.
                    accArr = np.array(dpInputData)
                    accStd = 100. * np.std(accArr) / np.average(accArr)
                    if (eventObj['type'].lower() == 'seizure'):
                        if (accStd <configObj['accSdThreshold']):
                            print("Warning: Ignoring Low SD Seizure Datapoint: Event ID=%s: %s, %s - diff=%.1f, accStd=%.1f%%" % (eventId, eventTime, dpTime, timeDiffSec, accStd))
                            includeDp = False

                if (includeDp):
                    outArr.append(dpInputData)
                    classArr.append(type2id(eventType))
                    if useNoiseAugmentation and (eventObj['type'].lower() == 'seizure'):
                        if (debug): print("Applying Noise Augmentation - factor=%d, value=%.2f%%" % (noiseAugmentationFactor, noiseAugmentationValue))
                        noiseAugmentedDpData = generateNoiseAugmentedData(dpInputData,
                            noiseAugmentationValue, noiseAugmentationFactor, debug)
                        for augDp in noiseAugmentedDpData:
                            outArr.append(augDp)
                            classArr.append(type2id(eventType))
                    if usePhaseAugmentation and (eventObj['type'].lower() == 'seizure'):
                        if (debug): print("Applying Phase Augmentation to Seizure data")
                        phaseAugmentedDpData = generatePhaseAugmentedData(dpInputData, lastDpInputData, True)
                        for augDp in phaseAugmentedDpData:
                            outArr.append(augDp)
                            classArr.append(type2id(eventType))
                    lastDpInputData = dpInputData
                else:
                    #print("Out of Time Range - skipping")
                    pass

    return (outArr, classArr)

def getTestTrainData(nnModel, osd, configObj, debug=False):
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
    splitByEvent = libosd.configUtils.getConfigParam("splitTestTrainByEvent", configObj)
    testProp = libosd.configUtils.getConfigParam("testProp", configObj)
    oversample = libosd.configUtils.getConfigParam("oversample", configObj)
    undersample = libosd.configUtils.getConfigParam("undersample", configObj)
    phaseAugmentation = libosd.configUtils.getConfigParam("phaseAugmentation", configObj)
    randomSeed = libosd.configUtils.getConfigParam("randomSeed", configObj)
    if (debug): print("getTestTrainData: configObj=",configObj)
 
    outArr = []
    classArr = []

    if (osd is not None):
        print("getTestTrainData() - osd is not none, so preparing new test/train dataset")
        eventIdsLst = osd.getEventIds()
        if (splitByEvent):
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

            if (libosd.configUtils.getConfigParam('saveTestTrainData',configObj)):
                fname = configObj['trainDataFile']
                osd.saveEventsToFile(trainIdLst, fname, pretty=False, useCacheDir=True)
                print("Training Data written to file %s" % fname)
                fname = configObj['testDataFile']
                osd.saveEventsToFile(testIdLst, fname, pretty=False, useCacheDir=True)
                print("Test Data written to file %s" % fname)

            outTrain, classTrain = getDataFromEventIds(trainIdLst, nnModel, osd, configObj, debug)
            outTest, classTest = getDataFromEventIds(testIdLst, nnModel, osd, configObj, debug)
        else:  
            print("getTestTrainData(): Splitting data by Datapoint")
            # Split by datapoint rather than by event.
            outArr, classArr = getDataFromEventIds(eventIdsLst, nnModel, osd, configObj, debug)
            # Split into test and train data sets.
            outTrain, outTest, classTrain, classTest =\
                sklearn.model_selection.train_test_split(outArr, classArr,
                                                        test_size=testProp,
                                                        random_state=randomSeed,
                                                        stratify=classArr)
    else:
        print("getTestTrainData() - osd is None so attempting to read pre-prepared test/train data")
        osdTest = libosd.osdDbConnection.OsdDbConnection(debug=debug)
        fname = configObj['testDataFile']
        print("Loading OSDB File: %s" % fname)
        eventsObjLen = osdTest.loadDbFile(fname)
        testIdLst = osdTest.getEventIds()

        osdTrain = libosd.osdDbConnection.OsdDbConnection(debug=debug)
        fname = configObj['trainDataFile']
        print("Loading OSDB File: %s" % fname)
        eventsObjLen = osdTrain.loadDbFile(fname)
        trainIdLst = osdTrain.getEventIds()

        print("Generating training dataset")
        outTrain, classTrain = getDataFromEventIds(trainIdLst, nnModel, osdTrain, configObj, debug)
        print("Generating test dataset")
        outTest, classTest = getDataFromEventIds(testIdLst, nnModel, osdTest, configObj, debug)

    #print("Training Dataset")
    #for row in outTrain:
    #    print (len(row), row)
    #    if len(row)!=125:
    #        exit(-1)

    if (oversample is not None and oversample.lower()!="none"):
        # Oversample data to balance the number of datapoints in each of
        #    the seizure and false alarm classes.
        if (oversample.lower() == "random"):
            print("Using Random Oversampling")
            oversampler = imblearn.over_sampling.RandomOverSampler(random_state=0)
        elif (oversample.lower() == "smote"):
            print("Using SMOTE Oversampling")
            oversampler = imblearn.over_sampling.SMOTE()
        else:
            print("Not Using Oversampling")
            oversampler = None

        if oversampler != None:
            # Oversample training data
            if (debug): print("Resampling.  Shapes before:",len(outTrain), len(classTrain))
            x_resampled, y_resampled = oversampler.fit_resample(outTrain, classTrain)
            #print(".....After:", x_resampled.shape, y_resampled.shape)
            outTrain = x_resampled
            classTrain = y_resampled

            # Oversample test data
            x_resampled, y_resampled = oversampler.fit_resample(outTest, classTest)
            #print(".....After:", x_resampled.shape, y_resampled.shape)
            outTest = x_resampled
            classTest = y_resampled
        else:
            print("Not using Oversampling")


    if (undersample is not None and oversample != "None"):
        # Under data to balance the number of datapoints in each of
        #    the seizure and false alarm classes.
        if (undersample.lower() == "random"):
            print("Using Random Undersampling")
            undersampler = imblearn.under_sampling.RandomUnderSampler(random_state=0)
        else:
            print("Not using undersampling")
            undersampler = None

        if undersampler != None:
            # Undersample training data
            if (debug): print("Resampling.  Shapes before:",len(outTrain), len(classTrain))
            x_resampled, y_resampled = undersampler.fit_resample(outTrain, classTrain)
            #print(".....After:", x_resampled.shape, y_resampled.shape)
            outTrain = x_resampled
            classTrain = y_resampled

            # Undersample test data
            x_resampled, y_resampled = undersampler.fit_resample(outTest, classTest)
            #print(".....After:", x_resampled.shape, y_resampled.shape)
            outTest = x_resampled
            classTest = y_resampled
        else:
            print("Not using Undersampling")

    #print(outTrain, outTest, classTrain, classTest)
    # Convert into numpy arrays
    outTrainArr = np.array(outTrain)
    classTrainArr = np.array(classTrain)
    outTestArr = np.array(outTest)
    classTestArr = np.array(classTest)
    return(outTrainArr, outTestArr, classTrainArr, classTestArr)
 
def trainModel(configObj, modelFnameRoot="model", debug=False):
    ''' Create and train a new neural network model, saving it with filename starting 
    with the modelFnameRoot parameter.
    '''
    print("trainModel - configObj="+json.dumps(configObj))

    invalidEvents = libosd.configUtils.getConfigParam("invalidEvents", configObj)
    epochs = libosd.configUtils.getConfigParam("epochs", configObj)
    batch_size = libosd.configUtils.getConfigParam("batchSize", configObj)
    nLayers = libosd.configUtils.getConfigParam("nLayers", configObj)
    lrFactor = libosd.configUtils.getConfigParam("lrFactor", configObj)
    lrPatience = libosd.configUtils.getConfigParam("lrPatience", configObj)
    lrMin = libosd.configUtils.getConfigParam("lrMin", configObj)
    earlyStoppingPatience = libosd.configUtils.getConfigParam("earlyStoppingPatience", configObj)
    validationProp = libosd.configUtils.getConfigParam("validationProp", configObj)
    trainingVerbosity = libosd.configUtils.getConfigParam("trainingVerbosity", configObj)
    modelFname = "%s.h5" % modelFnameRoot
    nnModuleId = configObj['modelClass'].split('.')[0]
    nnClassId = configObj['modelClass'].split('.')[1]

    print("Importing nn Module %s" % nnModuleId)
    nnModule = importlib.import_module(nnModuleId)
    nnModel = eval("nnModule.%s()" % nnClassId)



    # Load each of the three events files (tonic clonic seizures,
    # all seizures and false alarms).

    if (configObj['saveTestTrainData']):
        print("saveTestTrainData is True - Reading whole database to prepare test/train data")
        print("Loading all seizures data")
        osdAllData = libosd.osdDbConnection.OsdDbConnection(debug=debug)
        for fname in configObj['dataFiles']:
            print("Loading OSDB File: %s" % fname)
            eventsObjLen = osdAllData.loadDbFile(fname)
        #eventsObjLen = osdAllData.loadDbFile(configObj['allSeizuresFname'])
        #eventsObjLen = osdAllData.loadDbFile(configObj['falseAlarmsFname'])
        #eventsObjLen = osdAllData.loadDbFile(configObj['ndaEventsFname'])
        print("Removing invalid events...")
        osdAllData.removeEvents(invalidEvents)
        #print("all Data eventsObjLen=%d" % eventsObjLen)
    else:
        print("saveTestTrainData is not set - using pre-prepared test/train dataset")
        osdAllData = None

    print("Splitting test/train data")
    # Convert the data into the format required by the neural network, and split it into a train and test dataset.
    xTrain, xTest, yTrain, yTest = getTestTrainData(nnModel, 
                                                    osdAllData, 
                                                    configObj, 
                                                    debug)

    xTrain = xTrain.reshape((xTrain.shape[0], xTrain.shape[1], 1))
    xTest = xTest.reshape((xTest.shape[0], xTest.shape[1], 1))

    
    
    nClasses = len(np.unique(yTrain))
    
    print("nClasses=%d" % nClasses)
    print("Training using %d seizure datapoints and %d false alarm datapoints"
            % (np.count_nonzero(yTrain == 1),
            np.count_nonzero(yTrain == 0)))
    print("Testing using %d seizure datapoints and %d false alarm datapoints"
            % (np.count_nonzero(yTest == 1),
            np.count_nonzero(yTest == 0)))

   
    if (os.path.exists(modelFname)):
        print("Model %s already exists - loading existing model as starting point for training" % modelFname)
        model = keras.models.load_model(modelFname)
    else:
        print("Creating new Model")
        model = nnModel.makeModel(input_shape=xTrain.shape[1:], num_classes=nClasses,
            nLayers=nLayers)
    
    keras.utils.plot_model(model, show_shapes=True)


    callbacks = [
        keras.callbacks.ModelCheckpoint(
            modelFname, save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=lrFactor, patience=lrPatience, min_lr=lrMin),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=earlyStoppingPatience, 
            verbose=trainingVerbosity),
    ]
    
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    
    history = model.fit(
        xTrain,
        yTrain,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_split=validationProp,
        verbose=trainingVerbosity
    )

    # After training, load the best model back from disk and test it.
    model = keras.models.load_model(modelFname)

    test_loss, test_acc = model.evaluate(xTest, yTest)

   
    print("Trained using %d seizure datapoints and %d false alarm datapoints"
        % (np.count_nonzero(yTrain == 1),
        np.count_nonzero(yTrain == 0)))
    print("Tesing using %d seizure datapoints and %d false alarm datapoints"
        % (np.count_nonzero(yTest == 1),
        np.count_nonzero(yTest == 0)))

    print("Test accuracy", test_acc)
    print("Test loss", test_loss)

    #Train and Validation: multi-class log-Loss & accuracy plot
    plt.figure(figsize=(12, 8))
    plt.plot(np.array(history.history['val_sparse_categorical_accuracy']), "r--", label = "val_sparse_categorical_accuracy")
    plt.plot(np.array(history.history['sparse_categorical_accuracy']), "g--", label = "sparse_categorical_accuracy")
    plt.plot(np.array(history.history['loss']), "y--", label = "Loss")
    plt.plot(np.array(history.history['val_loss']), "p-", label = "val_loss")
    plt.title("Training session's progress over iterations")
    plt.legend(loc='lower left')
    plt.ylabel('Training Progress (Loss/Accuracy)')
    plt.xlabel('Training Epoch')
    plt.ylim(0)
    plt.savefig("%s_training.png" % modelFnameRoot)
    plt.close()
    
    
    metric = "sparse_categorical_accuracy"
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.savefig("%s_training2.png" % modelFnameRoot)
    plt.close()
 
    calcConfusionMatrix(configObj, modelFnameRoot, xTest, yTest, debug)

    return(model)


def calcConfusionMatrix(configObj, modelFnameRoot="best_model", 
                        xTest=None, yTest=None, debug=False):
    oversample = libosd.configUtils.getConfigParam("oversample", configObj)

    if (xTest is None or yTest is None):
        # Load test and train data from the database if they are not passed to this function directly.
        invalidEvents = configObj['invalidEvents']    
        if ('seizureTimeRange' in configObj):
            seizureTimeRange = configObj['seizureTimeRange']
        else:
            seizureTimeRange = None

        print("Loading all seizures data")
        osdAllData = libosd.osdDbConnection.OsdDbConnection(debug=debug)
        #eventsObjLen = osdAllData.loadDbFile(configObj['allSeizuresFname'])
        #eventsObjLen = osdAllData.loadDbFile(configObj['falseAlarmsFname'])
        for fname in configObj['dataFiles']:
            print("Loading OSDB File: %s" % fname)
            eventsObjLen = osdAllData.loadDbFile(fname)
        osdAllData.removeEvents(invalidEvents)
        print("all Data eventsObjLen=%d" % eventsObjLen)

        nnModuleId = configObj['modelClass'].split('.')[0]
        nnClassId = configObj['modelClass'].split('.')[1]
        print("Importing nn Module %s" % nnModuleId)
        nnModule = importlib.import_module(nnModuleId)
        nnModel = eval("nnModule.%s()" % nnClassId)

        # Run each event through each algorithm
        xTrain, xTest, yTrain, yTest = getTestTrainData(nnModel, osdAllData,configObj, debug)

        xTest = xTest.reshape((xTest.shape[0], xTest.shape[1], 1))


    # Load the trained model back from disk and test it.
    modelFname = "%s.h5" % modelFnameRoot
    model = keras.models.load_model(modelFname)
    test_loss, test_acc = model.evaluate(xTest, yTest)

    nClasses = len(np.unique(yTest))
    print("nClasses=%d" % nClasses)
    print("Testing using %d seizure datapoints and %d false alarm datapoints"
          % (np.count_nonzero(yTest == 1),
             np.count_nonzero(yTest == 0)))

   
    print("Tesing using %d seizure datapoints and %d false alarm datapoints"
        % (np.count_nonzero(yTest == 1),
        np.count_nonzero(yTest == 0)))
 
    #define ytruw
    y_true=[]
    for element in yTest:
        y_true.append(np.argmax(element))
    prediction_proba=model.predict(xTest)
    prediction=np.argmax(prediction_proba,axis=1)
    
       
    # Confusion Matrix
    import seaborn as sns
    LABELS = ['No-Alarm','Seizure']
    cm = metrics.confusion_matrix(prediction, yTest)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, xticklabels=LABELS, yticklabels=LABELS, annot=True,
                linewidths = 0.1, fmt="d", cmap = 'YlGnBu');
    plt.title("%s: Confusion matrix" % modelFnameRoot, fontsize = 15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fname = "%s_confusion.png" % modelFnameRoot
    plt.savefig(fname)
    plt.close()
    print("Confusion Matrix Saved as %s." % fname)
    
    fname = "%s_stats.txt" % modelFnameRoot
    with  open(fname,"w") as outFile:
        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        total1=sum(sum(cm))
        outFile.write("\n|====================================================================|\n")
        outFile.write("****  Open Seizure Detector Classififcation Metrics Metrics  ****\n")
        outFile.write("****  Analysis of %d seizure and non seizure events Classififcation Metrics  ****\n" % total1)
        outFile.write("|====================================================================|\n")
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN)
        #print(TPR, TPR.shape, TPR[0])
        outFile.write("Sensitivity/recall or true positive rate: %.2f  %.2f\n" % tuple(TPR))
        # Specificity or true negative rate
        TNR = TN/(TN+FP)
        #print(TNR)
        outFile.write("Specificity or true negative rate: %.2f  %.2f\n" % tuple(TNR))
        # Precision or positive predictive value
        PPV = TP/(TP+FP)
        outFile.write("Precision or positive predictive value: %.2f  %.2f\n" % tuple(PPV))
        # Negative predictive value
        NPV = TN/(TN+FN)
        outFile.write("Negative predictive value: %.2f  %.2f\n" % tuple(NPV))
        # Fall out or false positive rate
        FPR = FP/(FP+TN)
        outFile.write("Fall out or false positive rate: %.2f  %.2f\n" % tuple(FPR))
        # False negative rate
        FNR = FN/(TP+FN)
        outFile.write("False negative rate: %.2f  %.2f\n" % tuple(FNR))
        # False discovery rate
        FDR = FP/(TP+FP)
        outFile.write("False discovery rate: %.2f  %.2f\n" % tuple(FDR))
        # Overall accuracy
        ACC = (TP+TN)/(TP+FP+FN+TN)
        outFile.write("Classification Accuracy: %.2f  %.2f\n" % tuple(ACC))
        outFile.write("|====================================================================|\n")
        report = classification_report(yTest, prediction)
        outFile.write(report)
        outFile.write("\n|====================================================================|\n")
        x=keras.metrics.sparse_categorical_accuracy(xTest, yTest)

        # summarize filter shapes
        for layer in model.layers:
        # check for convolutional layer
         if 'conv' not in layer.name:
             continue

        # get filter weights
        filters, biases = layer.get_weights()
        filterStr = layer.name
        for n in filters.shape:
            filterStr="%s, %d" % (filterStr,n)
        filterStr="%s\n" % filterStr
        outFile.write(filterStr)


        # summarize feature map shapes
        for i in range(len(model.layers)):
            layer = model.layers[i]
            # check for convolutional layer
            if 'conv' not in layer.name:
                continue
            # summarize output shape
            outFile.write("%d:  %s : " % (i, layer.name))
            for n in layer.output.shape:
                if n is not None:
                    outFile.write("%d, " % n)
            outFile.write("\n")

    print("Statistics Summary saved as %s." % fname)


def trainSvmModel(configObj, modelFnameRoot="model", debug=False):
    ''' Create and train a new SVM model, saving it with filename starting 
    with the modelFnameRoot parameter.
    '''
    print("trainSVMModel - configObj="+json.dumps(configObj))

    invalidEvents = libosd.configUtils.getConfigParam("invalidEvents", configObj)
    modelFname = "%s.joblib" % modelFnameRoot

    nnModuleId = configObj['modelClass'].split('.')[0]
    nnClassId = configObj['modelClass'].split('.')[1]

    print("Importing nn Module %s" % nnModuleId)
    nnModule = importlib.import_module(nnModuleId)
    nnModel = eval("nnModule.%s()" % nnClassId)



    # Load each of the three events files (tonic clonic seizures,
    # all seizures and false alarms).

    if (configObj['saveTestTrainData']):
        print("saveTestTrainData is True - Reading whole database to prepare test/train data")
        print("Loading all seizures data")
        osdAllData = libosd.osdDbConnection.OsdDbConnection(debug=debug)
        for fname in configObj['dataFiles']:
            print("Loading OSDB File: %s" % fname)
            eventsObjLen = osdAllData.loadDbFile(fname)
        #eventsObjLen = osdAllData.loadDbFile(configObj['allSeizuresFname'])
        #eventsObjLen = osdAllData.loadDbFile(configObj['falseAlarmsFname'])
        #eventsObjLen = osdAllData.loadDbFile(configObj['ndaEventsFname'])
        print("Removing invalid events...")
        osdAllData.removeEvents(invalidEvents)
        #print("all Data eventsObjLen=%d" % eventsObjLen)
    else:
        print("saveTestTrainData is not set - using pre-prepared test/train dataset")
        osdAllData = None

    print("Splitting test/train data")
    # Convert the data into the format required by the neural network, and split it into a train and test dataset.
    xTrain, xTest, yTrain, yTest = getTestTrainData(nnModel, 
                                                    osdAllData, 
                                                    configObj, 
                                                    debug)

    xTrain = xTrain.reshape((xTrain.shape[0], xTrain.shape[1], 1))
    xTest = xTest.reshape((xTest.shape[0], xTest.shape[1], 1))

    
    
    if (os.path.exists(modelFname)):
        print("Model %s already exists - loading existing model as starting point for training" % modelFname)
        model = joblib.load(modelFname)
    else:
        print("Creating new Model")
        model = sklearn.svm.SVC(kernel='rbf') # , C=C, gamma=gamma)
    
    # Train the model
    model.fit(
        xTrain,
        yTrain
   )
    
    # Write the model to disk.
    joblib.dump(model, modelFname)

    # After training, load the best model back from disk and test it.
    model = joblib.load(modelFname)

    test_loss, test_acc = model.evaluate(xTest, yTest)

   
    print("Trained using %d seizure datapoints and %d false alarm datapoints"
        % (np.count_nonzero(yTrain == 1),
        np.count_nonzero(yTrain == 0)))
    print("Tesing using %d seizure datapoints and %d false alarm datapoints"
        % (np.count_nonzero(yTest == 1),
        np.count_nonzero(yTest == 0)))

    print("Test accuracy", test_acc)
    print("Test loss", test_loss)

    #Train and Validation: multi-class log-Loss & accuracy plot
    plt.figure(figsize=(12, 8))
    plt.plot(np.array(history.history['val_sparse_categorical_accuracy']), "r--", label = "val_sparse_categorical_accuracy")
    plt.plot(np.array(history.history['sparse_categorical_accuracy']), "g--", label = "sparse_categorical_accuracy")
    plt.plot(np.array(history.history['loss']), "y--", label = "Loss")
    plt.plot(np.array(history.history['val_loss']), "p-", label = "val_loss")
    plt.title("Training session's progress over iterations")
    plt.legend(loc='lower left')
    plt.ylabel('Training Progress (Loss/Accuracy)')
    plt.xlabel('Training Epoch')
    plt.ylim(0)
    plt.savefig("%s_training.png" % modelFnameRoot)
    plt.close()
    
    
    metric = "sparse_categorical_accuracy"
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.savefig("%s_training2.png" % modelFnameRoot)
    plt.close()
 
    calcConfusionMatrix(configObj, modelFnameRoot, xTest, yTest, debug)

    return(model)



def main():
    print("gj_model.main()")
    parser = argparse.ArgumentParser(description='Seizure Detection Test Runner')
    parser.add_argument('--config', default="nnConfig.json",
                        help='name of json file containing test configuration')
    parser.add_argument('--model', default="cnn",
                        help='Root of filename of model to be generated or tested (without .h5 extension)')
    parser.add_argument('--debug', action="store_true",
                        help='Write debugging information to screen')
    parser.add_argument('--test', action="store_true",
                        help='Test existing model, do not re-train.')
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

    debug = configObj['debug']
    if args['debug']:
        debug = True;

    if not args['test']:
        trainModel(configObj, args['model'], debug)
    else:
        calcConfusionMatrix(configObj, modelFnameRoot = args['model'])
        
    


if __name__ == "__main__":
    main()

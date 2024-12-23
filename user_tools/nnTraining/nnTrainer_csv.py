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

import augmentData

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




def df2trainingData(df, nnModel, debug=False):
    ''' Converts a pandas dataframe df into a list of data and a list of associated seizure classes
    for use by model nnModel.
    This works by taking each row in the dataframe and converting it into a dict with the same
    values as an OpenSeizureDetector datapoint.   It then calls the dp2vector method of the specified
    model to pre-process the data into the format required by the model.

    FIXME:  It uses a simple for loop to loop through the dataframe - there is probably a quicker
    way of applying a function to each row in the dataframe in turn.
    '''

    accStartCol = df.columns.get_loc('M001')-1
    accEndCol = df.columns.get_loc('M124')+1
    hrCol = df.columns.get_loc('hr')
    typeCol = df.columns.get_loc('type')

    outLst = []
    classLst = []
    for n in range(0,len(df)):
        dpDict = {}
        if (debug): print("n=%d" % n)
        rowArr = df.iloc[n]
        if (debug): print("rowArrLen=%d" % len(rowArr), type(rowArr), rowArr)
        accArr = rowArr.iloc[accStartCol:accEndCol].values.astype(int).tolist()
        if (debug): print("accArr=", accArr, type(accArr))
        dpDict['rawData'] = accArr
        dpDict['hr'] = int(rowArr.iloc[hrCol])
        if (debug): print("dpDict=",dpDict)
        dpInputData = nnModel.dp2vector(dpDict, normalise=False)
        outLst.append(dpInputData)
        classLst.append(rowArr.iloc[typeCol])
        dpDict = None
        dpInputData = None

    return(outLst, classLst)


def augmentSeizureData(df, configObj, debug=False):
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
    TAG = "nnTrainer.augmentSeizureData()"
    useNoiseAugmentation = libosd.configUtils.getConfigParam("noiseAugmentation", configObj)
    noiseAugmentationFactor = libosd.configUtils.getConfigParam("noiseAugmentationFactor", configObj)
    noiseAugmentationValue = libosd.configUtils.getConfigParam("noiseAugmentationValue", configObj)
    usePhaseAugmentation = libosd.configUtils.getConfigParam("phaseAugmentation", configObj)
    useUserAugmentation = libosd.configUtils.getConfigParam("userAugmentation", configObj)
    oversample = libosd.configUtils.getConfigParam("oversample", configObj)
    undersample = libosd.configUtils.getConfigParam("undersample", configObj)

    df.to_csv("before_aug.csv")

    if usePhaseAugmentation:
        if (debug): print("%s: %d datapoints. Applying Phase Augmentation to Seizure data" % (TAG, len(df)))
        augDf = augmentData.phaseAug(df)
        df = augDf
        df.to_csv("after_phaseAug.csv")

    if useUserAugmentation:
        if (debug): print("%s: %d datapoints. Applying User Augmentation to Seizure data" % (TAG, len(df)))
        augDf = augmentData.userAug(df)
        df = augDf
        df.to_csv("after_userAug.csv")

    if useNoiseAugmentation: 
        if (debug): print("%s: %d datapoints.  Applying Noise Augmentation - factor=%d, value=%.2f%%" % (TAG, len(df), noiseAugmentationFactor, noiseAugmentationValue))
        augDf = augmentData.noiseAug(df, 
                                    noiseAugmentationValue, 
                                    noiseAugmentationFactor, 
                                    debug=False)
        df = augDf
        df.to_csv("after_noiseAug.csv")


    # Oversample Data to balance positive and negative data
    if (oversample is not None and oversample.lower()!="none"):
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
                
    if (debug): print("%s: returning %d datapoints" % (TAG, len(df)))

    return (df)

def trainModel(configObj, modelFnameRoot="model", debug=False):
    ''' Create and train a new neural network model, saving it with filename starting 
    with the modelFnameRoot parameter.
    '''
    TAG = "nnTrainer.trainmodel()"
    print("%s" % (TAG))
    epochs = libosd.configUtils.getConfigParam("epochs", configObj)
    batch_size = libosd.configUtils.getConfigParam("batchSize", configObj)
    nLayers = libosd.configUtils.getConfigParam("nLayers", configObj)
    lrFactor = libosd.configUtils.getConfigParam("lrFactor", configObj)
    lrPatience = libosd.configUtils.getConfigParam("lrPatience", configObj)
    lrMin = libosd.configUtils.getConfigParam("lrMin", configObj)
    earlyStoppingPatience = libosd.configUtils.getConfigParam("earlyStoppingPatience", configObj)
    validationProp = libosd.configUtils.getConfigParam("validationProp", configObj)
    trainingVerbosity = libosd.configUtils.getConfigParam("trainingVerbosity", configObj)
    nnModelClassName = libosd.configUtils.getConfigParam("modelClass", configObj)

    modelFname = "%s.h5" % modelFnameRoot
    nnModuleId = nnModelClassName.split('.')[0]
    nnClassId = nnModelClassName.split('.')[1]

    print("%s: Importing nn Module %s" % (TAG, nnModuleId))
    nnModule = importlib.import_module(nnModuleId)
    nnModel = eval("nnModule.%s()" % nnClassId)

    # Load the training data from file
    trainDataFname = libosd.configUtils.getConfigParam("trainDataCsvFile", configObj)
    print("%s: Loading Training Data from File %s" % (TAG, trainDataFname))
    df = augmentData.loadCsv(trainDataFname, debug=debug)
    print("%s: Loaded %d datapoints" % (TAG, len(df)))
    #augmentData.analyseDf(df)

    # Apply data augmentation
    print("%s: Augmenting Data" % (TAG))
    augDf = augmentSeizureData(df, configObj, debug=debug)
    df = augDf

    print("%s: After Augmentation data contains %d datapoints" % (TAG, len(df)))
    #augmentData.analyseDf(df)

    print("%s: Re-formatting data for training" % (TAG))
    xTrain, yTrain = df2trainingData(df, nnModel)

    print("%s: Converting to np arrays" % (TAG))
    xTrain = np.array(xTrain)
    yTrain = np.array(yTrain)

    print("%s: re-shaping array for training" % (TAG))
    xTrain = xTrain.reshape((xTrain.shape[0], xTrain.shape[1], 1))

    
    
    nClasses = len(np.unique(yTrain))
    
    print("nClasses=%d" % nClasses)
    print("Training using %d seizure datapoints and %d false alarm datapoints"
            % (np.count_nonzero(yTrain == 1),
            np.count_nonzero(yTrain == 0)))

   
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
    print("Trained using %d seizure datapoints and %d false alarm datapoints"
        % (np.count_nonzero(yTrain == 1),
        np.count_nonzero(yTrain == 0)))

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


def testModel(configObj, modelFnameRoot="model", debug=False):
    TAG = "nnTrainer.testModel()"
    print("%s" % (TAG))
    nnModelClassName = libosd.configUtils.getConfigParam("modelClass", configObj)
    testDataFname = libosd.configUtils.getConfigParam("testDataCsvFile", configObj)
    modelFname = "%s.h5" % modelFnameRoot
    nnModuleId = nnModelClassName.split('.')[0]
    nnClassId = nnModelClassName.split('.')[1]

    print("%s: Importing nn Module %s" % (TAG, nnModuleId))
    nnModule = importlib.import_module(nnModuleId)
    nnModel = eval("nnModule.%s()" % nnClassId)

    # Load the test data from file
    print("%s: Loading Test Data from File %s" % (TAG, testDataFname))
    df = augmentData.loadCsv(testDataFname, debug=debug)
    print("%s: Loaded %d datapoints" % (TAG, len(df)))
    #augmentData.analyseDf(df)

    print("%s: Re-formatting data for testing" % (TAG))
    xTest, yTest = df2trainingData(df, nnModel)

    print("%s: Converting to np arrays" % (TAG))
    xTest = np.array(xTest)
    yTest = np.array(yTest)

    print("%s: re-shaping array for testing" % (TAG))
    xTest = xTest.reshape((xTest.shape[0], xTest.shape[1], 1))

    # Load the best model back from disk and test it.
    model = keras.models.load_model(modelFname)

    test_loss, test_acc = model.evaluate(xTest, yTest)

   
    print("Tesing using %d seizure datapoints and %d false alarm datapoints"
        % (np.count_nonzero(yTest == 1),
        np.count_nonzero(yTest == 0)))

    print("Test accuracy", test_acc)
    print("Test loss", test_loss)

 
    calcConfusionMatrix(configObj, modelFnameRoot, xTest, yTest, debug)

    return(model)


def calcConfusionMatrix(configObj, modelFnameRoot="best_model", 
                        xTest=None, yTest=None, debug=False):

    TAG = "nnTrainer.calcConfusionMatrix()"
    print("%s" % (TAG))
    nnModelClassName = libosd.configUtils.getConfigParam("modelClass", configObj)
    testDataFname = libosd.configUtils.getConfigParam("testDataCsvFile", configObj)
    modelFname = "%s.h5" % modelFnameRoot
    nnModuleId = nnModelClassName.split('.')[0]
    nnClassId = nnModelClassName.split('.')[1]

    print("%s: Importing nn Module %s" % (TAG, nnModuleId))
    nnModule = importlib.import_module(nnModuleId)
    nnModel = eval("nnModule.%s()" % nnClassId)

    if (xTest is None or yTest is None):
        # Load the test data from file
        print("%s: Loading Test Data from File %s" % (TAG, testDataFname))
        df = augmentData.loadCsv(testDataFname, debug=debug)
        print("%s: Loaded %d datapoints" % (TAG, len(df)))
        #augmentData.analyseDf(df)

        print("%s: Re-formatting data for testing" % (TAG))
        xTest, yTest = df2trainingData(df, nnModel)

        print("%s: Converting to np arrays" % (TAG))
        xTest = np.array(xTest)
        yTest = np.array(yTest)

        print("%s: re-shaping array for testing" % (TAG))
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
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    total1=sum(sum(cm))
    with open(fname,"w") as outFile:
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





def main():
    print("nnTrainer_csv.main()")
    parser = argparse.ArgumentParser(description='Seizure Detection Neural Network Trainer')
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

    print("configObj=",configObj.keys())

    debug = configObj['debug']
    if args['debug']: debug=True

    if not args['test']:
        trainModel(configObj, args['model'], debug)
        testModel(configObj, args['model'], debug)
    else:
        testModel(configObj, args['model'], debug)
        #calcConfusionMatrix(configObj, modelFnameRoot = args['model'])
        
    


if __name__ == "__main__":
    main()

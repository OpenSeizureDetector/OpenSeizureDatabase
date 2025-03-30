#!/usr/bin/env python3

import argparse
from re import X
import sys
import os
import importlib
#from tkinter import Y
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
import nnTester




def df2trainingData(df, nnModel, debug=False):
    ''' Converts a pandas dataframe df into a list of data and a list of associated seizure classes
    for use by model nnModel.
    This works by taking each row in the dataframe and converting it into a dict with the same
    values as an OpenSeizureDetector datapoint.   It then calls the dp2vector method of the specified
    model to pre-process the data into the format required by the model.

    FIXME:  It uses a simple for loop to loop through the dataframe - there is probably a quicker
    way of applying a function to each row in the dataframe in turn.

    FIXME: This only works on acceleration magnitude values at the moment - add an option to use 3d data.
    '''

    accStartCol = df.columns.get_loc('M001')-1
    accEndCol = df.columns.get_loc('M124')+1
    hrCol = df.columns.get_loc('hr')
    typeCol = df.columns.get_loc('type')
    eventIdCol = df.columns.get_loc('id')

    outLst = []
    classLst = []
    lastEventId = None
    print("Processing Events:")
    for n in range(0,len(df)):
        dpDict = {}
        if (debug): print("n=%d" % n)
        rowArr = df.iloc[n]
        if (debug): print("rowArrLen=%d" % len(rowArr), type(rowArr), rowArr)

        eventId = rowArr.iloc[eventIdCol]
        if (eventId != lastEventId):
            sys.stdout.write("%d/%d (%.1f %%) : %s\r" % (n,len(df),100.*n/len(df), eventId))
            lastEventId = eventId

        accArr = rowArr.iloc[accStartCol:accEndCol].values.astype(int).tolist()
        if (debug): print("accArr=", accArr, type(accArr))
        dpDict['rawData'] = accArr
        dpDict['hr'] = int(rowArr.iloc[hrCol])
        if (debug): print("dpDict=",dpDict)
        dpInputData = nnModel.dp2vector(dpDict, normalise=True)
        if (dpInputData is not None):
            outLst.append(dpInputData)
            classLst.append(rowArr.iloc[typeCol])
        dpDict = None
        dpInputData = None
    print(".")
    return(outLst, classLst)


def trainModel(configObj, dataDir='.', debug=False):
    ''' Create and train a new neural network model, saving it with filename starting 
    with the modelFnameRoot parameter.
    '''
    TAG = "nnTrainer.trainmodel()"
    print("%s" % (TAG))
    trainAugCsvFname = libosd.configUtils.getConfigParam('trainAugmentedFileCsv', configObj)
    valCsvFname = libosd.configUtils.getConfigParam('valDataFileCsv', configObj)
    testCsvFname = libosd.configUtils.getConfigParam("testDataFileCsv", configObj)

    modelFnameRoot = libosd.configUtils.getConfigParam("modelFname", configObj)
    epochs = libosd.configUtils.getConfigParam("epochs", configObj)
    batch_size = libosd.configUtils.getConfigParam("batchSize", configObj)
    nLayers = libosd.configUtils.getConfigParam("nLayers", configObj)
    lrFactor = libosd.configUtils.getConfigParam("lrFactor", configObj)
    lrPatience = libosd.configUtils.getConfigParam("lrPatience", configObj)
    lrStart = libosd.configUtils.getConfigParam("lrStart", configObj)
    lrMin = libosd.configUtils.getConfigParam("lrMin", configObj)
    earlyStoppingPatience = libosd.configUtils.getConfigParam("earlyStoppingPatience", configObj)
    validationProp = libosd.configUtils.getConfigParam("validationProp", configObj)
    trainingVerbosity = libosd.configUtils.getConfigParam("trainingVerbosity", configObj)
    nnModelClassName = libosd.configUtils.getConfigParam("modelClass", configObj)

    if (validationProp == 0):
        print("WARNING: validationProp set to 0 - no validation data used - using test data instead")
        valCsvFname = testCsvFname

    inputDims = libosd.configUtils.getConfigParam("dims", configObj)
    if (inputDims is None): inputDims = 1

    # Load Model class from nnModelClassName
    modelFname = "%s.keras" % modelFnameRoot
    modelFnamePath = os.path.join(dataDir, modelFname)
    nnModuleId = nnModelClassName.split('.')[0]
    nnClassId = nnModelClassName.split('.')[1]

    print("%s: Importing nn Module %s" % (TAG, nnModuleId))
    nnModule = importlib.import_module(nnModuleId)
    nnModel = eval("nnModule.%s(configObj)" % nnClassId)

    # Load the training data from file
    trainAugCsvFnamePath = os.path.join(dataDir, trainAugCsvFname)
    valCsvFnamePath = os.path.join(dataDir, valCsvFname)

    print("%s: Loading training data from file %s" % (TAG, trainAugCsvFnamePath))
    if not os.path.exists(trainAugCsvFnamePath):
        print("ERROR: File %s does not exist" % trainAugCsvFnamePath)
        exit(-1)


    valCsvFnamePath = os.path.join(dataDir, valCsvFname)
    print("%s: Loading validation data from file %s" % (TAG, valCsvFnamePath))
    if not os.path.exists(valCsvFnamePath):
        print("ERROR: File %s does not exist" % valCsvFnamePath)
        exit(-1)

    df = augmentData.loadCsv(trainAugCsvFnamePath, debug=debug)
    print("%s: Loaded %d datapoints from file %s" % (TAG, len(df), trainAugCsvFname))
    #augmentData.analyseDf(df)


    print("%s: Re-formatting data for training" % (TAG))
    xTrain, yTrain = df2trainingData(df, nnModel)

    print("xTrain=", type(xTrain))
    print("yTrain=", type(yTrain))

    #for n in range(0,len(xTrain)):
    #    print(type(xTrain[n]))

    print("%s: Converting to np arrays" % (TAG))
    try:
        xTrain = np.array(xTrain)
    except ValueError as e:
        print("Failed simple array conversion - trying concatenate...")
        xTrain = np.concatenate(xTrain)
    yTrain = np.array(yTrain)


    print("xTrain.shape=",xTrain.shape,", yTrain.shape=",yTrain.shape)
    print("%s: re-shaping array for training" % (TAG))
    #xTrain = xTrain.reshape((xTrain.shape[0], xTrain.shape[1], xTrain.shape[2], 1))

    if (inputDims == 1):
        xTrain = xTrain.reshape((xTrain.shape[0], xTrain.shape[1], 1))
    elif (inputDims ==2):
        xTrain = xTrain.reshape((xTrain.shape[0], xTrain.shape[1], xTrain.shape[2], 1))
    else:
        print("ERROR - inputDims out of Range: %d" % inputDims)
        exit(-1)


    # Load and preprocess the validation data
    print("%s: Loading validation data from file %s" % (TAG, valCsvFnamePath))
    df = augmentData.loadCsv(valCsvFnamePath, debug=debug)
    print("%s: Loaded %d datapoints from file %s" % (TAG, len(df), valCsvFname))
    #augmentData.analyseDf(df)


    print("%s: Re-formatting data for training" % (TAG))
    xVal, yVal= df2trainingData(df, nnModel)

    print("xVal=", type(xVal))
    print("yVal=", type(yVal))

    print("%s: Converting to np arrays" % (TAG))
    try:
        xVal = np.array(xVal)
    except ValueError as e:
        print("Failed simple array conversion - trying concatenate...")
        xVal = np.concatenate(xVal)
    yVal= np.array(yVal)


    print("xVal.shape=",xVal.shape,", yVal.shape=",yVal.shape)
    print("%s: re-shaping array for training" % (TAG))

    if (inputDims == 1):
        xVal = xVal.reshape((xVal.shape[0], xVal.shape[1], 1))
    elif (inputDims ==2):
        xVal = xVal.reshape((xVal.shape[0], xVal.shape[1], xVal.shape[2], 1))
    else:
        print("ERROR - inputDims out of Range: %d" % inputDims)
        exit(-1)

    
    nClasses = len(np.unique(yTrain))
    print("nClasses=%d" % nClasses)
    print("Training using %d seizure datapoints and %d false alarm datapoints"
            % (np.count_nonzero(yTrain == 1),
            np.count_nonzero(yTrain == 0)))

   
    if (os.path.exists(modelFnamePath)):
        print("Model %s already exists - loading existing model as starting point for training" % modelFnamePath)
        model = keras.models.load_model(modelFnamePath)
        print("Model %s loaded" % modelFnamePath)
    else:
        print("Creating new Model")
        model = nnModel.makeModel(input_shape=xTrain.shape[1:], num_classes=nClasses,
            nLayers=nLayers)
    
    keras.utils.plot_model(model, show_shapes=True)


    callbacks = [
        keras.callbacks.ModelCheckpoint(
            modelFnamePath, save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=lrFactor, patience=lrPatience, min_lr=lrMin),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=earlyStoppingPatience, 
            verbose=trainingVerbosity),
    ]
    
    opt = keras.optimizers.Adam(learning_rate=lrStart)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    
    history = model.fit(
        xTrain,
        yTrain,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        #validation_split=validationProp,
        validation_data=(xVal, yVal),
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




def main():
    print("nnTrainer_csv.main()")
    parser = argparse.ArgumentParser(description='Seizure Detection Neural Network Trainer')
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
        trainModel(configObj, debug)
        nnTester.testModel(configObj, debug)
    else:
        nnTester.testModel(configObj, debug)
        
    


if __name__ == "__main__":
    main()

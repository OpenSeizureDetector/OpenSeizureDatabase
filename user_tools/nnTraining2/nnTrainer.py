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


def trainModel(configObj, debug=False):
    ''' Create and train a new neural network model, saving it with filename starting 
    with the modelFnameRoot parameter.
    '''
    TAG = "nnTrainer.trainmodel()"
    print("%s" % (TAG))
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

    modelFname = "%s.keras" % modelFnameRoot
    nnModuleId = nnModelClassName.split('.')[0]
    nnClassId = nnModelClassName.split('.')[1]

    print("%s: Importing nn Module %s" % (TAG, nnModuleId))
    nnModule = importlib.import_module(nnModuleId)
    nnModel = eval("nnModule.%s()" % nnClassId)

    # Load the training data from file
    trainAugCsvFname = configObj['trainAugmentedFileCsv']
    df = augmentData.loadCsv(trainAugCsvFname, debug=debug)
    print("%s: Loaded %d datapoints" % (TAG, len(df)))
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

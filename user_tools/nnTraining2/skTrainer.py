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

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libosd.configUtils

import augmentData





def trainModel(configObj, dataDir='.', debug=False):
    ''' Create and train a new scikit-learn model, saving it with filename starting 
    with the modelFnameRoot parameter.
    '''
    TAG = "skTrainer.trainmodel()"
    print("%s" % (TAG))
    trainAugCsvFname = libosd.configUtils.getConfigParam('trainAugmentedFileCsv', configObj['dataFileNames'])
    valCsvFname = libosd.configUtils.getConfigParam('valDataFileCsv', configObj['dataFileNames'])
    testCsvFname = libosd.configUtils.getConfigParam("testDataFileCsv", configObj['dataFileNames'])

    modelFnameRoot = libosd.configUtils.getConfigParam("modelFname", configObj['modelConfig'])
    modelClassName = libosd.configUtils.getConfigParam("modelClass", configObj['modelConfig'])


    # Load Model class from nnModelClassName
    modelFname = "%s.sklearn" % modelFnameRoot
    modelFnamePath = os.path.join(dataDir, modelFname)
    moduleId = modelClassName.split('.')[0]
    modelClassId = modelClassName.split('.')[1]

    print("%s: Importing Module %s" % (TAG, moduleId))
    module = importlib.import_module(moduleId)
    model = eval("module.%s(configObj['modelConfig'])" % modelClassId)

    # Load the training data from file
    trainAugCsvFnamePath = os.path.join(dataDir, trainAugCsvFname)
    testCsvFnamePath = os.path.join(dataDir, testCsvFname)

    print("%s: Loading training data from file %s" % (TAG, trainAugCsvFnamePath))
    if not os.path.exists(trainAugCsvFnamePath):
        print("ERROR: File %s does not exist" % trainAugCsvFnamePath)
        exit(-1)
        
    df = augmentData.loadCsv(trainAugCsvFnamePath, debug=debug)
    print("%s: Loaded %d datapoints from file %s" % (TAG, len(df), trainAugCsvFname))
    print(df.head())
    #augmentData.analyseDf(df)

    xTrain = df[configObj['dataProcessing']['features']]
    yTrain = df['type']

    print(xTrain)
    print(yTrain)

    model = sklearn.ensemble.RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    model.fit(xTrain, yTrain)

    print("Model trained - saving to file %s" % modelFnamePath)
    # Save the model to a file
    import joblib
    joblib.dump(model, modelFnamePath)


    print("Model saved to %s" % modelFnamePath)

    print("Testing model on test data")
    testDf = augmentData.loadCsv(testCsvFnamePath, debug=debug)
    print("%s: Loaded %d datapoints from file %s" % (TAG, len(testDf), testCsvFname))
    xTest = testDf[configObj['dataProcessing']['features']]
    yTest = testDf['type']

    # Make predictions on the test set
    yPred = model.predict(xTest)

    # Calculate the accuracy of the model
    accuracy = sklearn.metrics.accuracy_score(yTest, yPred)
    print(f'Model Accuracy: {accuracy:.2f}')

    print(sklearn.metrics.classification_report(yTest, yPred))

    print(sklearn.metrics.confusion_matrix(yTest, yPred))

    exit(-1)


    print("Trained using %d seizure datapoints and %d false alarm datapoints"
        % (np.count_nonzero(yTrain == 1),
        np.count_nonzero(yTrain == 0)))

    #Train and Validation: multi-class log-Loss & accuracy plot
    print("Plotting training history")
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
    plt.savefig(os.path.join(dataDir,"%s_training.png" % modelFnameRoot))
    plt.close()
    
    
    metric = "sparse_categorical_accuracy"
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.savefig(os.path.join(dataDir,"%s_training2.png" % modelFnameRoot))
    plt.close()

    print("Training Complete")




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

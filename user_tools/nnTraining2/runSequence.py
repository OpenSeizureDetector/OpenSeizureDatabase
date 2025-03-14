#!/usr/bin/env python3
#
#
# Run the neural network training toolchain sequence.   It does the following:
#   - Select data, filtering the OSDB data to select the desired data
#   - Split into a test and train dataset
#   - Flatten the OSDB .json data files into .csv files.
#   - Apply data augmentation to the seizure data and save the augmented .csv files
#   - Train a new or pre-existing model based on the augmented files.
#   - Tests the resulting model calculating statistics using the test dataset.
#
# Note that only the required parts of the sequence are run, depending on which data
#  files are in the working directory - if the file does not exist, it is re-generated
#  based on the files produced earlier in the sequence.
# This means that if you want to re-generate the files completely, you need to start in
#    a clean directory, or remove the output data files.


import argparse
import sys
import os
import sklearn.model_selection
import sklearn.metrics

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libosd.osdDbConnection
import libosd.dpTools
import libosd.osdAlgTools
import libosd.configUtils

import selectData
import splitData
import flattenData
import augmentData

def deleteFileIfExists(fname, debug=True):
    ''' If the specified file named fname exists, delete it, otherwise do nothing.'''
    if (os.path.exists(fname)):
            if (debug): print("Deleting %s" % fname)
            os.remove(fname)


def getOutputPath(outPath = "./output", prefix="training"):
    '''
    Returns a path to the next sequentially numbered folder in outPath
    '''
    import pathlib, os

    outFolderPath = os.path.join(outPath, prefix)
    # Save files associated with this run into output folder.      
    os.makedirs(outFolderPath, exist_ok=True)

    allOutputFolders = list(pathlib.Path(outFolderPath).glob('*/'))

    print("Existing Output Folders:")
    for folder in allOutputFolders:
        print(folder)
    #print(folder for folder in allOutputFolders)
    print("Length of allOutputFolders = %d" % len(allOutputFolders))

    print()
    if (len(allOutputFolders) > 0):
        print("getting latestoutputfolder")
        latestOutputFolderPath = max(allOutputFolders, key=os.path.getmtime)
        latestOutputFolder = os.path.split(latestOutputFolderPath)[-1]
    else:
        print("starting new output folder list")
        latestOutputFolder = "0"

    print("latestOutputFolder = %s" % latestOutputFolder)

    newOutputFolder = 1 + int(latestOutputFolder)

    newOutputPath = os.path.join(outFolderPath,str(newOutputFolder))
    os.makedirs(newOutputPath, exist_ok = False)

    print(latestOutputFolder, newOutputFolder, newOutputPath)
    return newOutputPath

def main():
    print("runSequence.main()")
    parser = argparse.ArgumentParser(description='Run the Neural Network Training toolchain sequence')
    parser.add_argument('--config', default="nnConfig.json",
                        help='name of json file containing configuration')
    parser.add_argument('--outDir', default="./output",
                        help='folder for training output (stored in sequential numbered folders within this folder)')
    parser.add_argument('--train', action="store_true",
                        help='Train the model')
    parser.add_argument('--test', action="store_true",
                        help='Test the model')
    parser.add_argument('--clean', action="store_true",
                        help='Clean up output files before running')
    parser.add_argument('--debug', action="store_true",
                        help='Write debugging information to screen')
    argsNamespace = parser.parse_args()
    args = vars(argsNamespace)

    debug = args['debug']
    if (debug): print(args)

    configObj = libosd.configUtils.loadConfig(args['config'])
    if (debug): print("configObj=",configObj.keys())
    # Load a separate OSDB Configuration file if it is included.
    if ("osdbCfg" in configObj):
        osdbCfgFname = libosd.configUtils.getConfigParam("osdbCfg",configObj)
        print("Loading separate OSDB Configuration File %s." % osdbCfgFname)
        osdbCfgObj = libosd.configUtils.loadConfig(osdbCfgFname)
        # Merge the contents of the OSDB Configuration file into configObj
        configObj = configObj | osdbCfgObj

    if (debug): print("configObj=",configObj.keys())

    allDataFname = configObj['allDataFileJson']
    testDataFname = configObj['testDataFileJson']
    trainDataFname = configObj['trainDataFileJson']
    valDataFname = configObj['valDataFileJson']
    testCsvFname = configObj['testDataFileCsv']
    testBalCsvFname = configObj['testBalancedFileCsv']
    trainCsvFname = configObj['trainDataFileCsv']
    valCsvFname = configObj['valDataFileCsv']
    trainAugCsvFname = configObj['trainAugmentedFileCsv']


    if args['clean']:
        # Clean up all output files
        print("Cleaning up output files")
        deleteFileIfExists(allDataFname)
        deleteFileIfExists(testDataFname)
        deleteFileIfExists(trainDataFname)
        deleteFileIfExists(valDataFname)
        deleteFileIfExists(testCsvFname)
        deleteFileIfExists(trainCsvFname)
        deleteFileIfExists(valCsvFname)
        deleteFileIfExists(trainAugCsvFname)
        deleteFileIfExists(testBalCsvFname)

        deleteFileIfExists("%s.keras" % configObj['modelFname'])
        deleteFileIfExists("%s_confusion.png" % configObj['modelFname'])
        deleteFileIfExists("%s_probabilities.png" % configObj['modelFname'])
        deleteFileIfExists("%s_training.png" % configObj['modelFname'])
        deleteFileIfExists("%s_training2.png" % configObj['modelFname'])
        deleteFileIfExists("%s_stats.txt" % configObj['modelFname'])

        exit(0)



    if args['train']:
        import nnTrainer
        import nnTester
        import numpy as np
        import tensorflow as tf
        import random

        outFolder = getOutputPath(args['outDir'], configObj['modelFname'])
        print("Writing Output to folder %s" % outFolder)

        # Initialise random number generators
        seed = configObj['randomSeed'];
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed) 

        if (not os.path.exists(allDataFname)):
            print("All data file missing - re-generating")
            print("Removing raw, flattened and augmented files where they exist, so they are re-generated")
            deleteFileIfExists(testDataFname)
            deleteFileIfExists(trainDataFname)
            deleteFileIfExists(valDataFname)
            deleteFileIfExists(testCsvFname)
            deleteFileIfExists(trainCsvFname)
            deleteFileIfExists(valCsvFname)
            deleteFileIfExists(trainAugCsvFname)
            deleteFileIfExists(testBalCsvFname)
            selectData.selectData(configObj, debug) 

        if (not os.path.exists(testDataFname)) or (not os.path.exists(trainDataFname)):
            print("Test/Train data files missing - re-generating")
            print("Removing raw, flattened and augmented files where they exist, so they are re-generated")
            deleteFileIfExists(testDataFname)
            deleteFileIfExists(trainDataFname)
            deleteFileIfExists(valDataFname)
            deleteFileIfExists(testCsvFname)
            deleteFileIfExists(trainCsvFname)
            deleteFileIfExists(valCsvFname)
            deleteFileIfExists(trainAugCsvFname)
            deleteFileIfExists(testBalCsvFname)

            splitData.saveTestTrainData(configObj, debug)

        if (not os.path.exists(testCsvFname)) or (not os.path.exists(trainCsvFname)):
            print("Flattened test/train data files missing - re-generating")
            print("Removing augmented files where they exist so they are re-generated")
            deleteFileIfExists(testCsvFname)
            deleteFileIfExists(trainCsvFname)
            deleteFileIfExists(valCsvFname)
            deleteFileIfExists(trainAugCsvFname)
            deleteFileIfExists(testBalCsvFname)

            flattenData.flattenOsdb(testDataFname, testCsvFname, configObj)
            flattenData.flattenOsdb(trainDataFname, trainCsvFname, configObj)
            flattenData.flattenOsdb(valDataFname, valCsvFname, configObj)

        if (not os.path.exists(trainAugCsvFname)):
            print("Augmented data file missing - re-generating")
            augmentData.augmentSeizureData(configObj, debug)
            augmentData.balanceTestData(configObj, debug)

        print("Training Model")
        nnTrainer.trainModel(configObj, debug)
        nnTester.testModel2(configObj, balanced=False, debug=debug)  

    
    if args['test']:
        import nnTester
        outFolder = getOutputPath(args['outDir'], configObj['modelFname'])
        print("Writing Output to folder %s" % outFolder)
        print("Testing Model")
        nnTester.testModel2(configObj, balanced=False, debug=debug)  

    # Archive Results
    import shutil
    if (os.path.exists(testDataFname)):
        shutil.copy(testDataFname, outFolder)
    if (os.path.exists(trainDataFname)):
        shutil.copy(trainDataFname, outFolder)
    if (os.path.exists(valDataFname)):
        shutil.copy(valDataFname, outFolder)
    if (os.path.exists(testCsvFname)):
        shutil.copy(testCsvFname, outFolder)
    if (os.path.exists(trainCsvFname)):
        shutil.copy(trainCsvFname, outFolder)
    if (os.path.exists(valCsvFname)):
        shutil.copy(valCsvFname, outFolder)
    if (os.path.exists(trainAugCsvFname)):
        shutil.copy(trainAugCsvFname, outFolder)
    if (os.path.exists(testBalCsvFname)):
        shutil.copy(testBalCsvFname, outFolder)
    if (os.path.exists("%s.keras" % configObj['modelFname'])):
        shutil.copy("%s.keras" % configObj['modelFname'], outFolder)
    if (os.path.exists("%s_confusion.png" % configObj['modelFname'])):
        shutil.copy("%s_confusion.png" % configObj['modelFname'], outFolder)
    if (os.path.exists("%s_probabilities.png" % configObj['modelFname'])):
        shutil.copy("%s_probabilities.png" % configObj['modelFname'], outFolder)
    if (os.path.exists("%s_training.png" % configObj['modelFname'])):
        shutil.copy("%s_training.png" % configObj['modelFname'], outFolder)
    if (os.path.exists("%s_training2.png" % configObj['modelFname'])):
        shutil.copy("%s_training2.png" % configObj['modelFname'], outFolder)
    if (os.path.exists("%s_stats.txt" % configObj['modelFname'])):
        shutil.copy("%s_stats.txt" % configObj['modelFname'], outFolder)
    if (os.path.exists(args['config'])):
        shutil.copy(args['config'], outFolder)


    print("Finished - output in folder %s" % outFolder)


if __name__ == "__main__":
    #outFolder = getOutputPath("./output")
    #print(outFolder)
    #exit(-1)
    main()

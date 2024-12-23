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
import flattenData
import augmentData
import nnTrainer
import nnTester

def deleteFileIfExists(fname, debug=True):
    ''' If the specified file named fname exists, delete it, otherwise do nothing.'''
    if (os.path.exists(fname)):
            if (debug): print("Deleting %s" % fname)
            os.remove(fname)


def main():
    print("runSequence.main()")
    parser = argparse.ArgumentParser(description='Run the Neural Network Training toolchain sequence')
    parser.add_argument('--config', default="nnConfig.json",
                        help='name of json file containing configuration')
    parser.add_argument('--train', action="store_true",
                        help='Train the model (otherwise it only tests it)')
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

    testDataFname = configObj['testDataFileJson']
    trainDataFname = configObj['trainDataFileJson']
    testCsvFname = configObj['testDataFileCsv']
    trainCsvFname = configObj['trainDataFileCsv']
    trainAugCsvFname = configObj['trainAugmentedFileCsv']

    if args['train']:
        if (not os.path.exists(testDataFname)) or (not os.path.exists(trainDataFname)):
            print("Test/Train data files missing - re-generating")
            print("Removing raw, flattened and augmented files where they exist, so they are re-generated")
            deleteFileIfExists(testDataFname)
            deleteFileIfExists(trainDataFname)
            deleteFileIfExists(testCsvFname)
            deleteFileIfExists(trainCsvFname)
            deleteFileIfExists(trainAugCsvFname)

            selectData.saveTestTrainData(configObj, debug)

        if (not os.path.exists(testCsvFname)) or (not os.path.exists(trainCsvFname)):
            print("Flattened test/train data files missing - re-generating")
            print("Removing augmented files where they exist so they are re-generated")
            deleteFileIfExists(testCsvFname)
            deleteFileIfExists(trainCsvFname)
            deleteFileIfExists(trainAugCsvFname)

            flattenData.flattenOsdb(testDataFname, testCsvFname, configObj)
            flattenData.flattenOsdb(trainDataFname, trainCsvFname, configObj)

        if (not os.path.exists(trainAugCsvFname)):
            print("Augmented data file missing - re-generating")
            augmentData.augmentSeizureData(configObj, debug)

        print("Training Model")
        nnTrainer.trainModel(configObj, debug)
    
    print("Testing Model")
    nnTester.testModel2(configObj, debug)        
    


if __name__ == "__main__":
    main()

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
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libosd.configUtils

import selectData
import splitData
import flattenData
import augmentData
from user_tools.nnTraining2.extractFeatures import extractFeatures

def deleteFileIfExists(fname, debug=True):
    ''' If the specified file named fname exists, delete it, otherwise do nothing.'''
    if (os.path.exists(fname)):
            if (debug): print("Deleting %s" % fname)
            os.remove(fname)


def getOutputPath(outPath = "./output", rerun=0, prefix="training"):
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

    if (rerun==0):
        newOutputPath = os.path.join(outFolderPath,str(newOutputFolder))
        os.makedirs(newOutputPath, exist_ok = False)
    else:
        newOutputPath = os.path.join(outFolderPath,str(rerun))

    print(latestOutputFolder, newOutputFolder, newOutputPath)
    return newOutputPath


def main():
    print("runSequence.main()")
    parser = argparse.ArgumentParser(description='Run the Neural Network Training toolchain sequence')
    parser.add_argument('--config', default="nnConfig.json",
                        help='name of json file containing configuration')
    parser.add_argument('--kfold', default=1,
                        help='number of folds for cross-validation')
    parser.add_argument('--rerun', default=0,
                        help='re-run the specified run number.  If 0 then a new run is created.')
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

    kfold = int(args['kfold'])

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

    allDataFname = configObj['dataFileNames']['allDataFileJson']
    testDataFname = configObj['dataFileNames']['testDataFileJson']
    trainDataFname = configObj['dataFileNames']['trainDataFileJson']
    valDataFname = configObj['dataFileNames']['valDataFileJson']
    testCsvFname = configObj['dataFileNames']['testDataFileCsv']
    testBalCsvFname = configObj['dataFileNames']['testBalancedFileCsv']
    trainCsvFname = configObj['dataFileNames']['trainDataFileCsv']
    valCsvFname = configObj['dataFileNames']['valDataFileCsv']
    trainAugCsvFname = configObj['dataFileNames']['trainAugmentedFileCsv']
    modelFname = configObj['modelConfig']['modelFname']


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

        deleteFileIfExists("%s.keras" % modelFname)
        deleteFileIfExists("%s_confusion.png" % modelFname)
        deleteFileIfExists("%s_probabilities.png" % modelFname)
        deleteFileIfExists("%s_training.png" % modelFname)
        deleteFileIfExists("%s_training2.png" % modelFname)
        deleteFileIfExists("%s_stats.txt" % modelFname)

        exit(0)



    if args['train']:
        import numpy as np
        import random

        # Initialise random number generators
        if ('randomSeed' in configObj):
            print("runSequence: Setting random seed to %d" % configObj['randomSeed'])
            seed = configObj['randomSeed'];
            np.random.seed(seed)
            random.seed(seed) 

        outFolder = getOutputPath(outPath=args['outDir'], rerun=args['rerun'], prefix=modelFname)
        print("runSequence: Writing Output to folder %s" % outFolder)

        # Select Data
        allDataFnamePath = os.path.join(outFolder, allDataFname)
        if (not os.path.exists(allDataFnamePath)):
            print("runSequence: All data file missing - re-generating")
            print("runSequence: Removing raw, flattened and augmented files where they exist, so they are re-generated")
            deleteFileIfExists(testDataFname)
            deleteFileIfExists(trainDataFname)
            deleteFileIfExists(valDataFname)
            deleteFileIfExists(testCsvFname)
            deleteFileIfExists(trainCsvFname)
            deleteFileIfExists(valCsvFname)
            deleteFileIfExists(trainAugCsvFname)
            deleteFileIfExists(testBalCsvFname)
            selectData.selectData(configObj, outDir=outFolder, debug=debug) 
            splitData.splitData(configObj, kFold=kfold, outDir=outFolder, debug=debug)
        else:
            print("runSequence: All data file %s already exists - skipping selection step" % allDataFnamePath)

        foldResults = []
        for nFold in range(0, kfold):
            if (kfold > 1):
                print("runSequence: Fold %d" % nFold)
                foldOutFolder = os.path.join(outFolder, "fold%d" % nFold)
            else:
                print("runSequence: No folds - using output folder %s" % outFolder)
                foldOutFolder = outFolder

            testFoldFnamePath = os.path.join(foldOutFolder, testDataFname)
            testFoldCsvFnamePath = os.path.join(foldOutFolder, testCsvFname)
            if not os.path.exists(testFoldCsvFnamePath):
                print("runSequence: Flattening test data %s" % testFoldFnamePath)
                if not os.path.exists(testFoldFnamePath):
                    print("ERROR: Test data file %s does not exist" % testFoldFnamePath)
                    exit(-1)
                flattenData.flattenOsdb(testFoldFnamePath, testFoldCsvFnamePath, configObj)
            else:
                print("runSequence: Test data %s already flattened - skipping" % testFoldCsvFnamePath)

            trainFoldFnamePath = os.path.join(foldOutFolder, trainDataFname)
            trainFoldCsvFnamePath = os.path.join(foldOutFolder, trainCsvFname)
            if not os.path.exists(trainFoldCsvFnamePath):
                print("runSequence: Flattening train data %s" % trainFoldFnamePath)
                if not os.path.exists(trainFoldFnamePath):
                    print("ERROR: Train data file %s does not exist" % trainFoldFnamePath)
                    exit(-1)
                flattenData.flattenOsdb(trainFoldFnamePath, trainFoldCsvFnamePath, configObj)
            else:
                print("runSequence: Train data %s already flattened - skipping" % trainFoldCsvFnamePath)

            trainAugCsvFnamePath = os.path.join(foldOutFolder, trainAugCsvFname)
            if not os.path.exists(trainAugCsvFnamePath):
                print("runSequence: Augmenting training data %s" % trainFoldCsvFnamePath)
                augmentData.augmentSeizureData(configObj, dataDir=foldOutFolder, debug=debug)
            else:
                print("runSequence: Training data %s already augmented - skipping" % trainAugCsvFname)

            # After data augmentation
            trainAugCsvFnamePath = os.path.join(foldOutFolder, configObj['dataFileNames']['trainAugmentedFileCsv'])
            trainFeaturesCsvPath = os.path.join(foldOutFolder, configObj['dataFileNames']['trainFeaturesFileCsv'])
            testFoldCsvFnamePath = os.path.join(foldOutFolder, configObj['dataFileNames']['testDataFileCsv'])
            testFeaturesCsvPath = os.path.join(foldOutFolder, configObj['dataFileNames']['testFeaturesFileCsv'])

            # Extract features for training data
            if not os.path.exists(trainFeaturesCsvPath):
                print("runSequence: Extracting features for training data")
                extractFeatures(trainAugCsvFnamePath, trainFeaturesCsvPath, configObj)
            else:
                print(f"runSequence: Training features {trainFeaturesCsvPath} already exist - skipping")

            # Extract features for test data
            if not os.path.exists(testFeaturesCsvPath):
                print("runSequence: Extracting features for test data")
                extractFeatures(testFoldCsvFnamePath, testFeaturesCsvPath, configObj)
            else:
                print(f"runSequence: Test features {testFeaturesCsvPath} already exist - skipping")

            if configObj['modelConfig']['modelType'] == "sklearn":
                print("runSequence: Training sklearn model")
                import skTrainer
                #import skTester
                modelResults = skTrainer.trainModel(configObj, dataDir=foldOutFolder, debug=debug)
                foldResults.append(modelResults)
                print("runSequence: Model trained")
                #print("runSequence: Testing Model - FIXME - this is not yet implemented for sklearn")
                #skTester.testModel2(configObj, dataDir=foldOutFolder, balanced=False, debug=debug)
            elif configObj['modelConfig']['modelType'] == "tensorflow":
                import nnTrainer
                import nnTester
                import tensorflow as tf
                if ('randomSeed' in configObj):
                    print("runSequence: Setting tensorflow random seed to %d" % configObj['randomSeed'])
                    seed = configObj['randomSeed'];
                    tf.random.set_seed(seed)
                print("runSequence: Training tensorflow model")
                nnTrainer.trainModel(configObj, dataDir=foldOutFolder, debug=debug)
                print("runSequence: Testing Model")
                nnTester.testModel2(configObj, dataDir=foldOutFolder, balanced=False, debug=debug)  
            print("runSequence: Finished fold %d, data in folder %s" % (nFold, foldOutFolder))

        print("|--------|-------------------------------------------------------|-----------------------------------------------|")
        print( "|        |   Model Results                                       | OSD Algorithm Results                         |")
        print("| FoldID |   np  |  tn   |  fn   |  fp   |  tp   |  tpr  |  fpr  | tnOsd | fnOsd | fpOsd | tpOsd |tprOsd |fprOsd |")
        print("|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|")
        for nFold, foldResult in enumerate(foldResults):
            print(f"| Fold {nFold} | {foldResult['tp']+foldResult['fn']:5d} | {foldResult['tn']:5d} | {foldResult['fn']:5d} | {foldResult['fp']:5d} | {foldResult['tp']:5d} | {foldResult['tpr']:5.2f} | {foldResult['fpr']:5.2f} | {foldResult['tnOsd']:5d} | {foldResult['fnOsd']:5d} | {foldResult['fpOsd']:5d} | {foldResult['tpOsd']:5d} | {foldResult['tprOsd']:5.2f} | {foldResult['fprOsd']:5.2f} |")
        print("|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|")

        avgResults = {}
        for key in foldResults[0].keys():
            avgResults[key] = sum(foldResult[key] for foldResult in foldResults) / len(foldResults)
        # calculate standard deviation for each key
        for key in foldResults[0].keys():
            avgResults[key + "_std"] = np.std([foldResult[key] for foldResult in foldResults])

        print(f"| Avg    |       | {avgResults['tn']:5.0f} | {avgResults['fn']:5.0f} | {avgResults['fp']:5.0f} | {avgResults['tp']:5.0f} | {avgResults['tpr']:5.2f} | {avgResults['fpr']:5.2f} | {avgResults['tnOsd']:5.0f} | {avgResults['fnOsd']:5.0f} | {avgResults['fpOsd']:5.0f} | {avgResults['tpOsd']:5.0f} | {avgResults['tprOsd']:5.2f} | {avgResults['fprOsd']:5.2f} |")
        print(f"| Std    |       | {avgResults['tn_std']/avgResults['tn']:5.2f} | {avgResults['fn_std']/avgResults['fn']:5.2f} | {avgResults['fp_std']/avgResults['fp']:5.2f} | {avgResults['tp_std']/avgResults['tp']:5.2f} | {avgResults['tpr_std']:5.2f} | {avgResults['fpr_std']:5.2f} | {avgResults['tnOsd_std']/avgResults['tnOsd']:5.2f} | {avgResults['fnOsd_std']/avgResults['fnOsd']:5.2f} | {avgResults['fpOsd_std']/avgResults['fpOsd']:5.2f} | {avgResults['tpOsd_std']/avgResults['tpOsd']:5.2f} | {avgResults['tprOsd_std']:5.2f} | {avgResults['fprOsd_std']:5.2f} |")
        print("|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|")

        # Save the results to a file
        kfoldSummaryPath = os.path.join(outFolder, "kfold_summary.txt")
        f = open(kfoldSummaryPath, 'w')
        f.write("K-Fold Summary:\n")
        f.write("|--------|-------------------------------------------------------|-----------------------------------------------|\n")
        f.write( "|        |   Model Results                                       | OSD Algorithm Results                         |\n")
        f.write("| FoldID |   np  |  tn   |  fn   |  fp   |  tp   |  tpr  |  fpr  | tnOsd | fnOsd | fpOsd | tpOsd |tprOsd |fprOsd |\n")
        f.write("|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|\n")
        for nFold, foldResult in enumerate(foldResults):
            f.write(f"| Fold {nFold} | {foldResult['tp']+foldResult['fn']:5d} | {foldResult['tn']:5d} | {foldResult['fn']:5d} | {foldResult['fp']:5d} | {foldResult['tp']:5d} | {foldResult['tpr']:5.2f} | {foldResult['fpr']:5.2f} | {foldResult['tnOsd']:5d} | {foldResult['fnOsd']:5d} | {foldResult['fpOsd']:5d} | {foldResult['tpOsd']:5d} | {foldResult['tprOsd']:5.2f} | {foldResult['fprOsd']:5.2f} |\n")
        f.write("|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|\n")
        f.write(f"| Avg    |       | {avgResults['tn']:5.0f} | {avgResults['fn']:5.0f} | {avgResults['fp']:5.0f} | {avgResults['tp']:5.0f} | {avgResults['tpr']:5.2f} | {avgResults['fpr']:5.2f} | {avgResults['tnOsd']:5.0f} | {avgResults['fnOsd']:5.0f} | {avgResults['fpOsd']:5.0f} | {avgResults['tpOsd']:5.0f} | {avgResults['tprOsd']:5.2f} | {avgResults['fprOsd']:5.2f} |\n")
        f.write(f"| Std    |       | {avgResults['tn_std']/avgResults['tn']:5.2f} | {avgResults['fn_std']/avgResults['fn']:5.2f} | {avgResults['fp_std']/avgResults['fp']:5.2f} | {avgResults['tp_std']/avgResults['tp']:5.2f} | {avgResults['tpr_std']:5.2f} | {avgResults['fpr_std']:5.2f} | {avgResults['tnOsd_std']/avgResults['tnOsd']:5.2f} | {avgResults['fnOsd_std']/avgResults['fnOsd']:5.2f} | {avgResults['fpOsd_std']/avgResults['fpOsd']:5.2f} | {avgResults['tpOsd_std']/avgResults['tpOsd']:5.2f} | {avgResults['tprOsd_std']:5.2f} | {avgResults['fprOsd_std']:5.2f} |\n")
        f.write("|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|\n")
        f.close()
        print("K-Fold summary saved to %s" % (kfoldSummaryPath))    



        if (os.path.exists(args['config'])):
            shutil.copy(args['config'], outFolder)

        print("runSequence: Finished training and testing - output in folder %s" % outFolder)    


    
    if args['test']:
        import nnTester
        outFolder = getOutputPath(outPath=args['outDir'], rerun=True, prefix=configObj['modelConfig']['modelFname'])
        #outFolder = getOutputPath(args['outDir'], configObj['modelConfig']['modelFname'])
        print("runSequence: Testing in folder %s" % outFolder)
        nnTester.testModel2(configObj, dataDir=outFolder, balanced=False, debug=debug)  
        
    # Archive Results
    #import shutil
    #if (os.path.exists(testDataFname)):
    #    shutil.copy(testDataFname, outFolder)
    #if (os.path.exists(trainDataFname)):
    #    shutil.copy(trainDataFname, outFolder)
    #if (os.path.exists(valDataFname)):
    #    shutil.copy(valDataFname, outFolder)
    #if (os.path.exists(testCsvFname)):
    #    shutil.copy(testCsvFname, outFolder)
    #if (os.path.exists(trainCsvFname)):
    #    shutil.copy(trainCsvFname, outFolder)
    #if (os.path.exists(valCsvFname)):
    #    shutil.copy(valCsvFname, outFolder)
    #if (os.path.exists(trainAugCsvFname)):
    #    shutil.copy(trainAugCsvFname, outFolder)
    #if (os.path.exists(testBalCsvFname)):
    #    shutil.copy(testBalCsvFname, outFolder)
    #if (os.path.exists("%s.keras" % configObj['modelFname'])):
    #    shutil.copy("%s.keras" % configObj['modelFname'], outFolder)
    #if (os.path.exists("%s_confusion.png" % configObj['modelFname'])):
    #    shutil.copy("%s_confusion.png" % configObj['modelFname'], outFolder)
    #if (os.path.exists("%s_probabilities.png" % configObj['modelFname'])):
    #    shutil.copy("%s_probabilities.png" % configObj['modelFname'], outFolder)
    #if (os.path.exists("%s_training.png" % configObj['modelFname'])):
    #    shutil.copy("%s_training.png" % configObj['modelFname'], outFolder)
    #if (os.path.exists("%s_training2.png" % configObj['modelFname'])):
    #    shutil.copy("%s_training2.png" % configObj['modelFname'], outFolder)
    #if (os.path.exists("%s_stats.txt" % configObj['modelFname'])):
    #    shutil.copy("%s_stats.txt" % configObj['modelFname'], outFolder)
    #if (os.path.exists(args['config'])):
    #    shutil.copy(args['config'], outFolder)


    print("Finished - output in folder %s" % outFolder)


if __name__ == "__main__":
    #outFolder = getOutputPath("./output")
    #print(outFolder)
    #exit(-1)
    main()

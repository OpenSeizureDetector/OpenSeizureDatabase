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

try:
    from user_tools.nnTraining2 import selectData, splitData, flattenData, augmentData
except ImportError:
    import selectData
    import splitData
    import flattenData
    import augmentData
from user_tools.nnTraining2.extractFeatures import extractFeatures
from user_tools.nnTraining2.addFeatureHistory import add_feature_history

# Conditional imports: allow running as a script from this folder OR importing as a module for testing.
# This ensures compatibility with both 'python runSequence.py' and 'pytest ...' from repo root.
def deleteFileIfExists(fname, debug=True):
    ''' If the specified file named fname exists, delete it, otherwise do nothing.'''
    if (os.path.exists(fname)):
            if (debug): print("Deleting %s" % fname)
            os.remove(fname)


def getLatestOutputFolder(outPath = "./output", prefix="training"):
    import pathlib, os

    outFolderPath = os.path.join(outPath, prefix)
    allOutputFolders = list(pathlib.Path(outFolderPath).glob('*/'))

    print("Existing Output Folders:")
    for folder in allOutputFolders:
        print(folder)
    #print(folder for folder in allOutputFolders)
    print("Length of allOutputFolders = %d" % len(allOutputFolders))

    #print()
    if (len(allOutputFolders) > 0):
        latestOutputFolderPath = max(allOutputFolders, key=os.path.getctime)
    else:
        latestOutputFolderPath = None
    
    return(latestOutputFolderPath)



def getOutputPath(outPath = "./output", rerun=0, prefix="training"):
    '''
    Returns a path to the next sequentially numbered folder in outPath
    '''
    import pathlib, os
    outFolderPath = os.path.join(outPath, prefix)
    latestOutputFolderPath = getLatestOutputFolder(outPath=outPath, prefix=prefix)
    if latestOutputFolderPath is not None:
        print("latestOutputFolderPath = %s" % latestOutputFolderPath)
        latestOutputFolder = os.path.split(latestOutputFolderPath)[-1]
    else:
        #print("starting new output folder list")
        latestOutputFolder = "0"

    print("latestOutputFolder = %s" % latestOutputFolder)

    newOutputFolder = 1 + int(latestOutputFolder)

    if (rerun==0):
        newOutputPath = os.path.join(outFolderPath,str(newOutputFolder))
        os.makedirs(newOutputPath, exist_ok = False)
    else:
        newOutputPath = os.path.join(outFolderPath,str(rerun))

    print("getOutputPath() - outputPath=%s" %newOutputPath)
    return newOutputPath


def run_sequence(args):
    """
    Run the Neural Network Training toolchain sequence.

    Args:
        args (dict): Dictionary of arguments. Keys should match the command line options:
            'config' (str): Path to configuration JSON file.
            'kfold' (int or str): Number of folds for cross-validation.
            'rerun' (int or str): Re-run the specified run number. If 0, a new run is created.
            'outDir' (str): Output directory for results.
            'train' (bool): If True, train the model.
            'test' (bool): If True, test the model.
            'clean' (bool): If True, clean up output files before running.
            'debug' (bool): If True, print debugging information.
        Example:
            args = {
                'config': 'nnConfig.json',
                'kfold': 5,
                'rerun': 0,
                'outDir': './output',
                'train': True,
                'test': False,
                'clean': False,
                'debug': True
            }
    """
    print("runSequence.run_sequence()")
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
                flattenData.flattenOsdb(testFoldFnamePath, testFoldCsvFnamePath, debug=debug)
            else:
                print("runSequence: Test data %s already flattened - skipping" % testFoldCsvFnamePath)

            trainFoldFnamePath = os.path.join(foldOutFolder, trainDataFname)
            trainFoldCsvFnamePath = os.path.join(foldOutFolder, trainCsvFname)
            if not os.path.exists(trainFoldCsvFnamePath):
                print("runSequence: Flattening train data %s" % trainFoldFnamePath)
                if not os.path.exists(trainFoldFnamePath):
                    print("ERROR: Train data file %s does not exist" % trainFoldFnamePath)
                    exit(-1)
                flattenData.flattenOsdb(trainFoldFnamePath, trainFoldCsvFnamePath, debug=debug)
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

            # Generate feature history files if they do not exist
            trainFeaturesHistoryCsvPath = os.path.join(foldOutFolder, configObj['dataFileNames']['trainFeaturesHistoryFileCsv'])
            testFeaturesHistoryCsvPath = os.path.join(foldOutFolder, configObj['dataFileNames']['testFeaturesHistoryFileCsv'])
            if not (os.path.exists(trainFeaturesHistoryCsvPath) and os.path.exists(testFeaturesHistoryCsvPath)):
                print("runSequence: Generating feature history files")
                add_feature_history(configObj, foldOutFolder=foldOutFolder)
            else:
                print("runSequence: Feature history files already exist - skipping")

            # Update training to use history files
            configObj['dataFileNames']['trainFeaturesFileCsv'] = configObj['dataFileNames']['trainFeaturesHistoryFileCsv']
            configObj['dataFileNames']['testFeaturesFileCsv'] = configObj['dataFileNames']['testFeaturesHistoryFileCsv']

            if configObj['modelConfig']['modelType'] == "sklearn":
                print("runSequence: Training sklearn model")
                import skTrainer
                import skTester
                trainingResults = skTrainer.trainModel(configObj, dataDir=foldOutFolder, debug=debug)
                print("runSequence: Model trained")
                testResults = skTester.testModel(configObj, dataDir=foldOutFolder, debug=debug)
                foldResults.append(testResults)
            elif configObj['modelConfig']['modelType'] in ["tensorflow", "pytorch", "neuralNet"]:
                import nnTrainer
                import nnTester
                
                # Detect framework (tensorflow or pytorch)
                framework = configObj['modelConfig'].get('framework', 'tensorflow')
                
                # Set random seed based on framework
                if ('randomSeed' in configObj):
                    seed = configObj['randomSeed']
                    if framework == 'pytorch':
                        print("runSequence: Setting PyTorch random seed to %d" % seed)
                        import torch
                        torch.manual_seed(seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(seed)
                    else:
                        print("runSequence: Setting TensorFlow random seed to %d" % seed)
                        import tensorflow as tf
                        tf.random.set_seed(seed)
                
                print("runSequence: Training %s neural network model" % framework)
                nnTrainer.trainModel(configObj, dataDir=foldOutFolder, debug=debug)
                print("runSequence: Testing Model")
                nnTester.testModel(configObj, dataDir=foldOutFolder, balanced=False, debug=debug)  
            print("runSequence: Finished fold %d, data in folder %s" % (nFold, foldOutFolder))

        # Compute average results across folds
        avgResults = {}
        for key in foldResults[0].keys():
            avgResults[key] = sum(foldResult[key] for foldResult in foldResults) / len(foldResults)
        # calculate standard deviation for each key
        for key in foldResults[0].keys():
            avgResults[key + "_std"] = np.std([foldResult[key] for foldResult in foldResults])


        # Save the results to a file
        kfoldSummaryPath = os.path.join(outFolder, "kfold_summary.txt")
        kfoldSummaryPath = os.path.join(outFolder, "kfold_summary.txt")
        kfoldJsonPath = os.path.join(outFolder, "kfold_summary.json")
        import json
        with open(kfoldSummaryPath, 'w') as f:
            f.write("K-Fold Summary (epoch based analysis):\n")
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
            f.write("\n\n")
            f.write("K-Fold Summary (event based analysis):\n")
            f.write("|--------|-------------------------------------------------------|-----------------------------------------------|\n")
            f.write( "|        |   Model Results                                       | OSD Algorithm Results                         |\n")
            f.write("| FoldID |   np  |  tn   |  fn   |  fp   |  tp   |  tpr  |  fpr  | tnOsd | fnOsd | fpOsd | tpOsd |tprOsd |fprOsd |\n")
            f.write("|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|\n")
            for nFold, foldResult in enumerate(foldResults):
                f.write(f"| Fold {nFold} | {foldResult['event_tp']+foldResult['event_fn']:5d} | {foldResult['event_tn']:5d} | {foldResult['event_fn']:5d} | {foldResult['event_fp']:5d} | {foldResult['event_tp']:5d} | {foldResult['event_tpr']:5.2f} | {foldResult['event_fpr']:5.2f} | {foldResult['osd_event_tn']:5d} | {foldResult['osd_event_fn']:5d} | {foldResult['osd_event_fp']:5d} | {foldResult['osd_event_tp']:5d} | {foldResult['osd_event_tpr']:5.2f} | {foldResult['osd_event_fpr']:5.2f} |\n")
            f.write("|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|\n")
            f.write(f"| Avg    |       | {avgResults['event_tn']:5.0f} | {avgResults['event_fn']:5.0f} | {avgResults['event_fp']:5.0f} | {avgResults['event_tp']:5.0f} | {avgResults['event_tpr']:5.2f} | {avgResults['event_fpr']:5.2f} | {avgResults['osd_event_tn']:5.0f} | {avgResults['osd_event_fn']:5.0f} | {avgResults['osd_event_fp']:5.0f} | {avgResults['osd_event_tp']:5.0f} | {avgResults['osd_event_tpr']:5.2f} | {avgResults['osd_event_fpr']:5.2f} |\n")
            f.write(f"| Std    |       | {avgResults['event_tn_std']/avgResults['event_tn']:5.2f} | {avgResults['event_fn_std']/avgResults['event_fn']:5.2f} | {avgResults['event_fp_std']/avgResults['event_fp']:5.2f} | {avgResults['event_tp_std']/avgResults['event_tp']:5.2f} | {avgResults['event_tpr_std']:5.2f} | {avgResults['event_fpr_std']:5.2f} | {avgResults['osd_event_tn_std']/avgResults['osd_event_tn']:5.2f} | {avgResults['osd_event_fn_std']/avgResults['osd_event_fn']:5.2f} | {avgResults['osd_event_fp_std']/avgResults['osd_event_fp']:5.2f} | {avgResults['osd_event_tp_std']/avgResults['osd_event_tp']:5.2f} | {avgResults['osd_event_tpr_std']:5.2f} | {avgResults['osd_event_fpr_std']:5.2f} |\n")
            f.write("|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|\n")

        # Write foldResults as formatted JSON
        with open(kfoldJsonPath, 'w') as jf:
            json.dump(foldResults, jf, indent=2)
        print("K-Fold summary saved to %s" % (kfoldSummaryPath))    
        print("K-Fold JSON summary saved to %s" % (kfoldJsonPath))
        with open(kfoldSummaryPath, 'r') as summary_file:
            print(summary_file.read())
        print("===== End K-Fold Summary =====\n")



        if (os.path.exists(args['config'])):
            shutil.copy(args['config'], outFolder)

        print("runSequence: Finished training and testing - output in folder %s" % outFolder)    


    
    if args['test']:
        print("runSequence: Testing model")
        # if we specify an experiment to rerun, use the specified rerun folder, otherwise use the last run
        if (int(args['rerun']) > 0):
            outFolder = getOutputPath(outPath=args['outDir'], rerun=args['rerun'], prefix=modelFname)
        else:
            outFolder = getLatestOutputFolder(outPath=args['outDir'], prefix=modelFname)
        print("runSequence: Using Output to folder %s" % outFolder)
        if (configObj['modelConfig']['modelType'] == "sklearn"):
            import skTester
            # Test the model using sklearn
            skTester.testModel(configObj, dataDir=outFolder, debug=debug)
        elif configObj['modelConfig']['modelType'] in ["tensorflow", "pytorch", "neuralNet"]:
            import nnTester
            framework = configObj['modelConfig'].get('framework', 'tensorflow')
            print("runSequence: Testing %s neural network model" % framework)
            outFolder = getOutputPath(outPath=args['outDir'], rerun=True, prefix=configObj['modelConfig']['modelFname'])
            #outFolder = getOutputPath(args['outDir'], configObj['modelConfig']['modelFname'])
            print("runSequence: Testing in folder %s" % outFolder)
            nnTester.testModel2(configObj, dataDir=outFolder, balanced=False, debug=debug)  
        else:
            print("ERROR: Unsupported model type: %s" % configObj['modelConfig']['modelType'])
            exit(-1)
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
    import argparse
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
    run_sequence(args)

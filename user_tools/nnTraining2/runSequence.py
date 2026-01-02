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
import json
import numpy as np

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


def _countEventsInJson(filePath):
    """
    Count seizure and non-seizure events in a JSON file.
    
    Args:
        filePath (str): Path to the JSON file
        
    Returns:
        tuple: (seizure_count, non_seizure_count)
    """
    import json
    
    with open(filePath, 'r') as f:
        events = json.load(f)
    
    seizure_count = 0
    non_seizure_count = 0
    
    for event in events:
        event_type = event.get('type', '').lower()
        if event_type == 'seizure':
            seizure_count += 1
        else:
            non_seizure_count += 1
    
    return (seizure_count, non_seizure_count)


def _countEventsInCsv(filePath):
    """
    Count seizure and non-seizure events in a CSV file.
    
    Args:
        filePath (str): Path to the CSV file
        
    Returns:
        tuple: (seizure_count, non_seizure_count)
    """
    import pandas as pd
    
    df = pd.read_csv(filePath, low_memory=False)
    
    # Count unique events by eventId, using the 'type' column
    # type: 0 = false alarm/nda, 1 = seizure, 2 = other
    if 'eventId' in df.columns:
        # Group by eventId and get the type for each event
        event_types = df.groupby('eventId')['type'].first()
        seizure_count = (event_types == 1).sum()
        non_seizure_count = (event_types != 1).sum()
    else:
        # If no eventId, count rows
        seizure_count = (df['type'] == 1).sum()
        non_seizure_count = (df['type'] != 1).sum()
    
    return (seizure_count, non_seizure_count)


def calculateFileStats(filePath):
    """
    Calculate statistics for a data file containing seizure/non-seizure events.
    
    Args:
        filePath (str): Path to the file (must be .json or .csv)
        
    Returns:
        tuple: (seizure_count, non_seizure_count)
        
    Raises:
        ValueError: If file is not .json or .csv, or if file doesn't exist
    """
    if not os.path.exists(filePath):
        raise ValueError(f"File does not exist: {filePath}")
    
    file_ext = os.path.splitext(filePath)[1].lower()
    
    if file_ext == '.json':
        return _countEventsInJson(filePath)
    elif file_ext == '.csv':
        return _countEventsInCsv(filePath)
    else:
        raise ValueError(f"File must be .json or .csv, got: {file_ext}")


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
    nestedKfold = int(args['nestedKfold'])

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

            nSeizure, nNonseizure = calculateFileStats(allDataFnamePath)

            print("runSequence: Data selection complete - all data in file %s contains %d seizure events and %d non-seizure events" % (allDataFnamePath, nSeizure, nNonseizure))

            if nestedKfold > 1:
                print("runSequence: Splitting data into nested k-fold: %d outer folds x %d inner folds" % (nestedKfold, kfold))
            else:
                print("runSequence: Splitting data into %d folds" % kfold)
            splitData.splitData(configObj, kFold=kfold, nestedKfold=nestedKfold, outDir=outFolder, debug=debug)
        else:
            print("runSequence: All data file %s already exists - skipping selection step" % allDataFnamePath)

        foldResults = []
        
        # Determine iteration structure based on nested k-fold
        if nestedKfold > 1:
            # Nested k-fold: iterate through outer folds x inner folds
            outer_folds = range(0, nestedKfold)
            inner_folds = range(0, kfold)
        else:
            # Regular k-fold: treat as 1 outer fold with k inner folds
            outer_folds = range(0, 1)
            inner_folds = range(0, kfold)
        
        for nOuterFold in outer_folds:
            for nFold in inner_folds:
                if nestedKfold > 1:
                    print("runSequence: Outer Fold %d, Inner Fold %d" % (nOuterFold, nFold))
                    outerFoldFolder = os.path.join(outFolder, "outerfold%d" % nOuterFold)
                    if kfold > 1:
                        foldOutFolder = os.path.join(outerFoldFolder, "fold%d" % nFold)
                    else:
                        foldOutFolder = outerFoldFolder
                elif kfold > 1:
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
                    validateDatapoints = configObj.get('dataProcessing', {}).get('validateDatapoints', False)
                    flattenData.flattenOsdb(testFoldFnamePath, testFoldCsvFnamePath, debug=debug, validate_datapoints=validateDatapoints)
                else:
                    print("runSequence: Test data %s already flattened - skipping" % testFoldCsvFnamePath)

                trainFoldFnamePath = os.path.join(foldOutFolder, trainDataFname)
                trainFoldCsvFnamePath = os.path.join(foldOutFolder, trainCsvFname)
                if not os.path.exists(trainFoldCsvFnamePath):
                    print("runSequence: Flattening train data %s" % trainFoldFnamePath)
                    if not os.path.exists(trainFoldFnamePath):
                        print("ERROR: Train data file %s does not exist" % trainFoldFnamePath)
                        exit(-1)
                    validateDatapoints = configObj.get('dataProcessing', {}).get('validateDatapoints', False)
                    flattenData.flattenOsdb(trainFoldFnamePath, trainFoldCsvFnamePath, debug=debug, validate_datapoints=validateDatapoints)
                else:
                    print("runSequence: Train data %s already flattened - skipping" % trainFoldCsvFnamePath)

                nSeizure, nNonseizure = calculateFileStats(trainFoldCsvFnamePath)
                print(f"runSequence: Training data written to {trainFoldCsvFnamePath}, containing {nSeizure} seizure events and {nNonseizure} non-seizure events")

                # Augment training data
                trainAugCsvFnamePath = os.path.join(foldOutFolder, trainAugCsvFname)
                if not os.path.exists(trainAugCsvFnamePath):
                    print("runSequence: Augmenting training data %s" % trainFoldCsvFnamePath)
                    augmentData.augmentSeizureData(configObj, dataDir=foldOutFolder, debug=debug)
                    nSeizure, nNonseizure = calculateFileStats(trainAugCsvFnamePath)
                    print(f"runSequence: Augmented training data saved to {trainAugCsvFnamePath}, containing {nSeizure} seizure events and {nNonseizure} non-seizure events")
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

                # Generate feature history files if configured and needed
                addHistoryLength = configObj.get('dataProcessing', {}).get('addFeatureHistoryLength', 0)
                
                # Check if features list contains only raw acceleration (no calculated features)
                features = configObj.get('dataProcessing', {}).get('features', [])
                raw_acc_features = {'acc_magnitude', 'acc_x', 'acc_y', 'acc_z'}
                only_raw_acc = all(f in raw_acc_features for f in features) if features else False
                
                skip_history = (addHistoryLength == 0) or only_raw_acc
                
                if skip_history:
                    print("runSequence: Skipping feature history (addFeatureHistoryLength=0 or only raw acceleration features)")
                    # Use regular feature files directly
                    # Training files are already set correctly above
                else:
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

                # Get framework - check 'framework' field first, fall back to legacy 'modelType'
                framework = configObj['modelConfig'].get('framework')
                if framework is None:
                    framework = configObj['modelConfig'].get('modelType', 'tensorflow')
                
                if framework == "sklearn":
                    print("runSequence: Training sklearn model")
                    import skTrainer
                    import skTester
                    trainingResults = skTrainer.trainModel(configObj, dataDir=foldOutFolder, debug=debug)
                    print("runSequence: Model trained")
                    testResults = skTester.testModel(configObj, dataDir=foldOutFolder, debug=debug)
                    foldResults.append(testResults)
                elif framework in ["tensorflow", "pytorch"]:
                    import nnTrainer
                    import nnTester
                    
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
                    testResults = nnTester.testModel(configObj, dataDir=foldOutFolder, balanced=False, debug=debug) 
                    foldResults.append(testResults)
                    
                if nestedKfold > 1:
                    print("runSequence: Finished outer fold %d, inner fold %d, data in folder %s" % (nOuterFold, nFold, foldOutFolder))
                else:
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

        # NESTED K-FOLD: Test on truly independent outer fold test sets
        if nestedKfold > 1:
            print("\n" + "="*80)
            print("NESTED K-FOLD: Testing on Independent Outer Fold Test Sets")
            print("="*80 + "\n")
            
            outerFoldResults = []
            
            for nOuterFold in range(0, nestedKfold):
                print("\n" + "="*80)
                print("Testing OUTER FOLD %d on independent test set" % nOuterFold)
                print("="*80)
                
                outerFoldOutFolder = os.path.join(outFolder, "outerfold%d" % nOuterFold)
                outerfold_test_json = os.path.join(outerFoldOutFolder, "outerfold_test.json")
                
                if not os.path.exists(outerfold_test_json):
                    print("WARNING: Outer fold test file not found: %s" % outerfold_test_json)
                    continue
                
                # Find best inner fold model for this outer fold based on TPR
                innerFoldPaths = []
                innerFoldResults_forOuter = []
                
                for nFold in range(0, kfold):
                    if kfold > 1:
                        foldPath = os.path.join(outerFoldOutFolder, "fold%d" % nFold)
                    else:
                        foldPath = outerFoldOutFolder
                    
                    # Get the result for this inner fold
                    fold_index = nOuterFold * kfold + nFold
                    if fold_index < len(foldResults):
                        innerFoldResults_forOuter.append(foldResults[fold_index])
                        innerFoldPaths.append(foldPath)
                
                if len(innerFoldResults_forOuter) == 0:
                    print("WARNING: No inner fold results found for outer fold %d" % nOuterFold)
                    continue
                
                # Select best inner fold based on TPR (sensitivity)
                best_fold_idx = max(range(len(innerFoldResults_forOuter)), 
                                  key=lambda i: innerFoldResults_forOuter[i]['tpr'])
                best_fold_path = innerFoldPaths[best_fold_idx]
                best_fold_tpr = innerFoldResults_forOuter[best_fold_idx]['tpr']
                best_fold_fpr = innerFoldResults_forOuter[best_fold_idx]['fpr']
                
                print("runSequence: Best inner fold is %d with TPR=%.3f, FPR=%.3f" % 
                      (best_fold_idx, best_fold_tpr, best_fold_fpr))
                print("runSequence: Using model from: %s" % best_fold_path)
                
                # Prepare outer fold test data
                outerfold_test_csv = os.path.join(outerFoldOutFolder, "outerfold_test.csv")
                if not os.path.exists(outerfold_test_csv):
                    print("runSequence: Flattening outer fold independent test data")
                    validateDatapoints = configObj.get('dataProcessing', {}).get('validateDatapoints', False)
                    flattenData.flattenOsdb(outerfold_test_json, outerfold_test_csv, debug=debug, validate_datapoints=validateDatapoints)
                
                # Extract features for outer fold test data
                outerfold_test_features = os.path.join(outerFoldOutFolder, "outerfold_test_features.csv")
                if not os.path.exists(outerfold_test_features):
                    print("runSequence: Extracting features for outer fold test data")
                    extractFeatures(outerfold_test_csv, outerfold_test_features, configObj)
                
                # Test best model on outer fold independent test set
                print("runSequence: Testing best model (inner fold %d) on outer fold %d independent test set" % 
                      (best_fold_idx, nOuterFold))
                
                # Temporarily update config to point to outer fold test data in the correct directory
                original_test_csv = configObj['dataFileNames']['testDataFileCsv']
                original_test_features = configObj['dataFileNames']['testFeaturesFileCsv']
                
                configObj['dataFileNames']['testDataFileCsv'] = "outerfold_test.csv"
                configObj['dataFileNames']['testFeaturesFileCsv'] = "outerfold_test_features.csv"
                
                framework = configObj['modelConfig'].get('framework')
                if framework is None:
                    framework = configObj['modelConfig'].get('modelType', 'tensorflow')
                
                if framework == "sklearn":
                    import skTester
                    outerTestResults = skTester.testModel(configObj, dataDir=outerFoldOutFolder, debug=debug)
                elif framework in ["tensorflow", "pytorch"]:
                    import nnTester
                    outerTestResults = nnTester.testModel(configObj, dataDir=best_fold_path, balanced=False, debug=debug)
                
                # Restore original config
                configObj['dataFileNames']['testDataFileCsv'] = original_test_csv
                configObj['dataFileNames']['testFeaturesFileCsv'] = original_test_features
                
                # Store outer fold results
                outerTestResults['outer_fold'] = nOuterFold
                outerTestResults['best_inner_fold'] = best_fold_idx
                outerFoldResults.append(outerTestResults)
                
                print("\nrunSequence: Outer fold %d INDEPENDENT test results:" % nOuterFold)
                print("  TPR (Sensitivity): %.3f" % outerTestResults['tpr'])
                print("  FPR (False Positive Rate): %.3f" % outerTestResults['fpr'])
                print("  TP: %d, FP: %d, TN: %d, FN: %d" % 
                      (outerTestResults['tp'], outerTestResults['fp'], 
                       outerTestResults['tn'], outerTestResults['fn']))
                print("  Event TPR: %.3f, Event FPR: %.3f" % 
                      (outerTestResults['event_tpr'], outerTestResults['event_fpr']))
            
            # Compute and save outer fold summary
            if len(outerFoldResults) > 0:
                print("\n" + "="*80)
                print("NESTED K-FOLD: Summary of Independent Outer Fold Test Results")
                print("="*80)
                
                outerAvgResults = {}
                for key in outerFoldResults[0].keys():
                    if key not in ['outer_fold', 'best_inner_fold']:
                        outerAvgResults[key] = sum(result[key] for result in outerFoldResults) / len(outerFoldResults)
                        outerAvgResults[key + "_std"] = np.std([result[key] for result in outerFoldResults])
                
                print("Average across %d outer folds:" % len(outerFoldResults))
                print("  TPR (Sensitivity): %.3f ± %.3f" % (outerAvgResults['tpr'], outerAvgResults['tpr_std']))
                print("  FPR: %.3f ± %.3f" % (outerAvgResults['fpr'], outerAvgResults['fpr_std']))
                print("  Event TPR: %.3f ± %.3f" % (outerAvgResults['event_tpr'], outerAvgResults['event_tpr_std']))
                print("  Event FPR: %.3f ± %.3f" % (outerAvgResults['event_fpr'], outerAvgResults['event_fpr_std']))
                
                # Save outer fold results
                outerFoldSummaryPath = os.path.join(outFolder, "nested_kfold_outer_summary.txt")
                outerFoldJsonPath = os.path.join(outFolder, "nested_kfold_outer_summary.json")
                
                with open(outerFoldSummaryPath, 'w') as f:
                    f.write("NESTED K-FOLD: Independent Outer Fold Test Results\n")
                    f.write("="*80 + "\n\n")
                    f.write("These results are from TRULY INDEPENDENT test sets that were never used during training.\n")
                    f.write("Each outer fold's best inner fold model was tested on its corresponding independent test set.\n\n")
                    f.write("Outer Fold Results (epoch based analysis):\n")
                    f.write("|-----------|-------|-------|-------|-------|-------|-------|-------|\n")
                    f.write("| Outer     | Best  |   np  |  tn   |  fn   |  fp   |  tp   |  tpr  |  fpr  |\n")
                    f.write("| Fold ID   | Inner |       |       |       |       |       |       |       |\n")
                    f.write("|-----------|-------|-------|-------|-------|-------|-------|-------|-------|\n")
                    for result in outerFoldResults:
                        f.write(f"| Outer {result['outer_fold']:2d}  |   {result['best_inner_fold']:2d}  | {result['tp']+result['fn']:5d} | {result['tn']:5d} | {result['fn']:5d} | {result['fp']:5d} | {result['tp']:5d} | {result['tpr']:5.3f} | {result['fpr']:5.3f} |\n")
                    f.write("|-----------|-------|-------|-------|-------|-------|-------|-------|-------|\n")
                    f.write(f"| Average   |       |       | {outerAvgResults['tn']:5.0f} | {outerAvgResults['fn']:5.0f} | {outerAvgResults['fp']:5.0f} | {outerAvgResults['tp']:5.0f} | {outerAvgResults['tpr']:5.3f} | {outerAvgResults['fpr']:5.3f} |\n")
                    f.write(f"| Std Dev   |       |       | {outerAvgResults['tn_std']:5.1f} | {outerAvgResults['fn_std']:5.1f} | {outerAvgResults['fp_std']:5.1f} | {outerAvgResults['tp_std']:5.1f} | {outerAvgResults['tpr_std']:5.3f} | {outerAvgResults['fpr_std']:5.3f} |\n")
                    f.write("|-----------|-------|-------|-------|-------|-------|-------|-------|-------|\n")
                    f.write("\n\n")
                    f.write("Outer Fold Results (event based analysis):\n")
                    f.write("|-----------|-------|-------|-------|-------|-------|-------|-------|-------|\n")
                    f.write("| Outer     | Best  |   np  |  tn   |  fn   |  fp   |  tp   |  tpr  |  fpr  |\n")
                    f.write("| Fold ID   | Inner |       |       |       |       |       |       |       |\n")
                    f.write("|-----------|-------|-------|-------|-------|-------|-------|-------|-------|\n")
                    for result in outerFoldResults:
                        f.write(f"| Outer {result['outer_fold']:2d}  |   {result['best_inner_fold']:2d}  | {result['event_tp']+result['event_fn']:5d} | {result['event_tn']:5d} | {result['event_fn']:5d} | {result['event_fp']:5d} | {result['event_tp']:5d} | {result['event_tpr']:5.3f} | {result['event_fpr']:5.3f} |\n")
                    f.write("|-----------|-------|-------|-------|-------|-------|-------|-------|-------|\n")
                    f.write(f"| Average   |       |       | {outerAvgResults['event_tn']:5.0f} | {outerAvgResults['event_fn']:5.0f} | {outerAvgResults['event_fp']:5.0f} | {outerAvgResults['event_tp']:5.0f} | {outerAvgResults['event_tpr']:5.3f} | {outerAvgResults['event_fpr']:5.3f} |\n")
                    f.write(f"| Std Dev   |       |       | {outerAvgResults['event_tn_std']:5.1f} | {outerAvgResults['event_fn_std']:5.1f} | {outerAvgResults['event_fp_std']:5.1f} | {outerAvgResults['event_tp_std']:5.1f} | {outerAvgResults['event_tpr_std']:5.3f} | {outerAvgResults['event_fpr_std']:5.3f} |\n")
                    f.write("|-----------|-------|-------|-------|-------|-------|-------|-------|-------|\n")
                    f.write("\n\nIMPORTANT: Report these outer fold results in publications as they represent\n")
                    f.write("unbiased estimates of model generalization on truly independent test sets.\n")
                
                with open(outerFoldJsonPath, 'w') as jf:
                    json.dump(outerFoldResults, jf, indent=2)
                
                print("\nNested k-fold outer fold summary saved to: %s" % outerFoldSummaryPath)
                print("Nested k-fold outer fold JSON saved to: %s" % outerFoldJsonPath)
                
                with open(outerFoldSummaryPath, 'r') as summary_file:
                    print("\n" + summary_file.read())
                print("="*80 + "\n")


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
        # Get framework - check 'framework' field first, fall back to legacy 'modelType'
        framework = configObj['modelConfig'].get('framework')
        if framework is None:
            framework = configObj['modelConfig'].get('modelType', 'tensorflow')
        
        if framework == "sklearn":
            import skTester
            # Test the model using sklearn
            if kfold > 1:
                print("ERROR: K-fold testing not yet supported for sklearn models")
                exit(-1)
            skTester.testModel(configObj, dataDir=outFolder, debug=debug)
        elif framework in ["tensorflow", "pytorch"]:
            import nnTester
            print("runSequence: Testing %s neural network model" % framework)
            outFolder = getOutputPath(outPath=args['outDir'], rerun=args['rerun'], prefix=configObj['modelConfig']['modelFname'])
            #outFolder = getOutputPath(args['outDir'], configObj['modelConfig']['modelFname'])
            print("runSequence: Testing in folder %s" % outFolder)
            
            # Determine if we should rerun tests based on --rerun parameter
            rerunTests = int(args['rerun']) > 0
            
            if kfold > 1:
                print("runSequence: Running k-fold testing with %d folds (rerun=%s)" % (kfold, rerunTests))
                nnTester.testKFold(configObj, kfold=kfold, dataDir=outFolder, rerun=rerunTests, debug=debug)
            else:
                print("runSequence: Testing single model")
                nnTester.testModel(configObj, dataDir=outFolder, balanced=False, debug=debug)  
        else:
            print("ERROR: Unsupported framework: %s" % framework)
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
    parser.add_argument('--nestedKfold', default=1,
                        help='number of outer folds for nested k-fold validation (default 1 = no nesting)')
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

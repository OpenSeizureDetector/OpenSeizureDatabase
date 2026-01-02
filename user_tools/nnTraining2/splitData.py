#!/usr/bin/env python3

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



def splitData(configObj, kFold=1, nestedKfold=1, outDir=".", debug=False):
    """
    Using the osdb data in the 'allDataFileJson' configObj entry, load all the available seizure and non-seizure
    data, split the data into 'train' and 'test' data sets, and save them to files.
    
    Supports nested k-fold cross-validation for truly independent model evaluation.
    When nestedKfold > 1: Implements nested k-fold validation where:
      - Outer folds create completely independent test sets that are never touched by k-fold training
      - Inner k-fold splits the remaining data for each outer fold
      - This ensures truly independent generalization assessment
    When nestedKfold == 1: Traditional k-fold (kFold only) or simple train/test split
    
    Args:
        configObj: Configuration dictionary
        kFold: Number of inner folds (only used if nestedKfold == 1 or within each outer fold)
        nestedKfold: Number of outer folds for nested k-fold (if > 1, enables nested k-fold)
        outDir: Directory where the output files will be saved
        debug: Enable debug output
    """
    if (debug): print("saveTestTrainData: configObj=",configObj.keys())
    testProp = configObj['dataProcessing']['testProp']
    valProp = configObj['dataProcessing']['validationProp']
    randomSeed = libosd.configUtils.getConfigParam("randomSeed", configObj)
    allDataFname = configObj['dataFileNames']['allDataFileJson']
    testFname = configObj['dataFileNames']['testDataFileJson']
    trainFname = configObj['dataFileNames']['trainDataFileJson']
    valFname = configObj['dataFileNames']['valDataFileJson']
    if ("fixedTestEvents" in configObj['dataProcessing']):
        print("splitData: Using fixed test events from configObj")
        fixedTestEventsLst = configObj['dataProcessing']['fixedTestEvents']
    else:
        fixedTestEventsLst = None
    if ("fixedTrainEvents" in configObj['dataProcessing']):
        print("splitData: Using fixed train events from configObj")
        fixedTrainEventsLst = configObj['dataProcessing']['fixedTrainEvents']
    else:
        fixedTrainEventsLst = None
    if ("cacheDir" in configObj['osdbConfig']):
        print("splitData: Using cache directory from configObj")
        dbDir = configObj['osdbConfig']['cacheDir']
    else:
        print("splitData: Using default cache directory")
        dbDir = None

    print("splitData: Loading all data from file %s" % allDataFname)
    osd = libosd.osdDbConnection.OsdDbConnection(cacheDir=dbDir, debug=debug)

    allDataFnamePath = os.path.join(outDir, allDataFname)
    print("splitData: Loading file %s" % allDataFnamePath)
    eventsObjLen = osd.loadDbFile(allDataFnamePath, useCacheDir=False)
    print("splitData: Loaded %d events from file %s" % (eventsObjLen, allDataFname))   
    eventIdsLst = osd.getEventIds()

    eventsLst = osd.getEvents(eventIdsLst)
    seizureLst = []
    for event in eventsLst:
        if (event['type'] == "seizure"):
            seizureLst.append(1)
        else:
            seizureLst.append(0)

    # NESTED K-FOLD VALIDATION (outer fold + inner k-fold)
    if (nestedKfold > 1):
        print("\n" + "="*80)
        print("splitData: Using NESTED K-FOLD VALIDATION")
        print("  - Outer folds: %d (completely independent test sets)" % nestedKfold)
        print("  - Inner k-fold: %d (on remaining data in each outer fold)" % kFold)
        print("="*80 + "\n")
        
        # First split into outer folds (outer loop for truly independent test sets)
        skf_outer = sklearn.model_selection.StratifiedKFold(
            n_splits=nestedKfold, shuffle=True, random_state=randomSeed
        )
        
        for outer_fold, (trainval_outer_idx, test_outer_idx) in enumerate(
            skf_outer.split(eventIdsLst, seizureLst)
        ):
            print("\nsplitData: Processing OUTER FOLD %d" % outer_fold)
            print("  Total events in this outer fold: %d" % (len(trainval_outer_idx) + len(test_outer_idx)))
            
            # Get event IDs for outer fold's training/validation and test sets
            trainval_outer_ids = [eventIdsLst[i] for i in trainval_outer_idx]
            test_outer_ids = [eventIdsLst[i] for i in test_outer_idx]
            
            # Create outer fold directory
            outerFoldDataPath = os.path.join(outDir, "outerfold%d" % outer_fold)
            if not os.path.exists(outerFoldDataPath):
                os.makedirs(outerFoldDataPath)
            
            print("  Events for outer fold test set (held out completely): %d" % len(test_outer_ids))
            print("  Events for inner k-fold training: %d" % len(trainval_outer_ids))
            
            # Save the outer fold test set (completely independent)
            test_outer_fnamepath = os.path.join(outerFoldDataPath, "outerfold_test.json")
            osd.saveEventsToFile(test_outer_ids, test_outer_fnamepath, pretty=False, useCacheDir=False)
            print("  Saved outer fold test set to: %s" % test_outer_fnamepath)
            
            # Now do inner k-fold on the training/validation data
            if kFold > 1:
                print("  Performing inner k-fold (%d folds) on remaining data..." % kFold)
                skf_inner = sklearn.model_selection.StratifiedKFold(
                    n_splits=kFold, shuffle=True, random_state=randomSeed
                )
                
                # Get seizure labels for this outer fold's training data
                trainval_outer_seizure_labels = [seizureLst[i] for i in trainval_outer_idx]
                
                for inner_fold, (train_idx, test_idx) in enumerate(
                    skf_inner.split(trainval_outer_ids, trainval_outer_seizure_labels)
                ):
                    print("    Inner fold %d" % inner_fold)
                    
                    # Create inner fold directory
                    innerFoldDataPath = os.path.join(outerFoldDataPath, "fold%d" % inner_fold)
                    if not os.path.exists(innerFoldDataPath):
                        os.makedirs(innerFoldDataPath)
                    
                    # Get training and test event IDs for this inner fold
                    train_ids = [trainval_outer_ids[i] for i in train_idx]
                    test_ids = [trainval_outer_ids[i] for i in test_idx]
                    
                    print("      Training events: %d, Test events: %d" % (len(train_ids), len(test_ids)))
                    
                    # Save inner fold training data
                    train_fnamepath = os.path.join(innerFoldDataPath, trainFname)
                    osd.saveEventsToFile(train_ids, train_fnamepath, pretty=False, useCacheDir=False)
                    
                    # Save inner fold test data
                    test_fnamepath = os.path.join(innerFoldDataPath, testFname)
                    osd.saveEventsToFile(test_ids, test_fnamepath, pretty=False, useCacheDir=False)
                    
                    print("      Files saved to: %s" % innerFoldDataPath)
            else:
                # If kFold == 1, just save the outer fold's data as-is
                print("  No inner k-fold (kFold=1), saving outer fold data directly...")
                train_fnamepath = os.path.join(outerFoldDataPath, trainFname)
                test_fnamepath = os.path.join(outerFoldDataPath, testFname)
                
                osd.saveEventsToFile(trainval_outer_ids, train_fnamepath, pretty=False, useCacheDir=False)
                osd.saveEventsToFile(test_outer_ids, test_fnamepath, pretty=False, useCacheDir=False)
                
                print("  Files saved to: %s" % outerFoldDataPath)
        
        print("\n" + "="*80)
        print("splitData: NESTED K-FOLD VALIDATION SETUP COMPLETE")
        print("  Structure: outerfold0/{fold0,fold1,...}, outerfold1/{fold0,fold1,...}, etc.")
        print("  Each outerfold has independent test set in outerfold_test.json")
        print("="*80 + "\n")

    elif (kFold > 1):
        print("splitData: Using KFold Cross Validation - splitting data into %d folds" % kFold)
        #kf = sklearn.model_selection.KFold(n_splits=kFold, shuffle=True)
        kf = sklearn.model_selection.StratifiedKFold(n_splits=kFold, shuffle=True, random_state=randomSeed)
        for fold, (train_index, test_index) in enumerate(kf.split(eventIdsLst,seizureLst)):
            print(fold)
            print("splitData: TRAIN:", train_index, "TEST:", test_index)
            foldDataPath = os.path.join(outDir, "fold%d" % fold)
            if not os.path.exists(foldDataPath):
                os.makedirs(foldDataPath)
            print("splitData: Saving Fold %d Files into folder %s" % (fold, foldDataPath))

            trainIdLst = [eventIdsLst[i] for i in train_index]
            trainFoldFnamePath = os.path.join(foldDataPath, trainFname)
            osd.saveEventsToFile(trainIdLst, trainFoldFnamePath, pretty=False, useCacheDir=False)
            print("splitData: Training Data written to file %s" % trainFoldFnamePath)

            testIdLst = [eventIdsLst[i] for i in test_index]    
            testFoldFnamePath = os.path.join(foldDataPath, testFname)
            osd.saveEventsToFile(testIdLst, testFoldFnamePath, pretty=False, useCacheDir=False)
            print("splitData: Test Data written to file %s" % testFoldFnamePath)

            print("splitData:  Fold %d Files Saved" % fold)


            print("splitData: Total Number of Events = %d" % len(eventIdsLst))
            print("splitData: Number of Training Events = %d" % len(trainIdLst))
            print("splitData: Number of Test Events = %d" % len(testIdLst))
            print("splitData: Total Number of Events = %d" % (len(trainIdLst) + len(testIdLst)))

    else:
        print("splitData: kfold=1 so using single split of data into test and train datasets")
        if (fixedTestEventsLst is not None):
            print("splitData: Removing fixed test events")
            for testId in fixedTestEventsLst:
                print(testId)
                eventIdsLst.remove(testId)

        if (fixedTrainEventsLst):
            print("splitData: Removing fixed training events")
            for trainId in fixedTrainEventsLst:
                print(trainId)
                eventIdsLst.remove(trainId)


        print("splitData: Preparing new test/train dataset")

        print("splitData(): Splitting data by Event")
        # Split into test and train data sets.
        if (debug): print("Total Events=%d" % len(eventIdsLst))

        # Split events list into test and train data sets.
        print("splitData: Splitting data into test and train/validatin datasets")
        trainValIdLst, testIdLst =\
            sklearn.model_selection.train_test_split(eventIdsLst,
                                                    test_size=testProp,
                                                    random_state=randomSeed)
        if (debug): print("len(train)=%d, len(test)=%d" % (len(trainValIdLst), len(testIdLst)))
        #print("test=",testIdLst)

        if (fixedTestEventsLst is not None):
            print("splitData: Adding fixed test events to test data")
            for testId in fixedTestEventsLst:
                print(testId)
                testIdLst.append(testId)

        if (fixedTrainEventsLst):
            print("splitData: Adding fixed training events to test data")
            for trainId in fixedTrainEventsLst:
                print(trainId)
                testIdLst.append(trainId)

        # Split the training data set into training and validation data sets.
        if (valProp > 0):
            print("splitData: Validation proportion is %f" % valProp)
            if (debug): print("len(trainVal)=%d, len(test)=%d" % (len(trainValIdLst), len(testIdLst)))
            print("splitData: Splitting the training data set into training and validation data sets")
            trainIdLst, valIdLst =\
                sklearn.model_selection.train_test_split(trainValIdLst,
                                                        test_size=valProp,
                                                        random_state=randomSeed)
        else:
            print("splitData: No validation data set - using all training data")
            trainIdLst = trainValIdLst
            valIdLst = []


        print("splitData: Total Number of Events = %d" % len(eventIdsLst))
        print("splitData: Number of Training Events = %d" % len(trainIdLst))
        print("splitData: Number of Test Events = %d" % len(testIdLst))
        print("splitData: Number of Validation Events = %d" % len(valIdLst))
        print("splitData: Total Number of Events = %d" % (len(trainIdLst) + len(testIdLst) + len(valIdLst)))

        # 
        # Save the test and train data sets to files.   
        print("splitData: Saving Training Data File")
        trainFnamePath = os.path.join(outDir, trainFname)
        osd.saveEventsToFile(trainIdLst, trainFnamePath, pretty=False, useCacheDir=False)
        print("splitData: Training Data written to file %s" % trainFname)

        print("splitData: Saving Test Data File")
        testFnamePath = os.path.join(outDir, testFname)
        osd.saveEventsToFile(testIdLst, testFnamePath, pretty=False, useCacheDir=False)
        print("splitData: Test Data written to file %s" % testFname)

        print("splitData: Saving Validation Data File")
        if (len(valIdLst) == 0):
            print("splitData: No Validation Data - not saving validation file")
            valFname = None
        else:
            valFnamePath = os.path.join(outDir, valFname)
            osd.saveEventsToFile(valIdLst, valFnamePath, pretty=False, useCacheDir=False)
            print("splitData: Validation Data written to file %s" % valFname)

    print("splitData: Test and Train Data Sets Saved")
 

def main():
    print("selectData.main()")
    parser = argparse.ArgumentParser(description='Split OSDB Data into Test and Train Data sets')
    parser.add_argument('--config', default="nnConfig.json",
                        help='name of json file containing configuration')
    parser.add_argument('--kfold', default=1, type=int,
                        help='number of inner folds for cross-validation')
    parser.add_argument('--nestedKfold', default=1, type=int,
                        help='number of outer folds for nested k-fold validation (default 1 = no nesting)')
    parser.add_argument('--debug', action="store_true",
                        help='Write debugging information to screen')
    argsNamespace = parser.parse_args()
    args = vars(argsNamespace)
    print(args)



    configObj = libosd.configUtils.loadConfig(args['config'])
    print("configObj=",configObj.keys())
    # Load a separate OSDB Configuration file if it is included.
    if ("osdbCfg" in configObj):
        osdbCfgFname = libosd.configUtils.getConfigParam("osdbCfg",configObj)
        print("Loading separate OSDB Configuration File %s." % osdbCfgFname)
        osdbCfgObj = libosd.configUtils.loadConfig(osdbCfgFname)
        # Merge the contents of the OSDB Configuration file into configObj
        configObj = configObj | osdbCfgObj

    print("configObj=",configObj.keys())

    
    splitData(configObj, kFold=args['kfold'], nestedKfold=args['nestedKfold'], debug=args['debug'])
        
    


if __name__ == "__main__":
    main()

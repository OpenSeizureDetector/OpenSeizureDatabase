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



def splitData(configObj, kFold=1, outDir=".", debug=False):
    """
    Using the osdb data in the 'allDataFileJson' configObj entry, load all the available seizure and non-seizure
    data, split the data into 'train' and 'test' data sets, and save them to files.
    outDir is the directory where the output files will be saved.
    kFold is the number of folds to use for cross-validation.  If kFold is 1, then the data is split into a training and test set.
    If kFold > 1, then the data is split into kFold sets, and each set is used as the test set once.
    The configuration is specified in the configObj dict.   The following configObj elements
    are used:
       - allDataFileJson - the filename of the json file containing the data to be split.
       - testProp - the proportion of the events in the database to be used for the test data set.
       - randomSeed - the seed to use for the random number generator (to ensure repeatable results)
       - fixedTestEvents - list of event IDs to force into the test dataset
       - fixedTrainEvents - list of event IDs to force into the training dataset
       - eventFilters - dictionary specifying filters to apply to select required data.
       - trainDataFile - filename to use to save the training data set (relative to current working directory)
       - testDataFile - filename to use to save the test data set (relative to current working directory)
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

    if (kFold > 1):
        print("splitData: Using KFold Cross Validation - splitting data into %d folds" % kFold)
        kf = sklearn.model_selection.KFold(n_splits=kFold, shuffle=True)
        for fold, (train_index, test_index) in enumerate(kf.split(eventIdsLst)):
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

    
    splitData(configObj, args['debug'])
        
    


if __name__ == "__main__":
    main()

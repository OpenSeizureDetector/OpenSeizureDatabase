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



def saveTestTrainData(configObj, debug=False):
    """
    Using the osdb data in the 'allDataFileJson' configObj entry, load all the available seizure and non-seizure
    data, split the data into 'train' and 'test' data sets, and save them to files.
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
    testProp = libosd.configUtils.getConfigParam("testProp", configObj)
    valProp = libosd.configUtils.getConfigParam("validationProp", configObj)
    randomSeed = libosd.configUtils.getConfigParam("randomSeed", configObj)
    allDataFname = libosd.configUtils.getConfigParam("allDataFileJson", configObj)
    testFname = libosd.configUtils.getConfigParam("testDataFileJson", configObj)
    trainFname = libosd.configUtils.getConfigParam("trainDataFileJson", configObj)
    valFname = libosd.configUtils.getConfigParam("valDataFileJson", configObj)
    fixedTestEventsLst = libosd.configUtils.getConfigParam("fixedTestEvents", configObj)
    fixedTrainEventsLst = libosd.configUtils.getConfigParam("fixedTrainEvents", configObj)
    dbDir = libosd.configUtils.getConfigParam("cacheDir", configObj)

    print("Loading all data from file %s" % allDataFname)
    osd = libosd.osdDbConnection.OsdDbConnection(cacheDir=dbDir, debug=debug)

    print("splitData - loading file %s" % allDataFname)
    eventsObjLen = osd.loadDbFile(allDataFname, useCacheDir=False)
    print("loaded %d events from file %s" % (eventsObjLen, allDataFname))   

    
    eventIdsLst = osd.getEventIds()
    if (fixedTestEventsLst is not None):
        print("Removing fixed test events")
        for testId in fixedTestEventsLst:
            print(testId)
            eventIdsLst.remove(testId)

    if (fixedTrainEventsLst):
        print("Removing fixed training events")
        for trainId in fixedTrainEventsLst:
            print(trainId)
            eventIdsLst.remove(trainId)


    print("Preparing new test/train dataset")

    print("getTestTrainData(): Splitting data by Event")
    # Split into test and train data sets.
    if (debug): print("Total Events=%d" % len(eventIdsLst))

    # Split events list into test and train data sets.
    print("Splitting data into test and train/validatin datasets")
    trainValIdLst, testIdLst =\
        sklearn.model_selection.train_test_split(eventIdsLst,
                                                test_size=testProp,
                                                random_state=randomSeed)
    if (debug): print("len(train)=%d, len(test)=%d" % (len(trainValIdLst), len(testIdLst)))
    #print("test=",testIdLst)

    if (fixedTestEventsLst is not None):
        print("Adding fixed test events to test data")
        for testId in fixedTestEventsLst:
            print(testId)
            testIdLst.append(testId)

    if (fixedTrainEventsLst):
        print("Adding fixed training events to test data")
        for trainId in fixedTrainEventsLst:
            print(trainId)
            testIdLst.append(trainId)

    # Split the training data set into training and validation data sets.
    print("Splitting the training data set into training and validation data sets")
    trainIdLst, valIdLst =\
        sklearn.model_selection.train_test_split(trainValIdLst,
                                                test_size=valProp,
                                                random_state=randomSeed)


    print("Total Number of Events = %d" % len(eventIdsLst))
    print("Number of Training Events = %d" % len(trainIdLst))
    print("Number of Test Events = %d" % len(testIdLst))
    print("Number of Validation Events = %d" % len(valIdLst))
    print("Total Number of Events = %d" % (len(trainIdLst) + len(testIdLst) + len(valIdLst)))

    # 
    # Save the test and train data sets to files.   
    print("Saving Training Data File")
    osd.saveEventsToFile(trainIdLst, trainFname, pretty=False, useCacheDir=False)
    print("Training Data written to file %s" % trainFname)

    print("Saving Test Data File")
    osd.saveEventsToFile(testIdLst, testFname, pretty=False, useCacheDir=False)
    print("Test Data written to file %s" % testFname)

    print("Saving Validation Data File")
    osd.saveEventsToFile(valIdLst, valFname, pretty=False, useCacheDir=False)
    print("Validation Data written to file %s" % valFname)
 

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

    
    saveTestTrainData(configObj, args['debug'])
        
    


if __name__ == "__main__":
    main()

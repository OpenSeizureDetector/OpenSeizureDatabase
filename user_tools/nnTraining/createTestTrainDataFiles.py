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
    Using the osdb 'dataFiles' specified in configObj, load all the available seizure and non-seizure
    data, split the data into 'train' and 'test' data sets, and save them to files.
    The configuration is specified in the configObj dict.   The following configObj elements
    are used:
       - osdbCfg - file name of separate json configuration file which will be included
       - dataFiles - list of osdb data files to use to create data set
       - testProp - the proportion of the events in the database to be used for the test data set.
       - randomSeed - the seed to use for the random number generator (to ensure repeatable results)
       - invalidEvents - list of event IDs to exclude from the datasets
       - fixedTestEvents - list of event IDs to force into the test dataset
       - fixedTrainEvents - list of event IDs to force into the training dataset
       - trainDataFile - filename to use to save the training data set (relative to current working directory)
       - testDataFile - filename to use to save the test data set (relative to current working directory)
    """
    if (debug): print("getTestTrainData: configObj=",configObj.keys())
    invalidEvents = libosd.configUtils.getConfigParam("invalidEvents", configObj)
    excludedUserIds = libosd.configUtils.getConfigParam("excludeUserIds", configObj)
    includeUserIds = libosd.configUtils.getConfigParam("includeUserIds", configObj)
    includeTypes = libosd.configUtils.getConfigParam("includeTypes", configObj)
    includeSubTypes = libosd.configUtils.getConfigParam("includeSubTypes", configObj)
    testProp = libosd.configUtils.getConfigParam("testProp", configObj)
    randomSeed = libosd.configUtils.getConfigParam("randomSeed", configObj)
    testFname = libosd.configUtils.getConfigParam("testDataFile", configObj)
    trainFname = libosd.configUtils.getConfigParam("trainDataFile", configObj)
    fixedTestEventsLst = libosd.configUtils.getConfigParam("fixedTestEvents", configObj)
    fixedTrainEventsLst = libosd.configUtils.getConfigParam("fixedTrainEvents", configObj)
 
    print("Loading all data")
    osd = libosd.osdDbConnection.OsdDbConnection(debug=debug)
    for fname in configObj['dataFiles']:
        print("Loading OSDB File: %s" % fname)
        eventsObjLen = osd.loadDbFile(fname)
        print("Loaded %d events" % eventsObjLen)


    # Remove specified invalid events
    eventIdsLst = osd.getEventIds()
    print("A total of %d events read from database" % len(eventIdsLst))
    if (invalidEvents is not None):
        print("Removing invalid events...")
        osd.removeEvents(invalidEvents)
        eventIdsLst = osd.getEventIds()
        print("%d events remaining after removing invalid events" % len(eventIdsLst))

    # Remove Excluded users
    print("Removing data for excluded users.  ExcludedUserIds=",excludedUserIds)
    osd.excludeUserIds(excludedUserIds)
    eventIdsLst = osd.getEventIds()
    print("%d events remaining after removing excluded users" % len(eventIdsLst))

    # Retain included users
    print("Retaining specified included users.  IncludedUserIds=", includeUserIds)
    osd.includeUserIds(includeUserIds, debug=False)
    eventIdsLst = osd.getEventIds()
    print("%d events remaining after retaining included users" % len(eventIdsLst))

    # Retain included types
    print("Retaining specified types.  includeTypes=", includeTypes)
    osd.includeTypes(includeTypes, debug=False)
    eventIdsLst = osd.getEventIds()
    print("%d events remaining after retaining included types" % len(eventIdsLst))

    # Retain included subtypes
    print("Retaining specified subtypes.  includeTypes=", includeSubTypes)
    osd.includeSubTypes(includeSubTypes, debug=False)
    eventIdsLst = osd.getEventIds()
    print("%d events remaining after retaining included subtypes" % len(eventIdsLst))


    if (fixedTestEventsLst is not None):
        print("Removing fixed test events")
        eventIdsLst = osd.getEventIds()
        for testId in fixedTestEventsLst:
            print(testId)
            eventIdsLst.remove(testId)

    if (fixedTrainEventsLst):
        print("Removing fixed training events")
        eventIdsLst = osd.getEventIds()
        for trainId in fixedTrainEventsLst:
            print(trainId)
            eventIdsLst.remove(trainId)


    print("Preparing new test/train dataset")

    print("getTestTrainData(): Splitting data by Event")
    # Split into test and train data sets.
    if (debug): print("Total Events=%d" % len(eventIdsLst))

    # Split events list into test and train data sets.
    trainIdLst, testIdLst =\
        sklearn.model_selection.train_test_split(eventIdsLst,
                                                test_size=testProp,
                                                random_state=randomSeed)
    if (debug): print("len(train)=%d, len(test)=%d" % (len(trainIdLst), len(testIdLst)))
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


    print("Saving Training Data File")
    trainFname = configObj['trainDataFile']
    osd.saveEventsToFile(trainIdLst, trainFname, pretty=False, useCacheDir=False)
    print("Training Data written to file %s" % trainFname)

    print("Saving Test Data File")
    osd.saveEventsToFile(testIdLst, testFname, pretty=False, useCacheDir=False)
    print("Test Data written to file %s" % testFname)

 

def main():
    print("createTestTrainDataFiles.main()")
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

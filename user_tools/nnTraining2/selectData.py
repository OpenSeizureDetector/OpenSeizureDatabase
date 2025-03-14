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



def selectData(configObj, debug=False):
    """
    Using the osdb 'dataFiles' specified in configObj, load all the available seizure and non-seizure
    data, and filter it bsaed on the event filters in teh configuration object.  Save the filtered events to a single file
    The configuration is specified in the configObj dict.   The following configObj elements
    are used:
       - osdbCfg - file name of separate json configuration file which will be included
       - dataFiles - list of osdb data files to use to create data set
       - invalidEvents - list of event IDs to exclude from the datasets
       - eventFilters - dictionary specifying filters to apply to select required data.
       - allDataFileJson - filename to use to save the filtered data set (relative to current working directory)
    """
    if (debug): print("getTestTrainData: configObj=",configObj.keys())
    invalidEvents = libosd.configUtils.getConfigParam("invalidEvents", configObj)
    allDataFname = libosd.configUtils.getConfigParam("allDataFileJson", configObj)
    filterCfg = configObj['eventFilters']
    dbDir = libosd.configUtils.getConfigParam("cacheDir", configObj) 

    print("Loading all data")
    osd = libosd.osdDbConnection.OsdDbConnection(cacheDir=dbDir, debug=debug)
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


    filterCfg = configObj['eventFilters']
    print("filterCfg=", filterCfg)
    
    eventIdsLst = osd.getFilteredEventsLst(
            includeUserIds = filterCfg['includeUserIds'],
            excludeUserIds = filterCfg['excludeUserIds'],
            includeTypes = filterCfg['includeTypes'],
            excludeTypes = filterCfg['excludeTypes'],
            includeSubTypes = filterCfg['includeSubTypes'],
            excludeSubTypes = filterCfg['excludeSubTypes'],
            includeDataSources = filterCfg['includeDataSources'],
            excludeDataSources = filterCfg['excludeDataSources'],
            includeText = filterCfg['includeText'],
            excludeText = filterCfg['excludeText'],
            require3dData= filterCfg['require3dData'],
            requireHrData= filterCfg['requireHrData'],
            requireO2SatData= filterCfg['requireO2SatData'],
            debug = True

    )

    print("%d events remaining after applying filters" % len(eventIdsLst))
    #print(eventIdsLst)
    
    print("Total Number of Events = %d" % len(eventIdsLst))

    # 
    # Save the filtered events set.   
    print("Saving filtered data to file %s" % allDataFname)
    osd.saveEventsToFile(eventIdsLst, allDataFname, pretty=False, useCacheDir=False)
    print("Filtered Data written to file %s" % allDataFname)

    print("selectData - Done") 

def main():
    print("selectData.main()")
    parser = argparse.ArgumentParser(description='Select OSDB Data based on filter criteria')
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

    
    selectData(configObj, args['debug'])
        
    


if __name__ == "__main__":
    main()

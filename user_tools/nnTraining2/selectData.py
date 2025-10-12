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



def selectData(configObj, outDir=".", debug=False):
    """
    Using the osdb 'dataFiles' specified in configObj, load all the available seizure and non-seizure
    data, and filter it bsaed on the event filters in teh configuration object.  Save the filtered events to a single file in output directory outDir.
    The configuration is specified in the configObj dict.   The following configObj elements
    are used:
       - osdbCfg - file name of separate json configuration file which will be included
       - dataFiles - list of osdb data files to use to create data set
       - invalidEvents - list of event IDs to exclude from the datasets
       - eventFilters - dictionary specifying filters to apply to select required data.
       - allDataFileJson - filename to use to save the filtered data set (relative to current working directory)
    """
    if (debug): print("selectData: configObj=",configObj.keys())
    if ("invalidEvents" in configObj['osdbConfig']):
        print("selectData: Using invalid events from configObj")
        invalidEvents = configObj['osdbConfig']['invalidEvents']
    else:
        invalidEvents = None
    allDataFname = configObj['dataFileNames'].get('allDataFileJson', None)
    filterCfg = configObj['eventFilters']
    if ("cacheDir" in configObj['osdbConfig']):
        print("selectData: Using cache directory from configObj")
        dbDir = configObj['osdbConfig']['cacheDir']
    else:
        print("selectData: Using default cache directory")
        dbDir = None
    dbDir = libosd.configUtils.getConfigParam("cacheDir", configObj) 

    print("selectData: Loading all data from osdb folder %s" % dbDir)
    osd = libosd.osdDbConnection.OsdDbConnection(cacheDir=dbDir, debug=debug)
    for fname in configObj['osdbConfig']['osdbFiles']:
        print("selectData: Loading OSDB File: %s" % fname)
        eventsObjLen = osd.loadDbFile(fname)
        print("selectData: Loaded %d events" % eventsObjLen)
        # Debug: report seizure/non-seizure counts after each file is loaded
        if debug:
            try:
                events_now = osd.getAllEvents(includeDatapoints=False)
                total_now = len(events_now)
                seizures_now = sum(1 for e in events_now if str(e.get('type','')).lower() == 'seizure')
                non_seizures_now = total_now - seizures_now
                print(f"selectData: After loading {fname}: total events={total_now}, seizures={seizures_now}, non-seizures={non_seizures_now}")
            except Exception as e:
                print(f"selectData: debug count failed after loading {fname}: {e}")


    # Remove specified invalid events
    eventIdsLst = osd.getEventIds()
    print("selectData: A total of %d events read from database" % len(eventIdsLst))
    if (invalidEvents is not None):
        print("selectData: Removing invalid events...")
        osd.removeEvents(invalidEvents)
        eventIdsLst = osd.getEventIds()
        print("selectData: %d events remaining after removing invalid events" % len(eventIdsLst))


    filterCfg = configObj['eventFilters']
    print("selectData: filterCfg=", filterCfg)
    
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
            debug = debug

    )

    print("selectData: %d events remaining after applying filters" % len(eventIdsLst))
    #print(eventIdsLst)
    
    print("selectData: Total Number of Events = %d" % len(eventIdsLst))

    # Debug: report seizure/non-seizure counts after filtering
    if debug:
        try:
            filtered_events = osd.getEvents(eventIdsLst, includeDatapoints=False)
            total_filtered = len(filtered_events)
            seizures_filtered = sum(1 for e in filtered_events if str(e.get('type','')).lower() == 'seizure')
            non_seizures_filtered = total_filtered - seizures_filtered
            print(f"selectData: After filtering: total={total_filtered}, seizures={seizures_filtered}, non-seizures={non_seizures_filtered}")
        except Exception as e:
            print(f"selectData: debug filtered count failed: {e}")

    # 
    # Save the filtered events set.   
    allDataFnamePath = os.path.join(outDir, allDataFname)
    print("selectData: Saving filtered data to file %s" % allDataFnamePath)
    osd.saveEventsToFile(eventIdsLst, allDataFnamePath, pretty=False, useCacheDir=False)
    print("selectData: Filtered Data written to file %s" % allDataFnamePath)

    print("selectData - Done") 

def main():
    print("selectData.main()")
    parser = argparse.ArgumentParser(description='Select OSDB Data based on filter criteria')
    parser.add_argument('--config', default="nnConfig.json",
                        help='name of json file containing configuration')
    parser.add_argument('--outDir', default=".",
                        help='name of directory to write output files')
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

    
    selectData(configObj, outDir=args['outDir'], debug=args['debug'])
        
    


if __name__ == "__main__":
    main()

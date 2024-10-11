#!/usr/bin/env python

import sys
import os
import json
import argparse

from pymongo import MongoClient 

sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
#import libosd.analyse_event
import libosd.webApiConnection
import libosd.osdDbConnection
import libosd.osdAppConnection
import libosd.dpTools
import libosd.configUtils

def loadData(configObj, debug=False):
    print("loadData - configObj="+json.dumps(configObj))
    if ('dbDir' in configObj.keys()):
        dbDir = configObj['dbDir']
    else:
        dbDir = None

    invalidEvents = configObj['invalidEvents']
    print("invalid events", invalidEvents)

    # Load each of the three events files (tonic clonic seizures,
    #all seizures and false alarms).
    osd = libosd.osdDbConnection.OsdDbConnection(cacheDir=dbDir, debug=debug)
    for fname in configObj['dataFiles']:
        eventsObjLen = osd.loadDbFile(fname)
        print("loaded %d events from file %s" % (eventsObjLen, fname))
    osd.removeEvents(invalidEvents)
    osd.listEvents() 
    return osd

def writeToDb(osd): 
    # Making Connection
    myclient = MongoClient("mongodb://localhost:27017/") 
    
    # database 
    db = myclient["OSDB"]
    
    # Created or Switched to collection 
    # names: GeeksForGeeks
    Collection = db["events"]

    eventIds = osd.getEventIds()
    eventLst = osd.getEvents(eventIds)
    #print(eventLst)    
        
    Collection.insert_many(eventLst)  

def listDbEvents():
    print("listDbEvents")
    # Making Connection
    myclient = MongoClient("mongodb://localhost:27017/") 
    
    # database 
    db = myclient["OSDB"]
    
    # Created or Switched to collection 
    # names: GeeksForGeeks
    Collection = db["events"]

    cursor = Collection.find()

    for doc in cursor:
        print(doc)
        print()


   

def main():
    print("importOsdb.main()")
    parser = argparse.ArgumentParser(description='Import OSDB into Mongodb database')
    parser.add_argument('--config', default="config.json",
                        help='name of json file containing db configuration')
    #parser.add_argument('--out', default="output",
    #                    help='name of output CSV file')
    parser.add_argument('--debug', action="store_true",
                        help='Write debugging information to screen')
    argsNamespace = parser.parse_args()
    args = vars(argsNamespace)
    print(args)

    configObj = libosd.configUtils.loadConfig(args['config'])
    print("configObj=",configObj)
    # Load a separate OSDB Configuration file if it is included.
    if ("osdbCfg" in configObj):
        osdbCfgFname = libosd.configUtils.getConfigParam("osdbCfg",configObj)
        print("Loading separate OSDB Configuration File %s." % osdbCfgFname)
        osdbCfgObj = libosd.configUtils.loadConfig(osdbCfgFname)
        # Merge the contents of the OSDB Configuration file into configObj
        configObj = configObj | osdbCfgObj

    print("configObj=",configObj)

    osdb = loadData(configObj, args['debug'])
    writeToDb(osdb)
    


if __name__ == "__main__":
    main()

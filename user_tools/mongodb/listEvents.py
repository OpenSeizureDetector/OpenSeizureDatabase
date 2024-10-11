#!/usr/bin/env python

import sys
import os
import json
import argparse

from pymongo import MongoClient 



def listDbEvents():
    print("listDbEvents")
    # Making Connection
    myclient = MongoClient("mongodb://localhost:27017/") 
    
    # database 
    db = myclient["OSDB"]
    
    # Created or Switched to collection 
    # names: GeeksForGeeks
    Collection = db["events"]

    cursor = Collection.find({"type":"Seizure"})

    for doc in cursor:
        print("%s, %s, %s" % (doc['id'], doc['type'], doc['desc']))


   

def main():
    print("listEvents.main()")
    parser = argparse.ArgumentParser(description='List events in Mongodb database')
    #parser.add_argument('--config', default="config.json",
    #                    help='name of json file containing db configuration')
    #parser.add_argument('--out', default="output",
    #                    help='name of output CSV file')
    parser.add_argument('--debug', action="store_true",
                        help='Write debugging information to screen')
    argsNamespace = parser.parse_args()
    args = vars(argsNamespace)
    print(args)

    listDbEvents()    


if __name__ == "__main__":
    main()

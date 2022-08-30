#!/usr/bin/python3

import sys
import json

def loadConfig(configFname):
    # Opening JSON file
    try:
        f = open(configFname)
        print("Opened File")
        configObj = json.load(f)
        f.close()
        print("configObj=",configObj)
    except BaseException as e:
        print("Error Opening File %s" % configFname)
        print("OS error: {0}".format(e))
        print(sys.exc_info())
        configObj = None
    return configObj

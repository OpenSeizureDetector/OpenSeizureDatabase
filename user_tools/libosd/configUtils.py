#!/usr/bin/env python3

import sys
import json

def loadConfig(configFname):
    '''
    loadConfig : load a configuration JSON file from disk and return the parsed object (dict).

    Parameters
    ----------
    configFname : String
        File path of configuration file to load.

    Returns
    -------
    dict
        A dictionary representing the loaded file.
    '''
    try:
        f = open(configFname)
        #print("Opened File")
        configObj = json.load(f)
        f.close()
        #print("configObj=",configObj)
    except BaseException as e:
        print("Error Opening File %s" % configFname)
        print("OS error: {0}".format(e))
        print(sys.exc_info())
        configObj = None
    return configObj

def getConfigParam(param,configObj):
    '''
    getConfigParam : returns the value of the given parameter in configObj, or None if the parameter 
    does not exist.
    Note:  This only looks for top level keys in the dictionary - it does not look for nested parameters.

    Parameters
    ----------
    param : String
        name of parameter to return
    configObj : dict
        A dictionary of configuration parameters.

    Returns
    -------
    _type_
        _description_
    '''    
    if (param in configObj.keys()):
        return configObj[param]
    else:
        return None

# Tools for manipulating datapoints

import dateutil.parser
import json

def dateStr2secs(dateStr):
    ''' Convert a string representation of date/time into seconds from
    the start of 1970 (standard unix timestamp)
    '''
    parsed_t = dateutil.parser.parse(dateStr)
    return parsed_t.timestamp()


def dp2rawData(dp, debug=False):
    '''Accepts a dataPoint object from the osd Database and converts it into
    a 'raw data' JSON string that would have been received by the phone to
    create it.
    This is useful to reproduce the exact phone behaviour from datapoints
    stored in the database.
    '''
    if (debug): print("dp2rawData: dp=",dp)
    if ('dataTime' in dp):
        currTs = dateStr2secs(dp['dataTime'])
    else:
        currTs = None

    dataObj = None
    if ('rawData') in dp.keys():
        if (debug): print("V2 style datapoint object")
        # This is for the new style database that avoids dataJSON strings.
        dataObj = dp
    else:
        if ('dataJSON' in dp):
            dpObj = json.loads(dp['dataJSON'])
            if ('dataJSON' in dpObj):
                dataObj = json.loads(dpObj['dataJSON'])

    # if we do not have valid data, return a None object.
    if (dataObj is None):
        return None
    
    # Otherwise we try to parse the object.
    try:
        #if (debug): print("dataObj=",dataObj)
        # Create raw data list
        accelLst = []
        accelLst3d = []
        # FIXME:  It is not good to hard code the length of an array!
        for n in range(0,125):
            accelLst.append(dataObj['rawData'][n])
            if ("rawData3D" in dataObj.keys()):
                #print("3dData present")
                accelLst3d.append(dataObj['rawData3D'][n*3])
                accelLst3d.append(dataObj['rawData3D'][n*3 + 1])
                accelLst3d.append(dataObj['rawData3D'][n*3 + 2])

        rawDataObj = {"dataType": "raw", "Mute": 0}
        rawDataObj['HR'] = dataObj['hr']
        rawDataObj['data'] = accelLst
        rawDataObj['data3D'] = accelLst3d
        # FIXME - add o2sat
        dataJSON = json.dumps(rawDataObj)
    except (json.decoder.JSONDecodeError, TypeError):
        print("ERROR Decoding JSON String")
        print(dp)
        dataJSON = None
        raise
    
    return dataJSON

def dp2rawData_old(dp):
    '''Accepts a dataPoint object from the osd Database and converts it into
    a 'raw data' JSON string that would have been received by the phone to
    create it.
    This is useful to reproduce the exact phone behaviour from datapoints
    stored in the database.
    FIXME - move this to libosd?
    '''
    if ('dataTime' in dp):
        currTs = dateStr2secs(dp['dataTime'])
    else:
        currTs = None
        
    if ('dataJSON' in dp):
        dpObj = json.loads(dp['dataJSON'])
    else:
        dpObj = None
    if ('dataJSON' in dpObj):
        try:
            dataObj = json.loads(dpObj['dataJSON'])

            # Create raw data list
            accelLst = []
            accelLst3d = []
            # FIXME:  It is not good to hard code the length of an array!
            for n in range(0,125):
                accelLst.append(dataObj['rawData'][n])
                if ("data3D" in dataObj.keys()):
                    print("3dData present")
                    accelLst3d.append(dataObj['rawData3D'][n*3])
                    accelLst3d.append(dataObj['rawData3D'][n*3 + 1])
                    accelLst3d.append(dataObj['rawData3D'][n*3 + 2])

            rawDataObj = {"dataType": "raw", "Mute": 0}
            rawDataObj['HR'] = dataObj['HR']
            rawDataObj['data'] = accelLst
            rawDataObj['data3D'] = accelLst3d
            # FIXME - add o2sat
            dataJSON = json.dumps(rawDataObj)
        except (json.decoder.JSONDecodeError, TypeError):
            dataJSON = None
    else:
        dataJSON = None
    return dataJSON


def getAccelDataFromJson(jsonStr):
    if (jsonStr is not None):
        jsonObj = json.loads(jsonStr)
        accData = jsonObj['data']
        hrVal = jsonObj['HR']
    else:
        accData = None
        hrVal = None
    return(accData,hrVal)

    

def getParamFromDp(paramStr, dp):
    # New style format of data is easy - no dataJSON
    if (paramStr in dp.keys()):
        return(dp[paramStr])
    else:
        return(None)

    # Old Code before we expanded the dataJSON strings in the deliverable database.
    if not 'dataJSON' in dp.keys():
        print("ERROR:  getParamFromDp - searching for %s: dpDoes not contain dataJSON element" % paramStr)
        return None
    jsonStr = dp['dataJSON']
    jsonObj = json.loads(jsonStr)

    if not 'dataJSON' in jsonObj.keys():
        print("ERROR:  getParamFromDp - dp.dataJSON Does not contain dataJSON element")
        return None
    jsonStr2 = jsonObj['dataJSON']
    jsonObj2 = json.loads(jsonStr2)

    
    if (paramStr) in jsonObj2.keys():
        return(jsonObj2[paramStr])
    else:
        print("ERROR: getParamFromDp - parameter %s not found in dataJSON"
              % paramStr)
        print(dp, dp.keys())
        print(jsonObj2, jsonObj2.keys())
        return None

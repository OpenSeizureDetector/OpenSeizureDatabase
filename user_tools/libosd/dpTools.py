# Tools for manipulating datapoints

import dateutil.parser
import json

def dateStr2secs(dateStr):
    ''' Convert a string representation of date/time into seconds from
    the start of 1970 (standard unix timestamp)
    '''
    parsed_t = dateutil.parser.parse(dateStr)
    return parsed_t.timestamp()


def dp2rawData(dp):
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
            rawDataObj['HR'] = dataObj['hr']
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

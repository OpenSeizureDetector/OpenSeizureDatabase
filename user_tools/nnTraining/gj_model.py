#!/usr/bin/python3

import argparse
import sys
import os
import json
import importlib
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import libosd.osdDbConnection
import libosd.dpTools
import libosd.osdAlgTools


def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)




def dp2vector(dp):
    '''Convert a datapoint object into an input vector to be fed into the neural network.
    '''
    dpInputData = []
    rawDataStr = libosd.dpTools.dp2rawData(dp)
    accData, hr = libosd.dpTools.getAccelDataFromJson(rawDataStr)
    print(accData, hr)

    return dpInputData


def trainModel(configObj, outFile="model.pkl", debug=False):
    print("trainModel - configObj="+json.dumps(configObj))

    invalidEvents = configObj['invalidEvents']
    print("invalid events", invalidEvents)

    # Load each of the three events files (tonic clonic seizures,
    # all seizures and false alarms).
    osdTc = libosd.osdDbConnection.OsdDbConnection(debug=debug)
    eventsObjLen = osdTc.loadDbFile(configObj['tcSeizuresFname'])
    print("tcSeizures  eventsObjLen=%d" % eventsObjLen)
    osdTc.removeEvents(invalidEvents)
    osdTc.listEvents()

    osdAll = libosd.osdDbConnection.OsdDbConnection(debug=debug)
    eventsObjLen = osdAll.loadDbFile(configObj['allSeizuresFname'])
    osdAll.removeEvents(invalidEvents)
    print("all Seizures eventsObjLen=%d" % eventsObjLen)

    osdFalse = libosd.osdDbConnection.OsdDbConnection(debug=debug)
    eventsObjLen = osdFalse.loadDbFile(configObj['falseAlarmsFname'])
    osdFalse.removeEvents(invalidEvents)
    print("false alarms eventsObjLen=%d" % eventsObjLen)

    
    # Run each event through each algorithm
    tcTest, tcTrain = getTestTrainData(osdTc)
    #allSeizureResults, allSeizureResultsStrArr = testEachEvent(osdAll, algs)
    #falseAlarmResults, falseAlarmResultsStrArr = testEachEvent(osdFalse, algs)
    #results = falseAlarmResults

    model = make_model(input_shape=x_train.shape[1:])
    keras.utils.plot_model(model, show_shapes=True)
    return(model)
    


def getTestTrainData(osd, trainProp=0.7):
    """
    for each event in the OsdDbConnection 'osd', create a set of rows 
    of training data for the model - one row per datapoint.
    Returns the data as (test, train) split by the trainProp proportions.
    FIXME:  Filter datapoints to only use those within a specified time of the
    event time (to make sure we really capture seizure data and not
    normal data before or after the seizure)
    """
    # Now we loop through each event in the eventsList
    eventIdsLst = osd.getEventIds()
    nEvents = len(eventIdsLst)
    outArr = []
    for eventNo in range(0,nEvents):
        eventId = eventIdsLst[eventNo]
        print("Analysing event %s" % eventId)
        eventObj = osd.getEvent(eventId, includeDatapoints=True)
        eventResultsStrArr = []
        for dp in eventObj['datapoints']:
            dpInputData = dp2vector(dp)
            # FIXME  Decide whether to use this datapoint, then create
            # an array of data representing the datapoint if it is to be used.
            outArr.append(dpInputData)
            
    print(outArr)
    # FIXME - split into test and train datasets.
    testArr= outArr
    trainArr = outArr
    return(testArr, trainArr)
    

    

def main():
    print("gj_model.main()")
    parser = argparse.ArgumentParser(description='Seizure Detection Test Runner')
    parser.add_argument('--config', default="osdbConfig.json",
                        help='name of json file containing test configuration')
    parser.add_argument('--out', default="model.pkl",
                        help='name of output CSV file')
    parser.add_argument('--debug', action="store_true",
                        help='Write debugging information to screen')
    argsNamespace = parser.parse_args()
    args = vars(argsNamespace)
    print(args)


    inFile = open(args['config'],'r')
    configObj = json.load(inFile)
    inFile.close()
    trainModel(configObj, args['out'], args['debug'])
    


if __name__ == "__main__":
    main()

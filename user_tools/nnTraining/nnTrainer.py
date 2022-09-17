#!/usr/bin/python3

import argparse
import sys
import os
import json
import importlib
import sklearn.model_selection
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import libosd.osdDbConnection
import libosd.dpTools
import libosd.osdAlgTools

def type2id(typeStr):
    if typeStr.lower() == "seizure":
        id = 1
    elif typeStr.lower() == "false alarm":
        id = 0
    else:
        id = 2
    return id

def make_model(input_shape, num_classes):
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
    #print(accData, hr)

    if (accData is not None):
        for n in range(0,len(accData)):
            dpInputData.append(accData[n])
    else:
        print("*** Error in Datapoint: ", dp)
        print("*** No celeration data found with datapoint.")
        print("*** I recommend adding event %s to the invalidEvents list in the configuration file" % dp['eventId'])
        exit(-1)
    return dpInputData

def getTestTrainData(osd, seizureTimeRange = None, trainProp=0.7):
    """
    for each event in the OsdDbConnection 'osd', create a set of rows 
    of training data for the model - one row per datapoint.
    Returns the data as (test, train) split by the trainProp proportions.
    if seizureTimeRange is not None, it should be an array [min, max]
    which is the time range in seconds from the event time to include datapoints.
    The idea of this is that a seizure event may include datapoints before or
    after the seizure, which we do not want to include in seizure training data.
    So specifying seizureTimeRange as say [-20, 40] will only include datapoints
    that occur less than 20 seconds before the seizure event time and up to 
    40 seconds after the seizure event time.
    FIXME:  Filter datapoints to only use those within a specified time of the
    event time (to make sure we really capture seizure data and not
    normal data before or after the seizure)
    """
    # Now we loop through each event in the eventsList
    eventIdsLst = osd.getEventIds()
    nEvents = len(eventIdsLst)
    outArr = []
    classArr = []
    for eventNo in range(0,nEvents):
        eventId = eventIdsLst[eventNo]
        eventObj = osd.getEvent(eventId, includeDatapoints=True)
        eventType = eventObj['type']
        print("Processing event %s (type=%s, id=%d)" % (eventId, eventType, type2id(eventType)),type(eventObj['datapoints']))
        sys.stdout.flush()
        if (eventObj['datapoints'] is None):
            print("No datapoints - skipping")
        else:
            #print("nDp=%d" % len(eventObj['datapoints']))
            for dp in eventObj['datapoints']:
                dpInputData = dp2vector(dp)
                eventTime = eventObj['dataTime']
                dpTime = dp['dataTime']
                eventTimeSec = libosd.dpTools.dateStr2secs(eventTime)
                dpTimeSec = libosd.dpTools.dateStr2secs(dpTime)
                timeDiffSec = dpTimeSec - eventTimeSec
                includeDp = True
                if (eventObj['type'].lower() == 'seizure'):
                    if (seizureTimeRange is not None):
                        if (timeDiffSec < seizureTimeRange[0]):
                            includeDp=False
                        if (timeDiffSec > seizureTimeRange[1]):
                            includeDp=False

                if (includeDp):
                    print("%s, %s - diff=%.1f" % (eventTime, dpTime, timeDiffSec))
                    outArr.append(dpInputData)
                    classArr.append(type2id(eventType))
                else:
                    #print("Out of Time Range - skipping")
                    pass

    # Split into test and train data sets.
    outTrain, outTest, classTrain, classTest =\
        sklearn.model_selection.train_test_split(outArr, classArr,
                                                 test_size=0.25,
                                                 random_state=4,
                                                 stratify=classArr)

    #print(outTrain, outTest, classTrain, classTest)
    # Convert into numpy arrays
    outTrainArr = np.array(outTrain)
    classTrainArr = np.array(classTrain)
    outTestArr = np.array(outTest)
    classTestArr = np.array(classTest)
    return(outTrainArr, outTestArr, classTrainArr, classTestArr)
    



def trainModel(configObj, outFile="model.pkl", debug=False):
    print("trainModel - configObj="+json.dumps(configObj))

    invalidEvents = configObj['invalidEvents']
    print("invalid events", invalidEvents)
    if ('seizureTimeRange' in configObj):
        seizureTimeRange = configObj['seizureTimeRange']
    else:
        seizureTimeRange = None

    # Load each of the three events files (tonic clonic seizures,
    # all seizures and false alarms).

    print("Loading all seizures data")
    osdAllData = libosd.osdDbConnection.OsdDbConnection(debug=debug)
    eventsObjLen = osdAllData.loadDbFile(configObj['allSeizuresFname'])
    eventsObjLen = osdAllData.loadDbFile(configObj['falseAlarmsFname'])
    osdAllData.removeEvents(invalidEvents)
    print("all Data eventsObjLen=%d" % eventsObjLen)


    # Run each event through each algorithm
    xTrain, xTest, yTrain, yTest = getTestTrainData(osdAllData,seizureTimeRange)

    xTrain = xTrain.reshape((xTrain.shape[0], xTrain.shape[1], 1))
    xTest = xTest.reshape((xTest.shape[0], xTest.shape[1], 1))
    nClasses = len(np.unique(yTrain))
    print("nClasses=%d" % nClasses)
    print("Training using %d seizure datapoints and %d false alarm datapoints"
          % (np.count_nonzero(yTrain == 1),
             np.count_nonzero(yTrain == 0)))
    print("Testing using %d seizure datapoints and %d false alarm datapoints"
          % (np.count_nonzero(yTest == 1),
             np.count_nonzero(yTest == 0)))

    
    model = make_model(input_shape=xTrain.shape[1:], num_classes=nClasses)
    keras.utils.plot_model(model, show_shapes=True)

    epochs = 500
    batch_size = 32

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "best_model.h5", save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    ]
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    history = model.fit(
        xTrain,
        yTrain,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_split=0.2,
        verbose=1,
    )

    # After training, load the best model back from disk and test it.
    model = keras.models.load_model("best_model.h5")

    test_loss, test_acc = model.evaluate(xTest, yTest)

    print("nClasses=%d" % nClasses)
    print("Trained using %d seizure datapoints and %d false alarm datapoints"
          % (np.count_nonzero(yTrain == 1),
             np.count_nonzero(yTrain == 0)))
    print("Tesing using %d seizure datapoints and %d false alarm datapoints"
          % (np.count_nonzero(yTest == 1),
             np.count_nonzero(yTest == 0)))


    
    print("Test accuracy", test_acc)
    print("Test loss", test_loss)

    metric = "sparse_categorical_accuracy"
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.show()
    plt.close()



    
    return(model)
    



    

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

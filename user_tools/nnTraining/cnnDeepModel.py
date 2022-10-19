#!/usr/bin/env python

'''
nnModel is an abstract class to describe a generic seizure detection neural network
model.
It should be sub-classed to define the particular model geometry and provide
the function to convert a datapoint into a model input tensor.
'''
import sys
import os
import numpy as np
import keras

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libosd.dpTools

import nnModel

class CnnModel(nnModel.NnModel):
    def __init__(self):
        print("NnModel Constructor")

    def makeModel(self, input_shape, num_classes):
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

        conv4 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv3)
        conv4 = keras.layers.BatchNormalization()(conv4)
        conv4 = keras.layers.ReLU()(conv4)

        conv5 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv4)
        conv5 = keras.layers.BatchNormalization()(conv5)
        conv5 = keras.layers.ReLU()(conv5)

        gap = keras.layers.GlobalAveragePooling1D()(conv5)

        output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

        return keras.models.Model(inputs=input_layer, outputs=output_layer)

    def dp2vector(self, dpObj, normalise=False):
        '''Convert a datapoint object into an input vector to be fed into the neural network.   Note that if dp is not a dict, it is assumed to be a json string
        representation instead.
        if normalise is True, applies Z normalisation to accelerometer data
        to give a mean of zero and standard deviation of unity.
        https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-normalize-or-standardize-a-dataset-in-python.md
        '''
        dpInputData = []
        if (type(dpObj) is dict):
            rawDataStr = libosd.dpTools.dp2rawData(dpObj)
        else:
            rawDataStr = dpObj
        accData, hr = libosd.dpTools.getAccelDataFromJson(rawDataStr)
        #print(accData, hr)

        if (accData is not None):
            if (normalise):
                accArr = np.array(accData)
                accArrNorm = (accArr - np.average(accArr)) / (np.std(accArr))
                accData = accArrNorm.tolist()
            for n in range(0,len(accData)):
                dpInputData.append(accData[n])
        else:
            print("*** Error in Datapoint: ", dpObj)
            print("*** No acceleration data found with datapoint.")
            print("*** I recommend adding event %s to the invalidEvents list in the configuration file" % dp['eventId'])
            exit(-1)

        return dpInputData
        
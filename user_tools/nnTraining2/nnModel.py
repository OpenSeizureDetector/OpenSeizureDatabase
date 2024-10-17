#!/usr/bin/env python

'''
nnModel is an abstract class to describe a generic seizure detection neural network
model.
It should be sub-classed to define the particular model geometry and provide
the function to convert a datapoint into a model input tensor.
'''

class NnModel:
    def __init__(self):
        print("NnModel Constructor")

    def makeModel(self):
        print("NnModel.makeModel() - FIXME - Define this function to do something!")

    def dp2vector(self, dpObj, normalise=False):
        print("NnModel.dp2vector() - FIXME - Define this function to do something!")

    
#!/usr/bin/env python

'''
skModel is an abstract class to describe a generic seizure detection scikit-learn model
model.
It should be sub-classed to define the particular model geometry and provide
the function to convert a row of the training data into a model input array
.
ConfigObj is an optional dictionary of configuration parameters.
'''

class SkModel:
    def __init__(self, configObj=None, debug=False):
        self.configObj=configObj
        self.debug=debug
        print("SkModel Constructor")

    def makeModel(self):
        print("SkModel.makeModel() - FIXME - Define this function to do something!")

    def dp2vector(self, dpObj, normalise=False):
        print("SkModel.dp2vector() - FIXME - Define this function to do something!")

    
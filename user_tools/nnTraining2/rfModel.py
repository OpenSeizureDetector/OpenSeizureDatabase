#!/usr/bin/env python

'''
rfModel.py - Random Forest Model Class for Sklearn
This file defines the RfModel class which is used to create and manage a Random Forest model using the Sklearn library.
'''

class RfModel:
    def __init__(self, configObj=None, debug=False):
        self.configObj=configObj
        self.debug=debug
        print("RfModel Constructor - configObj=", configObj)

    def makeModel(self):
        print("RfModel.makeModel() - FIXME - Define this function to do something!")

    def dp2vector(self, dpObj, normalise=False):
        print("RfModel.dp2vector() - FIXME - Define this function to do something!")

    
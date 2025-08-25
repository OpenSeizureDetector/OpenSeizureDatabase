#!/usr/bin/env python


'''
rfModel.py - Random Forest Model Class for Sklearn

This file defines the RfModel class which is used to create and manage a Random Forest model using the Sklearn library.

ConfigObj Usage:
    configObj is expected to be a dictionary containing model parameters:
        - 'classWeights': (optional) dictionary mapping class labels to weights, e.g. {0: 1.0, 1: 2.0}
        - 'n_estimators': (optional) number of trees in the forest (default: 100)
        - 'max_depth': (optional) maximum depth of the trees (default: None)
    Other configObj keys may be used for data directories, debug flags, etc.
'''

import os
import joblib
import sklearn
import sklearn.utils.class_weight
import sklearn.ensemble
import numpy as np

from user_tools.nnTraining2 import skModel


class RfModel(skModel.SkModel):
    TAG = "rfModel.RfModel"
    modelFnameDefault = 'rfModel.joblib'

    def __init__(self, dataDir = '.', configObj=None, debug=False):
        """
        Initialize the RfModel object.
        Args:
            dataDir (str): Directory for saving/loading model files.
            configObj (dict, optional): Configuration dictionary (see top-level comment).
            debug (bool, optional): Enable debug output.
        """
        self.configObj = configObj
        self.debug = debug
        self.model = None
        self.dataDir = dataDir
        print("RfModel Constructor - configObj=", configObj)





    def fit(self, xTrain, yTrain):
        """
        Train the Random Forest model using the provided training data.
        Args:
            xTrain: Training feature vectors (array-like).
            yTrain: Training labels (array-like).

        Uses configObj for model parameters:
            - 'classWeights': (optional) dictionary mapping class labels to weights.
            - 'n_estimators': (optional) number of trees in the forest.
            - 'max_depth': (optional) maximum depth of the trees.
        """
        classWeights = None
        # Get class weights from configObj if provided
        if 'classWeights' in self.configObj:
            classWeightsStr = self.configObj['classWeights']
            classWeights = {int(k): v for k, v in classWeightsStr.items()}  
        else:
            print("%s: No class weights defined in configObj - using default weights" % self.TAG)
            classWeights = sklearn.utils.class_weight.compute_class_weight(
                'balanced', np.unique(yTrain), yTrain)

        print("%s: Using class weights: %s" % (self.TAG, classWeights))

        # Print training data statistics
        print("\n%s: Training using %d seizure datapoints and %d false alarm datapoints"
            % (self.TAG, np.count_nonzero(yTrain == 1),
            np.count_nonzero(yTrain == 0)))

        # Get model hyperparameters from configObj
        n_estimators = self.configObj.get('n_estimators', 100)
        max_depth = self.configObj.get('max_depth', None)

        print("%s: Training using n_estimators=%d, max_depth=%s" % (self.TAG, n_estimators, str(max_depth)))
        # Create the Random Forest model
        self.model = sklearn.ensemble.RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=classWeights,
            random_state=42)

        # Train the model
        self.model.fit(xTrain, yTrain)


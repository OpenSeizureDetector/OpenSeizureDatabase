#!/usr/bin/env python

'''
skModel is an abstract class to describe a generic seizure detection scikit-learn model.
It should be sub-classed to define the particular model geometry and provide
the function to convert a row of the training data into a model input array.

ConfigObj is an optional dictionary of configuration parameters. Typical parameters used include:
    - modelFnameDefault: Default filename for saving/loading the model.
    - Any model-specific hyperparameters (e.g., number of estimators, kernel type, etc.)
    - Feature selection or normalization options.
    - Debug or logging options.

Note: The exact parameters required depend on the subclass implementation and the specific model.
'''

import os
import joblib



class SkModel:
    TAG = "skModel.SkModel"

    def __init__(self, configObj=None, debug=False):
        """
        Initialize the SkModel object.
        Args:
            configObj (dict, optional): Configuration parameters for the model.
            debug (bool, optional): Enable debug output.
        """
        self.configObj = configObj
        self.debug = debug
        self.model = None
        print("SkModel Constructor")

    def makeModel(self):
        """
        Create and initialize the scikit-learn model.
        This method should be overridden in subclasses to define the model architecture.
        """
        print("SkModel.makeModel() - FIXME - Define this function to do something!")

    def dp2vector(self, dpObj, normalise=False):
        """
        Convert a data point object to a feature vector for model input.
        Args:
            dpObj: Data point object (e.g., row from training data).
            normalise (bool): Whether to normalize the vector.
        Returns:
            Feature vector (array-like).
        """
        print("SkModel.dp2vector() - FIXME - Define this function to do something!")

    def fit(self, xTrain, yTrain):
        """
        Train the model using the provided training data.
        Args:
            xTrain: Training feature vectors.
            yTrain: Training labels.
        """
        print("SkModel.fit() - FIXME - Define this function to do something!")

    def save(self, dataDir='.', modelFname=None):
        """
        Save the trained model to disk using joblib.
        Args:
            dataDir (str): Directory to save the model file.
            modelFname (str, optional): Filename for the model. Uses default if None.
        """
        if modelFname is None:
            modelFname = self.modelFnameDefault
        modelFnamePath = os.path.join(dataDir, modelFname)
        print("%s: Saving model to file %s" % (self.TAG, modelFnamePath))
        joblib.dump(self.model, modelFnamePath)
        print("%s: Model saved successfully" % (self.TAG))

    def load(self, dataDir='.', modelFname=None):
        """
        Load a trained model from disk using joblib.
        Args:
            dataDir (str): Directory containing the model file.
            modelFname (str, optional): Filename for the model. Uses default if None.
        """
        if modelFname is None:
            modelFname = self.modelFnameDefault
        modelFnamePath = os.path.join(dataDir, modelFname)
        print("%s: Loading model from file %s" % (self.TAG, modelFnamePath))
        self.model = joblib.load(modelFnamePath)
        print("%s: Model loaded successfully" % (self.TAG))

    def predict(self, xTest):
        """
        Predict labels for the given test data using the trained model.
        Args:
            xTest: Test feature vectors.
        Returns:
            Predicted labels (array-like) or None if model is not loaded.
        """
        if self.model is None:
            print("%s: ERROR: Model is not defined - cannot predict - load or fit the model first." % self.TAG)
            return None
        return self.model.predict(xTest)

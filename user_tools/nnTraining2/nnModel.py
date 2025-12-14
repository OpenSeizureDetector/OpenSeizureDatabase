#!/usr/bin/env python

'''
nnModel is an abstract class to describe a generic seizure detection neural network
model.
It should be sub-classed to define the particular model geometry and provide
the function to convert a datapoint into a model input tensor.
ConfigObj is an optional dictionary of configuration parameters.

This base class now provides framework detection utilities to support both
TensorFlow/Keras and PyTorch implementations.
'''

import os
import sys

class NnModel:
    def __init__(self, configObj=None, debug=False):
        self.configObj = configObj
        self.debug = debug
        self.framework = self._detect_framework(configObj)
        if debug:
            print(f"NnModel Constructor - framework: {self.framework}")

    def _detect_framework(self, configObj):
        """
        Detect which framework to use based on config or availability.
        
        Priority:
        1. Explicit 'framework' in configObj (if provided)
        2. Fall back to TensorFlow for backward compatibility
        
        Returns: 'tensorflow' or 'pytorch'
        """
        if configObj is not None and 'framework' in configObj:
            framework = configObj['framework'].lower()
            if framework in ['tensorflow', 'tf', 'keras']:
                return 'tensorflow'
            elif framework in ['pytorch', 'torch']:
                return 'pytorch'
        
        # Default to TensorFlow for backward compatibility
        return 'tensorflow'

    def get_framework(self):
        """Return the active framework name."""
        return self.framework

    def makeModel(self, input_shape=None, num_classes=2, nLayers=None):
        """
        Abstract method: Create and return the model.
        Subclasses must implement this.
        
        Args:
            input_shape: Shape of input data (framework-specific format)
            num_classes: Number of output classes
            nLayers: Number of layers (model-specific)
            
        Returns:
            Model object (Keras Model or PyTorch nn.Module)
        """
        raise NotImplementedError("Subclasses must implement makeModel()")

    def dp2vector(self, dpObj, normalise=False):
        """
        Abstract method: Convert a datapoint to input vector.
        Subclasses must implement this.
        
        Args:
            dpObj: Datapoint object or dict
            normalise: Whether to normalize the data
            
        Returns:
            Vector representation suitable for model input
        """
        raise NotImplementedError("Subclasses must implement dp2vector()")

    def save_model(self, filepath):
        """
        Save model to file (framework-agnostic).
        Subclasses may override for custom behavior.
        
        Args:
            filepath: Path to save model
        """
        if self.framework == 'tensorflow':
            if hasattr(self, 'model') and self.model is not None:
                self.model.save(filepath)
        elif self.framework == 'pytorch':
            import torch
            if hasattr(self, 'model') and self.model is not None:
                # Save state dict for PyTorch
                state = {
                    'model_state_dict': self.model.state_dict(),
                    'config': self.configObj
                }
                torch.save(state, filepath)
        else:
            raise ValueError(f"Unknown framework: {self.framework}")

    def load_model(self, filepath):
        """
        Load model from file (framework-agnostic).
        Subclasses may override for custom behavior.
        
        Args:
            filepath: Path to load model from
            
        Returns:
            Loaded model object
        """
        if self.framework == 'tensorflow':
            try:
                import keras
            except ImportError:
                from tensorflow import keras
            self.model = keras.models.load_model(filepath)
            return self.model
        elif self.framework == 'pytorch':
            import torch
            if not hasattr(self, 'model') or self.model is None:
                raise RuntimeError("PyTorch model must be created before loading weights")
            checkpoint = torch.load(filepath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            return self.model
        else:
            raise ValueError(f"Unknown framework: {self.framework}")

    
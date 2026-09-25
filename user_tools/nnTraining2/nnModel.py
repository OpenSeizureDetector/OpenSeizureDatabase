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

    # Buffer pre-fill: stationary sensor reading in milli-g (1.0 g).
    STATIONARY_ACC_MILLIG = 1000.0

    def getAccBufSize(self):
        """
        Return the length of the rolling acceleration buffer in samples,
        or 0 if this model does not use a rolling buffer.
        """
        for attr in ('bufferSamples', 'analysisSamp'):
            nBuf = getattr(self, attr, None)
            try:
                if nBuf is not None and int(nBuf) > 0:
                    return int(nBuf)
            except (TypeError, ValueError):
                continue
        return 0

    def prefillAccBuf(self, mode='stationary'):
        """
        Fill the rolling acceleration buffer with synthetic data before the first
        datapoint of an event, so that dp2vector() returns a vector immediately
        instead of returning None until the buffer has filled (which would drop
        the first samples of every event during testing).

        Only intended for test time - training keeps the empty (cold-start)
        buffer so that the model learns to cope with a warm-up period.

        Args:
            mode: 'stationary' fills the buffer with STATIONARY_ACC_MILLIG
                  (i.e. a stationary sensor at 1 g).

        Returns:
            True if the buffer was filled, False if this model has no buffer or
            the requested mode is not supported.
        """
        nBuf = self.getAccBufSize()
        if nBuf <= 0 or not hasattr(self, 'accBuf'):
            return False
        if str(mode).lower() not in ('stationary', 'static'):
            return False
        self.accBuf = [float(self.STATIONARY_ACC_MILLIG)] * nBuf
        return True

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
            checkpoint = torch.load(filepath, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            return self.model
        else:
            raise ValueError(f"Unknown framework: {self.framework}")

    
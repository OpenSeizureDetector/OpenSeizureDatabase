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

    def export_example_inputs(self, batch_size=1):
        """Example inputs for torch.export / ExecuTorch tracing.

        Each model architecture knows its own input geometry (sequence length,
        channel order), which the generic .pt -> .pte converter cannot guess.
        Returns a tuple of CPU float32 tensors shaped (batch, channels,
        total_samples) - exactly the layout the .pte runtime is fed at
        inference time (see nnTester.predict_model, is_pte path) - or None
        if this model does not declare an export layout (the converter then
        falls back to its legacy (1, 1, N) heuristic).

        Subclasses with non-standard input geometry must override this.
        """
        return None
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

    def get_warmup_datapoints(self, samples_per_datapoint=125):
        """Number of leading datapoints of a buffer segment whose model input
        window is not yet fully real data (i.e. still contains pre-fill).

        For a buffer of N samples fed S samples per datapoint, the first
        ceil(N/S)-1 datapoints are partial. Returns 0 for models without a
        rolling buffer. Used to mask warm-up datapoints out of alarm decisions.
        """
        nBuf = self.getAccBufSize()
        if nBuf <= 0:
            return 0
        try:
            spd = int(samples_per_datapoint)
        except (TypeError, ValueError):
            return 0
        if spd <= 0:
            return 0
        import math
        return max(0, int(math.ceil(nBuf / float(spd))) - 1)

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

    def prefillAccBuf(self, mode='repeat', ref=None, rng=None):
        """
        Fill the rolling acceleration buffer with synthetic data before the first
        datapoint of a buffer segment (event start, or restart after a data gap),
        so that dp2vector() returns a vector immediately instead of returning
        None until the buffer has filled (which would drop the first samples of
        every segment during testing).

        Only intended for test time - training keeps the empty (cold-start)
        buffer so that the model learns to cope with a warm-up period
        (nnTrainer.df2trainingData never calls this).

        Args:
            mode: 'repeat' fills the buffer by tiling the reference datapoint
                   (i.e. assume the device was doing what it was doing at the
                   first real datapoint). Deterministic; preferred default.
                  'noise' fills the buffer with Gaussian noise matched to the
                   reference datapoint's mean/SD (more conservative when the
                   reference itself may be unusual). Pass rng for reproducibility.
                  'stationary' fills the buffer with STATIONARY_ACC_MILLIG
                   (i.e. a stationary sensor at 1 g). Legacy behaviour; the
                   perfectly flat fill is out-of-distribution for the model and
                   tends to inflate seizure probabilities at segment starts.
            ref: reference acceleration samples (list/1D array, milli-g) used
                 by 'repeat' and 'noise'. If None, falls back to 'stationary'.
            rng: numpy Generator (or seed int) used by 'noise'. If None, a
                 non-deterministic generator is used.

        Returns:
            True if the buffer was filled, False if this model has no buffer or
            the requested mode is not supported.
        """
        nBuf = self.getAccBufSize()
        if nBuf <= 0 or not hasattr(self, 'accBuf'):
            return False
        mode = str(mode).lower() if mode is not None else 'repeat'
        if mode in ('stationary', 'static'):
            self.accBuf = [float(self.STATIONARY_ACC_MILLIG)] * nBuf
            return True
        if mode not in ('repeat', 'noise'):
            return False
        if ref is None:
            # No reference available - fall back to stationary fill.
            self.accBuf = [float(self.STATIONARY_ACC_MILLIG)] * nBuf
            return True
        try:
            import numpy as _np
            refArr = _np.asarray(ref, dtype=float).ravel()
            refArr = refArr[~_np.isnan(refArr)]
            if refArr.size == 0:
                raise ValueError("empty reference")
        except Exception:
            self.accBuf = [float(self.STATIONARY_ACC_MILLIG)] * nBuf
            return True
        if mode == 'repeat':
            tiled = _np.tile(refArr, int(_np.ceil(nBuf / float(refArr.size))))[:nBuf]
            self.accBuf = [float(v) for v in tiled]
            return True
        # mode == 'noise'
        mu = float(refArr.mean())
        sd = float(refArr.std())
        if not _np.isfinite(sd) or sd <= 0:
            sd = 5.0  # degenerate reference: assume quiet-sensor noise
        try:
            if rng is None:
                import numpy as _np2
                rng = _np2.random.default_rng()
            elif not hasattr(rng, 'normal'):
                import numpy as _np2
                rng = _np2.random.default_rng(int(rng))
            fill = rng.normal(mu, sd, nBuf)
        except Exception:
            self.accBuf = [float(self.STATIONARY_ACC_MILLIG)] * nBuf
            return True
        self.accBuf = [float(v) for v in fill]
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

    
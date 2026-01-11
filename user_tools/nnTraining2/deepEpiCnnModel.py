#!/usr/bin/env python
"""
DeepEpiCnnModel

Implements the 14-layer 1D-CNN described in the paper and exposes a
Harrell-Davis quantile helper for ensemble aggregation.

Source paper:
Spahr A., Bernini A., Ducouret P., et al., "Deep learning–based detection of
generalized convulsive seizures using a wrist-worn accelerometer", Epilepsia,
2025. DOI: https://doi.org/10.1111/epi.18406
PDF: https://onlinelibrary.wiley.com/doi/pdf/10.1111/epi.18406

Architecture summary implemented:
- Input: 1D sequence (e.g. 30s @ 25Hz -> 750 samples, shape (750,1))
- 14 Conv1D layers with kernel_size=5, padding='valid', dilation=1
    filter progression: [16, 32 (x11), 64, 64]
    strides: configurable but defaults to mostly 1 with occasional 2 for downsampling
- After each Conv1D: BatchNormalization -> ReLU
- After conv stack: GlobalAveragePooling1D -> 64-dim vector
- Dense head: 64 -> 64 -> 32 -> 16 -> num_classes (with BatchNorm + ReLU, dropout p=0.025)

Notes: Harrell-Davis estimator requires scipy.special.betainc; if scipy is
absent a helpful ImportError is raised.
"""

import sys
import os
import numpy as np
try:
    import keras
except Exception:
    from tensorflow import keras

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libosd.dpTools

try:
    # Harrell–Davis needs the regularized incomplete beta function
    from scipy.special import betainc
except Exception:
    betainc = None

try:
    from user_tools.nnTraining2 import nnModel
except Exception:
    import nnModel


class DeepEpiCnnModel(nnModel.NnModel):
    def __init__(self, configObj=None, debug=False):
        super().__init__(configObj, debug)
        # Default sampling and buffer behaviour (simple: use sampleFreq + window)
        self.sampleFreq = 25.0
        # model expects a 'window' parameter (samples) in configObj and an explicit sampleFreq
        self.window = None
        self.bufferSamples = None
        self.conv_dropout = 0.0
        self.dense_dropout = 0.025

        # stride pattern configuration: can provide 'strides' in configObj as list of 14 ints
        self.default_strides = None

        if configObj is not None:
            try:
                if 'sampleFreq' in configObj:
                    self.sampleFreq = float(configObj['sampleFreq'])
                # The model requires bufferSeconds to calculate number of samples
                if 'bufferSeconds' in configObj:
                    bufferSeconds = float(configObj['bufferSeconds'])
                    self.bufferSamples = int(self.sampleFreq * bufferSeconds)
                    self.window = self.bufferSamples
                else:
                    # fallback to default 30s if bufferSeconds not present
                    self.bufferSamples = int(self.sampleFreq * 30)
                    self.window = self.bufferSamples
                # Optional dropout configuration
                self.conv_dropout = float(configObj.get('convDropout', self.conv_dropout))
                self.dense_dropout = float(configObj.get('denseDropout', self.dense_dropout))
            except Exception:
                # fallback defaults
                self.sampleFreq = 25.0
                self.window = int(self.sampleFreq * 30)
                self.bufferSamples = self.window
                self.conv_dropout = 0.0
                self.dense_dropout = 0.025

        # internal acc buffer (vector magnitude)
        self.accBuf = []
        self.model = None

    def makeModel(self, input_shape=None, num_classes=2, nLayers=14):
        """Build the 14-layer CNN as described.

        input_shape: expected (bufferSamples, 1)
        num_classes: final softmax classes
        nLayers: should be 14 (ignored otherwise)
        """
        if input_shape is None or len(input_shape) < 1:
            input_shape = (self.bufferSamples, 1)

        input_layer = keras.layers.Input(shape=input_shape)
        x = input_layer

        # Define filter progression: first layer 16, then eleven 32s, then two 64s
        filters = [16] + [32]*11 + [64, 64]
        if len(filters) != 14:
            raise ValueError("Filter list must have 14 entries")

        # Stride pattern: allow gentle downsampling. Can be overridden by config.
        # Default: stride=1 everywhere except every 5th layer use stride=2 to reduce length.
        strides = [1 if ((i+1) % 5) != 0 else 2 for i in range(14)]

        # Build conv stack
        for i in range(14):
            f = filters[i]
            s = strides[i]
            x = keras.layers.Conv1D(filters=f, kernel_size=5, strides=s, padding='valid', dilation_rate=1)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)
            if self.conv_dropout > 0.0:
                print("Applying conv dropout p=", self.conv_dropout)
                x = keras.layers.Dropout(self.conv_dropout)(x)

        # Global average pooling to produce a 64-d representation (last filters should be 64)
        x = keras.layers.GlobalAveragePooling1D()(x)

        # Dense head: 64 -> 64 -> 32 -> 16 -> num_classes
        x = keras.layers.Dense(64)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        if self.dense_dropout > 0.0:
            x = keras.layers.Dropout(self.dense_dropout)(x)

        x = keras.layers.Dense(64)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

        x = keras.layers.Dense(32)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

        x = keras.layers.Dense(16)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

        output = keras.layers.Dense(num_classes, activation='softmax')(x)

        self.model = keras.models.Model(inputs=input_layer, outputs=output)
        return self.model

    def appendToAccBuf(self, accData):
        self.accBuf.extend(accData)
        if len(self.accBuf) > self.bufferSamples:
            self.accBuf = self.accBuf[-self.bufferSamples:]

    def resetAccBuf(self):
        self.accBuf = []

    def accData2vector(self, accData, normalise=False):
        """Convert acceleration data from mG to G and prepare for model input.
        
        Args:
            accData: List of acceleration magnitude values in mG (milliG)
            normalise: Whether to normalize the data
        
        Returns:
            List representation of data in G, or None if insufficient data
        """
        self.appendToAccBuf(accData)
        if len(self.accBuf) < self.bufferSamples:
            return None
        # Convert from mG to G (divide by 1000)
        vec = np.array(self.accBuf[-self.bufferSamples:], dtype=float) / 1000.0
        if normalise:
            std = vec.std()
            if std != 0:
                vec = (vec - vec.mean()) / std
            else:
                vec = vec - vec.mean()
        return vec.tolist()

    def dp2vector(self, dpObj, normalise=False):
        if (type(dpObj) is dict):
            rawDataStr = libosd.dpTools.dp2rawData(dpObj)
        else:
            rawDataStr = dpObj

        accData, hr = libosd.dpTools.getAccelDataFromJson(rawDataStr)
        if accData is None:
            return None
        return self.accData2vector(accData, normalise)

    @staticmethod
    def harrell_davis_quantile(sample, q=0.7):
        """Compute Harrell–Davis quantile estimate for a 1D array of sample model scores.

        sample: 1D iterable of numeric values (e.g. model scores from ensemble)
        q: quantile in (0,1)

        Returns weighted sum of order statistics.
        Requires scipy.special.betainc for accurate regularized incomplete beta.
        """
        if betainc is None:
            raise ImportError("scipy is required for Harrell–Davis quantile. Please install scipy.")

        x = np.asarray(sample)
        if x.size == 0:
            raise ValueError("sample must not be empty")

        n = x.size
        a = (n + 1) * q
        b = (n + 1) * (1 - q)
        # Sort sample
        xs = np.sort(x)

        # Weights w_i = I_{i/n}(a,b) - I_{(i-1)/n}(a,b)
        weights = np.zeros(n)
        for i in range(1, n + 1):
            u1 = i / n
            u0 = (i - 1) / n
            weights[i - 1] = betainc(a, b, u1) - betainc(a, b, u0)

        # Ensure weights sum to 1 (numerical correction)
        weights = weights / np.sum(weights)
        return float(np.sum(weights * xs))


def main():
    print("DeepEpiCnnModel quick test")
    m = DeepEpiCnnModel(debug=True)
    model = m.makeModel(input_shape=(m.bufferSamples, 1), num_classes=2)
    model.summary()
    # test Harrell–Davis
    try:
        print('HD(0.7) of [0.1,0.2,0.9,0.8] =', DeepEpiCnnModel.harrell_davis_quantile([0.1,0.2,0.9,0.8], 0.7))
    except ImportError as e:
        print('scipy missing: ', e)


if __name__ == '__main__':
    main()

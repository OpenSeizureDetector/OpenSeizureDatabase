#!/usr/bin/env python
"""
DeepEpiCnnModel - PyTorch Implementation

PyTorch implementation of the 14-layer 1D-CNN described in the paper.
This is a parallel implementation to deepEpiCnnModel.py (TensorFlow/Keras version).

Source paper:
Spahr A., Bernini A., Ducouret P., et al., "Deep learning–based detection of
generalized convulsive seizures using a wrist-worn accelerometer", Epilepsia,
2025. DOI: https://doi.org/10.1111/epi.18406

Architecture summary:
- Input: 1D sequence (e.g. 30s @ 25Hz -> 750 samples, shape (batch, 1, 750))
- 14 Conv1d layers with kernel_size=5, padding=0 (valid), dilation=1
    filter progression: [16, 32 (x11), 64, 64]
    strides: configurable but defaults to mostly 1 with occasional 2 for downsampling
- After each Conv1d: BatchNorm1d -> ReLU
- After conv stack: AdaptiveAvgPool1d(1) -> flatten -> 64-dim vector
- Dense head: 64 -> 64 -> 32 -> 16 -> num_classes (with BatchNorm + ReLU, dropout p=0.025)
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libosd.dpTools

try:
    from user_tools.nnTraining2 import nnModel
except Exception:
    import nnModel


class DeepEpiCnn(nn.Module):
    """
    PyTorch nn.Module implementing the 14-layer 1D CNN architecture.
    """
    def __init__(self, input_length=750, num_classes=2, conv_dropout=0.0, dense_dropout=0.025):
        """
        Args:
            input_length: Number of samples in input (e.g. 750 for 30s@25Hz)
            num_classes: Number of output classes
            conv_dropout: Dropout probability after each conv block (0.0 = no dropout)
            dense_dropout: Dropout probability for dense layers (0.0 = no dropout)
        """
        super(DeepEpiCnn, self).__init__()
        
        self.conv_dropout = conv_dropout
        self.dense_dropout = dense_dropout
        
        # Filter progression: [16, 32 (x11), 64, 64]
        filters = [16] + [32]*11 + [64, 64]
        
        # Stride pattern: stride=1 everywhere except every 5th layer uses stride=2
        strides = [1 if ((i+1) % 5) != 0 else 2 for i in range(14)]
        
        # Build convolutional layers
        # Note: PyTorch Conv1d expects (batch, channels, length)
        # Input has 1 channel (magnitude), so in_channels starts at 1
        conv_layers = []
        in_channels = 1
        for i in range(14):
            out_channels = filters[i]
            stride = strides[i]
            
            conv_layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=5,
                stride=stride,
                padding=0,  # 'valid' padding
                dilation=1
            ))
            conv_layers.append(nn.BatchNorm1d(out_channels))
            conv_layers.append(nn.ReLU())
            
            # Add dropout after each conv block if conv_dropout > 0
            if self.conv_dropout > 0.0:
                conv_layers.append(nn.Dropout(self.conv_dropout))
            
            in_channels = out_channels
        
        self.conv_stack = nn.Sequential(*conv_layers)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Dense head: 64 -> 64 -> 32 -> 16 -> num_classes
        # After conv stack, we have 64 channels (last filter size)
        self.fc1 = nn.Linear(64, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        # Apply dropout only after first dense layer if dense_dropout > 0
        self.dropout1 = nn.Dropout(dense_dropout) if dense_dropout > 0.0 else nn.Identity()
        
        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.relu3 = nn.ReLU()
        
        self.fc4 = nn.Linear(32, 16)
        self.bn4 = nn.BatchNorm1d(16)
        self.relu4 = nn.ReLU()
        
        self.fc_out = nn.Linear(16, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 1, length) or (batch, length, 1)
               Will be reshaped to (batch, 1, length) if needed.
        
        Returns:
            Logits of shape (batch, num_classes)
        """
        # Handle input shape: convert (batch, length, 1) -> (batch, 1, length)
        if x.dim() == 3 and x.shape[2] == 1:
            x = x.permute(0, 2, 1)  # (batch, length, 1) -> (batch, 1, length)
        elif x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, length) -> (batch, 1, length)
        
        # Convolutional stack
        x = self.conv_stack(x)
        
        # Global average pooling: (batch, 64, length) -> (batch, 64, 1)
        x = self.global_avg_pool(x)
        
        # Flatten: (batch, 64, 1) -> (batch, 64)
        x = x.squeeze(-1)
        
        # Dense head
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        
        # Output layer (logits, no softmax - use with CrossEntropyLoss)
        x = self.fc_out(x)
        
        return x


class DeepEpiCnnModelPyTorch(nnModel.NnModel):
    """
    PyTorch wrapper implementing the nnModel interface for DeepEpiCnnModel.
    This provides the same interface as the TensorFlow version.
    """
    def __init__(self, configObj=None, debug=False):
        super().__init__(configObj, debug)
        
        # Default sampling and buffer behaviour
        self.sampleFreq = 25.0
        self.window = None
        self.bufferSamples = None
        self.conv_dropout = 0.0
        self.dense_dropout = 0.025
        
        if configObj is not None:
            try:
                if 'sampleFreq' in configObj:
                    self.sampleFreq = float(configObj['sampleFreq'])
                if 'bufferSeconds' in configObj:
                    bufferSeconds = float(configObj['bufferSeconds'])
                    self.bufferSamples = int(self.sampleFreq * bufferSeconds)
                    self.window = self.bufferSamples
                else:
                    # fallback to default 30s if bufferSeconds not present
                    self.bufferSamples = int(self.sampleFreq * 30)
                    self.window = self.bufferSamples
                # Optional dropout configuration (matching TensorFlow version)
                self.conv_dropout = float(configObj.get('convDropout', self.conv_dropout))
                self.dense_dropout = float(configObj.get('denseDropout', self.dense_dropout))
            except Exception:
                # fallback defaults
                self.sampleFreq = 25.0
                self.window = int(self.sampleFreq * 30)
                self.bufferSamples = self.window
                self.conv_dropout = 0.0
                self.dense_dropout = 0.025
        
        # Internal acc buffer (vector magnitude)
        self.accBuf = []
        self.model = None
        
        # Device selection (CPU or CUDA)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if debug:
            print(f"DeepEpiCnnModelPyTorch: Using device {self.device}")
    
    def makeModel(self, input_shape=None, num_classes=2, nLayers=14):
        """
        Build the 14-layer CNN as described.
        
        Args:
            input_shape: Expected (bufferSamples, 1) - will be transposed internally
            num_classes: Final output classes
            nLayers: Should be 14 (ignored otherwise)
        
        Returns:
            PyTorch nn.Module
        """
        if input_shape is None or len(input_shape) < 1:
            input_length = self.bufferSamples
        else:
            # input_shape is (bufferSamples, 1) from TensorFlow convention
            input_length = input_shape[0]
        
        self.model = DeepEpiCnn(
            input_length=input_length,
            num_classes=num_classes,
            conv_dropout=self.conv_dropout,
            dense_dropout=self.dense_dropout
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        if self.debug:
            print(f"Created DeepEpiCnn with input_length={input_length}, num_classes={num_classes}")
            if self.conv_dropout > 0.0:
                print(f"Applying conv dropout p={self.conv_dropout}")
            if self.dense_dropout > 0.0:
                print(f"Applying dense dropout p={self.dense_dropout}")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        return self.model
    
    def appendToAccBuf(self, accData):
        """Append acceleration data to buffer."""
        self.accBuf.extend(accData)
        if len(self.accBuf) > self.bufferSamples:
            self.accBuf = self.accBuf[-self.bufferSamples:]
    
    def resetAccBuf(self):
        """Reset acceleration buffer."""
        self.accBuf = []
    
    def accData2vector(self, accData, normalise=False):
        """
        Convert acceleration data to input vector by adding the data in accData
        to a buffer, and returning the last bufferSamples samples as a vector.
        
        Args:
            accData: List of acceleration magnitude values in mG (milliG)
            normalise: Whether to normalize the data
        
        Returns:
            List representation of normalized/raw data in G, or None if insufficient data.
            
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
        """
        Convert datapoint to input vector.
        
        Args:
            dpObj: Datapoint dict or raw data string
            normalise: Whether to normalize
        
        Returns:
            Vector representation suitable for model input
        """
        if type(dpObj) is dict:
            rawDataStr = libosd.dpTools.dp2rawData(dpObj)
        else:
            rawDataStr = dpObj
        
        accData, hr = libosd.dpTools.getAccelDataFromJson(rawDataStr)
        if accData is None:
            return None
        
        return self.accData2vector(accData, normalise)
    
    def predict(self, x):
        """
        Run inference on input data.
        
        Args:
            x: Input data (numpy array or torch tensor)
        
        Returns:
            Predictions as numpy array
        """
        self.model.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x).float()
            x = x.to(self.device)
            
            output = self.model(x)
            # Apply softmax to get probabilities
            probs = torch.softmax(output, dim=1)
            
            return probs.cpu().numpy()
    
    @staticmethod
    def harrell_davis_quantile(sample, q=0.7):
        """
        Compute Harrell–Davis quantile estimate for a 1D array of sample model scores.
        
        This is a static method shared with the TensorFlow implementation.
        
        Args:
            sample: 1D iterable of numeric values (e.g. model scores from ensemble)
            q: quantile in (0,1)
        
        Returns:
            Weighted sum of order statistics.
            Requires scipy.special.betainc for accurate regularized incomplete beta.
        """
        try:
            from scipy.special import betainc
        except ImportError:
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
    """Quick test of PyTorch implementation."""
    print("DeepEpiCnnModelPyTorch quick test")
    
    # Create model with default config
    config = {
        'sampleFreq': 25,
        'window': 750,
        'framework': 'pytorch'
    }
    
    m = DeepEpiCnnModelPyTorch(configObj=config, debug=True)
    model = m.makeModel(input_shape=(750, 1), num_classes=2)
    
    print("\nModel architecture:")
    print(model)
    
    # Print model summary with key parameters
    print("\n" + "="*80)
    print("MODEL LAYER DETAILS")
    print("="*80)
    total_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d):
            params = sum(p.numel() for p in module.parameters())
            total_params += params
            print(f"Conv1d {name}: in={module.in_channels}, out={module.out_channels}, "
                  f"kernel={module.kernel_size[0]}, stride={module.stride[0]}, params={params:,}")
        elif isinstance(module, nn.Linear):
            params = sum(p.numel() for p in module.parameters())
            total_params += params
            print(f"Linear {name}: {module.in_features} -> {module.out_features}, params={params:,}")
        elif isinstance(module, nn.Dropout):
            print(f"Dropout {name}: p={module.p}")
    print(f"\nTotal parameters: {total_params:,}")
    print("="*80)
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    test_input = torch.randn(batch_size, 750, 1)
    with torch.no_grad():
        output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (logits): {output}")
    
    # Test Harrell–Davis
    try:
        hd_result = DeepEpiCnnModelPyTorch.harrell_davis_quantile([0.1, 0.2, 0.9, 0.8], 0.7)
        print(f'\nHD(0.7) of [0.1,0.2,0.9,0.8] = {hd_result}')
    except ImportError as e:
        print(f'\nscipy missing: {e}')


if __name__ == '__main__':
    main()

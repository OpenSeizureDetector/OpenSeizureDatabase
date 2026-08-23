#!/usr/bin/env python
"""
CNN-LSTM Model - PyTorch Implementation

Combines a 1D CNN for feature extraction on short windows (1 second) with
an LSTM layer to capture longer-term temporal dynamics. This addresses
the high false alarm rate of the simple CNN by considering temporal context.

Architecture:
- Input: Accelerometer data stream at 25Hz
- Short-window CNN: Processes 1-second chunks (25 samples @ 25Hz)
  - Extracts feature vectors from each 1-second window
- LSTM: Processes sequence of CNN feature vectors (30-60 seconds)
  - Captures temporal patterns and dynamics
- Dense classifier head for final prediction

This model uses the same interface as deepEpiCnnModel_torch.py
to work seamlessly with runSequence.py
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


class CnnFeatureExtractor(nn.Module):
    """
    1D CNN for feature extraction on short windows (1 second).
    Outputs a feature vector from each 1-second accelerometer sample.
    """
    def __init__(self, window_samples=25, feature_dim=64, conv_dropout=0.0):
        """
        Args:
            window_samples: Number of samples in 1-second window (25 @ 25Hz)
            feature_dim: Dimension of extracted feature vector
            conv_dropout: Dropout probability after conv layers
        """
        super(CnnFeatureExtractor, self).__init__()
        
        self.window_samples = window_samples
        self.feature_dim = feature_dim
        
        # Compact CNN: 4 conv layers for 1-second windows
        # Layer 1: (1, 25) -> (16, 22) with kernel=5, stride=1
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(conv_dropout) if conv_dropout > 0.0 else nn.Identity()
        
        # Layer 2: (16, 22) -> (32, 19) with kernel=5, stride=1
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(conv_dropout) if conv_dropout > 0.0 else nn.Identity()
        
        # Layer 3: (32, 19) -> (32, 16) with kernel=5, stride=1
        self.conv3 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=0)
        self.bn3 = nn.BatchNorm1d(32)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(conv_dropout) if conv_dropout > 0.0 else nn.Identity()
        
        # Layer 4: (32, 16) -> (64, 12) with kernel=5, stride=1
        self.conv4 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=0)
        self.bn4 = nn.BatchNorm1d(64)
        self.relu4 = nn.ReLU()
        self.drop4 = nn.Dropout(conv_dropout) if conv_dropout > 0.0 else nn.Identity()
        
        # Global average pooling: (64, 12) -> (64,)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Feature projection: 64 -> feature_dim
        self.fc_feature = nn.Linear(64, feature_dim)
    
    def forward(self, x):
        """
        Extract features from 1-second acceleration window.
        
        Args:
            x: Input tensor of shape (batch, 1, window_samples) or (batch, window_samples)
        
        Returns:
            Feature tensor of shape (batch, feature_dim)
        """
        # Handle input shape
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, window_samples) -> (batch, 1, window_samples)
        
        # Convolutional layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.drop3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.drop4(x)
        
        # Global average pooling: (batch, 64, T) -> (batch, 64)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        
        # Project to feature dimension
        x = self.fc_feature(x)
        
        return x


class CnnLstm(nn.Module):
    """
    CNN-LSTM network for seizure detection.
    Combines 1-second CNN feature extraction with LSTM for temporal modeling.
    """
    def __init__(self, window_samples=25, lstm_seq_length=30, feature_dim=64, 
                 lstm_hidden_dim=128, num_layers=2, num_classes=2, 
                 conv_dropout=0.0, lstm_dropout=0.2, dense_dropout=0.025):
        """
        Args:
            window_samples: Samples in 1-second window (25 @ 25Hz)
            lstm_seq_length: Number of 1-second features for LSTM (30 = 30 seconds)
            feature_dim: CNN feature vector dimension
            lstm_hidden_dim: LSTM hidden state dimension
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            conv_dropout: Dropout after conv layers
            lstm_dropout: Dropout in LSTM (between layers)
            dense_dropout: Dropout in dense layers
        """
        super(CnnLstm, self).__init__()
        
        self.window_samples = window_samples
        self.lstm_seq_length = lstm_seq_length
        self.feature_dim = feature_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_classes = num_classes
        
        # CNN feature extractor for 1-second windows
        self.cnn_extractor = CnnFeatureExtractor(
            window_samples=window_samples,
            feature_dim=feature_dim,
            conv_dropout=conv_dropout
        )
        
        # LSTM to process sequence of CNN features
        # Input: (batch, seq_length, feature_dim)
        # Output: (batch, seq_length, lstm_hidden_dim)
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        
        # Dense classifier head on final LSTM output
        # After LSTM: (batch, lstm_seq_length, lstm_hidden_dim)
        # Take last timestep: (batch, lstm_hidden_dim)
        self.fc1 = nn.Linear(lstm_hidden_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dense_dropout) if dense_dropout > 0.0 else nn.Identity()
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dense_dropout) if dense_dropout > 0.0 else nn.Identity()
        
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dense_dropout) if dense_dropout > 0.0 else nn.Identity()
        
        self.fc_out = nn.Linear(32, num_classes)
    
    def forward(self, x):
        """
        Forward pass through CNN-LSTM network.
        
        Args:
            x: Input tensor - handles multiple input formats:
               - (batch, 750): 2D array from nnTrainer
               - (batch, 750, 1): 3D with trailing feature dimension
               - (batch, 1, 750): 3D with middle dimension = 1
               - (batch, 30, 25): Already properly shaped
        
        Returns:
            Logits of shape (batch, num_classes)
        """
        # First, squeeze any singleton dimensions that aren't needed
        # Keep going until we have either 2D or 3D with the right structure
        while x.dim() > 3:
            # Remove excessive dimensions
            if x.shape[-1] == 1:
                x = x.squeeze(-1)
            else:
                x = x.squeeze(1)
        
        # Now handle the remaining 2D or 3D cases
        if x.dim() == 2:
            # Shape: (batch, total_samples) = (batch, 750)
            batch_size = x.shape[0]
            total_samples = x.shape[1]
            
            expected_total = self.lstm_seq_length * self.window_samples
            if total_samples != expected_total:
                raise ValueError(
                    f"Input 2D shape {x.shape}: expected {total_samples}=={expected_total}"
                )
            
            x = x.reshape(batch_size, self.lstm_seq_length, self.window_samples)
        
        elif x.dim() == 3:
            batch_size = x.shape[0]
            dim1 = x.shape[1]
            dim2 = x.shape[2]
            
            # Case 1: (batch, 750, 1) - squeeze the 1, then reshape
            if dim2 == 1 and dim1 == self.lstm_seq_length * self.window_samples:
                x = x.squeeze(2)  # (batch, 750)
                x = x.reshape(batch_size, self.lstm_seq_length, self.window_samples)
            
            # Case 2: (batch, 1, 750) - squeeze the 1, then reshape
            elif dim1 == 1 and dim2 == self.lstm_seq_length * self.window_samples:
                x = x.squeeze(1)  # (batch, 750)
                x = x.reshape(batch_size, self.lstm_seq_length, self.window_samples)
            
            # Case 3: (batch, 30, 25) - already correct shape
            elif dim1 == self.lstm_seq_length and dim2 == self.window_samples:
                pass  # x is already in the right shape
            
            else:
                raise ValueError(
                    f"Cannot reshape 3D input {x.shape} to "
                    f"(batch, {self.lstm_seq_length}, {self.window_samples})"
                )
        
        else:
            raise ValueError(
                f"Input must be 2D or 3D, got {x.dim()}D with shape {x.shape}"
            )
        
        # At this point, x should definitely be (batch, lstm_seq_length, window_samples)
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        
        # Final verification
        if seq_length != self.lstm_seq_length or x.shape[2] != self.window_samples:
            raise ValueError(
                f"Shape mismatch: got {x.shape}, "
                f"expected (batch, {self.lstm_seq_length}, {self.window_samples})"
            )
        
        # Process each 1-second window through CNN to get features
        # Input: (batch, lstm_seq_length, window_samples)
        # Reshape for CNN: (batch*seq_length, 1, window_samples)
        x_cnn = x.reshape(batch_size * seq_length, 1, self.window_samples)
        
        # Extract features for each window
        features = self.cnn_extractor(x_cnn)  # (batch*seq_length, feature_dim)
        
        # Reshape back for LSTM: (batch, seq_length, feature_dim)
        features = features.reshape(batch_size, seq_length, self.feature_dim)
        
        # LSTM processing of feature sequence
        lstm_out, (h_n, c_n) = self.lstm(features)  # lstm_out: (batch, seq_length, lstm_hidden_dim)
        
        # Use last timestep output for classification
        last_output = lstm_out[:, -1, :]  # (batch, lstm_hidden_dim)
        
        # Dense classifier head
        x = self.fc1(last_output)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        # Output logits
        x = self.fc_out(x)
        
        return x


class CnnLstmModelPyTorch(nnModel.NnModel):
    """
    PyTorch wrapper implementing the nnModel interface for CNN-LSTM model.
    Provides same interface as DeepEpiCnnModelPyTorch for compatibility with runSequence.py
    """
    def __init__(self, configObj=None, debug=False):
        super().__init__(configObj, debug)
        
        # Sampling parameters
        self.sampleFreq = 25.0
        self.cnn_window_seconds = 1.0  # 1-second windows for CNN
        self.lstm_window_seconds = 30.0  # 30 seconds of data for LSTM
        
        # Calculate buffer sizes
        self.cnn_window_samples = int(self.sampleFreq * self.cnn_window_seconds)  # 25
        self.lstm_seq_length = int(self.lstm_window_seconds * self.sampleFreq / self.cnn_window_samples)  # 30
        self.bufferSamples = int(self.sampleFreq * self.lstm_window_seconds)  # 750
        
        # Feature and LSTM dimensions
        self.feature_dim = 64
        self.lstm_hidden_dim = 128
        self.lstm_num_layers = 2
        
        # Dropout settings
        self.conv_dropout = 0.0
        self.lstm_dropout = 0.2
        self.dense_dropout = 0.025
        
        if configObj is not None:
            try:
                if 'sampleFreq' in configObj:
                    self.sampleFreq = float(configObj['sampleFreq'])
                
                if 'cnnWindowSeconds' in configObj:
                    self.cnn_window_seconds = float(configObj['cnnWindowSeconds'])
                
                if 'lstmWindowSeconds' in configObj:
                    self.lstm_window_seconds = float(configObj['lstmWindowSeconds'])
                
                # Recalculate based on config
                self.cnn_window_samples = int(self.sampleFreq * self.cnn_window_seconds)
                self.lstm_seq_length = int(self.lstm_window_seconds * self.sampleFreq / self.cnn_window_samples)
                self.bufferSamples = int(self.sampleFreq * self.lstm_window_seconds)
                
                # Feature and LSTM dimensions
                if 'featureDim' in configObj:
                    self.feature_dim = int(configObj['featureDim'])
                if 'lstmHiddenDim' in configObj:
                    self.lstm_hidden_dim = int(configObj['lstmHiddenDim'])
                if 'lstmNumLayers' in configObj:
                    self.lstm_num_layers = int(configObj['lstmNumLayers'])
                
                # Dropout settings
                self.conv_dropout = float(configObj.get('convDropout', self.conv_dropout))
                self.lstm_dropout = float(configObj.get('lstmDropout', self.lstm_dropout))
                self.dense_dropout = float(configObj.get('denseDropout', self.dense_dropout))
            except Exception as e:
                if debug:
                    print(f"Error parsing config: {e}, using defaults")
                # Use defaults
                self.cnn_window_samples = 25
                self.lstm_seq_length = 30
                self.bufferSamples = 750
        
        # Internal acc buffer
        self.accBuf = []
        self.model = None
        
        # Device selection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if debug:
            print(f"CnnLstmModelPyTorch: Using device {self.device}")
            print(f"  CNN window: {self.cnn_window_seconds}s ({self.cnn_window_samples} samples)")
            print(f"  LSTM sequence: {self.lstm_window_seconds}s ({self.lstm_seq_length} timesteps)")
            print(f"  Total buffer: {self.bufferSamples} samples")
            print(f"  Feature dim: {self.feature_dim}, LSTM hidden dim: {self.lstm_hidden_dim}")
    
    def makeModel(self, input_shape=None, num_classes=2, nLayers=None):
        """
        Build the CNN-LSTM model.
        
        Args:
            input_shape: Expected (bufferSamples, 1) from TensorFlow convention
            num_classes: Number of output classes
            nLayers: Ignored (architecture is fixed)
        
        Returns:
            PyTorch nn.Module
        """
        self.model = CnnLstm(
            window_samples=self.cnn_window_samples,
            lstm_seq_length=self.lstm_seq_length,
            feature_dim=self.feature_dim,
            lstm_hidden_dim=self.lstm_hidden_dim,
            num_layers=self.lstm_num_layers,
            num_classes=num_classes,
            conv_dropout=self.conv_dropout,
            lstm_dropout=self.lstm_dropout,
            dense_dropout=self.dense_dropout
        )
        
        # Move to device
        self.model = self.model.to(self.device)
        
        if self.debug:
            print(f"Created CnnLstm with:")
            print(f"  Input shape: ({self.bufferSamples}, 1)")
            print(f"  CNN window: {self.cnn_window_samples} samples")
            print(f"  LSTM seq length: {self.lstm_seq_length}")
            print(f"  Feature dim: {self.feature_dim}")
            print(f"  LSTM hidden dim: {self.lstm_hidden_dim}")
            print(f"  LSTM layers: {self.lstm_num_layers}")
            print(f"  Num classes: {num_classes}")
            print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
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
        Convert acceleration data to input vector by accumulating in buffer.
        Returns vector only when sufficient data accumulated (bufferSamples).
        
        Args:
            accData: List of acceleration magnitude values in mG
            normalise: Whether to normalize the data
        
        Returns:
            List representation of data in G, or None if insufficient data
        """
        self.appendToAccBuf(accData)
        if len(self.accBuf) < self.bufferSamples:
            return None
        
        # Convert from mG to G
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
            x: Input data (numpy array or torch tensor) of shape (batch, bufferSamples)
               or (batch, 1, bufferSamples)
        
        Returns:
            Predictions as numpy array of shape (batch, num_classes)
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
        """
        try:
            from scipy.special import betainc
        except ImportError:
            raise ImportError("scipy is required for Harrell–Davis quantile. Please install scipy.")
        
        sample = np.asarray(sample, dtype=float)
        sample = sample[~np.isnan(sample)]
        
        n = len(sample)
        if n < 1:
            return np.nan
        
        sample_sorted = np.sort(sample)
        
        alpha = q * (n + 1)
        beta = (1 - q) * (n + 1)
        
        # Compute weights for each order statistic
        weights = np.zeros(n)
        for j in range(n):
            cdf_j = betainc(alpha, beta, (j + 1) / (n + 1))
            cdf_j_minus_1 = betainc(alpha, beta, j / (n + 1)) if j > 0 else 0
            weights[j] = cdf_j - cdf_j_minus_1
        
        return np.dot(weights, sample_sorted)


def main():
    """
    Testing script for CNN-LSTM model.
    """
    print("CnnLstmModel testing...")
    
    # Create test configuration
    configObj = {
        "sampleFreq": 25,
        "cnnWindowSeconds": 1.0,
        "lstmWindowSeconds": 30.0,
        "featureDim": 64,
        "lstmHiddenDim": 128,
        "lstmNumLayers": 2,
        "convDropout": 0.0,
        "lstmDropout": 0.2,
        "denseDropout": 0.025
    }
    
    model = CnnLstmModelPyTorch(configObj, debug=True)
    model.makeModel(num_classes=2)
    
    # Test with random data
    batch_size = 4
    test_data = np.random.randn(batch_size, 1, 750).astype(np.float32)
    
    print("\nTesting model prediction...")
    predictions = model.predict(test_data)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions:\n{predictions}")
    
    print("\nModel testing complete!")


if __name__ == "__main__":
    main()

# CNN-LSTM Model Quick Start Guide

## Files Created

1. **`cnnLstmModel_torch.py`** - Complete PyTorch implementation of CNN-LSTM model
2. **`nnConfig_cnn_lstm_pytorch.json`** - Configuration template for training
3. **`CNN_LSTM_README.md`** - Comprehensive architecture and design documentation

## Testing Results ✓

The model has been tested and verified to:
- ✓ Instantiate successfully with PyTorch
- ✓ Build architecture: **281,378 parameters**
- ✓ Run inference on batched data
- ✓ Return correct output shapes (batch, num_classes)
- ✓ Use CUDA when available

## Quick Assessment: Is This Approach Good?

### ✓ Yes, for these reasons:

1. **Addresses the Right Problem**: High false alarm rates in seizure detection often come from isolated spikes in acceleration. The LSTM's temporal memory inherently filters these out by requiring sustained patterns.

2. **Appropriate Timeframes**:
   - **1-second CNN windows**: Perfect for capturing local acceleration dynamics without noise
   - **30-60 second LSTM sequences**: Ideal for seizures (most tonic-clonic seizures last 30-120 seconds)
   - This matches the actual temporal characteristics of seizure events

3. **Sound Architecture**:
   - CNN extracts high-level features efficiently
   - LSTM captures temporal evolution that correlates with seizure progression
   - Maintains compatibility with existing pipeline

4. **Practical Advantages**:
   - Uses only accelerometer data (no additional sensors needed)
   - Compatible with `runSequence.py` training pipeline
   - Reasonable parameter count (~281K vs 50K for pure CNN)
   - Tested and functional

### ⚠ Potential Challenges:

1. **Increased Latency**: Won't detect seizures until 30+ seconds of data accumulated
   - *Mitigation*: Can reduce LSTM window to 15-20 seconds if needed
   
2. **Memory Requirements**: Batch size reduced from 512 to 256 due to LSTM
   - *Mitigation*: Should fit on modern GPUs; adjust if needed
   
3. **Training Time**: ~2-4 hours vs 1-2 hours for pure CNN
   - *Mitigation*: Acceptable trade-off for better accuracy

## Getting Started

### Step 1: Train the Model

```bash
cd /home/graham/osd/OpenSeizureDatabase/user_tools/nnTraining2/

# Use the configuration specifically designed for CNN-LSTM
python runSequence.py --config nnConfig_cnn_lstm_pytorch.json
```

### Step 2: Configuration Options

**For maximum false alarm reduction** (if you have GPU memory):
```json
{
  "lstmWindowSeconds": 60.0,        // Full 60 seconds
  "lstmNumLayers": 3,               // More LSTM depth
  "lstmHiddenDim": 256,             // Larger hidden state
  "batchSize": 128                  // Reduced batch size
}
```

**For faster training** (if initial results look good):
```json
{
  "lstmWindowSeconds": 30.0,        // 30 seconds (default)
  "lstmNumLayers": 1,               // Single LSTM layer
  "lstmHiddenDim": 64,              // Smaller hidden state
  "batchSize": 256                  // Reasonable batch
}
```

**For balance** (recommended starting point - uses provided config):
```json
{
  "lstmWindowSeconds": 30.0,        // Good temporal context
  "lstmNumLayers": 2,               // 2 layers (good depth)
  "lstmHiddenDim": 128,             // Standard hidden size
  "batchSize": 256                  // Fits most GPUs
}
```

### Step 3: Evaluate Performance

Compare against baseline CNN:
```bash
# After training completes:
python nnTester.py output/cnnLstmModel_pytorch_best.pth testData.csv
```

### Step 4: Check Improvement

Monitor:
- **Sensitivity**: Should be similar or better than DeepEpiCnnModel (~80%+)
- **False Positive Rate (FPR)**: Should be **significantly lower** due to temporal modeling
- **F-score or Youden's Index**: Overall discriminative ability

## Model Architecture Summary

```
Input: 750 accelerometer samples (30s @ 25Hz)
  ↓
Split into 30 windows of 25 samples (1s each)
  ↓
CNN Feature Extractor (per window):
  Conv1d × 4 → Global Average Pool → 64 features
  ↓
LSTM Processor:
  2 LSTM layers × 128 units on 30 timesteps
  ↓
Dense Classifier Head:
  Dense(128) → Dense(64) → Dense(32) → Output(2 classes)
  ↓
Probability: [P(not seizure), P(seizure)]
```

**Total Parameters: 281,378**
- CNN extractor: ~20K
- LSTM layers: ~240K
- Dense head: ~21K

## Integration with Existing Tools

The model is designed to work with:
- ✓ `runSequence.py` - Training pipeline (no changes needed)
- ✓ `nnTrainer.py` - Trainer framework (fully compatible)
- ✓ `nnTester.py` - Testing/evaluation (fully compatible)
- ✓ Model interface matches `DeepEpiCnnModelPyTorch` exactly

## Recommended Next Steps

1. **Try the configuration as-is** first with default settings
2. **Evaluate on test set** - Compare FPR/Sensitivity metrics vs DeepEpiCnnModel
3. **If FPR still high**: Increase `lstmWindowSeconds` to 45-60
4. **If results good**: Consider ensemble combining both models
5. **For deployment**: Export to TorchScript for optimization if needed

## Example: Using the Trained Model in Code

```python
import torch
import numpy as np
from user_tools.nnTraining2.cnnLstmModel_torch import CnnLstmModelPyTorch

# Load configuration and model
config = {
    "sampleFreq": 25,
    "cnnWindowSeconds": 1.0,
    "lstmWindowSeconds": 30.0,
}
model = CnnLstmModelPyTorch(config)
model.makeModel(num_classes=2)

# Load saved weights
checkpoint = torch.load('output/cnnLstmModel_pytorch_best.pth', 
                        weights_only=False)
model.model.load_state_dict(checkpoint['model_state_dict'])

# Predict on new data
acc_data = [...]  # List of acceleration magnitudes in mG
vector = model.accData2vector(acc_data, normalise=True)

if vector is not None:
    batch = np.array(vector).reshape(1, 1, -1).astype(np.float32)
    predictions = model.predict(batch)
    seizure_probability = predictions[0, 1]
    print(f"Seizure probability: {seizure_probability:.2%}")
```

## Questions?

Refer to:
- `CNN_LSTM_README.md` for detailed architecture and theory
- `nnConfig_cnn_lstm_pytorch.json` for all tunable parameters
- `cnnLstmModel_torch.py` source code for implementation details

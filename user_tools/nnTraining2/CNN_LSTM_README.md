# CNN-LSTM Model Implementation for Seizure Detection

## Overview

The **CNN-LSTM model** combines convolutional neural networks (CNN) with long short-term memory (LSTM) networks to address the high false alarm rate of the simple 14-layer CNN. This approach:

1. **Extracts short-term features** using CNN on 1-second windows (25 samples @ 25Hz)
2. **Models temporal dynamics** using LSTM to process sequences of CNN features over 30-60 seconds
3. **Achieves temporal context awareness** by considering how acceleration patterns evolve over time

## Architecture

### Design Rationale

The false alarm problem in seizure detection often stems from isolated high-acceleration events that don't represent actual seizures. By incorporating LSTM, the model can:

- Learn that genuine seizures have characteristic patterns over several seconds
- Distinguish between isolated spikes (false alarms) and sustained motion patterns
- Capture the temporal evolution of acceleration magnitude and variability
- Use bidirectional patterns: what comes before and after a potential seizure signature

### CNN Feature Extractor (1-second windows)

```
Input: 1-second accelerometer data (25 samples)
  ↓
Conv1d (1→16, kernel=5): 25 → 22 samples
  ↓ + BatchNorm + ReLU
Conv1d (16→32, kernel=5): 22 → 19 samples
  ↓ + BatchNorm + ReLU
Conv1d (32→32, kernel=5): 19 → 16 samples
  ↓ + BatchNorm + ReLU
Conv1d (32→64, kernel=5): 16 → 12 samples
  ↓ + BatchNorm + ReLU
Global Average Pooling: 64 → 64 features
  ↓
Dense Layer: 64 → feature_dim (default: 64)
  ↓
Output: Feature vector (shape: feature_dim)
```

This compact CNN is designed for speed—it processes very short windows efficiently and outputs a fixed feature vector.

### LSTM Temporal Processor

```
Input: Sequence of CNN features (30 timesteps × feature_dim)
  ↓
LSTM Layer 1: feature_dim → lstm_hidden_dim (default: 128)
  ↓ + Dropout (0.2)
LSTM Layer 2: lstm_hidden_dim → lstm_hidden_dim
  ↓
Take last timestep: lstm_hidden_dim
  ↓
Dense Head:
  lstm_hidden_dim → 128 → 64 → 32 → num_classes
  (with BatchNorm + ReLU + Dropout between layers)
  ↓
Output: Logits (shape: num_classes)
```

The LSTM captures sequential patterns in the CNN features, learning temporal dependencies that distinguish seizures from normal motion.

## Configuration Parameters

### File: `nnConfig_cnn_lstm_pytorch.json`

Key parameters specific to CNN-LSTM:

```json
{
  "modelClass": "user_tools.nnTraining2.cnnLstmModel_torch.CnnLstmModelPyTorch",
  
  "sampleFreq": 25,
  "cnnWindowSeconds": 1.0,      // 1-second CNN feature extraction windows
  "lstmWindowSeconds": 30.0,    // 30-second total sequence for LSTM
  
  "featureDim": 64,             // CNN output feature dimension
  "lstmHiddenDim": 128,         // LSTM hidden state size
  "lstmNumLayers": 2,           // Number of stacked LSTM layers
  
  "convDropout": 0.05,          // Dropout in CNN
  "lstmDropout": 0.2,           // Dropout between LSTM layers
  "denseDropout": 0.15,         // Dropout in classifier head
  
  "batchSize": 256,             // Reduced from 512 due to LSTM memory
  "epochs": 100,
  "useLrSchedule": true,        // Use Spahr et al. 2025 schedule
  "useBalancedBatches": true    // Balanced sampling
}
```

### Adjustable Timeframes

The timeframe design is fully configurable:

**Option 1 (Conservative):** 1s CNN + 30s LSTM
```json
"cnnWindowSeconds": 1.0,
"lstmWindowSeconds": 30.0
```
→ 30 LSTM timesteps, lighter computation, faster training

**Option 2 (Default):** 1s CNN + 45s LSTM
```json
"cnnWindowSeconds": 1.0,
"lstmWindowSeconds": 45.0
```
→ 45 LSTM timesteps, more temporal context

**Option 3 (Comprehensive):** 1s CNN + 60s LSTM
```json
"cnnWindowSeconds": 1.0,
"lstmWindowSeconds": 60.0
```
→ 60 LSTM timesteps, maximum temporal context but higher memory

**Option 4 (Fine-grained):** 0.5s CNN + 30s LSTM
```json
"cnnWindowSeconds": 0.5,
"lstmWindowSeconds": 30.0
```
→ Higher CNN frequency, 60 LSTM timesteps

## Usage with runSequence.py

The model integrates seamlessly with the existing pipeline:

```bash
# 1. Prepare data and train the model
cd /home/graham/osd/OpenSeizureDatabase/user_tools/nnTraining2/
python runSequence.py --config nnConfig_cnn_lstm_pytorch.json

# 2. The model saves as: output/cnnLstmModel_pytorch_best.pth
# 3. Use the saved model for inference with any standard PyTorch loader
```

### Model Interface (Compatible with existing tools)

The CNN-LSTM model implements the same interface as `DeepEpiCnnModelPyTorch`:

```python
from user_tools.nnTraining2.cnnLstmModel_torch import CnnLstmModelPyTorch

# Initialize with configuration
config = {
    "sampleFreq": 25,
    "cnnWindowSeconds": 1.0,
    "lstmWindowSeconds": 30.0,
    "featureDim": 64,
    "lstmHiddenDim": 128,
    "lstmNumLayers": 2
}

model = CnnLstmModelPyTorch(config, debug=True)
model.makeModel(num_classes=2)

# Convert raw accelerometer data to vector
raw_acc_data = [...]  # List of acceleration magnitudes in mG
input_vector = model.accData2vector(raw_acc_data, normalise=True)

# Predict
if input_vector is not None:
    # Reshape for batch: (750,) → (1, 1, 750)
    batch = np.array(input_vector).reshape(1, 1, -1).astype(np.float32)
    predictions = model.predict(batch)
    # predictions shape: (1, 2) - probability for each class
    seizure_prob = predictions[0, 1]  # Probability of seizure
```

## Expected Performance Improvement

### Why This Helps with False Alarms

1. **Context Matters**: Normal movement often has high-acceleration spikes, but they're brief and isolated. Seizures typically have sustained or repeated acceleration patterns.

2. **Temporal Coherence**: The LSTM learns that seizures have a characteristic temporal signature:
   - Initial muscle contraction
   - Sustained rhythmic motion (especially tonic-clonic seizures)
   - Post-ictal slowing or cessation

3. **Reduces False Positives**: By requiring temporal consistency over 30-60 seconds, the model rejects brief acceleration anomalies.

### Potential Trade-offs

- **Latency**: Detection latency increases to the LSTM window size (30-60 seconds) because the model needs sufficient history
- **Memory**: LSTM requires more GPU memory than pure CNN; batch size may need reduction
- **Training Time**: Slightly longer training due to LSTM computation and increased model complexity

## Hyperparameter Tuning Recommendations

### Start with Defaults, Then Adjust

1. **If false alarm rate is still high:**
   - Increase `lstmWindowSeconds` from 30 to 45-60
   - Increase `lstmNumLayers` from 2 to 3
   - Increase `lstmHiddenDim` from 128 to 256

2. **If detection sensitivity is too low:**
   - Decrease `lstmNumLayers` to 1
   - Decrease dropout rates (`lstmDropout`, `denseDropout`)
   - Reduce `lstmWindowSeconds` to 20-30 seconds

3. **For faster training:**
   - Reduce `lstmNumLayers` to 1
   - Reduce `featureDim` from 64 to 32
   - Increase `batchSize` (if GPU memory allows)

4. **For better generalization:**
   - Increase `lstmDropout` to 0.3-0.4
   - Increase `denseDropout` to 0.25
   - Enable `useBalancedBatches` and aggressive augmentation

## Comparison with DeepEpiCnnModel

| Aspect | DeepEpiCnnModel | CNN-LSTM |
|--------|-----------------|----------|
| Architecture | 14-layer pure CNN | CNN + 2-layer LSTM |
| Detection latency | ~1-2 seconds | 30-60 seconds |
| Parameter count | ~50K | ~200-300K |
| Training time | ~1-2 hours | ~2-4 hours |
| Memory per batch | ~2-4 GB (batch=512) | ~3-6 GB (batch=256) |
| Temporal awareness | Local (kernel window) | Global (LSTM sequence) |
| False alarm handling | Limited | Improved (temporal context) |

## Files Generated/Modified

1. **New File:** `cnnLstmModel_torch.py` - CNN-LSTM implementation
2. **New File:** `nnConfig_cnn_lstm_pytorch.json` - Configuration template
3. **Existing:** Works with `runSequence.py`, `nnTrainer.py`, `nnTester.py`

## Troubleshooting

**Issue:** Out of memory errors during training
- **Solution:** Reduce `batchSize` from 256 to 128 or 64, or reduce `lstmHiddenDim`

**Issue:** Model not converging
- **Solution:** Ensure `useLrSchedule` is enabled, or try different learning rate values

**Issue:** Still high false alarm rate
- **Solution:** Increase `lstmWindowSeconds` to 45-60, increase LSTM layers or hidden dimension

**Issue:** Model too slow for deployment
- **Solution:** Convert to TorchScript (`.pt` format) or ONNX for optimization

## Future Enhancements

1. **Bidirectional LSTM**: Use bidirectional layers for better temporal context (requires offline processing)
2. **Attention Mechanism**: Add self-attention to weight important timesteps
3. **Multi-task Learning**: Jointly predict seizure type and timing
4. **Ensemble**: Combine CNN-LSTM with other models for improved robustness

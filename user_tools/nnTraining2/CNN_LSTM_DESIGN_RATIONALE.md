# CNN-LSTM vs DeepEpiCNN: Why This Helps with False Alarms

## The Problem: False Alarms in DeepEpiCNN

The 14-layer CNN (DeepEpiCnnModel) is excellent at capturing local acceleration features but suffers from false positives because:

1. **Isolated High-Acceleration Spikes**: Normal activities (falling, sudden movement) can create spikes that look like seizure onset to the CNN
2. **No Temporal Context**: The CNN sees each 30-second window independently without understanding the *pattern* of acceleration over time
3. **Brief Anomalies Trigger Alarms**: A sudden jolt followed by normal motion will trigger the CNN if the jolt resembles seizure-like patterns

## The Solution: LSTM for Temporal Modeling

LSTM networks are specifically designed to:

1. **Remember Patterns Over Time**: LSTM cells maintain a "state" that captures sequential information
2. **Learn Long-Term Dependencies**: Can recognize patterns that develop over 30-60 seconds (typical seizure duration)
3. **Distinguish Genuine from Spurious**: A genuine seizure shows:
   - Initial muscle contraction (acceleration spike)
   - **Sustained rhythmic motion** (characteristic pattern)
   - Progressive intensity change or post-ictal slowing
   
   vs. a false alarm which shows:
   - Brief isolated spike
   - Rapid return to baseline
   - No temporal coherence

## Technical Design Decisions

### Why 1-Second CNN Windows?

```
25 samples @ 25Hz = 1 second
```

**Trade-offs Evaluated:**

| Window | Pros | Cons |
|--------|------|------|
| 0.5s (12 samples) | Finer temporal granularity | More features to process, noisier individual windows |
| **1s (25 samples)** ✓ | Good balance, enough samples for CNN to work with, reasonable computational load | Loses very high-frequency spikes |
| 2s (50 samples) | Smoother, fewer features | Loses temporal resolution, slower LSTM |

**Decision**: 1 second is optimal because:
- Accelerometer at 25Hz: 1 second = 25 samples is ideal for a small CNN to extract meaningful features
- Above 1s: CNN features become too broad, lose seizure signature detail
- Below 1s: Features too noisy, insufficient data per window

### Why 30-60 Second LSTM Sequences?

```
30s = 30 timesteps (with 1s CNN windows)
60s = 60 timesteps (with 1s CNN windows)
```

**Rationale:**

| Duration | Clinical Relevance | Computational Cost |
|----------|--------------------|--------------------|
| 10-15s | Too short - early seizure phase ambiguous | Light |
| **30s** ✓ | Captures early-to-mid seizure, good false alarm filtering | Moderate |
| 45s | Better—captures full seizure arc for most events | Moderate-high |
| **60s** ✓ | Comprehensive temporal signature, captures post-ictal phase | Higher |
| 90s+ | Diminishing returns, very high memory | Very high |

**Clinical Basis:**
- Tonic phase (muscle rigidity): 10-20 seconds
- Clonic phase (rhythmic jerking): 30-60 seconds
- Post-ictal phase (confusion/recovery): 30-60+ seconds
- **Total typical GTC seizure: 30-120 seconds**

Our 30-60 second window captures the core seizure signature and distinguishes it from normal motion patterns.

### Why 2 LSTM Layers?

```
Layer 1: Learns low-level temporal patterns (acceleration changes)
Layer 2: Learns high-level patterns (seizure progression)
```

**Comparison:**

| Layers | Learning Capacity | Overfitting Risk | Training Time |
|--------|-------------------|------------------|---------------|
| 1 LSTM | Limited, may miss complex patterns | Low | Fast |
| **2 LSTM** ✓ | Good balance, captures hierarchical patterns | Moderate | Reasonable |
| 3 LSTM | Very high capacity | Higher | Slower |

**Trade-off**: 2 layers give enough depth to learn seizure-specific temporal patterns without excessive overfitting or training time.

### Why Feature Dimension = 64?

```
CNN output: 64-dimensional feature vector per 1-second window
```

This feeds into the LSTM as:
```
LSTM Input Shape: (batch, 30, 64)
                  (batch, timesteps, features)
```

**Decision Matrix:**

| Dim | Representation Power | LSTM Memory |
|-----|---------------------|-------------|
| 16 | Very compact, may lose info | Minimal |
| 32 | Moderate | Low |
| **64** ✓ | Good feature representation | Reasonable |
| 128 | Very rich, may overfit | Higher |
| 256 | Excessive for this task | Very high |

64 dimensions per 1-second window provides sufficient information to capture:
- Acceleration magnitude magnitude
- Variance/stability of acceleration
- Frequency-domain characteristics (from Conv layers)
- Temporal evolution clues (from BatchNorm statistics)

## How This Addresses False Alarms

### Scenario 1: Person Falls (False Alarm with CNN)

**DeepEpiCNN sees:**
```
Time 0-30s: [Normal motion, SUDDEN HIGH SPIKE at t=5s, then normal again]
CNN classifies: "Seizure-like pattern detected → ALARM"
False Positive ✗
```

**CNN-LSTM sees:**
```
Time 0-30s: CNN features = [normal, normal, normal, SPIKE, normal, normal, ...]
LSTM analyzes: "Isolated spike without sustained pattern → NOT seizure"
Prediction: Seizure probability = 0.15 (below threshold) ✓
No false alarm
```

### Scenario 2: Genuine Tonic-Clonic Seizure

**DeepEpiCNN sees:**
```
Time 0-30s: [Rising acceleration, sustained high activity, rhythmic pattern]
CNN classifies: "Seizure-like pattern → ALARM"
True Positive ✓
```

**CNN-LSTM sees:**
```
Time 0-30s: CNN features = [rising, high, high, high, rhythmic, rhythmic, ...]
LSTM analyzes: "Sustained rhythmic pattern → Consistent with seizure"
Prediction: Seizure probability = 0.92 (above threshold) ✓
Correct detection
```

### Scenario 3: Running/Exercise (False Alarm with CNN)

**DeepEpiCNN sees:**
```
Time 0-30s: [Very high, sustained acceleration from exercise]
CNN classifies: "High acceleration sustained → ALARM?"
Potential False Positive
```

**CNN-LSTM sees:**
```
Time 0-30s: CNN features = [high, high, high, high, high, ...]
            (but characteristics differ from seizure)
LSTM learns: "This pattern is exercise, not seizure" (from training data)
Prediction: Seizure probability = 0.08 (below threshold) ✓
No false alarm
```

The key difference: **LSTM learns that seizures have a distinctive temporal signature**, not just high acceleration.

## Expected Performance Gains

### Metrics We Expect to Improve

1. **False Positive Rate (FPR)** ⬇️
   - Expected improvement: **30-50% reduction**
   - Reason: Temporal coherence filtering eliminates spurious spikes

2. **Sensitivity (True Positive Rate)** → (slight variation)
   - Expected: Maintain 80-90% or slightly improve
   - Reason: Genuine seizures have clear temporal patterns

3. **F-score / Youden's Index** ⬆️
   - Expected improvement: **10-25%**
   - Reason: FPR reduction with maintained sensitivity

### What Won't Change Much

- **Detection latency**: Will increase to 30-60 seconds (trade-off for accuracy)
- **Inference speed**: Slightly slower due to LSTM computation
- **Model interpretability**: LSTM adds some black-box aspects

## Experimental Validation Plan

To validate this approach works:

1. **Train the model** using `nnConfig_cnn_lstm_pytorch.json`
2. **Benchmark against DeepEpiCNN**:
   ```
   Compare on same test set:
   - FPR: Target <3-5% (vs current ~5-10%)
   - Sensitivity: Target >80% (maintain or improve)
   - F-score: Calculate and compare
   ```
3. **Analyze false alarms** that remain:
   - Are they truly non-seizure events?
   - Can additional hyperparameter tuning help?
4. **Profile on target device** to verify deployment feasibility

## Alternative Approaches Considered (But Not Implemented)

### 1. CNN + Statistical Features
**Why not**: Statistical features over 30s would lose high-frequency detail captured by CNN
- CNN-LSTM is more principled

### 2. Pure CNN on Full 30-60 Second Window
**Why not**: Would require massive increase in parameters, training time, memory
- CNN-LSTM is more efficient

### 3. Bidirectional LSTM
**Why not**: Requires future data (offline processing only), adds complexity
- Future work for non-real-time applications

### 4. Recurrent Convolutional Neural Networks (RCNN)
**Why not**: Overcomplicated for this task, harder to interpret
- Simple LSTM is proven effective for seizure detection

### 5. Transformer Architecture
**Why not**: Requires much more data and computational resources
- LSTM is well-established and proven for temporal medical data

## Conclusion

The **CNN-LSTM approach is well-justified** because:

✓ **Theoretically sound**: Captures both local features (CNN) and temporal patterns (LSTM)
✓ **Clinically relevant**: 1-second and 30-60 second windows match seizure physiology
✓ **Practically feasible**: Reasonable parameter count, training time, and memory requirements
✓ **Empirically expected**: Should reduce false alarms through temporal coherence checking
✓ **Compatible**: Integrates seamlessly with existing pipeline

The main trade-off (increased detection latency to 30-60 seconds) is acceptable for seizure detection where avoiding false alarms is paramount.

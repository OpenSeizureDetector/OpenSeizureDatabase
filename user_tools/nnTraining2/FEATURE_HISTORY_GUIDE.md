# Feature History Configuration Guide

## Overview

The `addFeatureHistoryLength` parameter controls whether temporal history columns are added to feature CSVs during preprocessing. This is important for optimizing processing time and storage.

**Relationship to `nHistory`:** These parameters do **exactly the same thing**. `addFeatureHistoryLength` is the new, more descriptive name. For backward compatibility, the code falls back to `nHistory` if `addFeatureHistoryLength` is not specified. Always use `addFeatureHistoryLength` in new configs.

## How Both Parameters Are Used

The same value is read in **three places** in the pipeline:

1. **`addFeatureHistory.py`**: Creates N history columns (e.g., `specPower_t-0`, `specPower_t-1`, ..., `specPower_t-4` when N=5)
2. **`skTrainer.py`**: Reads N to know which history column names to extract from the CSV
3. **`skTester.py`**: Reads N to know which history column names to extract from the CSV

**Important:** All three must read the same value, which is why we unified them to use `addFeatureHistoryLength` (with `nHistory` fallback).

## Configuration

Add to `dataProcessing` section in your config JSON:

```json
"dataProcessing": {
    "addFeatureHistoryLength": 0,  // 0 = skip, >0 = add N history columns
    ...
}
```

## When to Use Feature History

### ❌ Skip Feature History (`addFeatureHistoryLength: 0`)

Use when working with **raw acceleration data** for models that handle temporal windowing internally:

- **PyTorch models** (e.g., `deepEpiCnnModel_torch.py`)
- **TensorFlow models** (e.g., `deepEpiCnnModel.py`)
- **Any CNN/RNN** that processes time series directly

**Example configs:**
- `nnConfig_deep_pytorch.json` - Uses `features: ["acc_magnitude"]`
- `nnConfig_deep.json` - Uses `features: ["acc_magnitude"]`

**Why skip?**
- Models have internal buffers (`accBuf`) for temporal context
- Only `_t-0` (current timestep) columns are extracted anyway
- Saves massive processing time and storage
- Avoids creating 750+ redundant columns per feature

### ✅ Add Feature History (`addFeatureHistoryLength: > 0`)

Use when working with **calculated features** for traditional ML models:

- **sklearn models** (Random Forest, SVM, etc.)
- **Features like:** `specPower`, `roiPower`, `hrAlarm`, frequency domain features
- Models that need explicit temporal context in their input

**Example configs:**
- `nnConfig.json` - Uses `features: ["mean_x", "mean_freq_x", "total_power_x_seizure_main", ...]`

**Why use it?**
- sklearn models are stateless - need explicit history columns
- Each row becomes: `[specPower_t-4, specPower_t-3, ..., specPower_t-0, specPower_mean]`
- Model sees temporal patterns directly in the feature vector

## How It Works

### Without Feature History (`addFeatureHistoryLength: 0`)

**Input CSV from `extractFeatures`:**
```
eventId,type,M000,M001,M002,...,M749
evt1,1,100,102,99,...,105
```

**Used directly for training** - no additional columns added.

### With Feature History (`addFeatureHistoryLength: 5`)

**Input CSV from `extractFeatures`:**
```
eventId,type,specPower,roiPower
evt1,1,850,420
evt1,1,920,450
...
```

**After `addFeatureHistory`:**
```
eventId,type,specPower_t-4,specPower_t-3,...,specPower_t-0,specPower_mean,roiPower_t-4,...
evt1,1,800,830,...,920,864,400,...
```

## Important: Raw Acceleration is Never Historified

The code **automatically excludes** these columns from history processing:
- `x`, `y`, `z`, `magnitude` (raw values)
- `M000`, `M001`, ... `M749` (windowed acceleration samples)

**Reason:** Models that use raw acceleration handle windowing internally via buffering mechanisms.

## Backward Compatibility

For backward compatibility, the code falls back to the old `nHistory` parameter:

```python
n_history = configObj.get('dataProcessing', {}).get('addFeatureHistoryLength', 
            configObj.get('dataProcessing', {}).get('nHistory', 5))
```

## Performance Impact

### Example: `nnConfig_deep_pytorch.json` with `window=750`

**Before (with feature history):**
- Process 750 history columns per feature
- Create massive CSV files with 750+ columns
- **All columns ignored** during training (only `_t-0` used)
- Processing time: ~10-20 minutes for large datasets

**After (`addFeatureHistoryLength: 0`):**
- Skip history step entirely
- Keep original feature CSV (1 column per feature)
- Processing time: **0 seconds** (step skipped)
- **No loss in accuracy** - model handles windowing

## Migration Guide

### Updating Existing Configs

**For raw acceleration models (CNN/RNN):**
```json
// OLD (unnecessary processing)
"dataProcessing": {
    "window": 750,
    "features": ["acc_magnitude"],
    "simpleMagnitudeOnly": true
    // nHistory would be used by default
}

// NEW (optimized)
"dataProcessing": {
    "addFeatureHistoryLength": 0,  // ← Add this
    "window": 750,
    "features": ["acc_magnitude"],
    "simpleMagnitudeOnly": true
}
```

**For sklearn models with calculated features:**
```json
// OLD
"dataProcessing": {
    "nHistory": 5,  // ← Old parameter
    "features": ["specPower", "roiPower"]
}

// NEW
"dataProcessing": {
    "addFeatureHistoryLength": 5,  // ← New parameter
    "features": ["specPower", "roiPower"]
}
```

## Auto-Detection

The pipeline also **automatically skips** feature history when:
1. `addFeatureHistoryLength == 0`, OR
2. `features` list contains ONLY raw acceleration features

```python
raw_acc_features = {'acc_magnitude', 'acc_x', 'acc_y', 'acc_z'}
only_raw_acc = all(f in raw_acc_features for f in features)
skip_history = (addHistoryLength == 0) or only_raw_acc
```

## Summary

| Model Type | Features | `addFeatureHistoryLength` | Reason |
|------------|----------|--------------------------|---------|
| PyTorch CNN | `acc_magnitude` | `0` | Model handles windowing |
| TensorFlow CNN | `acc_magnitude` | `0` | Model handles windowing |
| sklearn RF | `specPower`, `roiPower` | `5` | Stateless model needs history |
| sklearn SVM | Frequency features | `10` | More history for complex patterns |

**Rule of thumb:** If your features start with `acc_`, use `0`. If they're calculated metrics, use `> 0`.

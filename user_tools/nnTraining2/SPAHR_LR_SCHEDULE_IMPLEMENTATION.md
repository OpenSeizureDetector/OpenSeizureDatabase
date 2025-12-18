# Spahr et al. 2025 Learning Rate Schedule Implementation

## Overview

This document describes the implementation of the three-phase learning rate schedule and training approach described in Spahr et al. 2025 for the OpenSeizureDatabase neural network training pipeline.

## Changes Made

### 1. Configuration File (`nnConfig_deep_pytorch.json`)

Added new parameters to `modelConfig` section:

#### New Parameters:
- **`useLrSchedule`**: `true` - Enable three-phase LR schedule (set to `false` for original behavior)
- **`useAdamW`**: `true` - Use AdamW optimizer instead of Adam
- **`adamwBeta1`**: `0.9` - AdamW beta1 parameter
- **`adamwBeta2`**: `0.999` - AdamW beta2 parameter
- **`weightDecay`**: `0.0` - AdamW weight decay parameter

#### Learning Rate Schedule Parameters:
- **`lrPeak`**: `1e-3` - Peak learning rate during warmup
- **`lrMainEnd`**: `3e-5` - Final LR after cosine annealing
- **`warmupSteps`**: `2500` - Linear warmup duration
- **`mainSteps`**: `45000` - Cosine annealing duration
- **`cooldownSteps`**: `2500` - Linear cooldown duration
- **`totalTrainingSteps`**: `50000` - Total training steps

#### Advanced Checkpointing Parameters (Spahr et al. 2025 criteria):
- **`evalEverySteps`**: `5000` - Evaluation frequency
- **`saveBestOnBothImprovement`**: `true` - Save when both sensitivity and FAR improve
- **`saveBestOnFarReduction`**: `0.10` - Save if FAR reduces by >10%
- **`saveBestOnSensitivityTolerance`**: `0.05` - Sensitivity must stay within 5%

#### Modified Parameters:
- **`batchSize`**: Changed from `64` to `512` (per paper)
- **`lrStart`**: Changed from `1e-3` to `1e-5` (initial LR for warmup)

### 2. Training Code (`nnTrainer.py`)

#### A. Updated `load_config_params()` Function

Added loading of all new configuration parameters with sensible defaults for backward compatibility.

#### B. Updated `trainModel_tensorflow()` Function

**Changes:**
- Added support for AdamW optimizer with configurable beta parameters
- Existing three-phase LR schedule now uses `adamw_beta1` and `adamw_beta2` parameters
- Falls back gracefully to Adam if AdamW not available
- Maintains backward compatibility with original ReduceLROnPlateau scheduler

**Optimizer Selection:**
```python
if params['use_adamw']:
    optimizer = keras.optimizers.AdamW(
        learning_rate=schedule,
        beta_1=params['adamw_beta1'],
        beta_2=params['adamw_beta2'],
        weight_decay=params['weight_decay']
    )
```

#### C. Completely Rewrote `trainModel_pytorch()` Function

**Key Changes:**

1. **Optimizer Configuration:**
   ```python
   if params['use_adamw']:
       optimizer = optim.AdamW(
           model.parameters(),
           lr=params['lrStart'],
           betas=(params['adamw_beta1'], params['adamw_beta2']),
           weight_decay=params['weight_decay']
       )
   ```

2. **Three-Phase Learning Rate Schedule:**
   ```python
   def get_three_phase_lr(step):
       # Phase 1: Warmup (linear increase)
       if step < warmup:
           lr = lr_start + progress * (lr_peak - lr_start)
       # Phase 2: Cosine annealing
       elif step < warmup + main:
           cosine_factor = 0.5 * (1 + cos(π * progress))
           lr = lr_main_end + (lr_peak - lr_main_end) * cosine_factor
       # Phase 3: Cooldown (linear decrease to 0)
       else:
           lr = lr_main_end * (1 - progress)
   ```

3. **Dual Training Mode:**
   - **Step-based training** (when `useLrSchedule=true`): Runs for fixed number of steps (50,000)
   - **Epoch-based training** (when `useLrSchedule=false`): Original behavior with epochs
   
4. **Enhanced Validation Metrics:**
   - Calculates **sensitivity** (True Positive Rate): `TP / (TP + FN)`
   - Calculates **FAR** (False Alarm Rate): `FP / (FP + TN)`
   - Tracks learning rate in history

5. **Advanced Checkpoint Logic (Spahr et al. 2025):**
   ```python
   # Save model if:
   # 1. Both sensitivity AND FAR improve, OR
   # 2. FAR reduces >10% while sensitivity stays within 5%
   
   both_improved = (sensitivity > best_sensitivity) and (far < best_far)
   far_reduction = (best_far - far) / best_far
   sensitivity_tolerance = abs(sensitivity - best_sensitivity)
   
   if both_improved:
       save_model("both metrics improved")
   elif far_reduction > 0.10 and sensitivity_tolerance <= 0.05:
       save_model("FAR reduced with sensitivity tolerance")
   ```

6. **Extended Training History:**
   - Now includes `'lr'` key tracking learning rate per epoch/evaluation

## Usage

### To Use Spahr et al. 2025 Schedule:

Set these in your config file:
```json
{
  "modelConfig": {
    "useLrSchedule": true,
    "useAdamW": true,
    "batchSize": 512,
    "totalTrainingSteps": 50000
  }
}
```

### To Use Original Training Method:

Set these in your config file:
```json
{
  "modelConfig": {
    "useLrSchedule": false,
    "useAdamW": false,
    "epochs": 100,
    "batchSize": 64
  }
}
```

## Backward Compatibility

All changes maintain full backward compatibility:

1. If `useLrSchedule` is not present or `false`, uses original ReduceLROnPlateau scheduler
2. If `useAdamW` is not present or `false`, uses Adam optimizer
3. If advanced checkpoint parameters are missing, uses simple validation loss improvement
4. All new parameters have sensible defaults
5. Epoch-based training still works exactly as before when schedule is disabled

## Training Output

With the new schedule enabled, training output includes:

```
Step 5000/50000 (Epoch 12) - loss: 0.1234 - acc: 0.9567 - 
  val_loss: 0.1456 - val_acc: 0.9423 - 
  sensitivity: 0.8912 - FAR: 0.0543 - lr: 9.87e-04

nnTrainer.trainModel_pytorch(): Saving best model to deepEpiCnnModel_pytorch.pt 
  (FAR reduced by 12.3% with sensitivity within tolerance)
```

## Key Differences from Paper

1. **No ensemble**: Single model training (ensemble can be implemented via runSequence.py with multiple runs)
2. **No data filtering**: Uses existing data augmentation pipeline (may need separate modification)
3. **Validation data**: Uses existing validation split rather than full dataset

## Reference

Spahr et al. (2025). "DeepEpi: A Deep Learning-Based Seizure Detection Model for Epilepsy Patients"
- Optimizer: AdamW with β1=0.9, β2=0.999, weight_decay=0.0
- Learning rate schedule: 1e-5 → 1e-3 (2500 steps) → 3e-5 (45000 steps) → 0 (2500 steps)
- Batch size: 512
- Total steps: 50,000
- Evaluation: Every 5000 steps

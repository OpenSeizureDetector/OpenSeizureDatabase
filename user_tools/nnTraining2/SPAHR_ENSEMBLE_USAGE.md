# Spahr et al. 2025 Implementation Guide

## Overview

Implementation of the deep learning seizure detection approach from:
**Spahr et al. (2025). "Deep learning based detection of generalized convulsive seizures using a wrist-worn accelerometer." Epilepsia.**

This implementation includes:
- Three-phase learning rate schedule (warmup → cosine annealing → cooldown)
- AdamW optimizer with configurable beta parameters
- Step-based training (50,000 steps)
- Balanced batch sampling
- Advanced checkpoint logic (dual criteria for model saving)
- Ensemble prediction with Harrell-Davis quantile aggregation

## Training a Single Model

### Configuration: `nnConfig_deep_pytorch.json`

Key parameters matching the paper:
```json
{
  "dataProcessing": {
    "oversample": "none",
    "undersample": "none",
    "features": ["acc_magnitude"]
  },
  "modelConfig": {
    "framework": "pytorch",
    "batchSize": 512,
    "useLrSchedule": true,
    "useAdamW": true,
    "useBalancedBatches": true,
    "lrStart": 1e-5,
    "lrPeak": 1e-3,
    "lrMainEnd": 3e-5,
    "warmupSteps": 2500,
    "mainSteps": 45000,
    "cooldownSteps": 2500,
    "totalTrainingSteps": 50000,
    "evalEverySteps": 5000,
    "adamwBeta1": 0.9,
    "adamwBeta2": 0.999,
    "weightDecay": 0.0
  }
}
```

### Run Single Model Training

```bash
# Activate virtual environment
source /home/graham/pyEnvs/osdb/bin/activate

# Train single model (70% train, 30% test)
cd /home/graham/osd/OpenSeizureDatabase/user_tools/nnTraining2
python runSequence.py --config nnConfig_deep_pytorch.json --train --kfold 1

# Output will be in: ./output/deepEpiCnnModel/[run_number]/
```

## Training Ensemble Models (Paper Approach)

The paper trains **30 models** through cross-validation and selects the **10 best** for the final ensemble.

### Step 1: Train Multiple Models with K-Fold

```bash
# Train 5-fold cross-validation (creates 5 models)
python runSequence.py --config nnConfig_deep_pytorch.json --train --kfold 5

# For more models (closer to paper's 30), run multiple times with different seeds:
# Edit nnConfig_deep_pytorch.json and change "randomSeed" between runs
python runSequence.py --config nnConfig_deep_pytorch.json --train --kfold 5
# Then change seed and run again...
```

**Output structure:**
```
./output/deepEpiCnnModel/[run_number]/
├── fold0/
│   ├── deepEpiCnnModel_best.pth
│   ├── test_trainDataFeaturesHistory.csv
│   └── test_testDataFeaturesHistory.csv
├── fold1/
│   └── ...
├── fold2/
│   └── ...
├── fold3/
│   └── ...
├── fold4/
│   └── ...
└── kfold_summary.txt
```

### Step 2: Evaluate Ensemble with Quantile Aggregation

```bash
# Single quantile evaluation (paper recommends q=0.6)
python ensemblePredictor.py \
    --config nnConfig_deep_pytorch.json \
    --outputDir ./output/deepEpiCnnModel/1 \
    --quantile 0.6

# Sweep through multiple quantiles to find optimal
python ensemblePredictor.py \
    --config nnConfig_deep_pytorch.json \
    --outputDir ./output/deepEpiCnnModel/1 \
    --sweep

# Test on specific data
python ensemblePredictor.py \
    --config nnConfig_deep_pytorch.json \
    --outputDir ./output/deepEpiCnnModel/1 \
    --quantile 0.6 \
    --testData path/to/custom_test.csv
```

## Understanding the Ensemble Approach

### How Ensemble Prediction Works

1. **Load all N models** from fold directories
2. **For each test window:**
   - Pass window through all N models
   - Collect N prediction scores: s₁, s₂, ..., sₙ
   - Apply Harrell-Davis quantile function with parameter q
   - Get final ensemble score
   - Detect seizure if ensemble_score > 0.5

### Quantile Parameter Effects

| Quantile | Meaning | Sensitivity | False Alarms |
|----------|---------|-------------|--------------|
| 0.4 | 60% of models must agree | High (↑) | More (↑) |
| 0.6 | 40% of models must agree | Optimal | Balanced |
| 0.7 | 30% of models must agree | Lower (↓) | Fewer (↓) |
| 0.9 | 10% of models must agree | Lowest (↓↓) | Very few (↓↓) |

**Paper's results with q=0.6:**
- Training set: 98% sensitivity, 1/6 day FAR
- Test set: 96% sensitivity, 1/8 day FAR

### Tunable Sensitivity

The key advantage of this approach is **post-training adjustability**:
- Change quantile parameter without retraining
- Adapt to user preferences (high sensitivity vs. low false alarms)
- Same ensemble, different operating points

## Configuration Files

### For Paper's Approach (PyTorch)
- **Config:** `nnConfig_deep_pytorch.json`
- **Features:** `acc_magnitude` (3D accelerometer magnitude)
- **Framework:** PyTorch with step-based training
- **Training:** 50,000 steps, batch size 512
- **LR Schedule:** Three-phase (warmup/cosine/cooldown)

### For Legacy Approach (TensorFlow)
- **Config:** `nnConfig_deep.json` or `nnConfig_deep_run.json`
- **Framework:** TensorFlow with epoch-based training
- **Training:** 100 epochs, ReduceLROnPlateau
- **Backward compatible:** Set `useLrSchedule: false`

## Key Differences from Original Code

### What Changed:
1. ✅ **Step-based training** (50k steps instead of 100 epochs)
2. ✅ **Three-phase LR schedule** (warmup → cosine → cooldown)
3. ✅ **AdamW optimizer** with β₁=0.9, β₂=0.999
4. ✅ **Balanced batches** via WeightedRandomSampler
5. ✅ **Advanced checkpoint logic** (dual criteria: both improve OR FAR↓10% + sensitivity within 5%)
6. ✅ **No undersampling** (keep all data, balance during training)

### Backward Compatibility:
All existing configs still work! Set `useLrSchedule: false` to use original approach.

## Validation

Test the pipeline with minimal dataset:
```bash
# Run quick test with test dataset
python runSequence.py --config nnConfig_test.json --train --kfold 1

# Validate data processing integrity
python test_data_processing.py --config nnConfig_test.json
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `batchSize` from 512 to 256 or 128
- Or use CPU: PyTorch will automatically fall back

### Training Too Slow
- Reduce `totalTrainingSteps` from 50000 to 25000
- But this may impact convergence

### Models Not Loading in Ensemble
- Check that fold directories exist: `./output/[model_name]/[run]/fold*/`
- Verify model files: `deepEpiCnnModel_best.pth` in each fold
- Ensure framework matches config (pytorch vs tensorflow)

## References

1. **Spahr et al. (2025)**. "Deep learning based detection of generalized convulsive seizures using a wrist-worn accelerometer." *Epilepsia*, 66(S3), 53-63.

2. **Implementation details**: See `SPAHR_LR_SCHEDULE_IMPLEMENTATION.md` for complete technical documentation.

## Quick Start Checklist

- [ ] Activate virtual environment: `source /home/graham/pyEnvs/osdb/bin/activate`
- [ ] Verify config: `nnConfig_deep_pytorch.json` has correct parameters
- [ ] Train ensemble: `python runSequence.py --config nnConfig_deep_pytorch.json --train --kfold 5`
- [ ] Wait for training (50k steps × 5 folds = ~hours depending on GPU)
- [ ] Evaluate ensemble: `python ensemblePredictor.py --config nnConfig_deep_pytorch.json --outputDir ./output/deepEpiCnnModel/1 --sweep`
- [ ] Check results: `./output/deepEpiCnnModel/1/ensemble_quantile_sweep.json`
- [ ] Adjust quantile based on desired sensitivity/FAR trade-off

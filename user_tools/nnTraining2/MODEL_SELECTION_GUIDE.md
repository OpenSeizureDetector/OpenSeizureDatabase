# Model Selection Criteria - Quick Reference

## Problem

During training, the model sometimes selects checkpoints with very high TPR (sensitivity) but also very high FPR (false positive rate), leading to too many false alarms. This is not useful in practice.

## Solution

New configuration parameters allow you to enforce better TPR/FPR balance:

### 1. **Maximum FPR Threshold** (`saveBestMaxFpr`)

Prevents saving any model with FPR exceeding this threshold, regardless of sensitivity.

```json
"saveBestMaxFpr": 0.15  // Never save if FPR > 15%
```

**Example:**
- Model A: TPR=0.95, FPR=0.30 → ❌ NOT SAVED (FPR too high)
- Model B: TPR=0.85, FPR=0.10 → ✓ SAVED (good balance)

### 2. **Model Selection Metric** (`modelSelectionMetric`)

Choose how to evaluate and compare model checkpoints:

#### Options:

**`"f_beta"`** (Recommended for seizure detection)
- Weighted F-score that balances sensitivity and specificity
- Use `fBeta: 2.0` to favor sensitivity (catching seizures) over low FPR
- Use `fBeta: 0.5` to favor low FPR over sensitivity
- Range: 0 to 1 (higher is better)

**`"youden"`** (Balanced approach)
- Youden's J statistic: TPR - FPR
- Finds optimal balance point on ROC curve
- Range: -1 to 1 (higher is better)

**`"f1"`** (Equal weighting)
- Equal balance between sensitivity and specificity
- Range: 0 to 1 (higher is better)

**`"dual_improvement"`** (Legacy)
- Original Spahr et al. 2025 logic
- Saves when both metrics improve OR FAR reduces significantly

### Configuration Examples

#### For Seizure Detection (Prioritize catching seizures)
```json
{
  "modelSelectionMetric": "f_beta",
  "fBeta": 2.0,
  "saveBestMaxFpr": 0.15,
  "saveBestMinSensitivity": 0.70
}
```

#### For Balanced Performance
```json
{
  "modelSelectionMetric": "youden",
  "saveBestMaxFpr": 0.10,
  "saveBestMinSensitivity": 0.80
}
```

#### For Minimizing False Alarms
```json
{
  "modelSelectionMetric": "f_beta",
  "fBeta": 0.5,
  "saveBestMaxFpr": 0.05,
  "saveBestMinSensitivity": 0.60
}
```

## How It Works

During training, before saving a checkpoint:

1. ✓ **Check FPR threshold**: If `FPR > saveBestMaxFpr`, reject immediately
2. ✓ **Check minimum sensitivity**: If `sensitivity < saveBestMinSensitivity`, reject
3. ✓ **Compare metric**: Calculate selection metric and save if it improved

## Training Output

The improved logging shows:
```
nnTrainer.trainModel_pytorch(): ✓ Saving checkpoint to output/model.pt
nnTrainer.trainModel_pytorch():   Reason: f_beta=0.8654 improved (was 0.8201, sens=0.850, FAR=0.100)
```

Or when rejected:
```
nnTrainer.trainModel_pytorch(): ✗ Not saving: FPR=0.320 exceeds max threshold=0.15
```

## Testing

Run the demo script to see how different metrics compare:
```bash
python user_tools/nnTraining2/test_model_selection_criteria.py
```

## Parameters Summary

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `saveBestMaxFpr` | float or null | null | Maximum acceptable FPR (e.g., 0.15 = 15%) |
| `saveBestMinSensitivity` | float | 0.25 | Minimum acceptable sensitivity |
| `modelSelectionMetric` | string | "dual_improvement" | Metric for model comparison |
| `fBeta` | float | 2.0 | Beta parameter for F-beta score |
| `saveBestOnFarReduction` | float | 0.10 | (Legacy) FAR reduction threshold |
| `saveBestOnSensitivityTolerance` | float | 0.05 | (Legacy) Sensitivity tolerance |

## Files Modified

- [nnTrainer.py](nnTrainer.py): Added `calculate_selection_metric()` and improved checkpoint logic
- [nnConfig_deep_pytorch.json](nnConfig_deep_pytorch.json): Added example configuration
- [test_model_selection_criteria.py](test_model_selection_criteria.py): Demo and testing script

# K-Fold Testing in nnTester.py

## Overview

`nnTester.py` now supports k-fold cross-validation testing to aggregate results across multiple fold models.

## New Command-Line Options

- `--kfold=N` : Test N folds and aggregate results
- `--rerun` : Re-run tests even if results already exist (only used with `--kfold`)

## Usage

### Test All Folds (Use Existing Results)
```bash
python nnTester.py --config nnConfig.json --kfold=5
```

This will:
- Load existing `testResults.json` from each fold directory
- Skip testing if results already exist (fast)
- Aggregate statistics across all folds
- Create summary files

### Test All Folds (Force Re-run)
```bash
python nnTester.py --config nnConfig.json --kfold=5 --rerun
```

This will:
- Re-run `testModel()` for each fold even if results exist
- Useful after modifying test data or model
- Create fresh results for all folds

### Test Single Model (Original Behavior)
```bash
python nnTester.py --config nnConfig.json
```

No change to original behavior when `--kfold` is not specified.

## Directory Structure

The tool expects fold directories to exist:

```
./
├── fold0/
│   ├── best_model.pt (or .keras)
│   ├── testDataFeatures.csv
│   └── testResults.json (created/loaded)
├── fold1/
│   ├── best_model.pt
│   ├── testDataFeatures.csv
│   └── testResults.json
├── fold2/
│   └── ...
└── ...
```

## Output Files

### kfold_summary.txt
Human-readable summary with:
- Mean ± standard deviation for all metrics
- Epoch-based analysis (accuracy, TPR, FPR)
- Event-based analysis (TPR, FPR)
- Detailed results by fold

### kfold_summary.json
Machine-readable JSON with:
- Number of folds
- Timestamp
- Aggregated averages with std dev
- Complete individual fold results

## Validation

The tool **validates** that the specified fold count matches the actual directory structure:

```bash
# If you have 3 folds but specify 5:
python nnTester.py --kfold=5

# Error: "Fold directory not found: ./fold3. Expected 5 folds but fold3 does not exist."
```

This prevents silently computing statistics on the wrong number of folds.

## Statistics Computed

For each metric, the tool computes:
- **Mean**: Average across all folds
- **Standard Deviation**: Variability across folds

### Metrics Included

**Epoch-based (datapoint level):**
- Accuracy (Model & OSD)
- TPR/Sensitivity (Model & OSD)
- FPR (Model & OSD)
- TP, FP, TN, FN counts

**Event-based:**
- TPR/Sensitivity (Model & OSD)
- FPR (Model & OSD)
- TP, FP, TN, FN counts

## Example Output

```
======================================================================
K-FOLD CROSS-VALIDATION SUMMARY
======================================================================
Number of folds: 5

EPOCH-BASED ANALYSIS:
  Model - Accuracy: 0.8542 ± 0.0123
  Model - TPR: 0.8210 ± 0.0156
  Model - FPR: 0.0987 ± 0.0089
  OSD   - Accuracy: 0.7512 ± 0.0067
  OSD   - TPR: 0.7001 ± 0.0112
  OSD   - FPR: 0.1534 ± 0.0098

EVENT-BASED ANALYSIS:
  Model - TPR: 0.8456 ± 0.0134
  Model - FPR: 0.1123 ± 0.0092
  OSD   - TPR: 0.7234 ± 0.0145
  OSD   - FPR: 0.1789 ± 0.0103
======================================================================
```

## Integration with runSequence.py

When using `runSequence.py` with k-fold training, it automatically runs `nnTester.testModel()` for each fold. You can then use this new feature to re-aggregate or update statistics:

```bash
# After training with runSequence.py --kfold=5
cd output_directory
python ../user_tools/nnTraining2/nnTester.py --config nnConfig.json --kfold=5
```

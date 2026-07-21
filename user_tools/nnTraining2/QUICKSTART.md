# Quick Start Guide: Nested K-Fold Cross-Validation

## Installation

Already configured in your workspace. Just activate the virtual environment:

```bash
cd /home/graham/osd/OpenSeizureDatabase
source venv/bin/activate
```

## Standard K-Fold (Existing Behavior)

Train with 5-fold cross-validation:

```bash
python3 user_tools/nnTraining2/runSequence.py \
  --config nnConfig_deep_pytorch.json \
  --kfold 5 \
  --train \
  --outDir ./output
```

Result: Trains 5 models, one per fold.

## Nested K-Fold (New Feature)

Train with nested k-fold: 3 outer folds × 5 inner folds = 15 total models

```bash
python3 user_tools/nnTraining2/runSequence.py \
  --config nnConfig_deep_pytorch.json \
  --kfold 5 \
  --nestedKfold 3 \
  --train \
  --outDir ./output
```

Result: 
- 3 completely independent test sets
- 5 cross-validation splits per outer fold
- Unbiased generalization estimates

## Recommended Configuration

Update your config file to use nested k-fold effectively:

```json
{
  "dataProcessing": {
    "testProp": 0.0,
    "validationProp": 0.0
  }
}
```

This ensures all data is split via k-fold, not fixed proportions.

## Checking the Help

```bash
python3 user_tools/nnTraining2/runSequence.py --help
```

Look for:
- `--kfold KFOLD` - number of inner folds
- `--nestedKfold NESTEDKFOLD` - number of outer folds (new!)

## Output Files

Results saved to `output/deepEpiCnnModel_pytorch/{run_number}/`

### Standard K-Fold Output
```
fold0/{models, results}
fold1/{models, results}
fold2/{models, results}
fold3/{models, results}
fold4/{models, results}
kfold_summary.txt        (average results)
kfold_summary.json       (detailed results)
```

### Nested K-Fold Output
```
outerfold0/
  ├── outerfold_test.json  (independent test set #1)
  ├── fold0/{models, results}
  ├── fold1/{models, results}
  └── ...

outerfold1/
  ├── outerfold_test.json  (independent test set #2)
  ├── fold0/{models, results}
  └── ...

outerfold2/
  ├── outerfold_test.json  (independent test set #3)
  └── ...

kfold_summary.txt        (averages across all 15 model evaluations)
kfold_summary.json       (detailed metrics)
```

## Common Scenarios

### Scenario 1: Quick Testing (5 minutes)
```bash
python3 user_tools/nnTraining2/runSequence.py \
  --config nnConfig_deep_pytorch.json \
  --kfold 2 \
  --train
```
2 models, ~10 minutes training.

### Scenario 2: Standard Evaluation (1-2 hours)
```bash
python3 user_tools/nnTraining2/runSequence.py \
  --config nnConfig_deep_pytorch.json \
  --kfold 5 \
  --train
```
5 models, ~50 minutes training.

### Scenario 3: Publication-Quality Results (3-6 hours)
```bash
python3 user_tools/nnTraining2/runSequence.py \
  --config nnConfig_deep_pytorch.json \
  --kfold 5 \
  --nestedKfold 3 \
  --train
```
15 models (3 outer × 5 inner), provides truly independent test sets.

## Reporting Results

### Standard K-Fold Reporting
"We used 5-fold cross-validation to evaluate the model..."
(Note: test sets are not completely independent from hyperparameter selection)

### Nested K-Fold Reporting (Recommended)
"We used nested 3×5 cross-validation where 3 outer folds provided completely 
independent test sets and 5 inner folds were used for model selection and 
hyperparameter tuning. Final performance is reported as the average across 
the 3 outer folds..."
(Unbiased generalization estimate!)

## Advanced Usage

### Data Splitting Only (No Training)

Just split data without training:

```bash
python3 user_tools/nnTraining2/splitData.py \
  --config nnConfig_deep_pytorch.json \
  --kfold 5 \
  --nestedKfold 3
```

Creates nested directory structure with train/test splits.

### Custom Fold Numbers

3 outer folds, 3 inner folds:
```bash
python3 user_tools/nnTraining2/runSequence.py \
  --config nnConfig_deep_pytorch.json \
  --kfold 3 \
  --nestedKfold 3 \
  --train
```
Result: 9 models trained.

## What Gets Printed

When running nested k-fold, you'll see output like:

```
================================================================================
runSequence: Training with nested k-fold validation
  Outer folds: 3 (independent test sets)
  Inner folds: 5 (per outer fold)
================================================================================

================================================================================
runSequence: OUTER FOLD 0
================================================================================

runSequence: Outer Fold 0, Inner Fold 0
runSequence: Flattening test data ...
runSequence: Flattening train data ...
runSequence: Training pytorch neural network model
runSequence: Testing Model
runSequence: Finished outer fold 0, inner fold 0, data in folder ...

runSequence: Outer Fold 0, Inner Fold 1
...
```

This confirms nested k-fold is working correctly.

## Troubleshooting

**Q: How long will nested k-fold take?**
A: ~3× longer than standard k-fold (since 3× more models to train)

**Q: Can I use the same config as before?**
A: Yes! Default `--nestedKfold 1` gives standard k-fold behavior.

**Q: Where are the test results saved?**
A: In `output/deepEpiCnnModel_pytorch/{run_number}/kfold_summary.txt`

**Q: Which results should I report in a paper?**
A: For nested k-fold, report the average from `kfold_summary.txt` which 
represents the average performance across all outer fold test sets.

## Further Reading

- [NESTED_KFOLD_USAGE.md](../../NESTED_KFOLD_USAGE.md) - Complete usage guide
- [NESTED_KFOLD_README.md](./NESTED_KFOLD_README.md) - Implementation details
- [nnConfig_nested_kfold_example.json](./nnConfig_nested_kfold_example.json) - Example config

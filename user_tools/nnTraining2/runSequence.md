## runSequence.py — pipeline overview and configuration

This document describes the data processing steps implemented by `user_tools/nnTraining2/runSequence.py`, the configuration values the script expects, and example configuration snippets you can adapt.

## Purpose

`runSequence.py` implements the end-to-end toolchain used to prepare data, augment it, extract features, and train/test models for the Open Seizure Database neural-network training flow. The script is conservative: it only re-runs steps that are missing or explicitly requested via the `clean`/`rerun` control, so you can re-run the pipeline incrementally and avoid repeating expensive work.

## High-level processing sequence

When run in training mode (`train=True`), the script performs the following steps for each k-fold (or a single run when `kfold=1`):

- Create an output folder for the run (under `outDir/<modelFname>/<run_number>/`, unless `rerun` is specified).
- Selection and splitting:
  - `selectData.selectData(...)` is called to create the master selected dataset (JSON). If missing, the script will re-generate selection and split files.
  - `splitData.splitData(...)` creates train/test/val JSON files (one per fold when k-folding).
- Flattening: convert the JSON OSDB event files to flattened CSVs via `flattenData.flattenOsdb(...)`. Files produced:
  - train CSV (`trainDataFileCsv`) and test CSV (`testDataFileCsv`) per fold.
- Augmentation: run `augmentData.augmentSeizureData(...)` to produce an augmented training CSV (`trainAugmentedFileCsv`).
- Feature extraction: call `extractFeatures(...)` to compute epoch-level features for train and test CSVs. Produces `trainFeaturesFileCsv` and `testFeaturesFileCsv`.
- Feature history: `add_feature_history(...)` is called to build history-feature versions (rolling windows / temporal history) of the feature files. These replace the raw features files for training.
- Training and testing:
  - For `modelType == "sklearn"`, `skTrainer.trainModel(...)` and `skTester.testModel(...)` are used.
  - For `modelType == "tensorflow"`, `nnTrainer.trainModel(...)` and `nnTester.testModel(...)` are used.

Notes:
- The script checks for existence of each output file and will skip steps whose outputs are already present (unless you remove them or use `clean`).
- `randomSeed` (if present in config) is used to seed NumPy and `random` before each fold to make results reproducible.

## Command / launcher usage

`runSequence.py` exposes a programmatic function `run_sequence(args)` which accepts a dictionary with keys matching the CLI options used by the surrounding tooling. Example invocation from Python:

```python
from user_tools.nnTraining2.runSequence import run_sequence

args = {
    'config': 'nnConfig.json',
    'kfold': 5,
    'rerun': 0,
    'outDir': './output',
    'train': True,
    'test': False,
    'clean': False,
    'debug': True
}

run_sequence(args)
```

If you prefer to run the pipeline manually, you can create a small wrapper script that parses command-line arguments and calls `run_sequence(...)` (the repo does not ship a CLI wrapper inside `runSequence.py`).

## Key configuration parameters

`runSequence.py` reads the configuration JSON file (path provided in `args['config']`) using `libosd.configUtils.loadConfig(...)`. The configuration must supply several sections and keys used by the script and the modules it calls. Below are the most important keys and their meanings.

Top-level sections and keys used by `runSequence.py`:

- `dataFileNames` — filenames used by the pipeline (relative to fold output directories). Common keys referenced by the script:
  - `allDataFileJson` — selected master JSON (produced by `selectData`).
  - `testDataFileJson` — test split JSON filename.
  - `trainDataFileJson` — train split JSON filename.
  - `valDataFileJson` — validation split JSON filename.
  - `testDataFileCsv` — flattened CSV filename for test data.
  - `trainDataFileCsv` — flattened CSV filename for train data.
  - `valDataFileCsv` — flattened CSV filename for val data.
  - `trainAugmentedFileCsv` — augmented training CSV filename (produced by `augmentData`).
  - `testBalancedFileCsv` — balanced test CSV filename (optional, produced by `augmentData.balanceTestData` if used).
  - `trainFeaturesFileCsv` — feature CSV filename for training (produced by `extractFeatures`).
  - `testFeaturesFileCsv` — feature CSV filename for testing.
  - `trainFeaturesHistoryFileCsv` / `testFeaturesHistoryFileCsv` — history-augmented feature filenames used for training/testing after `add_feature_history`.

- `modelConfig` — model settings and filenames:
  - `modelType` — either `sklearn` or `tensorflow`. Controls which training/test modules are used.
  - `modelFname` — model name / prefix used to create the output folder under `outDir`.

- `randomSeed` (optional) — integer used to seed NumPy and Python random for reproducibility.

Additional knobs (used indirectly by other modules called from the pipeline):

- `dataProcessing` — controls behavior of augmentation and feature extraction modules (examples):
  - `noiseAugmentation` (bool)
  - `noiseAugmentationFactor` (int)
  - `noiseAugmentationValue` (float)
  - `phaseAugmentation` (bool)
  - `userAugmentation` (bool)
  - `oversample`, `undersample` (str or None): 'none', 'random', 'smote'
  - `window`, `step` (for feature extraction windows)
  - `highPassFreq`, `highPassOrder` (filtering)
  - `worker_count`, `batch_size`, `stream_chunksize`, `progress_interval` (performance tuning for streaming/multiprocessing in `extractFeatures` and similar)

Refer to the individual module docs for more keys used by `flattenData`, `augmentData`, `extractFeatures`, and the trainers.

## Minimal example `nnConfig.json`

Below is a compact example config that demonstrates the main sections used by `runSequence.py`. Adapt file names and paths to your environment.

```json
{
  "dataFileNames": {
    "allDataFileJson": "all_data.json",
    "testDataFileJson": "test_split.json",
    "trainDataFileJson": "train_split.json",
    "valDataFileJson": "val_split.json",
    "testDataFileCsv": "test_flat.csv",
    "trainDataFileCsv": "train_flat.csv",
    "valDataFileCsv": "val_flat.csv",
    "trainAugmentedFileCsv": "train_augmented.csv",
    "trainFeaturesFileCsv": "train_features.csv",
    "testFeaturesFileCsv": "test_features.csv",
    "trainFeaturesHistoryFileCsv": "train_features_hist.csv",
    "testFeaturesHistoryFileCsv": "test_features_hist.csv",
    "testBalancedFileCsv": "test_balanced.csv"
  },
  "modelConfig": {
    "modelType": "sklearn",
    "modelFname": "training"
  },
  "randomSeed": 42,
  "dataProcessing": {
    "noiseAugmentation": true,
    "noiseAugmentationFactor": 2,
    "noiseAugmentationValue": 0.05,
    "phaseAugmentation": true,
    "userAugmentation": false,
    "oversample": "random",
    "undersample": "none",
    "window": 125,
    "step": 125,
    "worker_count": 4,
    "batch_size": 1000,
    "stream_chunksize": 20000
  }
}
```

## Output layout

When a run is created the script writes outputs under `outDir/<modelFname>/<run_number>/`. For k-fold experiments it writes each fold under `.../<run_number>/foldN/` and places the per-fold CSVs and model artifacts there. Typical files produced per fold:

- `train_flat.csv`, `test_flat.csv` — flattened train/test CSVs
- `train_augmented.csv` — augmented training CSV
- `train_features.csv`, `test_features.csv` — extracted features (or their history variants)
- model artifacts and plots: e.g. `<modelFname>.keras`, `<modelFname>_confusion.png`, `<modelFname>_training.png`, etc.

The script also produces a k-fold summary (`kfold_summary.txt` and `kfold_summary.json`) in the run folder.

## Debugging and re-run control

- `clean=True` will delete many of the output files listed in the script and exit.
- `rerun` allows re-using or overwriting a specific run folder instead of creating a new sequential one.
- `debug=True` prints additional diagnostic messages.

## Data processing options (detailed)

The `dataProcessing` section in the config controls augmentation, feature extraction, streaming and performance knobs used across the pipeline. Below are the keys that the tools read, their meanings, types and default values (taken from `user_tools/nnTraining2/nnConfig.json` where available).

- `nHistory` (int, default: 5)
  - Number of historical feature steps to include when building feature-history CSVs (used by `addFeatureHistory`).

- `highPassFreq` (float, default: 0.5)
  - Cutoff frequency (Hz) for high-pass filtering applied to accelerometer time series prior to feature extraction.

- `highPassOrder` (int, default: 2)
  - Filter order for the Butterworth high-pass filter.

- `window` (int, default: 125)
  - Number of accelerometer samples per epoch/window used for feature extraction (125 = 5s at 25 Hz).

- `step` (int, default: same as `window`)
  - Step size (samples) between successive epochs; smaller than `window` produces overlapping epochs.

- `oversample` (str, default: "none")
  - Event-level oversampling strategy. Supported values: `"none"`, `"random"`, `"smote"` (SMOTE falls back to random for single-feature event ids).

- `undersample` (str, default: "random")
  - Event-level undersampling strategy. Supported values: `"none"`, `"random"`.

- `worker_count` (int or null)
  - Number of worker processes for multiprocessing in streaming extraction. If `null` the code uses CPU count - 1.

- `batch_size` (int, default: 1000)
  - Number of feature rows to buffer before writing a batch to disk in streaming mode.

- `stream_chunksize` (int, default: 20000)
  - Number of CSV rows to read per pandas chunk when streaming flattened CSVs by event.

- `stream_low_memory` (bool, default: false)
  - Forwarded to `pandas.read_csv(..., low_memory=...)` when loading streamed temporary outputs; tune if you see dtype warnings.

- `stream_dtype_map` (dict or null)
  - Optional dtype mapping passed to `pandas.read_csv` for more reliable streaming reads and to avoid mixed-type inference.

- `noiseAugmentation` (bool, default: false)
  - Enable additive Gaussian noise augmentation on seizure rows.

- `noiseAugmentationFactor` (int, default: 20)
  - Number of noisy copies to create per seizure row when `noiseAugmentation` is enabled.

- `noiseAugmentationValue` (float, default: 30.0)
  - Standard deviation (same units as accelerometer columns) of the Gaussian noise added.

- `phaseAugmentation` (bool, default: false)
  - Enable phase (temporal shift) augmentation which constructs sliding windows from adjacent rows within an event.

- `userAugmentation` (bool, default: false)
  - Enable user-level resampling such that seizure samples are oversampled to balance user contributions.

- `splitTestTrainByEvent` (bool, default: true)
  - When `true` the split into train/test is performed at the event-level so that datapoints from the same event don't appear in both sets.

- `accSdThreshold` (float, default: 0.0)
  - Threshold applied to accelerometer standard deviation for event filtering (small values remove very quiet events).

- `testProp`, `validationProp` (float)
  - Proportion of events reserved for test/validation when splitting (e.g. 0.3 for 30% test).

- `seizureTimeRange` (list of two ints, default: [-20, 20])
  - Time window (seconds) around labelled seizure time used by some selection or labeling code.

- `features` (list of strings)
  - Subset of available calculated features to include in training/testing. The pipeline extracts a comprehensive set of features but only passes the selected names downstream. See the `Available features` section below.

- `progress_interval` (int, default: 100)
  - How often (in events) to print progress updates when streaming/multiprocessing.


### Performance tuning recommendations

- For moderate datasets (<100k rows) the default in-memory path (no streaming) is convenient. For larger flattened CSVs enable streaming/multiprocessing via `extractFeatures` and set `worker_count` to a sensible number (cpu_count - 1 is a good start).
- Increase `batch_size` to amortise disk writes but watch memory use. Decrease `stream_chunksize` to reduce memory spikes when pandas reads large chunks.
- Provide a `stream_dtype_map` when you know column dtypes up-front (prevents mixed-type columns and speeds up `read_csv`).


## Available features (calculated by accelFeatures)

The feature extractor (`accelFeatures.calculate_epoch_features`) computes both time-domain and frequency-domain features for each axis (x, y, z) and the acceleration magnitude plus some metadata features. The complete list of available feature names (as found in `user_tools/nnTraining2/nnConfig.json` `_all_features`) is:

```
osdSpecPower
osdRoiPower
specPower
roiPower
activity_count_magnitude
mean_magnitude
std_magnitude
skewness_magnitude
kurtosis_magnitude
zcr_magnitude
mean_freq_magnitude
entropy_magnitude
total_power_magnitude_osdRoi
peak_psd_magnitude_osdRoi
total_power_magnitude_osdSpec
peak_psd_magnitude_osdSpec
total_power_magnitude_osdFlap
peak_psd_magnitude_osdFlap
total_power_magnitude_seizure_main
peak_psd_magnitude_seizure_main
total_power_magnitude_initial_clonus
peak_psd_magnitude_initial_clonus
total_power_magnitude_late_clonus
peak_psd_magnitude_late_clonus
total_power_magnitude_1-3Hz
peak_psd_magnitude_1-3Hz
total_power_magnitude_2-4Hz
peak_psd_magnitude_2-4Hz
total_power_magnitude_3-5Hz
peak_psd_magnitude_3-5Hz
total_power_magnitude_4-6Hz
peak_psd_magnitude_4-6Hz
total_power_magnitude_5-7Hz
peak_psd_magnitude_5-7Hz
total_power_magnitude_6-8Hz
peak_psd_magnitude_6-8Hz
total_power_magnitude_7-9Hz
peak_psd_magnitude_7-9Hz
total_power_magnitude_8-10Hz
peak_psd_magnitude_8-10Hz
hr
o2sat
specPower
roiPower
acc_magnitude
meanLineLengthMag
powerMag_0.5-2.5
powerMag_2.5-4.5
powerMag_4.5-6.5
powerMag_6.5-8.5
powerMag_8.5-10.5
powerMag_10.5-12.5
meanLineLengthX
powerX_0.5-2.5
powerX_2.5-4.5
powerX_4.5-6.5
powerX_6.5-8.5
powerX_8.5-10.5
powerX_10.5-12.5
meanLineLengthY
powerY_0.5-2.5
powerY_2.5-4.5
powerY_4.5-6.5
powerY_6.5-8.5
powerY_8.5-10.5
powerY_10.5-12.5
meanLineLengthZ
powerZ_0.5-2.5
powerZ_2.5-4.5
powerZ_4.5-6.5
powerZ_6.5-8.5
powerZ_8.5-10.5
powerZ_10.5-12.5
```

Note: feature names include both per-axis aggregated statistics (mean, std, skewness, kurtosis, zero-crossing rate, mean spectral frequency, entropy) and band-specific PSD totals/peaks for multiple named bands (e.g. `seizure_main`, `initial_clonus`, `1-3Hz`, etc.).

To reduce model input dimensionality list only the `features` you want to use in the config (the example `nnConfig.json` includes a recommended compact set). The trainers (`skTrainer`/`nnTrainer`) select feature columns according to that list.


## Implementation notes and pointers

- `runSequence.py` is a coordinating script: most heavy lifting is delegated to these modules in `user_tools/nnTraining2`:
  - `selectData` — selects events and produces the master JSON
  - `splitData` — splits the master JSON into train/test/val JSONs
  - `flattenData` — flattens JSON -> CSV (per-event rows expanded to accelerometer samples columns)
  - `augmentData` — applies phase/noise/user augmentation to flattened CSVs
  - `extractFeatures` — streams flattened CSVs, computes epoch features, supports multiprocessing and batched writes
  - `addFeatureHistory` — builds history-enriched feature CSVs

- If you plan to convert everything to streaming mode (as discussed in code review), focus first on `flattenData` output shape and column names so that downstream streaming readers (for augmentation and feature extraction) can rely on constant schemas.

## Troubleshooting

- If a step is repeatedly skipped, inspect the target output file path in the run folder — the script will skip a step when the expected file already exists.
- For reproducibility between runs, set `randomSeed` in the configuration and ensure `worker_count` is set sensibly.
- For very large datasets prefer the streaming/multiprocessing config knobs in `dataProcessing` (see `extractFeatures` docs in `user_tools/nnTraining2/extractFeatures.py`).

## Contact

If you need a tailored example config for a particular model type (`tensorflow` vs `sklearn`) or want me to convert the pipeline to always use streaming mode end-to-end, tell me which modules you'd like converted first and I will make a small, backward-compatible plan.

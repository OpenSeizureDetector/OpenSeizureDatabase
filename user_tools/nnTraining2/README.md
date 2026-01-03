Running Tests
-------------

Unit tests for the data processing pipeline are provided in files named `test_*.py` in this folder. These tests use pytest and check the expected behaviour of selectData, splitData, flattenData, and extractFeatures using the provided simulated_events.json.

To run all tests, make sure you have pytest installed:

```bash
pip install pytest
```

Then run:

```bash
pytest .
```

Or to run a specific test file:

```bash
pytest test_flattenData.py
```

Do not use `python test_xxx.py` directly, as pytest is required to discover and run the tests properly.


nnTraining2
===========

It is intended that this folder will be a replacement for the scripts in nnTraining - it will provide a tool chain to train a neural network based seizure detector using OSDB data.
The original nnTraining scripts used a lot of system memory - it is intended to split the data processing pipeline into completely separate processes to try to reduce the memory requirement.

**NEW**: This toolchain now supports both **TensorFlow/Keras** and **PyTorch** frameworks! Choose your preferred framework via configuration.

Framework Support
-----------------

The nnTraining2 toolchain supports both TensorFlow and PyTorch for model training:

### TensorFlow/Keras (Default)
- Uses `tensorflow` and `keras` for model definition and training
- Config example: `nnConfig_deep.json`
- Model file extension: `.keras`
- Requirements: `pip install tensorflow keras`

### PyTorch
- Uses `torch` for model definition and training
- Config example: `nnConfig_deep_pytorch.json`
- Model file extension: `.pt`
- Requirements: `pip install -r requirements_pytorch.txt`

### Selecting a Framework

Set the `framework` field in your config file's `modelConfig` section:

```json
"modelConfig": {
    "framework": "pytorch",
    "modelClass": "user_tools.nnTraining2.deepEpiCnnModel_torch.DeepEpiCnnModelPyTorch",
    ...
}
```

Or for TensorFlow:

```json
"modelConfig": {
    "framework": "tensorflow",
    "modelClass": "user_tools.nnTraining2.deepEpiCnnModel.DeepEpiCnnModel",
    ...
}
```

**Note**: If `framework` is not specified, TensorFlow is used by default for backward compatibility.

### Available Models

| Framework   | Model Class | File |
|-------------|-------------|------|
| TensorFlow  | `DeepEpiCnnModel` | `deepEpiCnnModel.py` |
| PyTorch     | `DeepEpiCnnModelPyTorch` | `deepEpiCnnModel_torch.py` |

Both implementations provide the same 14-layer 1D CNN architecture from the DeepEpi paper (Spahr et al., 2025).

### Installation

**For TensorFlow:**
```bash
pip install tensorflow keras numpy scipy matplotlib pandas scikit-learn imbalanced-learn python-dateutil
```

**For PyTorch:**
```bash
pip install -r requirements_pytorch.txt
```

### GPU Support

- **TensorFlow**: Requires `tensorflow-gpu` or TensorFlow 2.x with CUDA support
- **PyTorch**: Automatically uses CUDA if available. Check with:
  ```python
  import torch
  print(torch.cuda.is_available())
  ```

Data Processing Pipeline
------------------------

  - Select data - select a subset of the OSDB data based on specified filter criteria (e.g. only specific users, only events containing 3d accelerometer data etc.)
  - Split data - split the dataset into test and train parts.
  - Flatten data - convert the data from .json files into .csv files with one row per event
  - Augment data - provide various data augmentation functions
  - Balance data - downsample the negative data events to balance the positive and negative datasets
  - Train network - train the neural network based on the final set of data.


Neural Network Input Formats
----------------------------

We define a number of data formats that will be the input format for the neural network.   Possible formats are:

  - 1: Simple 1d accelerometer data (125 samples at 25Hz (=5 seconds of data) of vector magnitude values)
  - 2: 1d accelerometer data with heart rate (as for 1 above, plus an additional column for heart rate measurement - heart rate is recorded once in each 5 second period).
  - 3: 3d accelerometer data (3 rows, X, Y and Z with 125 columns, sampled at 25 Hz to give 5 seconds of 3d data.
  - 4: 3d accelerometer data with heart rate (as for 3 above plus an additional column for heart rate measurement (heart rate value is repeated in each of the three rows)))

Select Data (selectData.py)
-----------
Reads the osdb json files specified in osdbcfg.json and applies filters (specified in osdbcfg.json) to remove data which is not required.
Splits the data into a test and train dataset, based on the testProp parameter to specify the proportion of the data to be used for testing.
Saves the test and train data file into the current working directory.

Flatten Data (flattenData.py)
----------
Reads the test and train .json files and converts each datapoint into a row in a .csv file, saving the .csv files into the current working directory.

Usage:  flattenData.py -i testData.json -o testData.csv


nnTraining2
===========

It is intended that this folder will be a replacement for the scripts in nnTraining - it will provide a tool chain to train a neural network based seizure detector using OSDB data.
The original nnTraining scripts used a lot of system memory - it is intended to split the data processing pipeline into completely separate processes to try to reduce the memory requirement.

Data Processing Pipeline
------------------------

  - Select data - select a subset of the OSDB data based on specified filter criteria (e.g. only specific users, only events containing 3d accelerometer data etc.)
  - Split data - split the dataset into test and train parts.
  - Flatten data - convert the data from .json files into .csv files with one row per event
  - Augment data - provide various data augmentation functions
  - Balance data - downsample the negative data events to balance the positive and negative datasets
  - Train network - train the neural network bsaed on the final set of data.


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


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


Network Input Formats
---------------------

We define a number of data formats that will be the input format for the neural network.   Possible formats are:

  - 1: Simple 1d accelerometer data (125 samples at 25Hz (=5 seconds of data) of vector magnitude values)
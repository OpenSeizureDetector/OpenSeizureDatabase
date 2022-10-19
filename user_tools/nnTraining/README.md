nnTraining
==========

Overview
--------
nnTrainer is a tool to train a Keras (Tensorflow) based neural network model using
OpenSeizureDatabase data.
It will split the OpenSeizureDatabase data into train, validation and test datasets.  It then
trains the model using the train and validation data and saves the trained model as a .h5 file.

The makeTfliteModel.py script is used to convert the .h5 model file into the format used by TensorFlow Lite in the Android App.

Usage
-----
nnTrainer.py accepts a number of command line parameters:
  * --config <filename> - the filename of a JSON configuration file to use (See below)
  * --model <name> - the root file name of the model to be created (without the .h5 extension)
  * --test - if this parameter is specified the model is not trained, but instead tested using the test dataset to create a number of output files showing reliability statistics.
  * --debug - if set, enables more detailed debugging output.

  Configuration File
  ------------------
  The configuration file specified on the command line shoudl be a file containing a single JSON object
  as shown below (annotated with comments starting '#' which must NOT be present in the actual file.):
  {
    "debug": false,                     # Enable or Disable detailed debugging output
    "osdbCfg":"../osdbCfg.json",        # Filename of an additional configuration file with details of the OpenSeizureDatabase installation
    "modelClass": "cnnModel.CnnModel",  # Module and Class ID of the python definition of the neural network model.
    "epochs": 500,                      # Maximum number of epochs to train.
    "batchSize": 100,                   # Batch Size for training
    "seizureTimeRange":[-20,20],        # For any seizure events in the database which do not include a 'seizureTimes' element, use this value as the range in seconds from the event time to include data.
    "oversample": "random",              # Method of oversampling to balance seizure and false alarm data (random, SMOTE or None)
    "phaseAugmentation": false,          # If true, phase Augmentation is enabled to generate more seizure data.
    "splitTestTrainByEvent": true,       # If true the test/train split is done at the event level.  If false, it is done by datapoint.
    "testProp" : 0.2,                   # Proportion of the data to be kept back for the test dataset.
    "validationProp": 0.2,              # Proportion of the data to be kept back for validation.
    "randomSeed" : 4,                   # Random number generator seed (to give repeatable results)
    "lrFactor" : 0.9,                   # Learning rate reduction factor.
    "lrPatience": 20,                   # Learning Rate reduction patience parameter.
    "lrMin": 0.0001,                    # Minimum learning rate
    "earlyStoppingPatience": 200,       # Patience parameter used for early stopping to avoid over training.
    "trainingVerbosity": 1              # If 1 keras training progress output is sent to console.
  }

  OSDB Configuration
  ------------------
  The osdbCfg parameter should point to a JSON file which contains a single object as follows:
  {
    "allSeizuresFname":"osdb_3min_allSeizures.json",  # Filename for seizure data
    "falseAlarmsFname":"osdb_3min_falseAlarms.json",  # Filename for non-seizure data.
    "invalidEvents": [886, 1850, 1933 ]               # List of event IDs to be excluded from processing
  }

  seizureTimes
 -----
  The seizure data in the database typically contains 3 minutes of data, centred on the 'Event Time'
  It is possible that the seizure only starts a short time before the Event Time, so it will harm
  the training accuracy if all the data is assumed to be representing seizure movements.
  The Event data in the database may contain a 'seizureTimes' value which specifies the time that
  the data curator estimates the seizure to start and end, based on examination of the acceleromter data.
  If the 'seizureTimes' value is not included in the database, the value of the configuration file
  'seizureTimeRange' is used instead.

  Augmentation
  ----
  With image processing it is common to increase the number of training images by using the 
  available images rotated or translated slightly in different directions to create additional
  training images.
  The equivalent of this for the accelerometer data is 'Phase Augmentation' which can be enabled using
  the 'phaseAugmentation' configuration parameter.
  When phaseAugmentation is enabled, two consecutive 5 second data points are used to generate a number
  of intermediate 5 second sequences by offseting the data by one reading at a time.
  NOTE:  This is NOT implemented yet 

  Oversampling
  ----
  We have much more non-seizure data in the database than genuine seizure data.  If we used the data
  'raw' it would bias the model to predicting non-seizure results, so would give poor seizure detection reliability.
  To avoid this we can oversample the seizure data to increase the number of seizure datapoints to be
  the same as the number of non-seizure datapoints.
  Two ways are available to do this 
    * 'random' selects seizure datapoints at random from the training data.
    * 'smote' creates synthetic datapoints by interpolating between two randomly selected datapoints from the training data.


Output
----
The trained model is saved to the command line specified 'model' value with a .h5 extension.
Several graphs and a text summary of test statistics are also saved using the specified 'model' parameter as the prefix and either a .txt or .png extension.

Support / Issues
----
If you find an issue with this system, please create an issue in the Github issue tracker and contact
graham@openseizuredetector.org.uk

Licence
---
The code licence is GPL3 or later.

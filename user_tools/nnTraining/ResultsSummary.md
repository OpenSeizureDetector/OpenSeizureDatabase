# Neural Network Seizure Detector Development - Results Summary

Starting Point - Convolutional neural network based on https://keras.io/examples/timeseries/timeseries_classification_from_scratch/

Using only accelerometer magnitude data

Used 10min grouped database allSeizures and falseAlarms files.
75% train, 25% test.
Set for 500 epochs, but early stopping function often stopped it early.

Version 0.01:
-----------
Raw accelerometer data.
Uses all available datapoints for each event.
Trained on 1344 seizure datapoints and 3648 false alarm datapoints
Tested on 448 seizure datapoints and 1217 false alarm datapoints.
Stopped at epoch 248
Test Accouracy 0.86, Test Loss 0.35

Version 0.02:
-----------
As for V0.01 except
Only uses datapoints between -20 sec and +40 sec from event time for seizure
events to limit the number of 'non-sezure' datapoints marked as seizures in
V0.01.
Trained on 571 seizure datapoints and 3649 false alarm datapoints
Tested on 191 seizure datapoints and 1216 false alarm datapoints.
Stopped at epoch 239
Test Accouracy 0.92, Test Loss 0.30


Summary
-------
Ver.	Test	Loss	
V0.01 	0.86	0.35
V0.02 	0.92	0.30


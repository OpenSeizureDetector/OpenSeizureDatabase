# Neural Network Seizure Detector Development - Results Summary

Starting Point - Convolutional neural network based on https://keras.io/examples/timeseries/timeseries_classification_from_scratch/

Using only accelerometer magnitude data

Used 10min grouped database allSeizures and falseAlarms files.
75% train, 25% test.
Set for 500 epochs, but early stopping function often stopped it early.

Version 0.01:
-----------
  * Raw accelerometer data.
  * Uses all available datapoints for each event.
  * Trained on 1344 seizure datapoints and 3648 false alarm datapoints
  * Tested on 448 seizure datapoints and 1217 false alarm datapoints.
  * Stopped at epoch 248
  * Test Accouracy 0.86, Test Loss 0.35

Version 0.02:
-----------
As for V0.01 except
  * Only uses datapoints between -20 sec and +40 sec from event time for seizure
events to limit the number of 'non-sezure' datapoints marked as seizures in
V0.01.
  * Trained on 571 seizure datapoints and 3649 false alarm datapoints
  * Tested on 191 seizure datapoints and 1216 false alarm datapoints.
  * Stopped at epoch 239
  * Test Accouracy 0.92, Test Loss 0.30
  * Stopped at epoch 320
  * Test Accouracy 0.91, Test Loss 0.31
  * Stopped at epoch 188
  * Test Accouracy 0.92, Test Loss 0.29
  * Stopped at epoch 234
  * Test Accouracy 0.92, Test Loss 0.28

Version 0.03:
-------------
As for V0.02 except:
  * Added more false alarms generated using V0.02 (walking and running caused V0.02 to alarm).
  * Used 3min grouped database to increase amount of data.
  * Used warnings as well as alarm events to increase the amount of false alarm data used to train.
  * Trained using 809 seizure dtapoints and 18338 false alarm datapoints
  * Tested using 270 seizure datapoints and 6113 false alarm datapoints
  * Stoped at epoch 130.
  * Test Accuracy 0.95, Test Loss 0.14.
Ran through testRunner:
  * Results Summary
	  * Category, OSD_v1, nn_v0.03
	  * tcSeizures, 0.78, 0.43
	  * allSeizures, 0.73, 0.23
	  * falseAlarms, 0.61, 1.00
So false alarm performance very good, but seizure detection performance
has deteriorated - presumably because we have trained with much more
false alarm data.



Version 0.04:
-------------
As for V0.03 except:
  * Introduced re-sampling to increase the number of seizure datapoints to be the
same as the number of false alarm datapoints.
  * Trained using 18338 seizure datapoints and 18338 false alarm datapoints
  * Tesing using 6113 seizure datapoints and 6113 false alarm datapoints
  * Stopped at Epoch No. 204.
  * Test accuracy 0.903
  * Test loss 0.277

TestRunner Results Summary
  * Category, OSD_v1, nn_v0.04
  * tcSeizures, 0.78, 1.00
  * allSeizures, 0.73, 0.98
  * falseAlarms, 0.61, 0.90

These results suggest that based on the available data, the neural network
has very high seizure detection rate and around a quarter of the false alarm
rate of the OSD algorithm (10% false alarms compared to 39% for OSD).
So build a new version of the Android App with V0.04 model.


Summary
-------
In the table below:
  *'Test' and 'Loss' are the values obtained from the test dataset as part of training (so we want to see Test high and Loss Low)
  * TC_Seiz, All_Seiz and FalseALarms are the proportion of events that were detected correctly using the OSD TestRunner, so we want to see values approaching 1.0

| Ver.		| Test	| Loss	| TC_Seiz	| All_Seiz	| FalseAlarms | Notes |
|---------------| ------| ------| --------------| --------------| ------------| ------ |
| OSD_v1	| -     |  -	| 0.78    	| 0.73		| 0.61 | 		|
| V0.01 	| 0.86	| 0.35  |   -		| -		| - |    		|
| V0.02 	| 0.92	| 0.30	| -		| -		| - | Good detection performance, but false alarmed when walking.|
| V0.03   	| 0.95	| 0.14	| 0.43		| 0.23		| 1.00 | Poor seizure detection performance so not used	|
| V0.04       	| 0.903	| 0.277 |   1.0     	| 0.98 		| 0.90 | Good false alarm performance, but difficult to simulate a seizure to make it alarm - is it trained to be too specific to the seizures in the training set?		|


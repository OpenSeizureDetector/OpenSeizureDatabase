# Neural Network Seizure Detector Development - Results Summary

Summary
=======
In the table below:
  *'Test' and 'Loss' are the values obtained from the test dataset as part of training (so we want to see Test high and Loss Low)
  * TC_Seiz, All_Seiz and FalseALarms are the proportion of events that were detected correctly using the OSD TestRunner, so we want to see values approaching 1.0

| Ver.					| Test	| Loss	| TC_Seiz	| All_Seiz	| FalseAlarms 	  | Notes	  |
|---------------			| ------| ------| --------------| --------------| ------------	  | ------ 	  |
| OSD_v1				| -     |  -	| 0.78    	| 0.73		| 0.61 		  | This is the current algorithm used by OpenSeizureDetector with default settings.		  |
| V0.01 (all data)			| 0.86	| 0.35	|   -		| -		| -    		  |    		  |
| V0.02 (-20 to +40s of event)		| 0.92	| 0.30	| 0.76		| 0.74		| 0.82 		  | OSD V4.1.0a &b: Good detection performance, but false alarmed when walking.|
| V0.03 (additional training data)  	| 0.95	| 0.14	| 0.43		| 0.23		| 1.00 		  | Poor seizure detection performance so not used	|
| V0.04 (Random Oversample)     	| 0.903	| 0.277 |   1.0     	| 0.98 		| 0.90 		  | OSD V4.1.0c: Good false alarm performance, but difficult to simulate a seizure to make it alarm - is it trained to be too specific to the seizures in the training set?		|
| V0.05 (SMOTE)  			| 0.87	| 0.29	| 0.97		| 0.98		| 0.86 		  | OSD V4.1.0d: Good false alarm and detection performance against testdataset.	|
| V0.06 (More normal data)		| 0.88	| 0.28	| 0.95		| 0.96		| 0.87     	  | OSD V4.1.0e:      	  |
| V0.07 (More normal data)		| 0.87	| 0.30	| 0.97		| 0.98		| 0.82		  | OSD V4.1.0f |
| V0.08 (More normal data)		| 0.87	| 0.32	| 0.89		| 0.89		| 0.92		  | not used because spoted some bad seizure data in database |
| V0.09 (Removed bad seizure data)	| 0.92	| 0.19	| 1.00		| 1.00		| 0.94		  | OSD V4.1.0g |


Detailed Description
====================

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

Used this version in V4.1.0a of the OSD phone app for testing.   It definitely avoids most of the previous false alarm causes (such as typing)....BUT it alarms if you run up the stairs or walk briskly swinging your arm - the original OSD algorithm filtered these movements out becuase they have a lot of low frequency movement in them, so the training dataset did not contain any....

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
false alarm data.   Did not test this on the phone becuase of the poor performance with the test data.



Version 0.04:
-------------
As for V0.03 except:
  * Introduced re-sampling to increase the number of seizure datapoints to be the
same as the number of false alarm datapoints. (used imblearn RandomOverSampler function)
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
So build a new version of the Android App with V0.04 model.   This is V4.1.0c.apk (https://github.com/OpenSeizureDetector/Android_Pebble_SD/blob/V4.1.x/app/release/app-release-4.1.0c.apk).

False alarm performance appears good - the main false alarms were from standing still talking, so very subtle movements.   

The main concern is that I can not simulate a tonic-clonic seizure to set it off (with high frequency shaking of the arm).  Sometimes it detects the simulation movement as seizure-like, but then jumps to indicating 0% seizure probability without me intentionally changing the movement.

I think this means it is trained to detect very specific movements from the training dataset and not more generic 'movements like this'.   Try oversampling and adding random noise into the data next to see if that makes it more generic.

Version 0.05:
-------------
As for V0.04 except:
  * Changed from random oversampling to SMOTE oversampling which creates synthetic data based on the minority class data (so should be better than just adding noise into the data maybe).
  
 * Trained using 18338 seizure datapoints and 18338 false alarm datapoints
 * Tesing using 6113 seizure datapoints and 6113 false alarm datapoints
 * Stopped at Epoch No 334 (after 26 minutes)
 * Test accuracy 0.874
 * Test loss 0.288

TestRunner Results:
 * Category, OSD_v1, nn_v0.05
 * tcSeizures, 0.78, 0.97
 * allSeizures, 0.73, 0.98
 * falseAlarms, 0.60, 0.86

So good detection and false alarm performance against test dataset - try on live system to see how it behaves.
Using this model on the live system caused false alarms during slow walking or standing still, so collected a lot of data for these events to add to the database.
These were not previously covered in the databse because the OSD algorithm filters out events with a high proportion of the power <3Hz so had not generated
false alarms to cause them to be included in the databse.

Version 0.06:
-------------
As for V0.06 except:
  * Added more normal data to the database - mostly slow walking and false alarms with little movement aused by V0.05
  
 * Trained using 19945 seizure datapoints and 19944 False alarm datapoints
 * Tesing using 6648 seizure datapoints and  6649 false alarm datapoints
 * Stopped at Epoch No 308 (after 27 minutes)
 * Test accuracy 0.88
 * Test loss 0.28

TestRunner Results:
 * Category, OSD_v1, nn_v0.06
 * tcSeizures, 0.78, 0.95
 * allSeizures, 0.73, 0.96
 * falseAlarms, 0.61, 0.87

Still gave false alarms for very small movements (Which would have been filtered out by OSD algorithm.   Collected 24 hrs of
false alarm data to use to re-train next version.

Version 0.07:
-------------
As for V0.6 except:
  * Added more false alarm data generated using V0.06

Trained using 22196 seizure datapoints and 22196 false alarm datapoints
Tesing using 7399 seizure datapoints and 7399 false alarm datapoints
Test accuracy 0.87
Test loss 0.30
real	18m22.154s
TestRunner Results:
 * Category, OSD_v1, nn_v0.07
 * tcSeizures, 0.78, 0.97
 * allSeizures, 0.73, 0.98
 * falseAlarms, 0.63, 0.82

Incorpored into V4.1.0f of Android App - initial results suggest it is giving less false alarms than V0.06.

Version 0.08:
-------------
As for V0.7 except:
  * Added more false alarm data generated using V0.06, up to 30/09/2022

Trained using 25251 seizure datapoints and 25251 false alarm datapoints
Tesing using 8417 seizure datapoints and 8417 false alarm datapoints
Test accuracy 0.87
Test loss 0.32
real	23m
TestRunner Results:
 * Category, OSD_v1, nn_v0.08
 * tcSeizures, 0.78, 0.89
 * allSeizures, 0.73, 0.89
 * falseAlarms, 0.63, 0.92

This version was not used because the review of the testrunner results showed that some of the failed seizure detections were related to bad data (either
using the phone datasource, or the data point timing being odd).  These were removed and the model re-trained to produce V0.09 below.

Version 0.09:
-------------
As for V0.8 except:
  * Removed 4 seizure type events which used the phone data source, so data probably invalid (events 9828, 12973, 14101, 15208)
  * Removed one seizure type event which had invalid data - times of datapoints were very odd (8661)

Trained using 25513 seizure datapoints and 25512 false alarm datapoints
Tesing using 8504 seizure datapoints and 8505 false alarm datapoints
Test accuracy 0.92
Test loss 0.19

real	41m11.553s

TestRunner Results:
 * Category, OSD_v1, nn_v0.09
 * tcSeizures, 0.78, 1.00
 * allSeizures, 0.73, 1.00
 * falseAlarms, 0.63, 0.94

The training statistis are much better than obtained previously - probably because of removing the bad seizure data.   The training and validation test/loss
values were much closer together than we obtained previously too.   This is confirmed by the testRunner data where all seizure events were detected correctly.   The false alarm rate is much lower than the original OSD algorithm by around a factor of 6 (6% of events produced false alarms compared to 37% with the OSD algorithm).

Built into V4.1.0g of the phone app for testing.


Summary
-------
In the table below:
  *'Test' and 'Loss' are the values obtained from the test dataset as part of training (so we want to see Test high and Loss Low)
  * TC_Seiz, All_Seiz and FalseALarms are the proportion of events that were detected correctly using the OSD TestRunner, so we want to see values approaching 1.0

| Ver.					| Test	| Loss	| TC_Seiz	| All_Seiz	| FalseAlarms | Notes |
|---------------			| ------| ------| --------------| --------------| ------------| ------ |
| OSD_v1				| -     |  -	| 0.78    	| 0.73		| 0.61 | 		|
| V0.01 (all data)			| 0.86	| 0.35	|   -		| -		| - |    		|
| V0.02 (-20 to +40s of event)		| 0.92	| 0.30	| 0.76		| 0.74		| 0.82 | OSD V4.1.0a &b: Good detection performance, but false alarmed when walking.|
| V0.03 (additional training data)  	| 0.95	| 0.14	| 0.43		| 0.23		| 1.00 | Poor seizure detection performance so not used	|
| V0.04 (Random Oversample)     	| 0.903	| 0.277 |   1.0     	| 0.98 		| 0.90 | OSD V4.1.0c: Good false alarm performance, but difficult to simulate a seizure to make it alarm - is it trained to be too specific to the seizures in the training set?		|
| V0.05 (SMOTE)  			| 0.87	| 0.29	| 0.97		| 0.98		| 0.86 		  | OSD V4.1.0d: Good false alarm and detection performance against testdataset.	|
| V0.06 (More normal data)		| 0.88	| 0.28	| 0.95		| 0.96		| 0.87     	  | OSD V4.1.0e: May be slightly too sensitive to very small movements still - collect more real-world data to find out.     	  |
| V0.07 (More normal data)		| 0.87	| 0.30	| 0.97		| 0.98		| 0.82		  | OSD V4.1.0f |
| V0.08 (More normal data)		| 0.87	| 0.32	| 0.89		| 0.89		| 0.92		  | not used because spoted some bad seizure data in database |
| V0.09 (Removed bad seizure data)	| 0.92	| 0.19	| 1.00		| 1.00		| 0.94		  | OSD V4.1.0g |

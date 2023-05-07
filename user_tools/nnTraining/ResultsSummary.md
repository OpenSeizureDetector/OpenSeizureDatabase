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
| V0.10 (Additional false alarm data)   | 0.91  | 0.23  | 1.00          | 0.99          | 0.92            | Noticed that it alarms with phone datasouce when stationary so not used |
| V0.12 (removed bad seizure data)	| 0.95	| 0.15	| 0.97		| 0.94		| 0.93		  | Used in OSD V4.1.0 which is to be released for beta testing |
| V0.17 (manually set seizure times)  	| 0.72  | 0.61  | 0.97		| 0.98		| 0.59		  | Seizure detection good, but poor false alarm performance.
| V0.18 (as v0.17 but split by event not datapoint| 0.53 | 0.91 | 0.95 | 0.97		| 0.29		  | false alarm performance worse than test/train split by datapoint (v0.17)
| V0.19 (as v0.18 but used random oversampling, not SMOTE | 0.75 | 0.55 | 0.97 | 0.96		| 0.62		  | Better seizure detection than OSD algorithm with comparable false alarms.
| V0.20 (as v0.19 5 layers, not 3 | 0.86 | 0.49 | 0.93 | 0.91		| 0.89		  | Good seizure detection and false alarm performance.
| V0.26 (phase augmentation) | 0.66 | 1.76 | | 0.48 | 0.97 | Good false alarm performance, but disappointing seizure detection reliability | 


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
As for V0.08 except:
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

Version 0.10:
-------------
As for V0.09 except included additional false alarms to reduce false alarm rate.

Trained using 27176 seizure datapoints and 27175 false alarm datapoints
Tesing using 9058 seizure datapoints and 9059 false alarm datapoints
Test accuracy 0.91
Test loss 0.23


Category, OSD_v1, nn_v0.10
tcSeizures, 0.78, 1.00
allSeizures, 0.72, 0.99
falseAlarms, 0.65, 0.92

Realised that the phone accelerometer generates alarms when stationary, so this is no good for public use - generated some phone accelerometer stationary data and re-trained....

Version 0.11:
-------------
As for V0.10 except included additional zero movement negative data to try to avoid it alarming if the phone datasource was enabled and the phone was just
sitting on the bench.
Not successful - in testing some of the zero movement results still alarmed.
Looked more closely into the reasons and it appears that some of the seizure data used had very low level of movement.  So added a printout of the
standard deviation of the accelerometer data for each seizure datapoint.   If more than 3 datapoints had <1% stdev, marked the seizure as invalid for training
purposes.   Also reduced the time range for seizure data to -20 to +20 sec as some seizures are of short duration so using +40 included some very low movement
data as seizure datapoints.
Note:  Could automate this and not use a datapoint as a seizure datapoint if Stdev<1%?

Added more zero movement negative data and re-trained as V0.12....

Version 0.12:
-------------
As v0.11 but
  * Reduced seizure time range to -20 to +20 sec
  * Excluded seizures with more than 3 datapoints with acceleration standard deviation <1%
  * Added more negative data that had zero movement (to avoid zero movement of the phone generating an alarm (around event number 20000, userid 38).

Trained using 27990 seizure datapoints and 27990 false alarm datapoints
Tesing using 9330 seizure datapoints and 9330 false alarm datapoints
Test accuracy 0.95
Test loss 0.15

Results Summary
Category, OSD_v1, nn_v0.12
tcSeizures, 0.78, 0.97
allSeizures, 0.72, 0.94
falseAlarms, 0.66, 0.93

There were 5 failures to alarm in the 'All Seizures' category and 1 tonic-clonic.
These were:
  * 1046 (Seizure/None)
  * 7357 (Seizure/Other)
  * 12763 (Seizure/Other/"micro event")
  * 15417 (Seizure/Tonic-Clonic/"was at 5 AM")
  * 15452 (Seizure/Aura/"absence seizure")

False alarms included Talking, Sorting, washing / cleaning, Motor Vehicle, typing.
Both the false alarm performance and seizure detection performance is significantly better than the original OSD algorithm, so published this version as V4.1.0 of the OSD Android App.

Version 0.13:
-------------
As v0.12 but
  * Used new 'split by event' option rather than splitting database by datapoint.    This means that the test events are completely separate from the training ones - previously datapoints were split between test and training sets so it is likely that some datapoints from every seizure were included in the training data set.

Stopped at Epocch 116
Trained using 31938 seizure datapoints and 31938 false alarm datapoints
Tesing using 152 seizure datapoints and 8017 false alarm datapoints
Test accuracy 0.69
Test loss 0.96

Category, OSD_v1, nn_v0.13
tcSeizures, 0.76, 0.95
allSeizures, 0.71, 0.94
falseAlarms, 0.66, 0.58

This shows that the seizure detection reliability remains good, and comparable to model v0.12, but the false alarm performance is worse.
This is quite encouraging because it means that even  when we keep back a portion of the real seizure data and do not use it for training at all, we still get good seizure detection reliability - suggesting that we are not over fitting the model?

Version 0.14:
-------------
As v0.13 but:
  * Increased the early stopping "patience" prameter from 50 to 200 to make the training run for longer.

Stopped at Epoch 268
Trained using 31938 seizure datapoints and 31938 false alarm datapoints
Tesing using 152 seizure datapoints and 8017 false alarm datapoints
Test accuracy 0.57
Test loss 0.97

Version 0.17:
-------------
Set seizure start and end times manually and selected seziure datapoints only within that time range.
Test/Train split by datapoint (As per V0.12)
Ran through to Epoch 500
Trained using 36248 seizure datapoints and 36248 false alarm datapoints
Tesing using 303 seizure datapoints and 9062 false alarm datapoints
Test accuracy 0.72
Test loss 0.61

Jamie's Statistics:
Sensitivity/recall or true positive rate: 0.99  0.08
Specificity or true negative rate: 0.08  0.99
Precision or positive predictive value: 0.73  0.70
Negative predictive value: 0.70  0.73
Fall out or false positive rate: 0.92  0.01
False negative rate: 0.01  0.92
False discovery rate: 0.27  0.30
Classification Accuracy: 0.72  0.72
 
TestRunner Results
Category, OSD_v1, nn_v0.17
tcSeizures, 0.75, 0.97
allSeizures, 0.71, 0.98
falseAlarms, 0.65, 0.59 



Version 0.18
------------
As for v0.17, but enabled test train split by event rather than by datapoint.

Stopped after Epoch 217
Trained using 36132 seizure datapoints and 36132 false alarm datapoints
Tesing using 310 seizure datapoints and 9178 false alarm datapoints
Test accuracy 0.53
Test loss 0.91

Jamie's Statistics
Sensitivity/recall or true positive rate: 0.99  0.05
Specificity or true negative rate: 0.05  0.99
Precision or positive predictive value: 0.53  0.78
Negative predictive value: 0.78  0.53
Fall out or false positive rate: 0.95  0.01
False negative rate: 0.01  0.95
False discovery rate: 0.47  0.22
Classification Accuracy: 0.53  0.53

TestRunner Results
Category, OSD_v1, nn_v0.17
tcSeizures, 0.75, 0.95
allSeizures, 0.71, 0.97
falseAlarms, 0.65, 0.29 

Version 0.19
------------
As for v0.18, but went back to random oversampling rather than SMOTE in case that was generating
the false alarm issues.

Stopped after Epoch 312
Trained using 36132 seizure datapoints and 36132 false alarm datapoints
Tesing using 310 seizure datapoints and 9178 false alarm datapoints
Test accuracy 0.75
Test loss 0.55

 Jamie's Statistics
Sensitivity/recall or true positive rate: 0.98  0.08
Specificity or true negative rate: 0.08  0.98
Precision or positive predictive value: 0.76  0.59
Negative predictive value: 0.59  0.76
Fall out or false positive rate: 0.92  0.02
False negative rate: 0.02  0.92
False discovery rate: 0.24  0.41
Classification Accuracy: 0.75  0.75

TestRunner Results
Category, OSD_v1, nn_v0.17
tcSeizures, 0.75, 0.97
allSeizures, 0.71, 0.96
falseAlarms, 0.65,  0.62

So we have much better seizure detection rate than the original OSD, and simlar false alarm rate, so quite good :).

Version 0.20
------------
As for v0.19, but used deeper network (5 convolution layers rather than 3).

Stopped after epoch 277
Trained using 36132 seizure datapoints and 36132 false alarm datapoints
Tesing using 310 seizure datapoints and 9178 false alarm datapoints
Test accuracy 0.86
Test loss 0.49

Sensitivity/recall or true positive rate: 0.98  0.11
Specificity or true negative rate: 0.11  0.98
Precision or positive predictive value: 0.87  0.48
Negative predictive value: 0.48  0.87
Fall out or false positive rate: 0.89  0.02
False negative rate: 0.02  0.89
False discovery rate: 0.13  0.52
Classification Accuracy: 0.86  0.86

TestRunner Results
Category, OSD_v1, nn_v0.20
tcSeizures, 0.75, 0.93
allSeizures, 0.71, 0.91
falseAlarms, 0.65,  0.89

V0.22 - 26oct2022
As for V0.20, but included Normal Daily Living (NDA) events in training set.
Used 0.3 test/validation proportion.
Seemed to be pretty good as far as false alarms are concerned, except
it alarmed constantly during deep sleep (must be detecting very small but non-zero movements and treating this as a seizure.

V0.23 - 27oct2022
As for V0.22 but added more NDA and false alarm data.
Trained using 64705 seizure datapoints and 64705 false alarm datapoints
Tesing using 27672 seizure datapoints and 27672 false alarm datapoints
Test accuracy 0.72
Test loss 1.96

Sensitivity/recall or true positive rate: 0.65  0.90
Specificity or true negative rate: 0.90  0.65
Precision or positive predictive value: 0.95  0.49
Negative predictive value: 0.49  0.95
Fall out or false positive rate: 0.10  0.35
False negative rate: 0.35  0.10
False discovery rate: 0.05  0.51
Classification Accuracy: 0.72  0.72

TestRunner Results:
Category, OSD_v1, cnn_v0.23
tcSeizures,  0.75, 0.72
allSeizures, 0.73, 0.82
falseAlarms, 0.65, 0.93  
NDA Events,  0.85, 1.00  (NDA)

So this model gives reasonable seizure detection (an improvement on OSD for all seizures), but much fewer false alarms - the NDA results represent 0.44 false alarms per day.

v0.24 - 28oct2022
Introduced noise augmentation where each seizure data point is copied a specified number
of times, with random noise added to each measurement.
Used a noise level of 10 mg and created 5 copies per datapoint (was going to use 10
copies per data point but my computer crashed because I used up all 16GB of memory
with a factor 10 specified.....)

Training using 413580 seizure and 413580 non-seizure data ponts.  Now takes over 6 minutes per epoch.....

Stopped manually because we had such good validation results (val_loss 0.02, val_acc 0.99).
Ran test manually:
Sensitivity/recall or true positive rate: 0.63  0.88
Specificity or true negative rate: 0.88  0.63
Precision or positive predictive value: 0.94  0.46
Negative predictive value: 0.46  0.94
Fall out or false positive rate: 0.12  0.37
False negative rate: 0.37  0.12
False discovery rate: 0.06  0.54
Classification Accuracy: 0.70  0.70

TestRunner Output:
All Seizures: 88% detection acuracy
Category, OSD_v1, cnn_v0.24
tcSeizures, 0.76, 0.82
allSeizures, 0.72, 0.88
falseAlarms, 0.65, 0.92
ndaEvents,  0.84, 1.00

So this appears to be giving us 88% seizure detection reliability and very few false alarms, so a good candidate for use.
NOTE:  The testRunner set up used all data, not just the 'Test' data set, so the test runner data includes 70% data that was used during training.

v0.26 - 06nov2022
With testrunner set up to use only dedicated test data, not including the training and validation data sets, the performance looked a lot worse, so implemented Phase Augmentation to generate more datapoints from seizure data by offsetting the 125 datapoint window.
Trained using 117048 seizure datapoints and 117048 false alarm datapoints.
It appeared to train nicely with the training and validation statistics being similar,
so it looked as though our overfitting issues had been overcome:
Epoch 122/500
1639/1639 [==============================] - 19s 12ms/step - loss: 0.0882 - sparse_categorical_accuracy: 0.9715 - val_loss: 0.0793 - val_sparse_categorical_accuracy: 0.9778 - lr: 1.2500e-04

Tesing using 48182 seizure datapoints and 48182 false alarm datapoints
Test accuracy 0.66
Test loss 1.76
So the results using the test data do not seem as good as I would have hoped.   The test runner results for the test data alone detected 12 of the 25 seizures, so about 48%
False alarm performance was better with 97% of the false alarms detected correctly.  For the normal daily activities data, all 411 events were detected correctly as not being seizures.

So for this version we are seeing very good false alarm performance, but somewhat disappointing seizure detection performance for the test data.

Repeating the testRunner run with the alarmTime parameter set down to 5 seconds from 10 seconds (which means it will alarm instantly without going through the 'WARNING' stage)
increased the seizure detection reliability to 64% without a significant deterioration in false alarm performance, so this is an option if this model were to be used.

It looks though as if the best option is to increase the amount of seizure data used for training to increase the relative weight of seizure to non-seizures.

Do this by increasing noise augmentation factor to 30 (for seizure data only),
and reducing the seizure SD threshold from 1.0 to 0.5 to include more seizure datapoints (v0.27).

v0.28 - increased noise augmentation to factor 100 and 20 mg

v0.28
-----
Increased noise augmentation to factor 100 with 20mg standard deviation.

Epoch 117/500
3275/3275 [==============================] - 38s 12ms/step - loss: 0.1238 - sparse_categorical_accuracy: 0.9542 - val_loss: 0.2015 - val_sparse_categorical_accuracy: 0.9131 - lr: 1.2500e-04
Epoch 117: early stopping
5937/5937 [==============================] - 12s 2ms/step - loss: 1.8669 - sparse_categorical_accuracy: 0.6941  

  * Trained using 233928 seizure datapoints and 233928 false alarm datapoints
  * Tesing using 94984 seizure datapoints and 94984 false alarm datapoints
  * Test accuracy 0.69
  * Test loss 1.867

TestRunner results on test data only:
| DataSet      |  osdAlg  |  cnn_v0.28 |
| --------     |  ---     |  ---       |
| all Seizures |  72%     |   52%      |
| false alarms |  65%     |   84%      |
| NDA          |  85%     |   100%     |
| --------     |  ---     |  ---       |

So we are seeing very good NDA performance still but with disappointing seizure detection.  I wonder if we have too much NDA data which is biasing it?

Try altering phase and noise augmentation so noise augmentation is applied to
every phase augmented data point, to increase the amount of seizure data significantly.

v0.29
Combined phase and noise augmentation.
  * Epoch 124: early stopping
  * 9672/9672 [==============================] - 14s 1ms/step - loss: 2.1874 - sparse_categorical_accuracy: 0.7024
  * Trained using 381384 seizure datapoints and 381384 false alarm datapoints
  * Tesing using 154752 seizure datapoints and 154752 false alarm datapoints
  * Test accuracy 0.7024012804031372
  * Test loss 2.18743634223938

TestRunner results on test data only:
| DataSet      |  osdAlg  |  cnn_v0.28 |
| --------     |  ---     |  ---       |
| all Seizures |  72%     |   48%      |
| false alarms |  65%     |   89%      |
| NDA          |  85%     |   100%     |
| --------     |  ---     |  ---       |

So increasing the augmentation of the seizure data has improved false alarm performance, but
seizure detection performance has deteriorated.  I don't know why!

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
| V0.10 (Additional false alarm data)   | 0.91  | 0.23  | 1.00          | 0.99          | 0.92            | Noticed that it alarms with phone datasouce when stationary so not used |
| V0.12 (removed bad seizure data)	| 0.95	| 0.15	| 0.97		| 0.94		| 0.93		  | Used in OSD V4.1.0 which is to be released for beta testing |
| V0.17 (manually set seizure times)  	| 0.72  | 0.61  | 0.97		| 0.98		| 0.59		  | Seizure detection good, but poor false alarm performance.
| V0.18 (as v0.17 but split by event not datapoint| 0.53 | 0.91 | 0.95 | 0.97		| 0.29		  | false alarm performance worse than test/train split by datapoint (v0.17)
| V0.19 (as v0.18 but used random oversampling, not SMOTE | 0.75 | 0.55 | 0.97 | 0.96		| 0.62		  | Better seizure detection than OSD algorithm with comparable false alarms.
| V0.20 (as v0.19 5 layers, not 3 | 0.86 | 0.49 | 0.93 | 0.91		| 0.89		  | Good seizure detection and false alarm performance.
| V0.26 (phase augmentation) | 0.66 | 1.76 | | 0.48 | 0.97 | Good false alarm performance, but disappointing seizure detection reliability | 
| V0.29 (phase and noise augmentation) | 0.70 | 2.19 | | 0.48 | 0.89 |
| V0.30 |  |  |  |  |
| V0.31 (Random undersampling) |  |  | | 0.97 | 0.78 | Very good seizure detection, reasonable false alarm performance |
| V0.32 (Random oversampling split by datapoint) |  | | | 0.96 | 0.95 | Very good seizure detection and false alarm performance - but is it over-trained? |

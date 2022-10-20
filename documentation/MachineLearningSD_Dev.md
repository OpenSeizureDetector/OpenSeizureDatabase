% Development of a Practical Machine Learning based Seizure Detector

# Abstract
The free, Open Source epileptic seizure detector, [OpenSeizureDetector](https://openseizuredetector.org.uk) utilises a deterministic algorithm based on detecting an excess of movement in a defined frequency range, measured using a smart watch.
The algorithm has been shown to detect tonic-clonic seizures reliably, but is prone to producing false alarms if the user makes 'seizure-like' movements.
In this document we describe the development of an improved, machine learning based seizure detector trained using crowd sourced data provided by OpenSeizureDetector users.
The new machine learning algorithm is tested against 'real world' data to demonstrate it practical applicability.

# Background
The original OpenSeizureDetector algorithm [\[1\]](#ref1) is deterministic and uses the following process:
  * Collect 5 seconds of accelerometer data (At 25Hz), and calculate the vector magnitude of each of the 125 measurements in the 5 second period.
  * Calculate the fourier transform of the vector magnitudes over the 5 second period
  * Calculate the average power over the whole spectrum, and the average power over a 3-8 Hz Region of Interest (ROI).
  * If ROI power is above a specified threshold and the ratio of ROI to whole spectrum power is above a second threshold (signifying an excess of movement in the ROI frequencies), then the movement is considered 'seizure-like'.
  * If two consecutive 5 second data points are 'seizure-like', a WARNING is generated.
  * If three consecutive 5 second data points are 'seizure-like', an ALARM is generated.

The choice of 25 Hz sample frequency was a balance between data transfer requirements, and ensuring that movements of up to 10Hz could be detected (a 25 Hz sample frequency allows movements of up to 12.5 Hz to be measured).   The use of a 5 second sample period was chosen becasue with 5 seconds, a frequency resolution of 0.2 Hz is achieved in the measurement, which was judged to be adequate to distinguish different movement frequencies.

A more detailed description of the algorithm is presented in Reference [\[1\]](#ref1).

The issue with the algorithm is that although it has been shown to detect tonic-clonic seizures reliably (detection reliability in excess of 75%), it also generates false alarms for some common activities (such as brishing teeth or hair, which involve movements in the ROI frequency range).

An improved algorithm is required if the detection reliability is to be maintained (and hopefully improved), and the false alarm rate reduced.    A major issue preventing development of the detection algorithm is the availability of suitable test data to tune a detection model and gain confidence in its effectiveness.

This project has therefore comprised two distinct parts:
  * Development of a crowd-sourced dataset of seizure and seizure-like data measured using smart watches (OpenSeizureDatabase).
  * Using the database of seizure data to develop a machine learning model which can be deployed in OpenSeizureDetector to improve the system for users.

# Seizure Database Development
## Data Collection Approach
The [OpenSeizureDetector Android phone app](https://play.google.com/store/apps/details?id=uk.org.openseizuredetector) was modified to allow users to contribute data collected by the system to a central database, using a system we called ['Data Sharing'](https://www.openseizuredetector.org.uk/?page_id=1818).  The data sharing system does the following:
  * Each time OpenSeizureDetector enters an ALARM or WARNING state, an 'Event' is created in the database.
  * The data recorded by the system either side of the event are uploaded as a series of 'Datapoints' containing 5 second snapshots of accelerometer, heart rate and O2 saturation data.    Data is recorded for 75 seconds before and after the event time.
  * The user receives an Android notification that they have 'unvalidated data' recorded in the database.
  * When they select the notification they are presented with a list of their events and can select them to categorise the type of event, and include a text description of what happened.   The available categories are:
    * Seizure (with various sub-types)
    * Fall
    * False Alarm (with various sub types of possible causes)
    * Unknown

The system was made live for all users in June 2022 and since that time, over 200 users have signed up to contribute data.


## Database Contents
At the time of writing (October 2022), the database contains over 22000 events and over 500000 5 second datapoints.   Almost 400 of these events have been tagged by users as being seizures and over 3000 as false alarms (the remaining events are a very small number of falls (16) and unknown or unclassified data).

The (data collection approach)[#Data Collection Approach] adopted is simple, so is prone to duplication (because a WARNING and an ALARM state could be generated in rapid succcession creating two practically overlapping events).

For this reason the raw data in the database is post-processed into text files containing JSON objects describing each 'Unique' event.   Unique events are determined by grouping the events into 3 minute periods and selecting a single representative event to cover that period.

The accuracy of the seizure data is particularly important so the unique seizure events were reviewed manually and a number excluded because of bad data, or the user reporting that they were not wearing the watch at the time of the seizure.    Following this screening 98 unique seizure events have been used, of which 40 were tagged as being tonic-clonic seizures.

There were 1635 unique 'False Alarm' events recorded in the database and used as negative results for training purposes.

The large number of 'unknown' events have not been used, but it is expected that the vast majority of these will be false alarms so could be used as additional negative results if necessary.

It should be noted that the database contains only data that OpenSeizureDetector identified as being 'seizure-like' or which were reported manually by the user.   An exercise to collect continuous 'Normal Daily Activity' data is intended in the future to allow a true false positive rate to be determined for future algorithms.

# Machine Learning Model Development
## Options
A number of options were considered for a machine learning based seizure detector, some of which were:
  * State Vector Machine (SVM) - similar to that used by Empatica in their [Embrace](https://onlinelibrary.wiley.com/doi/10.1111/j.1528-1167.2012.03444.x) product.
  * A Long-Short Term Memory (LSTM) Artificial Neural Network - often used for time series analysis.
  * A Convolutional Neural Network (CNN) - often used for image processing.
  * An autoencoder based anomaly detector - often used for detecting rare occurrences in data.

The SVM and autoencoder options have not been assessed yet.
Initial trials of an LSTM based seizure detector produced good seizure detection reliability, but very poor false alarm performance, so would require additional development to improve it.

The best results obtained to date have been using a Convolutional Neural Network (CNN), as described below.

## Convolutional Neural Network Structure
A Convolutional Neural Nework (CNN) was set up in line with the structure suggested by Wang, Yang and Oates [2](#ref2), using an example provided in the [Keras documentation](https://keras.io/examples/timeseries/timeseries_classification_from_scratch/).

The model accepts a 125 point vector, which is 5 seconds of acceleration magnitude measurements at 25 Hz.  The original version contained 3 convolution layers, as shown below (click on it to see full size version):

<p align="center"><a href="./model_structure.png"><img src="./model_structure.png" width=150></a></p>

## Training Process Development
The process for training the model has been developed iteratively.  The following describes the sequence of events and main issues encountered.   Further details can be found on the [Results Summary](./resultsSummary.md) page.

  * The available data was split into test, train and validation datasets (20% test, 64% train, 16% validation)
  * The accuracy was lower than anticipated (86%), because the seizure events contained some non-seizure data (e.g. before or after the seizure).
  * The seizure events were truncated to be between -20 sec and +40 sec from the event time, to increase the likelihood that only seizure movements were included.  This reduced the number of seizure datapoints used in training by around a factor of 2.  It did however give a higher test accuracy of 92% (loss 0.30) (cnn_v0.02)
    * BUT when this was implemented into OpenSeizureDetector it gave lots of false alarms for normal activities which the original OpenSeizureDetector algorithm filtered out.   This is an issue with the false alarm data in the database - it only includes events which OpenSeizureDetector considered to be seizure-like - all other activities were excluded.
    *  A user wore the system using this CNN altorithm (Android App version 4.1.0a) for several days to generate events for activities that gave false alarms using this network, to build up the 'non-seizure' data in the database.
  * The model was re-trained using the additional false alarm data, and also using data grouped using 3 minute time periods to increase the amount of data included - this increased the number of training datapoints to 809 and the number of false alarms to 18338.
    *  The training statistics for this model (cnn_v0.03) were good (test acuracy 95%, loss 0.14), but when the system was run through the testRunner system to simulate real-life usage, seizure detection performance was very poor (43% for tonic-clonic seizures and 23% for all seizures).
    *  This suggests that too much false alarm data was used in training relative to the seizure data, biasing the model towards predicting non-seizure results.
  *  Introduced Random Oversampling to increase the number of seizure datapoints used for training to equal the number of false alarm datapoints.
    * Training statistics were good (90% test acuracy and 0.28 loss), and the testRunner showed very good performance 100% detection reliability for tonic-clonic seizures and 98% for all seizures.   False alarm performance was good with 90% of the events being identified correctly as non-seizure.
    * So the system appeared to be very good so a new version of the Android App was built ot test it (V4.1.0c).   The main issues were:
      * It generates false alarms for very tiny movements, which would be ignored by the OSD algorithm.
      * It is difficult to simulate a seizure to make it generate an alarm - this may be because real seizures are not the same as the movement being used to simulate them, or it may be that the model has been trained to be too specific to exact movements from the seizures in the database and not simlilar movements.
  *  Switched from Random oversampling to SMOTE oversampling which simulates data by interpolating between two randomly seleted data points (cnn_v0.05).   
    *  The training statistics were still good (87% accuracy and 0.29 loss).  Test Runner results wereslightly worse than the previous version, but very similar.
    *  It still generates false alarms for small movements (slow walking or standing still)
    *  Used this version to generate a lot of false alarms to include in future training versions.
  *  Worked through several versions, using additional training data as more false alarms were generated from the CNN algorithm.
  *  Examined which seizures were failing to be detected and found that they had bad data, so were excluded from training and testing.
  *  Found that using the phone data source with the phone stationary on a table generated alarms - so generated a lot of zero movement false alarms to include in the training data set.
  *  Found that small movement was still being detected as a seizure, so reduced the time span of seizure data to -20 to +20 seconds to reduce the chances of including zero movement data points as seizures.   Also excluded seizures that had more than 3 data points with \<1% standard deviation from the training data set.
     * Trainded version was cnn_v0.12 which gave very good training statistics (test accuracy 95%, loss 0.15) and testrunner results (all seizures 94% accurate non-seizure false alarms 93% accurate).   Released this version as V4.1.0 of the Android app for beta testers.
  *  It was noted that collecting all the datapoints together and then doing the test/train/validation split meant that it is likely that some datapoints from every seizure are included in the training set, so testing is not truely independent of the training data.   When the data was instead split by event rather than datapoint (which means that no data from a test event appears in the training dataset), the performance was noted to deteriorate - while the seizure detection reliability remained high at around 95%, the proportion of false alarm data that was correctly identified as a false alarm fell dramatically from 93% to 29%.
    * The reason for the false alarm performance deteriorating is not fully understood - it had been suspected that the seizure detection reliability would fall instead.
    * One possibility is that the SMOTE interpolation is generating some synthetic 'seizure' data points that look like normal non-seizure movement.
  * Reverted back to random oversampling to avoid interpolating seizure data (cnn_v0.19)
    * This improved the false alarm performance to be simlar to the original OSD algorithm, with good seizure detection reliability (97%).
    * It was disappointing that the false alarm performance was not better than the original OSD algorithm though.
  * Increased the depth of the neural network from 3 to 5 convolution layers (cnn_v0.20).  This gave good training statistics (test accuracy 86%, loss 0.49), and good test runner performance (seizure detection reliability 91%, false alarm performance 89%).
    * It is likely that increasing the depth fo the network allows it to identify more features in the data that the shallower network could detect during training.
    * The plan is now to deploy Version 0.20 for testing on a live system.


# References
  * [<a name="ref1">1</a>] [Seizure Detection Algorithm](https://www.openseizuredetector.org.uk/?page_id=455)
  * [<a name="ref2">2</a>] [Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline](https://arxiv.org/abs/1611.06455)


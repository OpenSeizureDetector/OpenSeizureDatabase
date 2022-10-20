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
A Convolutional Neural Nework (CNN) was set up in line with the structure suggested by Wang, Yang and Oates [2]

# References
[<a name="ref1">1</a>] [Seizure Detection Algorithm](https://www.openseizuredetector.org.uk/?page_id=455)
[<a name="ref2">2</a>] [Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline](https://arxiv.org/abs/1611.06455)


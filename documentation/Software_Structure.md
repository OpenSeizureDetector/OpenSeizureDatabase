# OpenSeizureDatabase Software Structure

## Curator Tools
The curator tools folder should not be needed by most users - it contains the tools to download the data from the 'Data Sharing' server and create the OpenSeizureDatabase .json distribution files.

## libosd
The libosd folder contains a library of python scripts for using and manipulating the OpenSeizureDatabase data.   The software in the user_tools folder uses this library to process the data.

## user_tools
User_tools contains a number of software tools to work with OpenSeizureDatabase Data as described below.

### dataSummariser
summariseData.py will create html summary pages for events stored in the database, plotting raw accelerometer and heart rate data, and also the results from the OSD seizure detection algorithm.

### nnTraining
nnTrainer.py uses OpenSeizureDatabase to train a convolutional neural network (CNN) seizure detector model.

### testRunner
testRunner.py passes OpenSeizureDatabase through user specified seizure detection algorithms in the same way as it would receive the data when implemented on a phone using OpenSeizureDetector.   It calculates statistics for the seizure detection reliability and reliability of detecting non-seizure false alarms corretly.

# Licenses
  * The software is licenced under GPL3 or later
  * When published, the data will be licenced under a variation of a Creative Commons Attribution licence, which will require the results of any work using the data to be published so that the users of OpenSeizureDetector who contributed the data can benefit from it (see [Licence](./Licence.md))


# Contact
If you have any questions regarding this software, please contact graham@openseizuredetector.org.uk
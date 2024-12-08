# User Tools README

This folder contains a number of python tools that users of the Open Seizure Database might find useful for processing the data:

  - dataSummariser - produces an html page that summarises the data for a particular event.
  - testRunner - runs the data in the database through user defined seizure detection algorithms and calculates succsess/failure statistics to allow the algorithms to be evaluated.
  - nnTraining - first go at code to train a neural network for seizure detection (including data augmentation) - *now obsolite* - develop nnTraining2 instead.
  - nnTraining2 - updated tool to train neural network (or other machine learning model) using OpenSeizureDatabase data.   This produces intermediate .csv files for test/tain data and
    before and after augmentation is applied to reudce processing time and memory requirements for repeat runs.
  - colabScripts - scripts used by Jamie Pordoy during his PhD research.
  - mongodb - not used at the moment - it might be worth loading osdb into a mongodb database so we can use mongodb's filtering capability to select data?
  - See also https://github.com/jpordoy/AMBER/tree/Amber_beta_1.0.1 which is Jamie Pordoy's LSTM based seizure detector model he developed during his PhD (I want to get this working on
    the current dataset and try it in the OpenSeizureDetector android app.   There is a fork of this repository (https://github.com/OpenSeizureDetector/AMBER) that I am modifying to take the .csv
    files produced by the nnTraining2 tool chain.

    

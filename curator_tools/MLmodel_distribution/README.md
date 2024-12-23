MLmodel_distribution
====================

This folder contains the files and scripts associated with distributing machine learning models to the 
main OpenSeizureDetector Android App.

There is a JSON file MLmodels.json which describes the available machine learning models, and then several
.tflite files which are the tensorflow lite files for implementing hte model on Android.

upload.sh uploads the files so they appear at https://openseizuredetector.org.uk/static/MLmodels/MLmodels.json etc.

The MLmodels.json file is an array of objects, with each object representing a single machine learning model.
The elements of the object are:
  * id - a unique integer identifier for the particular publishe dmodel.
  * desc - text description of the model
  * url - url of the model file (relative to the MLmodels folder on the server)
  * notes - text notes about the model
  * recommended - boolean specifying which model is the recommended one to use.
  * inputFormat - integer representing the particular input format.  Valid valuse of inputFormat are:
     1  - 125 acceleration vector magnitude values (in milli-g), representing 5 seconds of data at 25 Hz sample frequency.


     
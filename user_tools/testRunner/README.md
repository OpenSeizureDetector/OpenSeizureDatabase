testRunner - OpenSeizureDetector Algorithm Testing
===============================================

This folder contains the test framework to assess potential new seizure
detection algorithms by running real data collected using the Data Sharing
system through the proposed algorithm and comparing it to the original
OpenSeizureDetector algorithm results.

Each Algorithm should be implemented as a sub-class of sdAlg.SdAlg
As a minimum it should provide a single function processDp(dpStr) which
accepts a JSON string representing a single datapoint.
It should return a JSON string which contains as a minimum an alarmState
integer element, whose values are:
	0 : OK
	1 : WARNING
	2 : ALARM
Other elements may be included in the return value.

The current state of the code is that it is working to do a simple test using a physical device
running OpenSeizureDetector to provide the processing.  See the example testConfig.json file.

testConfig.json is should contain a single JSON object with teh following elements:
		testName - a human readable description
		download - true to download data from the database, false to use cached data
		credentialsFname - file name of a file containing database login credentials
				 (see client.cfg.template).
		eventsList - an array of eventId numbers to process
		algorithms - an array of algorithm objects, each of which contains the following:
			   "name" - a human readable name
			   "alg" - the module and class name of the algorithm to test.
			   "settings" - an algorithm specific settings object which is passed
			   	      to the class constructor.


Algorithms
==========

deviceAlg.DeviceAlg
-------------------
This algorithm uses the web interface to a running instance of the OpenSeizureDetector Android
App on an a Android device.   The only used setting is "ipAddr" which is the IP address of the
Android device running OpenSeizureDetector as shown on the OpenSeizureDetector main screen.

osdAlg.OsdAlg
-------------
This is be a python implementation of the original OpenSeizureDetector detection algorithm, with user configurable options for improvements.
It is configured in testConfig.json as per the example below:

		{
			"name": "OSD_v3",
			"alg": "osdAlg.OsdAlg",
			"enabled": true,
			"settings" : {
				"mode": "V3",
				"offset": 2500.0,
				"sampleFreq" : 25,
				"samplePeriod" : 5.0,
				"alarmFreqMin" : 3,
				"alarmFreqMax" : 8,
				"alarmThresh" : 100,
				"alarmRatioThresh" : 30,
				"warnTime": 5,
				"alarmTime": 10
			}
		}

The settings parameters are:
  * mode: V0 = original OSD algorithm, using the provided vector magnitudes in the data element for each datapoint.
			V1 = reproduction of the original OSD algorithm using 3d data to calculate the approximate vector magnitude.
			V2 = Version 2 of the OSD algorithm, which uses a mathematically correct vector magnitude.
			V3 = as for Version 2, but a fixed value is added to each accelerometer reading before calculating the vector magnitude.   The offset is specified as the 'offset' setting.
			V4 = as for Version 3, but the algorithm is calculated on each accelerometer axis independently, and the result based on a 1oo3 voting. (NOT YET IMPLEMENTED)


Usage
=====
Edit testConfig.json to suit your requirements.   Note that it points to a client.cfg file
which should contain database login credentials.   If you want to analyse other users' data
you must authenticate as a 'researcher' user rather than a normal user.
Contact graham@openseizuredetector.org.uk for access.

Then just run ./testRunner.py

There are a few dependencies such as numpy which must be satisfied for it to work.
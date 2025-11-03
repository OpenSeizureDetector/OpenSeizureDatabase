OpenSeizureDatabase Structure
=============================

This file provides a brief overview of the database files and their structure.   A more detailed description of the methodology is provided in a research paper preprint [here](https://www.researchgate.net/publication/373173396_The_Open_Seizure_Database_Facilitating_Research_Into_Non-EEG_Seizure_Detection).  It is also stored in this repository [here](./The_Open_Seizure_Database_Version_1.0.0.pdf)


Data Files
==========

The OpenSeizureDatabase is a collection of seizures or seizure-like 'Events' which have been contributed by the users of the OpenSeizureDetector system.
The events are categorised so they can be used in the development of testing of seizure detection systems.   The categories are:
  * Seizure
  * Fall
  * False Alarm
  * Unknown (unknown events are probably false alarms which have not been categorised by the user, but may contain genuine seizures)

Because tonic-clonic seizures are of particular interest in seizure detection (becasue they are easier to detect!), tonic clonic seizures are separated out into a dedicated data set.

A summary of the data is provided as a .csv file for each category with one row per event to give a high level overview of the events in the database.    The main data is provided as one .json file for each category which contains both the event description and the sensor measurements during the event.  The file structures are described below.

The OpenSeizureDetector Data Sharing system generates an Event every time the system produces a Warning, Alarm or Fall state.   This can result in significant duplication because a single seizure may generate several Warning and Alarm states during the seizure.  For this reason the raw events in the system are grouped into a smaller number of 'Unique Events' which are included in the published database.   (It is currently assumed that all alarms and warnings generated within a 3 minute period are part of the same event so they are grouped together.   In the future a more detailed dataset may be published that groups over a shorter time period so may produce more unique events)


Database Version Numbers
------------------------
The OpenSeizureDatabase data releases are stored under 3 figure version numbers, Va.b.c (such as V1.2.1).

The first figure (a) is the primary release version and is only updated if there is a change to the structure of the database that will break existing tools (such as re-naming a data element).

The second figure (b) is updated when additional data is added to the database - this will occur several times a year.

The final figure (c) is updated for data corrections - for example if a user reports that an event contains invalid data, the event may be deleted and a corrected version of the database issued.

A database release will be a compressed tar archive named OSDB_Va.b.c.tgz, whic contains several data files as described below.


Category Summary Files
----------------------
The category summary files are:
  * osdb_3min_tcSeizures.csv - overview of all the tonic-clonic seizues in the database
  * osdb_3min_allSeizures.csv - overview of all the seizure events in the database
  * osdb_3min_fallEvents.csv - overview of the fall events in the database
  * osdb_3min_falseAlarm.csv - overview of the false alarm events in the database
  * osdb_3min_unknownEvents.csv - overview of the unknown (=uncategorised) events in the database

Note that the '3min' in the filename refers to the grouping time window used to identify 'unique' events - it is assumed that all alarms and warnings generated within the time window are part of the same Event to avoid duplication.

Each file contains the following data fields:
  * eventId - the unique identifier for the event.
  * userId - the ID of the user who created the event
  * dataTime - the date/time of the event
  * type - event type (Seizure, False Alarm, Fall etc.)
  * subType - event sub-type (Tonic-Clonic, false alarm cause)
  * osdAlarmState - the status of OpenSeizureDetector at the time of the event (0=OK, 1=WARNING, 2=ALARM, 5=Manual Alarm)
  * dataSource - the OpenSeizureDetector data source used to collect the data (e.g. Garmin)
  * phoneAppVersion - the version of the OpenSeizureDetector phone app that created the event.
  * watchAppVersion - the version of the OpenSeizureDetector watch app that created the event
  * desc - user entered description of the event (if available).
  * Description

JSON Data Files
---------------
The JSON data file names are similar to the equivalent event category summary files described above, but end '.json'.
The format of each file is JSON.  It contains the following data.
  * An array of Event Objects (dictionaries).
  * Each Event object contains the following fieds:
      * id:  Unique Event ID
      * dataTime: Event date/time in yyy-mm-ddThh:mm:ssZ format
      * desc:  User entered description of the event
      * type:  Event Type
      * subType:  Event subType
      * userId:  User ID of user who created the event.
      * osdAlarmState: 0=OK, 1=WARNING, 2=ALARM, 5=Manual Alarm
      *  batteryPc - watch battery charge state (%)
      *  alarmState - 0=OK, 1=WARNING, 2=ALARM, 5=Manual Alarm
      *  alarmPhrase - text version of alarmState
      *  sdMode - not used
      *  sampleFreq - accelerometer sample frequency in Hz (should be 25 Hz)
      *  analysisPeriod - not used
      *  seizuretimes - [start, end] - the start and end of the clonic part of the seizure in seconds from the event dataTime, set by manual examination of the data.
      *  alarmFreqMin - lower frequency limit of ROI (Hz)
      *  alarmFreqMax - upper frequency limit of ROI (Hz)
      *  alarmThresh - absolute power threshold to enable seizure detection.
      *  alarmRatioThresh - ratio of ROI / whole spectrum power that will rigger an alarm (actually ratio*10, so 57 means 5.7).
      *  hrAlarmActive - boolean to state if the Heart Rate alarm was enabled at the time of the event.
      *  hrAlarmStanding - boolean to state if the heart rate alarm was standing at the time of the event.
      *  hrThreshMin - lower threshold to generate a heart rate alarm (bpm)
      *  hrThreshMax - upper threshold to generate an alarm (bpm)
      *  o2SatAlarmActive - bolean to state if the oxygen saturation alarm is enabled.
      *  o2SatAlarmStanding - boolean to state if the oxygen saturation alarm was standing at the time of the event.
      * o2SatThreshMin - lower threshold to generate an oxygen saturation alarm (%)
      * dataSourceName - the OpenSeizureDetector data source in use to generate the event.
      * watch Part No - manufacturer's part number of the watch in use.
      *  watchSdName - name of the watch app in use.
      * watchFwVersion - firnware version of the watch being used.
      * watchSdVersion - version numner of the watch app being used.
      * datapoints:  An array of datapoint objects, where each datapoint represents a 5 second period.  Each datapoint object contains the followning fields:
        * dataTime: the date time of the datapoint in yyyy-mm-ddThh:mm:ssZ format.
        * eventId: The ID of the event associated with this datapoint.
        * hr:  Measured heart rate (bpm)
        * id:  data point ID (not unique)
        * userId - the ID number of the user who created the event.
        * specPower - average power per bin over the whole spectrum.
        * roiPower - average power per bin over the region of interest frequencies.
        * roiRatio - 10*specPower/roiPower (compared to alarmRatioThresh value)
        * alarmState - 0=OK, 1=WARNING, 2=ALARM, 5=Manual Alarm
        * alarmPhrase - text version of alarmState
        * hr: measured heart rate (bpm)
        * o2Sat: measured oxygen saturation (%) (-1 = error)
        * simpleSpec: array of powers within 1 Hz bands between 0 and 10 Hz.
        * rawData: array of 125 accelerometer magnitude readings, followed by zeros
        * rawData3D: array of 3x125 accelerometer readings x1,y1,z1, x2,y2,z2, x3, y3, z3 etc.
           
Note that some of the fields in the files may not pre present for data collected using older versions of OpenSeizureDetector, so the presence of a particular filed should be checked for each event used.

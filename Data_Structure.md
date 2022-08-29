OpenSeizureDatabase Structure
=============================

The OpenSeizureDatabase is a collection of seizures or seizure-like 'Events' which have been contributed by the users of the OpenSeizureDetector system.
The events are categorised so they can be used in the development of testing of seizure detection systems.   The categories are:
  * Seizure
  * Fall
  * False Alarm
  * Unknown (unknown events are probably false alarms which have not been categorised by the user, but may contain genuine seizures)

Because tonic-clonic seizures are of particular interest in seizure detection (becasue they are easier to detect!), tonic clonic seizures are separated out into a dedicated data set.

The data is provided as a .csv file for each category with one row per event to give a high level overview of the events in the database.    The main data is provided as one .json file for each category which contains both the event description and the sensor measurements during the event.  The file structures are described below.

The OpenSeizureDetector Data Sharing system generates an Event every time the system produces a Warning, Alarm or Fall state.   This can result in significant duplication because a single seizure may generate several Warning and Alarm states during the seizure.  For this reason the raw events in the system are grouped into a smaller number of 'Unique Events' which are included in the published database.   (It is currently assumed that all alarms and warnings generated within a 10 minute period are part of the same event so they are grouped together.   In the future a more detailed dataset may be published that groups over a shorter time period so may produce more unique events)

Data Files
==========
Category Summary Files
----------------------
The category summary files are:
  * osdb_10min_tcSeizures.csv - overview of all the tonic-clonic seizues in the database
  * osdb_10min_allSeizures.csv - overview of all the seizure events in the database
  * osdb_10min_fallEvents.csv - overview of the fall events in the database
  * osdb_10min_falseAlarm.csv - overview of the false alarm events in the database
  * osdb_10min_unknownEvents.csv - overview of the unknown (=uncategorised) events in the database
Note that the '10min' in the filename refers to the grouping time window used to identify 'unique' events - it is assumed that all alarms and warnings generated within the time window are part of the same Event to avoid duplication.

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
      * dataJSON:  a JSON encoded string of data describing the OpenSeizureDetector state at the time of the event.   The fields in dataJSON are:
        *  dataTime - date/time of the event in dd-mm-yyyy hh:mm:ss format
        *  dataTimeStr - as data time but in yyymmddThhmmss format 
        *  batteryPc - watch battery charge state (%)
        *  alarmState - 0=OK, 1=WARNING, 2=ALARM, 5=Manual Alarm
        *  alarmPhrase - text version of alarmState
        *  sdMode - not used
        *  sampleFreq - accelerometer sample frequency in Hz (should be 25 Hz)
        *  analysisPeriod - not used
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

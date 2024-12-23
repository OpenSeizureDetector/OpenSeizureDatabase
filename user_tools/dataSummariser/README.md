OSTK Data Summariser
====================

This folder contains example tools to summarise data stored in the OpenSeizureDatabase JSON files.

Configuration
-------------
The configuration file osdbCfg.json contains a JSON object with the following elements:
   * cacheDir - this is the folder that contains the OSDB JSON files.   If cacheDir is not present it defaults to ~/osd/osdb.
   * dataFiles - a list of OSDB filenames to load for analysis (element _dataFiles is a complete list for reference).
   * invalidEvents - a list of event ID numbers to exclude as invalid (useful if an event is identified as having corrupt data so should be ignored - please inform osdb@openseizuredetector.org.uk if you find corrupt data so it can be excluded from future releases of the database).
   * credentialsFname - file name of a text file containing login credentials for the OSDB web API (only relevant for analysing 'live' data to support user queries over reasons for false alarms etc.)
   * skipElements - data object elements to be excluded from raw data output for clarity (users should not need to use this unless you want to produce a simplified JSON file for a particular event).


Usage
-----
usage: summariseData.py [-h] [--config CONFIG] [--remote] [--event EVENT]
                        [--outDir OUTDIR] [--index] [--debug]

options:
  *  -h, --help       show this help message and exit
  *  --config CONFIG  name of json configuration file
  *  --remote         Load events data from remote database, not locally cached OSDB
  *  --event EVENT    event to summarise (or comma separated list of event IDs)
  *  --outDir OUTDIR  output directory
  *  --index          Re-build index, not all summaries
  *  --debug          Write debugging information to screen


Usage Examples
--------------
  * summariseData --index  :  Produces an html index of the data in the loaded data files.
  * summariseData --event=78001  :  Produces an html summary page of event ID 78001
  * summariseData  :   Produces html summary pages for each event in the loaded data files.
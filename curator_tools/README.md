OpenSeizureDatabase Curator Tools
=================================

This folder contains tools for the creation and maintanance
of the anonymised, distributable OpenSeizureDatabase files.
It is not needed to use OpenSeizureDatabase for analysis of seizures.

Create an OpenSeizureDatabase Distribution
==========================================
The OpenSeizureDatabse distribution is a set of text files (json and csv format) that are generated from data in the OpenSeizureDetector Data Sharing
system live database.

Set-Up
------
Copy client.cfg.template to client.cfg and edit it to include the login
credentials of a user with 'staff' rights to the Data Sharing database.

Copy the osdb.cfg.template to osdb.cfg  
Edit osdb.cfg to alter the settings for the database to be created.
The fields in osdb.cfg are:
  * groupingPeriod - the time period over which events are grouped to consider them to be unique.  Valid frequency specifications are specified at https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
  * includeWarnings - 0 = only include events that are either marked as a seizure or resulted in an ALARM state.   1 = include all events including those which only resulted in a WARNING state.
  * invalidEvents - a list of eventIds to exclude from the database
  * invalidEventsNotes - Used to describe why events are excluded in case we forget.
  * credentialsFname - filename of the configuration file that specifies the user credentials to access the OpenSeizureDetector Data Sharing database (client.cfg)

Running
-------
run python makeOsdDb.py --out='osdb_10min'
the --out parameter specifies the prefix for the output files - above we
are including the grouping frequency in the filenames.

Generation can take some time as the Data Sharing database is slow when retrieving raw data - it can take around 90 minutes.

The output files are generated in the current working directory.


Contact
-------
Contact graham@openseizuredetector.org.uk with any issues.

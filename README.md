# OpenSeizureDatabase

This repository is for the database of seizure and seizure-like data that has been contributed by OpenSeizureDetector users to contribute to research to improve seizure detection algorithms

## Installation Instructions

  * Create a folder in your home directory called "osd" and change directory into it.
  * Clone this repository (git clone 
  * Create a folder in your home directory called "osd" with a subdirectory "osdb"
  * Obtain a copy of the OSDB JSON text files (contact graham@openseizuredetector.org.uk) and copy them into ~/osd/osdb


## Test Installation
  * Go to osdb/user_tools/dataSummariser
  * execute python ./summariseData.py --index.   This should produce a file output/index.html which lists all the data in the database.   Note that there will be missing image files because these are only generated when a full summary is created.
  * Select an event from the list in index.html and note its event ID
  * exeute python ./summariseData.py --event=<eventId>.   This should produce a folder, output/Event_<eventId>_summary which contanis an html file (index.html) and associated images to display a summary of the event.

## Documentation
For details of the data structure and the software included in this repository, pleaase refer to the [Documentation](./documentation/README.md)
# OpenSeizureDatabase

This repository is for the toolkit to create and proocess the database of seizure and seizure-like data that has been contributed by OpenSeizureDetector users to contribute to research to improve seizure detection algorithms (Open Seizure Toolkit (OSDK)).

The repository contains a number of python tools that form a toolkit to assist users of the database in working with the JSON files containing the seizure data.

## Licence
The python tools are licenced under the GNU Public Licence, V3.
The OpenSeizureDatabase data is licenced under a variation of the Creative Commons Share Alike licence, with an additional licence condition to publish results obtained using the data - see the [LICENCE](./documentation/LICENCE.md) description for more details.

## Data Users
The following people have access to the anonymised OpenSeizureDatabase data.

| Name    |  Contact           | Research Area   |  Web Site |
| -----   | ----               | ----            | ----      |
| Graham Jones | graham@openseizuredetector.org.uk | Development of Improved SeizureDetection Algorithms for OpenSeizureDetector | [OpenSeizureDetector](http://openseizuredetector.org.uk) |
| Jamie Pordoy |               | PhD Research into Seizure Detection |       |


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
For details of the data structure and the software included in this repository, please refer to the [Documentation](./documentation/README.md)

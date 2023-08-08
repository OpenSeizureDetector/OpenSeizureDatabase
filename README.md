# OpenSeizureDatabase

This repository contains the toolkit to create and proocess the database (the Open Seizure Database) of seizure and seizure-like data that has been contributed by OpenSeizureDetector users to contribute to research to improve seizure detection algorithms (Open Seizure Toolkit (OSTK)).

The repository contains a number of python tools that form a toolkit to assist users of the database in working with the JSON files containing the seizure data.

## Licence
The python tools are licenced under the GNU General Public Licence, V3.
The OpenSeizureDatabase data is licenced under a variation of the Creative Commons Share Alike licence, with an additional licence condition to publish a description of the system developed and results obtained using the data for the benefit of users of the OpenSeizureDetector project - see the [LICENCE](./documentation/LICENCE.md) description for more details.

## Data Users
The following people have access to the anonymised OpenSeizureDatabase data.

| Name    |  Contact           | Research Area   |  Web Site |
| -----   | ----               | ----            | ----      |
| Graham Jones | graham@openseizuredetector.org.uk | Development of Improved SeizureDetection Algorithms for OpenSeizureDetector | [OpenSeizureDetector](http://openseizuredetector.org.uk) |
| Jamie Pordoy | pordjam@uwl.ac.uk    | PhD Research into Seizure Detection |   [IntSaV Group](https://www.uwl.ac.uk/research/research-centres-and-groups/intelligent-sensing)    |
| Benjamin Mickler |               | Research into seizure detection |       |

Researchers who would like access to the data should email osdb@openseizuredetector.org.uk explaining what they intend to use the data for, and confirming that they will comply with the requirements of the [LICENCE](./documentation/LICENCE.md) for the data.    They will then be given access to a repository containing the JSON files that make up the Open Seizure Database.


## Installation Instructions

  * Create a folder in your home directory called "osd" and change directory into it.
  * Clone this repository (git clone https://github.com/OpenSeizureDetector/OpenSeizureDatabase.git)
  * change directory to ~/osd/OpenSeizureDatabase
  * create a python virtual environment with python -m venv ~/pyEnvs/osdb, and activate it with source ~/pyEnvs/osdb/bin/activate.
  * execute pip install -r requirements.txt (note the tensorflow requirement is not essential so can be removed if you do not intend to use tensorflow to train neural networks).
  * Create a folder in your home directory called "osd/osdb"
  * Copy the OSDB JSON text files into ~/osd/osdb


## Test Installation
  * Go to ~osd/OpenSeizureDatabase/user_tools/dataSummariser
  * execute python ./summariseData.py --index.   This should produce a file output/index.html which lists all the data in the database.   Note that there will be missing image files because these are only generated when a full summary is created.
  * Select an event from the list in index.html and note its event ID
  * exeute python ./summariseData.py --event=<eventId>.   This should produce a folder, output/Event_<eventId>_summary which contanis an html file (index.html) and associated images to display a summary of the event, similar to the example output in the [documentation](./documentation/) folder.


## Documentation
For details of the data structure and the software included in this repository, please refer to the [Documentation](./documentation/README.md).
The [documentation](./documentation/) folder also contains an Example set of data for a seizure for reference.

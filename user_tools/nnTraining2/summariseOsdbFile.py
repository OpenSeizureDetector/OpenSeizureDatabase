#!/usr/bin/env python3

import argparse
from re import X
import sys
import os
import json
import importlib
from urllib.parse import _NetlocResultMixinStr
#from tkinter import Y
import sklearn.model_selection
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt

from fpdf import FPDF

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libosd.osdDbConnection
import libosd.dpTools
import libosd.osdAlgTools
import libosd.configUtils


def create_pdf_with_charts(outFname, nSeizures, nNonSeizures, userIdSeizureCounts, seizureTypesCounts, nonSeizureTypesCounts):
    '''
    Create a PDF with charts for the summary statistics.
    PDF generation and chart creation logic contributed by DeepSeek (https://www.deepseek.com).

    Parameters:
    outFname (str): The name of the output PDF file.
    nSeizures (int): The number of seizures.
    nNonSeizures (int): The number of non-seizures.
    userIdSeizureCounts (dict): A dictionary of user IDs and the number of seizures they have.
    seizureTypesCounts (dict): A dictionary of seizure types and the number of each type.
    nonSeizureTypesCounts (dict): A dictionary of non-seizure types and the number of each type.
    '''
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add summary statistics to the PDF
    pdf.cell(200, 10, txt="Open Seizure Database Summary", ln=True, align="C")
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Total Seizures: {nSeizures}", ln=True)
    pdf.cell(200, 10, txt=f"Total Non-Seizures: {nNonSeizures}", ln=True)
    pdf.ln(10)

    # Create a pie chart for seizure vs non-seizure
    labels = ['Seizures', 'Non-Seizures']
    sizes = [nSeizures, nNonSeizures]
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Seizure vs Non-Seizure Events')
    plt.savefig('seizure_pie_chart.png')
    plt.close()
    pdf.image('seizure_pie_chart.png', x=10, y=None, w=50)

    # Create a bar chart for seizure types
    plt.figure(figsize=(10, 6))
    plt.bar(seizureTypesCounts.keys(), seizureTypesCounts.values())
    plt.title('Seizure Types Distribution')
    plt.xlabel('Seizure Type')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.tight_layout() 
    plt.savefig('seizure_types_bar_chart.png')
    plt.close()
    pdf.image('seizure_types_bar_chart.png', x=10, y=None, w=100)

    # Create a bar chart for non-seizure types
    plt.figure(figsize=(10, 8))
    plt.bar(nonSeizureTypesCounts.keys(), nonSeizureTypesCounts.values())
    plt.title('Non-Seizure Types Distribution')
    plt.xlabel('Non-Seizure Type')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.tight_layout() 
    plt.savefig('non_seizure_types_bar_chart.png')
    plt.close()
    pdf.image('non_seizure_types_bar_chart.png', x=10, y=None, w=180)

    # Create a bar chart for users with seizures
    plt.figure(figsize=(10, 6))
    plt.bar(userIdSeizureCounts.keys(), userIdSeizureCounts.values())
    plt.title('Numbers of Seizure Events Contributed by Users')
    plt.xlabel('User ID')
    plt.ylabel('Number of Seizures')
    plt.xticks(rotation=45)
    plt.savefig('users_seizures_bar_chart.png')
    plt.close()
    pdf.image('users_seizures_bar_chart.png', x=10, y=None, w=180)

    # Save the PDF
    pdf.output(outFname)


def summariseOsdbFile(inFname, outFname, makePdf=True, debug=False):
    '''
    Load the osdb data file inFname and output a text summary of the data contained in it.
    if outFname is None, sends output to stdout.
    '''

    osd = libosd.osdDbConnection.OsdDbConnection(debug=debug)

    if inFname is not None:
        print("flattenOsdb - loading file %s" % inFname)
        eventsObjLen = osd.loadDbFile(inFname, useCacheDir=False)
    else:
        print("No input file specified")
        return False    
    
    print("%d Events Loaded" % eventsObjLen)
    eventIdsLst = osd.getEventIds()
    nEvents = len(eventIdsLst)

    nNoDatapoints = 0
    nDatapoints = 0
    nSeizures = 0
    nNonSeizures = 0
    userIdCounts = {}
    userIdSeizureCounts = {}
    userIdNonSeizureCounts = {}
    seizureTypesCounts = {}
    nonSeizureTypesCounts = {}
    for eventNo in range(0,nEvents):
        eventId = eventIdsLst[eventNo]
        eventObj = osd.getEvent(eventId, includeDatapoints=True)
        userId = str(eventObj['userId'])
        if (not 'datapoints' in eventObj or eventObj['datapoints'] is None):
            print("Event %s: No datapoints - skipping" % eventId)
            nNoDatapoints += 1
        else:
            #print("nDp=%d" % len(eventObj['datapoints']))
            nDatapoints += 1
            if (eventObj['type'].lower() == 'seizure'):
                nSeizures += 1
                if userId in userIdSeizureCounts:
                    userIdSeizureCounts[userId] += 1
                else:
                    userIdSeizureCounts[userId] = 1
                if 'subType' in eventObj and eventObj['subType'] is not None and len(eventObj['subType']) > 0:
                    subType = eventObj['subType'].lower()
                else:
                    subType = "null"
                if subType in seizureTypesCounts:
                    seizureTypesCounts[subType] += 1
                else:
                    seizureTypesCounts[subType] = 1
            else:
                nNonSeizures += 1
                if userId in userIdNonSeizureCounts:
                    userIdNonSeizureCounts[userId] += 1
                else:
                    userIdNonSeizureCounts[userId] = 1
                if 'subType' in eventObj and eventObj['subType'] is not None and len(eventObj['subType']) > 0:
                    subType = eventObj['subType'].lower()
                else:
                    subType = "null"
                if subType in nonSeizureTypesCounts:
                    nonSeizureTypesCounts[subType] += 1
                else:
                    nonSeizureTypesCounts[subType] = 1
            if userId in userIdCounts:
                userIdCounts[userId] += 1
            else:
                userIdCounts[userId] = 1

    # Tidy some of the data to reduce the number of non-seizure categories.
    # This is a bit of a hack, but it's useful for the summary statistics.
    try:
        nonSeizureTypesCounts['brushing hair/teeth'] += nonSeizureTypesCounts.pop('brushing hair')
        nonSeizureTypesCounts['brushing hair/teeth'] += nonSeizureTypesCounts.pop('brushing teeth')
        nonSeizureTypesCounts['pushing pram/wheelchair/lawn mower'] += nonSeizureTypesCounts.pop('pushing pram/wheelchair')
        nonSeizureTypesCounts['washing / cleaning'] += nonSeizureTypesCounts.pop('washing up')
        nonSeizureTypesCounts['cooking/washing/cleaning'] += nonSeizureTypesCounts.pop('washing / cleaning')
        nonSeizureTypesCounts['sorting/knitting'] += nonSeizureTypesCounts.pop('sorting')
        nonSeizureTypesCounts['sorting/knitting'] += nonSeizureTypesCounts.pop('knitting')
        nonSeizureTypesCounts['walking/running/cycling'] += nonSeizureTypesCounts.pop('walking')
        nonSeizureTypesCounts['walking/running/cycling'] += nonSeizureTypesCounts.pop('cycling')
        nonSeizureTypesCounts['walking/running/cycling'] += nonSeizureTypesCounts.pop('cycle/run/walk')
        nonSeizureTypesCounts['talking/standing still'] += nonSeizureTypesCounts.pop('talking')
        nonSeizureTypesCounts['typing/hand tools'] += nonSeizureTypesCounts.pop('typing')
        nonSeizureTypesCounts['other'] += nonSeizureTypesCounts.pop('other (please describe in notes)')
        nonSeizureTypesCounts['other'] += nonSeizureTypesCounts.pop('null')
        
    except:
        print("Error simplifying non-seizure statistics")

    # Remove user IDs which have zero seizure data.
    for userId in userIdSeizureCounts.keys():
        if userIdSeizureCounts[userId] == 0:
            userIdSeizureCounts.pop(userId)

    print(nonSeizureTypesCounts.keys())
    print(userIdSeizureCounts.keys())


    if (makePdf):
        # Create PDF with charts    
        print("Outputting summary to PDF")
        if outFname is None:
            outFname = "osdbSummary.pdf"
        create_pdf_with_charts(outFname, nSeizures, nNonSeizures, userIdSeizureCounts, seizureTypesCounts, nonSeizureTypesCounts)
        print(f"PDF output written to {outFname}")

    else:
        try:
            print("Outputting summary to text")
            if outFname is not None:
                outPath = os.path.join(".", outFname)
                print("sending output to file %s" % outPath)
                outFile = open(outPath,'w')
            else:
                print("sending output to stdout")
                outFile = sys.stdout

            outFile.write("Summary of %d events:\n" % nEvents)
            #outFile.write("  %d events with no datapoints\n" % nNoDatapoints)
            #outFile.write("  %d events with datapoints\n" % (nDatapoints))
            outFile.write("  %d seizures\n" % nSeizures)
            outFile.write("  %d non-seizures\n" % nNonSeizures)
            outFile.write("  %d unique users\n" % len(userIdCounts))
            outFile.write("  %d unique users with seizures\n" % len(userIdSeizureCounts))
            outFile.write("  %d unique users with non-seizures\n" % len(userIdNonSeizureCounts))

            outFile.write("\n")
            outFile.write("  Seizure types:\n")
            for seizureType in seizureTypesCounts.keys():
                outFile.write("    %s: %d (%.1f%%)\n" % (seizureType, seizureTypesCounts[seizureType], 100.*seizureTypesCounts[seizureType]/nSeizures))

            outFile.write("\n")
            outFile.write("  Non-seizure types:\n")
            for nonSeizureType in nonSeizureTypesCounts.keys():
                outFile.write("    %s: %d (%.1f%%)\n" % (nonSeizureType, nonSeizureTypesCounts[nonSeizureType], 100.*nonSeizureTypesCounts[nonSeizureType]/nNonSeizures))

            outFile.write("\n")
            outFile.write("  Users with seizures:\n")
            for userId in userIdSeizureCounts.keys():
                outFile.write("    %s: %d (%.1f%%)\n" % (userId, userIdSeizureCounts[userId], 100.*userIdSeizureCounts[userId]/nSeizures))
            #outFile.write("  Users with non-seizures:\n")
            #for userId in userIdNonSeizureCounts.keys():
            #    outFile.write("    %s: %d (%.1f%%)\n" % (userId, userIdNonSeizureCounts[userId], 100.*userIdNonSeizureCounts[userId]/nNonSeizures))
        finally:
            if (outFname is not None):
                outFile.close()
                print("Output written to file %s" % outFname)


    return True

 

def main():
    print("summariseOsdbFile.main()")
    parser = argparse.ArgumentParser(description='Produce a summary of the data stored in an OSDB JSON format file')
    parser.add_argument('-i', default=None,
                        help='Input filename (uses configuration datafiles list if not specified)')
    parser.add_argument('-o', default=None,
                        help='Output filename (uses stout if not specified)')
    parser.add_argument('--debug', action="store_true",
                        help='Write debugging information to screen')
    parser.add_argument('--pdf', action="store_true",
                        help='create PDF output with charts, rather than text output')
    argsNamespace = parser.parse_args()
    args = vars(argsNamespace)
    print(args)

    summariseOsdbFile(args['i'], args['o'], makePdf=args['pdf'], debug=args['debug'])

if __name__ == "__main__":
    main()

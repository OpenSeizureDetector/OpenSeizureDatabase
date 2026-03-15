"""results.py – Result saving and statistics for testRunner.

Provides:
- Event-type index constants (OTHERS, ALL, FALSE, NDA)
- ``saveResults2``  – primary result writer, CSV files + summary text
- ``saveResults``   – legacy single-file writer
- ``getResultsStats`` / ``summariseResults`` – statistical helpers
"""
import os
import json
import csv
import numpy as np

from alg_runner import getEventAlarmState
from report import generateSummaryReport

# ---------------------------------------------------------------------------
# Event-type index constants
# ---------------------------------------------------------------------------

OTHERS_INDEX = 0
ALL_INDEX    = 1
FALSE_INDEX  = 2
NDA_INDEX    = 3


def type2index(typeStr, subTypeStr=None):
    """Return the output-array index for the given event type string."""
    retVal = OTHERS_INDEX
    t = typeStr.lower()
    if t == "nda":
        retVal = NDA_INDEX
    elif t == "false alarm":
        retVal = FALSE_INDEX
    elif t == "seizure":
        retVal = ALL_INDEX
    return retVal


# ---------------------------------------------------------------------------
# Primary result writer
# ---------------------------------------------------------------------------

def saveResults2(outDir, results, resultsStrArr, eventIdsLst, osd, algNames,
                 perDpDataLst=None, debug=False):
    """Write per-event CSV files to ``outDir``, a summary text file, save
    per-datapoint data, then generate the visual summary report.
    """
    print("saveResults2 - outDir=%s" % outDir)
    nEvents = len(eventIdsLst)

    outputs = [""] * 4
    outputs[OTHERS_INDEX] = "otherEvents"
    outputs[ALL_INDEX]    = "allSeizures"
    outputs[FALSE_INDEX]  = "falseAlarms"
    outputs[NDA_INDEX]    = "nda"

    outfLst = []
    for output in outputs:
        fname = os.path.join(outDir, "output_%s.csv" % output)
        outfLst.append(open(fname, "w"))

    nAlgs   = results.shape[1]
    lineStr = "eventId, date, type, subType, userId, datasource"
    for algNo in range(nAlgs):
        lineStr += ", %s" % algNames[algNo]
    lineStr += ", reported"
    for algNo in range(nAlgs):
        lineStr += ", %s" % algNames[algNo]
    lineStr += ", desc"
    print(lineStr)
    for outf in outfLst:
        if outf is not None:
            outf.write(lineStr + "\n")

    NTP = np.zeros(nAlgs + 1)
    NTN = np.zeros(nAlgs + 1)
    NFP = np.zeros(nAlgs + 1)
    NFN = np.zeros(nAlgs + 1)

    fnEventIds = []
    tpEventIds = []
    fpEventIds = []

    correctCount = np.zeros((len(outfLst), nAlgs + 1))
    totalCount   = np.zeros(len(outfLst))
    tcTotalCount = 0
    tcCorrectCount = np.zeros(nAlgs + 1)

    def _is_tonic_clonic(event_obj):
        if not isinstance(event_obj, dict):
            return False
        if str(event_obj.get('type', '')).strip().lower() != 'seizure':
            return False
        return 'tonic-clonic' in str(event_obj.get('subType', '')).strip().lower()

    alarmPhrases = ['----', 'WARN', 'ALARM', 'FALL', 'unused', 'MAN_ALARM', 'NDA']

    for eventNo in range(nEvents):
        eventId    = eventIdsLst[eventNo]
        eventObj   = osd.getEvent(eventId, includeDatapoints=False)
        outputIndex = type2index(eventObj['type'])
        expectAlarm = eventObj['type'].lower() == "seizure"
        totalCount[outputIndex] += 1

        is_tc = _is_tonic_clonic(eventObj)
        if is_tc:
            tcTotalCount += 1

        lineStr = "%s, %s, %s, %s, %s" % (
            eventId,
            eventObj['dataTime'],
            eventObj['type'],
            eventObj['subType'],
            eventObj['userId']
        )
        lineStr += ", %s" % eventObj.get('dataSourceName', 'unknown')

        for algNo in range(nAlgs):
            alarmed = results[eventNo][algNo][2] > 0
            warned  = results[eventNo][algNo][1] > 0

            if alarmed:
                if expectAlarm:
                    correctCount[outputIndex, algNo] += 1
                    NTP[algNo] += 1
                    if is_tc:
                        tcCorrectCount[algNo] += 1
                else:
                    NFP[algNo] += 1
            else:
                if expectAlarm:
                    NFN[algNo] += 1
                else:
                    correctCount[outputIndex, algNo] += 1
                    NTN[algNo] += 1

            if alarmed:
                lineStr += ", ALARM"
            elif warned:
                lineStr += ", WARN"
            else:
                lineStr += ", ----"

        reportedAlarmState = getEventAlarmState(eventObj=eventObj, debug=False)
        lineStr += ", %s" % alarmPhrases[reportedAlarmState]
        if reportedAlarmState == 2:
            if expectAlarm:
                correctCount[outputIndex, nAlgs] += 1
                NTP[nAlgs] += 1
                if is_tc:
                    tcCorrectCount[nAlgs] += 1
            else:
                NFP[nAlgs] += 1
        else:
            if expectAlarm:
                NFN[nAlgs] += 1
            else:
                correctCount[outputIndex, nAlgs] += 1
                NTN[nAlgs] += 1

        for algNo in range(nAlgs):
            lineStr += ", %s" % resultsStrArr[eventNo][algNo]
        lineStr += ", \"%s\"" % eventObj['desc']
        print(lineStr)

        # Classify for summary report (first algorithm)
        if nAlgs > 0:
            if expectAlarm:
                (tpEventIds if results[eventNo][0][2] > 0 else fnEventIds).append(str(eventId))
            elif results[eventNo][0][2] > 0:
                fpEventIds.append(str(eventId))

        if outfLst[outputIndex] is not None:
            outfLst[outputIndex].write(lineStr + "\n")

    # ---- Footer rows ----
    for outputIndex in range(len(outfLst)):
        outf = outfLst[outputIndex]
        if outf is None:
            continue

        lineStr = "#Total, , , ,"
        for algNo in range(nAlgs + 1):
            lineStr += ", %d" % totalCount[outputIndex]
        print(lineStr); outf.write(lineStr + "\n")

        lineStr = "#Correct Count, , , ,"
        for algNo in range(nAlgs + 1):
            lineStr += ", %d" % correctCount[outputIndex, algNo]
        print(lineStr); outf.write(lineStr + "\n")

        lineStr = "#Correct Prop, , , ,"
        for algNo in range(nAlgs + 1):
            denom = totalCount[outputIndex]
            prop  = (1.0 * correctCount[outputIndex, algNo] / denom) if denom > 0 else float('nan')
            lineStr += ", %.2f" % prop
        print(lineStr); outf.write(lineStr + "\n")

        if outputIndex == ALL_INDEX:
            lineStr = "#TonicClonic Total, , , ,"
            for algNo in range(nAlgs + 1):
                lineStr += ", %d" % tcTotalCount
            print(lineStr); outf.write(lineStr + "\n")

            lineStr = "#TonicClonic Correct Count, , , ,"
            for algNo in range(nAlgs + 1):
                lineStr += ", %d" % tcCorrectCount[algNo]
            print(lineStr); outf.write(lineStr + "\n")

            lineStr = "#TonicClonic Correct Prop, , , ,"
            for algNo in range(nAlgs + 1):
                denom = tcTotalCount
                prop  = (1.0 * tcCorrectCount[algNo] / denom) if denom > 0 else float('nan')
                lineStr += ", %.2f" % prop
            print(lineStr); outf.write(lineStr + "\n")

        outf.close()
        print("Output written to file %s" % outputs[outputIndex])

    # ---- Summary statistics text file ----
    summary_path = os.path.join(outDir, "testRunner_Summary.txt")
    with open(summary_path, "w") as outf:
        outf.write("TestRunner Summary\n\n")
        for algNo in range(nAlgs + 1):
            label = algNames[algNo] if algNo < nAlgs else "reported"
            outf.write("Algorithm %d: %s\n" % (algNo, label))
            outf.write("  NTP = %d\n" % NTP[algNo])
            outf.write("  NFP = %d\n" % NFP[algNo])
            outf.write("  NTN = %d\n" % NTN[algNo])
            outf.write("  NFN = %d\n" % NFN[algNo])
            outf.write("\n")
            if (NTP[algNo] + NFN[algNo]) > 0:
                outf.write("TPR = %.1f%%\n" % (100. * NTP[algNo] / (NTP[algNo] + NFN[algNo])))
            else:
                outf.write("TPR Not Calculated - no positive samples\n")
            if (NTN[algNo] + NFP[algNo]) > 0:
                outf.write("TNR = %.1f%%\n" % (100. * NTN[algNo] / (NTN[algNo] + NFP[algNo])))
            else:
                outf.write("TNR not calculated - no negative samples\n")
            outf.write("\n")

    # ---- Save per-datapoint data for --analyze mode ----
    if perDpDataLst is not None:
        pdp_path = os.path.join(outDir, 'perDpData.json')
        try:
            with open(pdp_path, 'w') as pf:
                json.dump(perDpDataLst, pf)
            print(f"Per-datapoint data saved to {pdp_path}")
        except Exception as e:
            print(f"WARNING: could not save perDpData: {e}")

    # ---- Generate visual summary report ----
    base_alg_names = [n for n in algNames if '.' not in str(n)]
    if not base_alg_names:
        base_alg_names = algNames[:1] if algNames else []
    print(f"\nFN={len(fnEventIds)}, TP={len(tpEventIds)}, FP={len(fpEventIds)}")
    if perDpDataLst is not None:
        generateSummaryReport(outDir, fnEventIds, tpEventIds, fpEventIds,
                              eventIdsLst, osd, perDpDataLst, base_alg_names,
                              debug=debug)
    else:
        print("WARNING: no per-dp data available - skipping summary report graphs")


# ---------------------------------------------------------------------------
# Legacy single-file writer (kept for backward compatibility)
# ---------------------------------------------------------------------------

def saveResults(outFile, results, resultsStrArr, osd, algs, algNames,
                expectAlarm=True):
    print("Displaying Results")
    eventIdsLst = osd.getEventIds()
    nEvents = len(eventIdsLst)
    print("Displaying %d Events" % nEvents)

    with open(outFile, "w") as outf:
        nAlgs   = len(algs)
        lineStr = "eventId, type, subType, userId"
        for algNo in range(nAlgs):
            lineStr += ", %s" % algNames[algNo]
        lineStr += ", reported"
        for algNo in range(nAlgs):
            lineStr += ", %s" % algNames[algNo]
        lineStr += ", desc"
        print(lineStr)
        outf.write(lineStr + "\n")

        correctCount = [0] * (nAlgs + 1)
        for eventNo in range(nEvents):
            eventId   = eventIdsLst[eventNo]
            eventObj  = osd.getEvent(eventId, includeDatapoints=False)
            lineStr   = "%s, %s, %s, %s" % (
                eventId, eventObj['type'], eventObj['subType'], eventObj['userId'])

            for algNo in range(nAlgs):
                alarmed = results[eventNo][algNo][2] > 0
                if alarmed and expectAlarm:
                    correctCount[algNo] += 1
                if not alarmed and not expectAlarm:
                    correctCount[algNo] += 1

                if alarmed:
                    lineStr += ", ALARM"
                elif results[eventNo][algNo][1] > 0:
                    lineStr += ", WARN"
                else:
                    lineStr += ", ----"

            alarmPhrases = ['OK', 'WARN', 'ALARM', 'FALL', 'unused', 'MAN_ALARM', 'NDA']
            lineStr += ", %s" % alarmPhrases[eventObj['osdAlarmState']]
            if eventObj['osdAlarmState'] == 2 and expectAlarm:
                correctCount[nAlgs] += 1
            if eventObj['osdAlarmState'] != 2 and not expectAlarm:
                correctCount[nAlgs] += 1

            for algNo in range(nAlgs):
                lineStr += ", %s" % resultsStrArr[eventNo][algNo]
            lineStr += ", \"%s\"" % eventObj['desc']
            print(lineStr)
            outf.write(lineStr + "\n")

        lineStr = "#Total, ,"
        for algNo in range(nAlgs + 1):
            lineStr += ", %d" % nEvents
        print(lineStr)

        lineStr = "#Correct Count, ,"
        for algNo in range(nAlgs + 1):
            lineStr += ", %d" % correctCount[algNo]
        print(lineStr)
        outf.write(lineStr + "\n")

        lineStr = "#Correct Prop, , , ,"
        for algNo in range(nAlgs + 1):
            lineStr += ", %.2f" % (1. * correctCount[algNo] / nEvents)
        print(lineStr)
        outf.write(lineStr + "\n")

    print("Output written to file %s" % outFile)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def getResultsStats(results, expectAlarm=True):
    nEvents = results.shape[0]
    nAlgs   = results.shape[1]
    correctCount   = [0] * nAlgs
    correctPropLst = [0.0] * nAlgs
    for eventNo in range(nEvents):
        for algNo in range(nAlgs):
            alarmed = results[eventNo][algNo][2] > 0
            if alarmed and expectAlarm:
                correctCount[algNo] += 1
            if not alarmed and not expectAlarm:
                correctCount[algNo] += 1
    for algNo in range(nAlgs):
        correctPropLst[algNo] = 1.0 * correctCount[algNo] / nEvents
    return nEvents, nAlgs, correctPropLst


def summariseResults(tcResults, allSeizuresResults, falseAlarmResults, algNames):
    print("Results Summary")
    nTcEvents,       nAlgs, tcStats         = getResultsStats(tcResults, True)
    nAllEvents,      nAlgs, allSeizsStats   = getResultsStats(allSeizuresResults, True)
    nFalseAlmEvents, nAlgs, falseAlarmStats = getResultsStats(falseAlarmResults, False)
    nAlgs = tcResults.shape[1]

    lineStr = "Category"
    for algNo in range(nAlgs):
        lineStr += ", %s" % algNames[algNo]
    print(lineStr)

    lineStr = "tcSeizures"
    for algNo in range(nAlgs):
        lineStr += ", %.2f" % tcStats[algNo]
    print(lineStr)

    lineStr = "allSeizures"
    for algNo in range(nAlgs):
        lineStr += ", %.2f" % allSeizsStats[algNo]
    print(lineStr)

    lineStr = "falseAlarms"
    for algNo in range(nAlgs):
        lineStr += ", %.2f" % falseAlarmStats[algNo]
    print(lineStr)

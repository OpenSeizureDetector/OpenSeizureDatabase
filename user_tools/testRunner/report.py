"""report.py – Graph generation and HTML summary report for testRunner.

Generates a 3-panel PNG graph for each event (acceleration vector magnitude,
algorithm metric, alarm state vs device-reported) and wraps them in an
HTML index page categorised as False Negatives, True Positives and a
balanced random sample of False Positives.

Also provides ``analyzeExistingResults`` to regenerate the report from a
previously saved run folder without re-running the algorithms.
"""
import os
import sys
import csv
import json
import random

import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
sys.path.append(os.path.join(_HERE, '..', '..'))

from io_utils import loadDataFiles

_GRAPH_COLORS = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown']


# ---------------------------------------------------------------------------
# Single-event graph
# ---------------------------------------------------------------------------

def generateEventGraph(eventId, eventObj, perDpData, algNames, outFname, debug=False):
    """Generate a 3-panel PNG for one event and save it to ``outFname``.

    Panel 1 – Acceleration vector magnitude vs time (25 Hz raw samples).
    Panel 2 – Algorithm metric per datapoint (pSeizure / roiRatio / alarmRatio).
    Panel 3 – Alarm state step plot: algorithm output(s) and device-reported.
    """
    event_type = eventObj.get('type', 'Unknown')
    event_sub  = eventObj.get('subType', '')
    user_id    = eventObj.get('userId', '?')
    desc       = eventObj.get('desc', '')
    data_time  = eventObj.get('dataTime', '')

    title_str = (
        f"Event {eventId} – {event_type}/{event_sub}\n"
        f"User: {user_id}   Time: {data_time}\n"
        f"{desc}"
    )

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    fig.suptitle(title_str, fontsize=9)

    ts_all   = perDpData.get('timestamps', [])
    acc_all  = perDpData.get('accelMag', [])
    ts_dp    = perDpData.get('dpTimestamps', [])
    rep_as   = perDpData.get('reportedAlarmStates', [])
    alg_outs = perDpData.get('algOutputs', {})

    # ---- Panel 1: acceleration magnitude ----
    ax0 = axes[0]
    if ts_all and acc_all:
        ax0.plot(ts_all, acc_all, color='tab:blue', linewidth=0.8, alpha=0.8)
    ax0.set_ylabel('Accel magnitude (mg)')
    ax0.set_title('Acceleration Vector Magnitude')
    ax0.grid(True, alpha=0.3)

    # ---- Panel 2: algorithm metric ----
    ax1 = axes[1]
    has_metric = False
    for alg_idx, alg_name in enumerate(algNames):
        aout = alg_outs.get(alg_name)
        if aout is None:
            continue
        metrics    = aout.get('metrics', [])
        metricName = aout.get('metricName') or 'metric'
        if ts_dp and metrics and any(m is not None for m in metrics):
            m_vals = [m if m is not None else float('nan') for m in metrics]
            n = min(len(ts_dp), len(m_vals))
            ax1.plot(ts_dp[:n], m_vals[:n],
                     color=_GRAPH_COLORS[alg_idx % len(_GRAPH_COLORS)],
                     label=f'{alg_name} ({metricName})',
                     linewidth=1.5)
            has_metric = True
    if not has_metric:
        ax1.text(0.5, 0.5, 'No metric data available',
                 ha='center', va='center', transform=ax1.transAxes, color='grey')
    ax1.set_ylabel('Algorithm Metric')
    ax1.set_title('Seizure Probability / Alarm Ratio')
    if has_metric:
        ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # ---- Panel 3: alarm states ----
    ax2 = axes[2]
    if ts_dp and rep_as:
        n = min(len(ts_dp), len(rep_as))
        ax2.step(ts_dp[:n], rep_as[:n],
                 color='black', where='post', linewidth=2,
                 label='Reported (device)', linestyle='-')
    for alg_idx, alg_name in enumerate(algNames):
        aout = alg_outs.get(alg_name)
        if aout is None:
            continue
        alm_states = aout.get('alarmStates', [])
        if ts_dp and alm_states:
            n = min(len(ts_dp), len(alm_states))
            ax2.step(ts_dp[:n], alm_states[:n],
                     color=_GRAPH_COLORS[alg_idx % len(_GRAPH_COLORS)],
                     where='post', linewidth=1.5, linestyle='--',
                     label=alg_name)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['OK (0)', 'WARN (1)', 'ALARM (2)'])
    ax2.set_ylim(-0.2, 2.5)
    ax2.set_ylabel('Alarm State')
    ax2.set_xlabel('Time (seconds from event start)')
    ax2.set_title('Alarm State: Algorithm vs Device-Reported')
    ax2.legend(fontsize=8, loc='upper left')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    try:
        fig.savefig(outFname, dpi=100, bbox_inches='tight')
    except Exception as e:
        print(f"WARNING: Failed to save graph {outFname}: {e}")
    plt.close(fig)
    if debug:
        print(f"Graph written to {outFname}")


# ---------------------------------------------------------------------------
# Multi-event HTML summary report
# ---------------------------------------------------------------------------

def generateSummaryReport(outDir, fnEventIds, tpEventIds, fpEventIds,
                          eventIdsLst, osd, perDpDataLst, algNames, debug=False):
    """Generate PNG graphs and an HTML index for FN / TP and a random FP sample.

    The number of FP examples shown is capped to ``len(FN) + len(TP)`` so
    the report is balanced without being dominated by false positives.
    """
    print("\n--- Generating Summary Report ---")

    report_dir = os.path.join(outDir, 'report')
    os.makedirs(report_dir, exist_ok=True)

    id_to_dp_idx = {perDpDataLst[i]['eventId']: i for i in range(len(perDpDataLst))}

    n_seizures = len(fnEventIds) + len(tpEventIds)
    fp_sample  = random.sample(fpEventIds, n_seizures) if len(fpEventIds) > n_seizures else list(fpEventIds)

    categories = [
        ('falseNegatives', fnEventIds, 'False Negatives (missed seizures)'),
        ('truePositives',  tpEventIds, 'True Positives (detected seizures)'),
        ('falsePositives', fp_sample,  'False Positives (random sample)'),
    ]

    html_lines = [
        '<html><head>',
        '<style>',
        'body { font-family: Arial, sans-serif; margin: 20px; }',
        'h1 { color: #333; }',
        'h2 { color: #555; border-bottom: 1px solid #ccc; }',
        '.event { margin: 10px 0; border: 1px solid #ddd; padding: 10px; }',
        'img { max-width: 100%; border: 1px solid #ccc; }',
        '</style></head><body>',
        '<h1>TestRunner Summary Report</h1>',
        f'<p>Output folder: {outDir}</p>',
        f'<p>Algorithms: {", ".join(algNames)}</p>',
        f'<p>False Negatives: {len(fnEventIds)} &nbsp; '
        f'True Positives: {len(tpEventIds)} &nbsp; '
        f'False Positives (shown): {len(fp_sample)} of {len(fpEventIds)}</p>',
    ]

    for cat_dir, event_id_list, cat_title in categories:
        cat_full_dir = os.path.join(report_dir, cat_dir)
        os.makedirs(cat_full_dir, exist_ok=True)
        html_lines.append(f'<h2>{cat_title} ({len(event_id_list)} events)</h2>')

        for eid in event_id_list:
            eventObj = osd.getEvent(eid, includeDatapoints=False)
            if eventObj is None:
                print(f"WARNING: event {eid} not found for report - skipping")
                continue

            dp_idx = id_to_dp_idx.get(eid)
            if dp_idx is None:
                print(f"WARNING: no per-dp data for event {eid} - skipping graph")
                continue

            graph_fname = os.path.join(cat_full_dir, f'event_{eid}.png')
            generateEventGraph(eid, eventObj, perDpDataLst[dp_idx], algNames,
                                graph_fname, debug=debug)

            rel_path = os.path.relpath(graph_fname, report_dir)
            html_lines += [
                '<div class="event">',
                f'<b>Event {eid}</b> &nbsp; '
                f'{eventObj.get("type","")}/{eventObj.get("subType","")} '
                f'&nbsp; User: {eventObj.get("userId","")}<br>',
                f'Time: {eventObj.get("dataTime","")}<br>',
                f'Description: {eventObj.get("desc","")}<br>',
                f'<img src="{rel_path}" alt="Event {eid}">',
                '</div>',
            ]

        print(f"  {cat_title}: {len(event_id_list)} graphs written to {cat_full_dir}")

    html_lines.append('</body></html>')
    index_path = os.path.join(report_dir, 'index.html')
    with open(index_path, 'w') as hf:
        hf.write('\n'.join(html_lines))
    print(f"Summary report written to {index_path}")


# ---------------------------------------------------------------------------
# Re-generate report from a saved run folder (--analyze mode)
# ---------------------------------------------------------------------------

def analyzeExistingResults(outDir, configObj, debug=False):
    """Regenerate the summary report from a previously saved run folder.

    Requires ``perDpData.json`` and the four result CSVs to be present in
    ``outDir``.  Event objects are re-loaded from the data files listed in
    ``configObj``.
    """
    print(f"\nanalyzeExistingResults() – folder: {outDir}")

    pdp_path = os.path.join(outDir, 'perDpData.json')
    if not os.path.exists(pdp_path):
        print(f"ERROR: perDpData.json not found in {outDir}")
        return

    with open(pdp_path, 'r') as f:
        perDpDataLst = json.load(f)
    print(f"Loaded per-dp data for {len(perDpDataLst)} events")

    dbDir     = configObj.get('dbDir', None)
    dataFiles = configObj.get('dataFiles', [])
    if not dataFiles:
        print("ERROR: no dataFiles in config – cannot reload events")
        return
    osd = loadDataFiles(dataFiles, dbDir=dbDir, debug=debug)
    osd.removeEvents(configObj.get('invalidEvents', []))

    fnEventIds, tpEventIds, fpEventIds = [], [], []

    def _parse_results_csv(csv_path):
        """Yield (eventId, alarmed_bool); skip header and comment rows."""
        if not os.path.exists(csv_path):
            return
        with open(csv_path, 'r') as cf:
            reader  = csv.reader(cf)
            headers = None
            for row in reader:
                if not row:
                    continue
                if headers is None:
                    headers = [h.strip() for h in row]
                    continue
                if row[0].startswith('#'):
                    continue
                if len(row) > 6:
                    yield row[0].strip(), row[6].strip().upper() == 'ALARM'

    for ev_id, alarmed in _parse_results_csv(os.path.join(outDir, 'output_allSeizures.csv')):
        (tpEventIds if alarmed else fnEventIds).append(ev_id)

    for ev_id, alarmed in _parse_results_csv(os.path.join(outDir, 'output_falseAlarms.csv')):
        if alarmed:
            fpEventIds.append(ev_id)

    for ev_id, alarmed in _parse_results_csv(os.path.join(outDir, 'output_otherEvents.csv')):
        if alarmed:
            fpEventIds.append(ev_id)

    for ev_id, alarmed in _parse_results_csv(os.path.join(outDir, 'output_nda.csv')):
        if alarmed:
            fpEventIds.append(ev_id)

    algNames = list(perDpDataLst[0].get('algOutputs', {}).keys()) if perDpDataLst else []

    print(f"FN={len(fnEventIds)}, TP={len(tpEventIds)}, FP={len(fpEventIds)}")
    print(f"Algorithms: {algNames}")

    generateSummaryReport(
        outDir, fnEventIds, tpEventIds, fpEventIds,
        list(set(fnEventIds + tpEventIds + fpEventIds)),
        osd, perDpDataLst, algNames, debug=debug
    )

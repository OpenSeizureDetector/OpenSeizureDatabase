#!/usr/bin/env python3
"""testRunner.py – Slim orchestrator entry point.

All heavy lifting is delegated to the sub-modules inside this package:
  io_utils.py        – data loading (CSV / JSON / training-event exclusion)
  output_folders.py  – numbered sequential run-folder management
  alg_runner.py      – running events through algorithm instances
  results.py         – saving CSV result files and statistics
  report.py          – per-event PNG graphs and HTML summary report
"""

import argparse
import importlib
import json
import os
import sys

# ---- ensure this package directory and the repo root are on sys.path ----
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
sys.path.append(os.path.join(_HERE, '..', '..'))

import libosd.configUtils

from io_utils import _resolve_existing_path, loadDataFiles, exclude_training_events_from_osd
from output_folders import getOutputPath
from alg_runner import testEachEvent
from results import saveResults2
from report import analyzeExistingResults


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------

def runTest(configObj, debug=False, configPath=None, testDataPath=None,
            seizuresOnly=False, outDir="./output", rerun=0, cmdArgs=None):
    print("runTest - configObj=" + json.dumps(configObj))

    # ---- Create / resolve numbered output folder ----
    runFolder = getOutputPath(outDir, rerun=rerun, prefix="testRun")
    print(f"Output folder: {runFolder}")

    # Save configuration and CLI args so the run is fully reproducible
    saved_config_path = os.path.join(runFolder, 'testConfig.json')
    with open(saved_config_path, 'w') as _cf:
        json.dump(configObj, _cf, indent=2)
    if cmdArgs:
        saved_args_path = os.path.join(runFolder, 'cmdArgs.json')
        with open(saved_args_path, 'w') as _af:
            json.dump(cmdArgs, _af, indent=2)
    print(f"Config saved to {saved_config_path}")

    dbDir = configObj.get('dbDir', None)

    configDir = None
    if configPath:
        try:
            configDir = os.path.dirname(os.path.abspath(configPath))
        except Exception:
            configDir = None

    invalidEvents = configObj['invalidEvents']
    print("invalid events", invalidEvents)

    # Optional command-line override for testData file
    dataFiles = configObj['dataFiles']
    if testDataPath:
        search_dirs = [d for d in [configDir, dbDir, os.getcwd()] if d]
        resolved_test_data = _resolve_existing_path(testDataPath, search_dirs=search_dirs)
        if resolved_test_data is None:
            print(
                f"ERROR: --testData file not found: '{testDataPath}'. "
                f"Searched CWD and: {search_dirs}",
                file=sys.stderr,
            )
            return
        dataFiles = [resolved_test_data]
        print(f"Using command-line test data override: {resolved_test_data}")

    # Load each of the data files (OSDB JSON or flattened CSV)
    osd = loadDataFiles(dataFiles, dbDir=dbDir, debug=debug)
    osd.removeEvents(invalidEvents)
    filterCfg = dict(configObj['eventFilters'])
    if seizuresOnly:
        print("Applying --seizuresOnly filter (includeTypes=['seizure'])")
        filterCfg['includeTypes'] = ['seizure']
        filterCfg['excludeTypes'] = []
    print("filterCfg=", filterCfg)

    # Optional: exclude events used during training to avoid train/test contamination
    train_data_path = filterCfg.get('excludeTrainingEvents', None)
    if train_data_path:
        search_dirs = [d for d in [configDir, dbDir, os.getcwd()] if d]
        removed = exclude_training_events_from_osd(
            osd,
            train_data_path,
            search_dirs=search_dirs,
            debug=debug,
        )
        print(f"Excluded {len(removed)} training events")

    osd.listEvents()

    eventIdsLst = osd.getFilteredEventsLst(
        includeUserIds=filterCfg['includeUserIds'],
        excludeUserIds=filterCfg['excludeUserIds'],
        includeTypes=filterCfg['includeTypes'],
        excludeTypes=filterCfg['excludeTypes'],
        includeSubTypes=filterCfg['includeSubTypes'],
        excludeSubTypes=filterCfg['excludeSubTypes'],
        includeDataSources=filterCfg['includeDataSources'],
        excludeDataSources=filterCfg['excludeDataSources'],
        includeText=filterCfg['includeText'],
        excludeText=filterCfg['excludeText'],
        require3dData=filterCfg['require3dData'],
        requireHrData=filterCfg['requireHrData'],
        requireO2SatData=filterCfg['requireO2SatData'],
        debug=True,
    )

    print("%d events remaining after applying filters" % len(eventIdsLst))
    print(eventIdsLst)

    # Dynamically import and instantiate each enabled algorithm class
    algs = []
    algNames = []
    for algObj in configObj['algorithms']:
        print(algObj['name'], algObj['enabled'])
        if algObj['enabled']:
            moduleId = algObj['alg'].split('.')[0]
            classId  = algObj['alg'].split('.')[1]
            print("Importing Module %s" % moduleId)
            module = importlib.import_module(moduleId)
            algObj['settings']['name'] = algObj['name']
            settingsStr = json.dumps(algObj['settings'])
            print("settingsStr=%s (%s)" % (settingsStr, type(settingsStr)))
            algs.append(eval("module.%s(settingsStr, debug)" % classId))
            algNames.append(algObj['name'])
        else:
            print("Algorithm %s is disabled in configuration file - ignoring"
                  % algObj['name'])

    # Run each event through each algorithm then save results / report
    tcResults, tcResultsStrArr, expandedAlgNames, perDpDataLst = testEachEvent(
        eventIdsLst, osd, algs, algNames, debug=debug)
    saveResults2(runFolder, tcResults, tcResultsStrArr, eventIdsLst, osd,
                 expandedAlgNames, perDpDataLst=perDpDataLst, debug=debug)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    print("testRunner.main()")
    parser = argparse.ArgumentParser(description='Seizure Detection Test Runner')
    parser.add_argument('--config', default="testConfig.json",
                        help='name of json file containing test configuration')
    parser.add_argument('--testData', default=None,
                        help='Override dataFiles from config with a single test data file '
                             '(.json or flattened .csv)')
    parser.add_argument('--seizuresOnly', action="store_true",
                        help='Test only seizure events (faster)')
    parser.add_argument('--outDir', default="./output",
                        help='Root directory for numbered run output folders (default: ./output)')
    parser.add_argument('--rerun', type=int, default=0,
                        help='Re-use (not create) existing numbered run folder N. '
                             '0 = create a new folder (default).')
    parser.add_argument('--analyze', action="store_true",
                        help='Skip running tests; regenerate the summary report graphs from '
                             'a previously saved run folder (requires --rerun N).')
    parser.add_argument('--debug', action="store_true",
                        help='Write debugging information to screen')
    args = vars(parser.parse_args())
    print(args)

    configObj = libosd.configUtils.loadConfig(args['config'])
    print("configObj=", configObj)

    # Merge optional separate OSDB configuration file
    if "osdbCfg" in configObj:
        osdbCfgFname = libosd.configUtils.getConfigParam("osdbCfg", configObj)
        print("Loading separate OSDB Configuration File %s." % osdbCfgFname)
        osdbCfgObj = libosd.configUtils.loadConfig(osdbCfgFname)
        configObj = configObj | osdbCfgObj

    print("configObj=", configObj)

    if args.get('analyze'):
        rerun_num = args.get('rerun', 0)
        if rerun_num == 0:
            print("ERROR: --analyze requires --rerun N to specify which run folder to analyze.")
            sys.exit(1)
        run_folder = os.path.join(args['outDir'], 'testRun', str(rerun_num))
        if not os.path.exists(run_folder):
            print(f"ERROR: Run folder not found: {run_folder}")
            sys.exit(1)
        saved_config = os.path.join(run_folder, 'testConfig.json')
        if os.path.exists(saved_config):
            with open(saved_config, 'r') as _scf:
                configObj = json.load(_scf)
            print(f"Loaded config from {saved_config}")
        analyzeExistingResults(run_folder, configObj, debug=args.get('debug', False))
        return

    runTest(
        configObj,
        args['debug'],
        configPath=args.get('config'),
        testDataPath=args.get('testData'),
        seizuresOnly=args.get('seizuresOnly', False),
        outDir=args.get('outDir', './output'),
        rerun=args.get('rerun', 0),
        cmdArgs=args,
    )


if __name__ == "__main__":
    main()

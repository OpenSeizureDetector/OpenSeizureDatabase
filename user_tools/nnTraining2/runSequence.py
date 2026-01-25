#!/usr/bin/env python3
#
#
# Run the neural network training toolchain sequence.   It does the following:
#   - Select data, filtering the OSDB data to select the desired data
#   - Flatten the selected OSDB .json data file into a single .csv file (allData.csv)
#   - Split the allData.csv into test and train datasets (as .csv files)
#   - Apply data augmentation to the seizure data in the training .csv files
#   - Train a new or pre-existing model based on the augmented files.
#   - Tests the resulting model calculating statistics using the test dataset.
#
# Note that only the required parts of the sequence are run, depending on which data
#  files are in the working directory - if the file does not exist, it is re-generated
#  based on the files produced earlier in the sequence.
# This means that if you want to re-generate the files completely, you need to start in
#    a clean directory, or remove the output data files.
#
# The sequence ensures that only the initial data selection operates on JSON files.
# All subsequent operations (flattening, splitting, augmentation, feature extraction,
# training, and testing) work exclusively with CSV files.


import argparse
import sys
import os
import shutil
import json
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libosd.configUtils


class TeeLogger:
    """Duplicate output to both console and a log file."""
    def __init__(self, log_file, original_stream):
        self.log_file = log_file
        self.original_stream = original_stream
    
    def write(self, message):
        self.original_stream.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure immediate write
    
    def flush(self):
        self.original_stream.flush()
        self.log_file.flush()


def setup_logging(output_folder):
    """
    Set up logging to capture all console output to a file.
    
    Args:
        output_folder (str): Path to output folder
        
    Returns:
        TeeLogger: TeeLogger object that's now assigned to sys.stdout
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_folder, f"runSequence_{timestamp}.log")
    
    log_file = open(log_path, 'w', buffering=1)  # Line buffered
    
    # Write header to log
    log_file.write("="*80 + "\n")
    log_file.write(f"runSequence.py execution log\n")
    log_file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write("="*80 + "\n\n")
    log_file.flush()
    
    print(f"Logging console output to: {log_path}")
    
    # Save original streams
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Replace stdout and stderr with TeeLogger
    tee_stdout = TeeLogger(log_file, original_stdout)
    tee_stderr = TeeLogger(log_file, original_stderr)
    
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr
    
    # Store file handle and original streams in tee_stdout for later restoration
    tee_stdout.file_handle = log_file
    tee_stdout.original_stdout = original_stdout
    tee_stdout.original_stderr = original_stderr
    
    return tee_stdout


def cleanup_logging(tee_logger):
    """
    Restore original stdout/stderr and close log file.
    
    Args:
        tee_logger: The TeeLogger object that was assigned to sys.stdout
    """
    # Get the actual file handle from the TeeLogger
    actual_file = tee_logger.file_handle if hasattr(tee_logger, 'file_handle') else tee_logger.log_file
    
    # Write footer to log
    actual_file.write("\n" + "="*80 + "\n")
    actual_file.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    actual_file.write("="*80 + "\n")
    
    # Restore original streams
    sys.stdout = tee_logger.original_stdout
    sys.stderr = tee_logger.original_stderr
    
    # Close log file
    actual_file.close()

try:
    from user_tools.nnTraining2 import selectData, splitData, flattenData, augmentData
except ImportError:
    import selectData
    import splitData
    import flattenData
    import augmentData
from user_tools.nnTraining2.extractFeatures import extractFeatures
from user_tools.nnTraining2.addFeatureHistory import add_feature_history

# Conditional imports: allow running as a script from this folder OR importing as a module for testing.
# This ensures compatibility with both 'python runSequence.py' and 'pytest ...' from repo root.
def deleteFileIfExists(fname, debug=True):
    ''' If the specified file named fname exists, delete it, otherwise do nothing.'''
    if (os.path.exists(fname)):
            if (debug): print("Deleting %s" % fname)
            os.remove(fname)


def _countEventsInJson(filePath):
    """
    Count seizure and non-seizure events in a JSON file.
    
    Args:
        filePath (str): Path to the JSON file
        
    Returns:
        tuple: (seizure_count, non_seizure_count)
    """
    import json
    
    with open(filePath, 'r') as f:
        events = json.load(f)
    
    seizure_count = 0
    non_seizure_count = 0
    
    for event in events:
        event_type = event.get('type', '').lower()
        if event_type == 'seizure':
            seizure_count += 1
        else:
            non_seizure_count += 1
    
    return (seizure_count, non_seizure_count)


def _countEventsInCsv(filePath):
    """
    Count seizure and non-seizure events in a CSV file.
    
    Args:
        filePath (str): Path to the CSV file
        
    Returns:
        tuple: (seizure_count, non_seizure_count)
    """
    import pandas as pd
    
    df = pd.read_csv(filePath, low_memory=False)
    
    # Count unique events by eventId, using the 'type' column
    # type: 0 = false alarm/nda, 1 = seizure, 2 = other
    if 'eventId' in df.columns:
        # Group by eventId and get the type for each event
        event_types = df.groupby('eventId')['type'].first()
        seizure_count = (event_types == 1).sum()
        non_seizure_count = (event_types != 1).sum()
    else:
        # If no eventId, count rows
        seizure_count = (df['type'] == 1).sum()
        non_seizure_count = (df['type'] != 1).sum()
    
    return (seizure_count, non_seizure_count)


def calculateFileStats(filePath):
    """
    Calculate statistics for a data file containing seizure/non-seizure events.
    
    Args:
        filePath (str): Path to the file (must be .json or .csv)
        
    Returns:
        tuple: (seizure_count, non_seizure_count)
        
    Raises:
        ValueError: If file is not .json or .csv, or if file doesn't exist
    """
    if not os.path.exists(filePath):
        raise ValueError(f"File does not exist: {filePath}")
    
    file_ext = os.path.splitext(filePath)[1].lower()
    
    if file_ext == '.json':
        return _countEventsInJson(filePath)
    elif file_ext == '.csv':
        return _countEventsInCsv(filePath)
    else:
        raise ValueError(f"File must be .json or .csv, got: {file_ext}")


def getLatestOutputFolder(outPath = "./output", prefix="training"):
    import pathlib, os

    outFolderPath = os.path.join(outPath, prefix)
    allOutputFolders = list(pathlib.Path(outFolderPath).glob('*/'))

    print("Existing Output Folders:")
    for folder in allOutputFolders:
        print(folder)
    #print(folder for folder in allOutputFolders)
    print("Length of allOutputFolders = %d" % len(allOutputFolders))

    #print()
    if (len(allOutputFolders) > 0):
        # Sort by folder number (numerically), not by creation time
        latestOutputFolderPath = max(allOutputFolders, key=lambda p: int(os.path.basename(p)))
    else:
        latestOutputFolderPath = None
    
    return(latestOutputFolderPath)



def getOutputPath(outPath = "./output", rerun=0, prefix="training"):
    '''
    Returns a path to the next sequentially numbered folder in outPath
    '''
    import pathlib, os
    
    if (rerun==0):
        # Creating new folder - use prefix
        outFolderPath = os.path.join(outPath, prefix)
        latestOutputFolderPath = getLatestOutputFolder(outPath=outPath, prefix=prefix)
        if latestOutputFolderPath is not None:
            print("latestOutputFolderPath = %s" % latestOutputFolderPath)
            latestOutputFolder = os.path.split(latestOutputFolderPath)[-1]
        else:
            #print("starting new output folder list")
            latestOutputFolder = "0"

        print("latestOutputFolder = %s" % latestOutputFolder)

        newOutputFolder = 1 + int(latestOutputFolder)
        newOutputPath = os.path.join(outFolderPath,str(newOutputFolder))
        os.makedirs(newOutputPath, exist_ok = False)
    else:
        # Rerunning existing folder - search for it using the prefix
        newOutputPath = os.path.join(outPath, prefix, str(rerun))
        if not os.path.exists(newOutputPath):
            raise FileNotFoundError(f"Rerun folder not found: {newOutputPath}. Expected folder structure: {outPath}/{prefix}/{rerun}")

    print("getOutputPath() - outputPath=%s" %newOutputPath)
    return newOutputPath


def analyze_test_results(test_output_folder, configObj, debug=False):
    """
    Run analyzeEventResults on the test results CSV file.
    
    Args:
        test_output_folder (str): Path to folder containing test results
        configObj: Configuration object
        debug (bool): Debug flag
        
    Returns:
        tc_metrics: Dictionary with Tonic-Clonic seizure TPR statistics
    """
    try:
        modelFname = configObj['modelConfig'].get('modelFname', 'model')
        event_results_csv = os.path.join(test_output_folder, f"{modelFname}_event_results.csv")
        
        if not os.path.exists(event_results_csv):
            if debug:
                print(f"analyzeTestResults: Event results CSV not found: {event_results_csv}")
            return
        
        print(f"\nanalyzeTestResults: Analyzing test results in {test_output_folder}")
        
        # Import analyzeEventResults module
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        try:
            import analyzeEventResults
        except ImportError:
            print("ERROR: Could not import analyzeEventResults module")
            return
        
        # Load and analyze the event results
        df = analyzeEventResults.load_event_results(event_results_csv)
        
        # Run analyses
        seizure_df, user_metrics_df, far_metrics_df = analyzeEventResults.analyze_by_user(df)
        seizure_df_st, subtype_metrics_df = analyzeEventResults.analyze_by_seizure_type(df)
        tc_metrics = analyzeEventResults.analyze_tonic_clonic_seizures(df)
        false_alarms_df, far_by_subtype_df = analyzeEventResults.analyze_false_alarms(df)
        
        # Extract false negatives for reporting
        false_negatives_df = df[(df['ActualLabel'] == 1) & (df['ModelPrediction'] == 0)].copy()
        
        # Try to load allData.json for enhanced false negatives details
        alldata_list = None
        alldata_json_path = os.path.join(test_output_folder, '..', 'allData.json')
        if not os.path.exists(alldata_json_path):
            # Try one level up
            alldata_json_path = os.path.join(test_output_folder, '..', '..', 'allData.json')
        fn_enhanced_details_df = analyzeEventResults.extract_false_negatives_details(df, alldata_list)
        
        # Save text report
        report_path = os.path.join(test_output_folder, "event_analysis_report.txt")
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("EVENT ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("USER ANALYSIS\n" + "-"*80 + "\n")
            f.write("TPR by User:\n")
            f.write(user_metrics_df.to_string(index=False) + "\n\n")
            f.write("FAR (False Alarm Rate) by User:\n")
            f.write(far_metrics_df.to_string(index=False) + "\n\n")
            
            f.write("\nSEIZURE TYPE ANALYSIS\n" + "-"*80 + "\n")
            f.write("TPR by Seizure SubType:\n")
            f.write(subtype_metrics_df.to_string(index=False) + "\n\n")
            
            f.write("\nTONIC-CLONIC SEIZURES ANALYSIS\n" + "-"*80 + "\n")
            f.write("TPR Statistics for Tonic-Clonic Seizures Only:\n")
            f.write(f"  Count: {tc_metrics['Count']}\n")
            f.write(f"  True Positives: {tc_metrics['TP']}\n")
            f.write(f"  False Negatives: {tc_metrics['FN']}\n")
            f.write(f"  Total Seizures: {tc_metrics['TP_FN_Total']}\n")
            f.write(f"  TPR (Sensitivity): {tc_metrics['TPR']:.3f}\n")
            f.write(f"  FPR (False Positive Rate): {tc_metrics['FPR']:.3f}\n\n")
            
            f.write("\nFALSE ALARM ANALYSIS\n" + "-"*80 + "\n")
            if len(far_by_subtype_df) > 0:
                f.write("FAR by SubType:\n")
                f.write(far_by_subtype_df.to_string(index=False) + "\n")
        
        print(f"analyzeTestResults: Text report saved to {report_path}")
        
        # Generate PDF plots (uses analyzeEventResults' comprehensive plotting, including FAR by subtype)
        try:
            pdf_path = analyzeEventResults.generate_plots(
                df,
                seizure_df,
                user_metrics_df,
                far_metrics_df,
                subtype_metrics_df,
                false_alarms_df,
                false_negatives_df,
                fn_enhanced_details_df,
                test_output_folder,
            )
            print(f"analyzeTestResults: PDF report saved to {pdf_path}")
        except Exception as e:
            print(f"analyzeTestResults: Warning - Could not generate PDF report: {e}")
        
        # Return Tonic-Clonic metrics
        return tc_metrics
    
    except Exception as e:
        print(f"analyzeTestResults: Warning - Analysis failed: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        # Return empty dict on error
        return {}


def run_sequence(args):
    """
    Run the Neural Network Training toolchain sequence.

    Args:
        args (dict): Dictionary of arguments. Keys should match the command line options:
            'config' (str): Path to configuration JSON file.
            'kfold' (int or str): Number of folds for cross-validation.
            'rerun' (int or str): Re-run the specified run number. If 0, a new run is created.
            'outDir' (str): Output directory for results.
            'train' (bool): If True, train the model.
            'test' (bool): If True, test the model.
            'clean' (bool): If True, clean up output files before running.
            'debug' (bool): If True, print debugging information.
        Example:
            args = {
                'config': 'nnConfig.json',
                'kfold': 5,
                'rerun': 0,
                'outDir': './output',
                'train': True,
                'test': False,
                'clean': False,
                'debug': True
            }
    """
    print("runSequence.run_sequence()")
    debug = args['debug']
    if (debug): print(args)


def test_outer_folds(configObj, kfold, nestedKfold, outFolder, foldResults, args=None, debug=False):
    """
    Test models on outer fold independent test sets.
    
    Args:
        configObj: Configuration object
        kfold: Number of inner folds
        nestedKfold: Number of outer folds
        outFolder: Output folder path
        foldResults: Results from inner fold training (used to select best model)
        args: Command line arguments dictionary (contains testPtl flag)
        debug: Debug flag
        
    Returns:
        outerFoldResults: List of test results for each outer fold
    """
    from user_tools.nnTraining2.extractFeatures import extractFeatures
    try:
        from user_tools.nnTraining2 import flattenData
    except ImportError:
        import flattenData
    
    outerFoldResults = []
    testCsvFname = configObj['dataFileNames'].get('testDataFileCsv', 'testData.csv')
    
    for nOuterFold in range(0, nestedKfold):
        print("\n" + "="*80)
        print("Testing OUTER FOLD %d on independent test set" % nOuterFold)
        print("="*80)
        
        outerFoldOutFolder = os.path.join(outFolder, "outerfold%d" % nOuterFold)
        outerfold_test_csv = os.path.join(outerFoldOutFolder, testCsvFname.replace('.csv', '_outer.csv'))
        
        if not os.path.exists(outerfold_test_csv):
            print("WARNING: Outer fold test CSV not found: %s" % outerfold_test_csv)
            continue
        
        # Find best inner fold model for this outer fold based on TPR
        innerFoldPaths = []
        innerFoldResults_forOuter = []
        
        for nFold in range(0, kfold):
            if kfold > 1:
                foldPath = os.path.join(outerFoldOutFolder, "fold%d" % nFold)
            else:
                foldPath = outerFoldOutFolder
            
            # Get the result for this inner fold
            fold_index = nOuterFold * kfold + nFold
            if fold_index < len(foldResults):
                innerFoldResults_forOuter.append(foldResults[fold_index])
                innerFoldPaths.append(foldPath)
        
        if len(innerFoldResults_forOuter) == 0:
            print("WARNING: No inner fold results found for outer fold %d" % nOuterFold)
            continue
        
        # Select best inner fold based on TPR (sensitivity)
        best_fold_idx = max(range(len(innerFoldResults_forOuter)), 
                          key=lambda i: innerFoldResults_forOuter[i]['tpr'])
        best_fold_path = innerFoldPaths[best_fold_idx]
        best_fold_tpr = innerFoldResults_forOuter[best_fold_idx]['tpr']
        best_fold_fpr = innerFoldResults_forOuter[best_fold_idx]['fpr']
        
        print("runSequence: Best inner fold is %d with TPR=%.3f, FPR=%.3f" % 
              (best_fold_idx, best_fold_tpr, best_fold_fpr))
        print("runSequence: Using model from: %s" % best_fold_path)
        
        # Create a separate test output folder to avoid overwriting fold validation results
        test_output_folder = os.path.join(outerFoldOutFolder, f"fold{best_fold_idx}_independent_test")
        os.makedirs(test_output_folder, exist_ok=True)
        print(f"runSequence: Independent test results will be saved to: {test_output_folder}")
        
        # Copy model files to the test output folder for reference
        modelFname = configObj['modelConfig'].get('modelFname', 'model')
        framework = configObj['modelConfig'].get('framework')
        if framework is None:
            framework = configObj['modelConfig'].get('modelType', 'tensorflow')
        
        if framework == 'pytorch':
            model_ext = '.pt'
        else:
            model_ext = '.keras'
        
        model_src = os.path.join(best_fold_path, f"{modelFname}{model_ext}")
        model_dst = os.path.join(test_output_folder, f"{modelFname}{model_ext}")
        if os.path.exists(model_src):
            import shutil
            shutil.copy2(model_src, model_dst)
            print(f"runSequence: Copied model file to test folder: {model_dst}")
        
        # Also copy .ptl file if it exists (PyTorch Mobile format)
        if framework == 'pytorch':
            ptl_src = os.path.join(best_fold_path, f"{modelFname}.ptl")
            ptl_dst = os.path.join(test_output_folder, f"{modelFname}.ptl")
            if os.path.exists(ptl_src):
                shutil.copy2(ptl_src, ptl_dst)
                print(f"runSequence: Copied .ptl model file to test folder: {ptl_dst}")
            
            # Also copy .pte file if it exists (ExecuTorch format)
            pte_src = os.path.join(best_fold_path, f"{modelFname}.pte")
            pte_dst = os.path.join(test_output_folder, f"{modelFname}.pte")
            if os.path.exists(pte_src):
                shutil.copy2(pte_src, pte_dst)
                print(f"runSequence: Copied .pte model file to test folder: {pte_dst}")
        
        # Extract features for outer fold test data
        outerfold_test_features = os.path.join(outerFoldOutFolder, "outerfold_test_features.csv")
        if not os.path.exists(outerfold_test_features):
            print("runSequence: Extracting features for outer fold test data")
            extractFeatures(outerfold_test_csv, outerfold_test_features, configObj)
        
        # Test best model on outer fold independent test set
        print("runSequence: Testing best model (inner fold %d) on outer fold %d independent test set" % 
              (best_fold_idx, nOuterFold))
        
        # Get framework
        framework = configObj['modelConfig'].get('framework')
        if framework is None:
            framework = configObj['modelConfig'].get('modelType', 'tensorflow')
        
        if framework == "sklearn":
            import skTester
            outerTestResults = skTester.testModel(configObj, dataDir=outerFoldOutFolder, debug=debug)
        elif framework in ["tensorflow", "pytorch"]:
            import nnTester
            # Use absolute path to test features file so nnTester doesn't try to join it with dataDir
            test_features_path = os.path.abspath(os.path.join(outerFoldOutFolder, "outerfold_test_features.csv"))
            # Test .ptl model only if --testPtl flag is provided
            # Use separate output folder and clear title prefix to distinguish from fold validation results
            title_prefix = f"Outer Fold {nOuterFold} - Independent Test (Best Inner Fold {best_fold_idx})"
            outerTestResults = nnTester.testModel(
                configObj, 
                dataDir=test_output_folder,  # Model files are in test_output_folder
                balanced=False, 
                debug=debug, 
                testDataCsv=test_features_path, 
                test_ptl=args.get('testPtl', False) if args else False,
                test_pte=args.get('testPte', False) if args else False,
                test_percent=args.get('testPercent', 100.0) if args else 100.0,
                outputDir=test_output_folder,  # Save results to test_output_folder
                titlePrefix=title_prefix  # Clear title for plots
            )
            
            # Analyze test results and get Tonic-Clonic metrics
            tc_metrics = analyze_test_results(test_output_folder, configObj, debug=debug)
        
        # Store outer fold results
        outerTestResults['outer_fold'] = nOuterFold
        outerTestResults['best_inner_fold'] = best_fold_idx
        outerTestResults['best_model_path'] = best_fold_path
        outerTestResults['test_output_path'] = test_output_folder
        
        # Store Tonic-Clonic metrics if available
        if tc_metrics:
            outerTestResults['tc_count'] = tc_metrics.get('Count', 0)
            outerTestResults['tc_tp'] = tc_metrics.get('TP', 0)
            outerTestResults['tc_fn'] = tc_metrics.get('FN', 0)
            outerTestResults['tc_tpr'] = tc_metrics.get('TPR', 0.0)
            outerTestResults['tc_fpr'] = tc_metrics.get('FPR', 0.0)
        
        outerFoldResults.append(outerTestResults)
        
        print("\nrunSequence: Outer fold %d INDEPENDENT test results:" % nOuterFold)
        print("  TPR (Sensitivity): %.3f" % outerTestResults['tpr'])
        print("  FPR (False Positive Rate): %.3f" % outerTestResults['fpr'])
        print("  TP: %d, FP: %d, TN: %d, FN: %d" % 
              (outerTestResults['tp'], outerTestResults['fp'], 
               outerTestResults['tn'], outerTestResults['fn']))
        print("  Event TPR: %.3f, Event FPR: %.3f" % 
              (outerTestResults['event_tpr'], outerTestResults['event_fpr']))
        
        # Generate FP/FN analysis files (use test output folder for event results)
        print("\nrunSequence: Generating False Positive/Negative analysis files")
        _generate_fp_fn_analysis(test_output_folder, outerFoldOutFolder, nOuterFold, best_fold_idx, configObj, debug)
        
        # Extract and save training history
        print("runSequence: Extracting training history from best model")
        _extract_training_history(best_fold_path, outerFoldOutFolder, nOuterFold, framework, configObj)
    
    return outerFoldResults


def _generate_fp_fn_analysis(model_path, test_path, outer_fold_id, best_fold_idx, configObj, debug=False):
    """
    Generate CSV files for false positives and false negatives with metadata.
    Files are saved to the outerfold directory with outer fold ID in the filename.
    Metadata is loaded from allData.json (same source as nnTester uses).
    """
    modelFname = configObj['modelConfig'].get('modelFname', 'model')
    event_results_csv = os.path.join(model_path, f"{modelFname}_event_results.csv")
    
    if debug:
        print(f"  DEBUG: Looking for event results at: {event_results_csv}")
        print(f"  DEBUG: File exists: {os.path.exists(event_results_csv)}")
        if os.path.exists(model_path):
            print(f"  DEBUG: Files in {model_path}:")
            for f in os.listdir(model_path)[:20]:
                if 'event' in f.lower() or 'result' in f.lower():
                    print(f"    - {f}")
    
    if not os.path.exists(event_results_csv):
        print(f"  WARNING: Event results file not found: {event_results_csv}")
        return
    
    try:
        df_events = pd.read_csv(event_results_csv)
        
        if debug:
            print(f"  DEBUG: Loaded {len(df_events)} events from {event_results_csv}")
            print(f"  DEBUG: Columns: {list(df_events.columns)}")
        
        # The event_results_csv should already have metadata enriched by nnTester
        # (it loads from allData.json and enriches with UserID, Type, SubType, Description)
        # But verify the columns exist
        if 'UserID' not in df_events.columns:
            print(f"  WARNING: Metadata columns not found in event results - attempting to load from allData.json")
            
            # Load metadata from allData.json (same approach as nnTester)
            # allData.json is at the training output root, not in the fold subdirectory
            # Search up the directory tree to find it
            allDataFilename = configObj['dataFileNames'].get('allDataFileJson', 'allData.json')
            allDataPath = None
            
            # First try in the current test_path
            candidate_path = os.path.join(test_path, allDataFilename)
            if os.path.exists(candidate_path):
                allDataPath = candidate_path
            else:
                # Search up the directory tree (for fold subdirectories in outer fold testing)
                current_dir = test_path
                for _ in range(5):  # Search up to 5 levels up
                    parent_dir = os.path.dirname(current_dir)
                    if parent_dir == current_dir:  # Reached root directory
                        break
                    candidate_path = os.path.join(parent_dir, allDataFilename)
                    if os.path.exists(candidate_path):
                        allDataPath = candidate_path
                        break
                    current_dir = parent_dir
            
            event_metadata = {}
            
            if allDataPath and os.path.exists(allDataPath):
                try:
                    with open(allDataPath, 'r') as f:
                        allData = json.load(f)
                    
                    # Build metadata map from allData
                    events_list = allData if isinstance(allData, list) else allData.get('events', [])
                    for event in events_list:
                        # Note: allData.json uses 'id' not 'eventId' as the key
                        event_id = event.get('id')
                        if event_id is not None:
                            # Convert to string for consistent type matching
                            event_id_str = str(event_id)
                            event_metadata[event_id_str] = {
                                'userId': event.get('userId', 'N/A'),
                                'typeStr': event.get('type', 'N/A'),
                                'subType': event.get('subType', 'N/A'),
                                'desc': event.get('desc', 'N/A')
                            }
                    
                    if debug:
                        print(f"  DEBUG: Loaded metadata for {len(event_metadata)} events from {allDataPath}")
                except Exception as e:
                    print(f"  WARNING: Could not load metadata from {allDataPath}: {e}")
            else:
                print(f"  WARNING: allData file not found (searched in {test_path} and parent directories)")
            
            # Enrich event results with metadata - convert EventID to string for consistent lookup
            df_events['UserID'] = df_events['EventID'].map(lambda eid: event_metadata.get(str(eid), {}).get('userId', 'N/A'))
            df_events['Type'] = df_events['EventID'].map(lambda eid: event_metadata.get(str(eid), {}).get('typeStr', 'N/A'))
            df_events['SubType'] = df_events['EventID'].map(lambda eid: event_metadata.get(str(eid), {}).get('subType', 'N/A'))
            df_events['Description'] = df_events['EventID'].map(lambda eid: event_metadata.get(str(eid), {}).get('desc', 'N/A'))
        
        # Add flags for easy filtering
        df_events['isFalsePositive'] = (df_events['ActualLabel'] == 0) & (df_events['ModelPrediction'] == 1)
        df_events['isFalseNegative'] = (df_events['ActualLabel'] == 1) & (df_events['ModelPrediction'] == 0)
        
        # Save unified results file with seizure probability and classification flags
        # Select columns in desired order - include probability and flags for analysis
        columns_order = ['EventID', 'UserID', 'Type', 'SubType', 
                        'ActualLabel', 'ModelPrediction', 'MaxSeizureProbability', 
                        'isFalsePositive', 'isFalseNegative', 'OSDPrediction', 'Description']
        
        # Only include columns that exist
        available_columns = [col for col in columns_order if col in df_events.columns]
        results_df = df_events[available_columns].copy()
        
        # Save combined results file to the outerfold directory (test_path), not a subfolder
        # Include outer_fold_id and best_fold_idx in the filename for reference
        results_path = os.path.join(test_path, f"outerfold{outer_fold_id}_fold{best_fold_idx}_test_results.csv")
        results_df.to_csv(results_path, index=False)
        
        # Print summary statistics
        num_fp = (df_events['isFalsePositive']).sum()
        num_fn = (df_events['isFalseNegative']).sum()
        num_total = len(df_events)
        
        print(f"  Saved test results for {num_total} events to outerfold{outer_fold_id}_fold{best_fold_idx}_test_results.csv")
        print(f"    - False Positives: {num_fp}")
        print(f"    - False Negatives: {num_fn}")
        print(f"    - Correct Classifications: {num_total - num_fp - num_fn}")
            
    except Exception as e:
        print(f"  ERROR generating test results analysis: {e}")
        import traceback
        traceback.print_exc()



def _extract_training_history(model_path, test_path, outer_fold_id, framework, configObj):
    """
    Extract and save training history from the trained model.
    """
    try:
        if framework == "pytorch":
            # For PyTorch, look for training history JSON
            history_file = os.path.join(model_path, "training_history.json")
            
            print(f"  Looking for PyTorch training history at: {history_file}")
            print(f"  File exists: {os.path.exists(history_file)}")
            
            if os.path.exists(history_file):
                # Check file size first
                file_size = os.path.getsize(history_file)
                if file_size == 0:
                    print(f"  WARNING: training_history.json is empty (0 bytes)")
                else:
                    try:
                        with open(history_file, 'r') as f:
                            history = json.load(f)
                        
                        # Save history to output folder
                        history_path = os.path.join(test_path, f"outerfold_{outer_fold_id}_training_history.json")
                        with open(history_path, 'w') as f:
                            json.dump(history, f, indent=2)
                        print(f"  Saved training history to {os.path.basename(history_path)}")
                    except json.JSONDecodeError as e:
                        print(f"  WARNING: training_history.json is corrupted: {e}")
            else:
                print(f"  PyTorch training history file not found")
        
        elif framework == "tensorflow":
            # For TensorFlow/Keras, look for training history JSON or pickle
            history_file = os.path.join(model_path, "training_history.json")
            
            print(f"  Looking for TensorFlow training history at: {history_file}")
            
            if os.path.exists(history_file):
                try:
                    with open(history_file, 'r') as f:
                        history = json.load(f)
                    
                    # Save to output folder
                    history_path = os.path.join(test_path, f"outerfold_{outer_fold_id}_training_history.json")
                    with open(history_path, 'w') as f:
                        json.dump(history, f, indent=2)
                    print(f"  Saved training history to {os.path.basename(history_path)}")
                except json.JSONDecodeError:
                    print(f"  TensorFlow training history file corrupted")
            else:
                # Try pickle format
                import pickle
                pickle_file = os.path.join(model_path, "training_history.pkl")
                if os.path.exists(pickle_file):
                    with open(pickle_file, 'rb') as f:
                        history = pickle.load(f)
                    history_path = os.path.join(test_path, f"outerfold_{outer_fold_id}_training_history.pkl")
                    with open(history_path, 'wb') as f:
                        pickle.dump(history, f)
                    print(f"  Saved training history to {os.path.basename(history_path)}")
                else:
                    print(f"  TensorFlow training history files not found")
    
    except Exception as e:
        print(f"  WARNING: Could not extract training history: {e}")


def run_sequence(args):

    kfold = int(args['kfold'])
    nestedKfold = int(args['nestedKfold'])
    debug = args.get('debug', False)

    # If rerun is specified and config is default, try to load from output folder
    config_path = args['config']
    if int(args['rerun']) > 0 and args['config'] == 'nnConfig.json':
        # Construct the output path to check for saved config
        rerun_folder = os.path.join(args['outDir'], str(args['rerun']))
        saved_config = os.path.join(rerun_folder, 'nnConfig.json')
        if os.path.exists(saved_config):
            config_path = saved_config
            print("runSequence: Using saved configuration from %s" % saved_config)
        else:
            print("runSequence: Warning - rerun specified but no saved config found at %s, using %s" % (saved_config, args['config']))
    
    configObj = libosd.configUtils.loadConfig(config_path)
    if (debug): print("configObj=",configObj.keys())
    # Load a separate OSDB Configuration file if it is included.
    if ("osdbCfg" in configObj):
        osdbCfgFname = libosd.configUtils.getConfigParam("osdbCfg",configObj)
        print("Loading separate OSDB Configuration File %s." % osdbCfgFname)
        osdbCfgObj = libosd.configUtils.loadConfig(osdbCfgFname)
        # Merge the contents of the OSDB Configuration file into configObj
        configObj = configObj | osdbCfgObj

    if (debug): print("configObj=",configObj.keys())

    allDataFname = configObj['dataFileNames']['allDataFileJson']
    allDataCsvFname = configObj['dataFileNames'].get('allDataFileCsv', 'allData.csv')
    testCsvFname = configObj['dataFileNames']['testDataFileCsv']
    testBalCsvFname = configObj['dataFileNames']['testBalancedFileCsv']
    trainCsvFname = configObj['dataFileNames']['trainDataFileCsv']
    valCsvFname = configObj['dataFileNames']['valDataFileCsv']
    trainAugCsvFname = configObj['dataFileNames']['trainAugmentedFileCsv']
    modelFname = configObj['modelConfig']['modelFname']
    
    # Determine output folder early for logging setup
    # Always use modelFname as prefix for consistency
    prefix = modelFname
    outFolder = getOutputPath(outPath=args['outDir'], rerun=int(args['rerun']), prefix=prefix)
    
    # Set up logging to capture all console output
    log_file = setup_logging(outFolder)
    
    try:
        # All output from here will be logged to file

        if args['clean']:
            # Clean up all output files
            print("Cleaning up output files")
            deleteFileIfExists(os.path.join(outFolder, allDataFname))
            deleteFileIfExists(os.path.join(outFolder, allDataCsvFname))
            deleteFileIfExists(os.path.join(outFolder, testCsvFname))
            deleteFileIfExists(os.path.join(outFolder, trainCsvFname))
            deleteFileIfExists(os.path.join(outFolder, valCsvFname))
            deleteFileIfExists(os.path.join(outFolder, trainAugCsvFname))
            deleteFileIfExists(os.path.join(outFolder, testBalCsvFname))

            deleteFileIfExists(os.path.join(outFolder, "%s.keras" % modelFname))
            deleteFileIfExists(os.path.join(outFolder, "%s_confusion.png" % modelFname))
            deleteFileIfExists(os.path.join(outFolder, "%s_probabilities.png" % modelFname))
            deleteFileIfExists(os.path.join(outFolder, "%s_training.png" % modelFname))
            deleteFileIfExists(os.path.join(outFolder, "%s_training2.png" % modelFname))
            deleteFileIfExists(os.path.join(outFolder, "%s_stats.txt" % modelFname))

            exit(0)

        if args['train']:
            import random

            # Initialise random number generators
            if ('randomSeed' in configObj):
                print("runSequence: Setting random seed to %d" % configObj['randomSeed'])
                seed = configObj['randomSeed'];
                np.random.seed(seed)
                random.seed(seed) 

            # Output folder already created above for logging
            print("runSequence: Writing Output to folder %s" % outFolder)
        
            # Copy configuration file to output folder for future reference
            config_basename = os.path.basename(args['config'])
            dest_config_path = os.path.join(outFolder, config_basename)
            if os.path.exists(dest_config_path):
                shutil.copy(dest_config_path, dest_config_path + ".bak")
                print("runSequence: Backed up existing configuration at %s.bak" % dest_config_path)
            shutil.copy(args['config'], dest_config_path)
            print("runSequence: Saved configuration to %s" % dest_config_path)
            

            # Select Data
            allDataFnamePath = os.path.join(outFolder, allDataFname)
            allDataCsvFname = configObj['dataFileNames'].get('allDataFileCsv', 'allData.csv')
            allDataCsvPath = os.path.join(outFolder, allDataCsvFname)
            
            if (not os.path.exists(allDataFnamePath)):
                print("runSequence: All data file missing - re-generating")
                print("runSequence: Removing flattened and augmented CSV files where they exist, so they are re-generated")
                # Remove CSV files so they'll be regenerated
                testCsvFname = configObj['dataFileNames'].get('testDataFileCsv', 'testData.csv')
                trainCsvFname = configObj['dataFileNames'].get('trainDataFileCsv', 'trainData.csv')
                valCsvFname = configObj['dataFileNames'].get('valDataFileCsv', 'valData.csv')
                trainAugCsvFname = configObj['dataFileNames'].get('trainAugmentedFileCsv', 'trainDataAugmented.csv')
                testBalCsvFname = configObj['dataFileNames'].get('testBalancedFileCsv', 'testDataBalanced.csv')
                deleteFileIfExists(os.path.join(outFolder, testCsvFname))
                deleteFileIfExists(os.path.join(outFolder, trainCsvFname))
                deleteFileIfExists(os.path.join(outFolder, valCsvFname))
                deleteFileIfExists(os.path.join(outFolder, trainAugCsvFname))
                deleteFileIfExists(os.path.join(outFolder, testBalCsvFname))
                deleteFileIfExists(allDataCsvPath)
                selectData.selectData(configObj, outDir=outFolder, debug=debug)

                nSeizure, nNonseizure = calculateFileStats(allDataFnamePath)
                print("runSequence: Data selection complete - all data in file %s contains %d seizure events and %d non-seizure events" % (allDataFnamePath, nSeizure, nNonseizure))

                # Flatten allData.json to allData.csv immediately after selection
                print("runSequence: Flattening all data from %s" % allDataFnamePath)
                validateDatapoints = configObj.get('dataProcessing', {}).get('validateDatapoints', False)
                flattenData.flattenOsdb(allDataFnamePath, allDataCsvPath, debug=debug, validate_datapoints=validateDatapoints)
                
                # Require allData.csv to exist before proceeding
                if not os.path.exists(allDataCsvPath):
                    print("ERROR: Flattening failed - allData.csv not found at %s" % allDataCsvPath)
                    print("       Aborting runSequence to avoid falling back to JSON.")
                    return

                nSeizure, nNonseizure = calculateFileStats(allDataCsvPath)
                print("runSequence: All data flattened to %s, containing %d seizure events and %d non-seizure events" % (allDataCsvPath, nSeizure, nNonseizure))

                # Split CSV into fold-specific CSVs (overwrite to ensure consistency after regeneration)
                if nestedKfold > 1:
                    print("runSequence: Splitting data into nested k-fold: %d outer folds x %d inner folds" % (nestedKfold, kfold))
                else:
                    print("runSequence: Splitting data into %d folds" % kfold)
                splitData.splitCsvData(configObj, allDataCsvPath, outDir=outFolder, kFold=kfold, nestedKfold=nestedKfold, debug=debug)
            else:
                print("runSequence: All data file %s already exists - skipping selection step" % allDataFnamePath)
                
                # Ensure CSV exists; flatten if missing
                if not os.path.exists(allDataCsvPath):
                    print("runSequence: allData.csv missing - flattening %s" % allDataFnamePath)
                    validateDatapoints = configObj.get('dataProcessing', {}).get('validateDatapoints', False)
                    flattenData.flattenOsdb(allDataFnamePath, allDataCsvPath, debug=debug, validate_datapoints=validateDatapoints)
                    if not os.path.exists(allDataCsvPath):
                        print("ERROR: Flattening failed - allData.csv not found at %s" % allDataCsvPath)
                        print("       Aborting runSequence to avoid falling back to JSON.")
                        return
                else:
                    print("runSequence: All data CSV file %s already exists - using existing CSV" % allDataCsvPath)

                # If fold CSVs already exist for all expected folds, skip splitting
                expected_outer = nestedKfold if nestedKfold > 1 else 1
                expected_inner = kfold if kfold > 1 else 1
                folds_present = True
                for outer in range(expected_outer):
                    outer_dir = os.path.join(outFolder, f"outerfold{outer}")
                    if not os.path.exists(outer_dir):
                        folds_present = False
                        break
                    for inner in range(expected_inner):
                        fold_dir = os.path.join(outer_dir, f"fold{inner}") if nestedKfold > 1 or kfold > 1 else outer_dir
                        if not os.path.exists(fold_dir):
                            folds_present = False
                            break
                        test_csv = os.path.join(fold_dir, testCsvFname)
                        train_csv = os.path.join(fold_dir, trainCsvFname)
                        if not (os.path.exists(test_csv) and os.path.exists(train_csv)):
                            folds_present = False
                            break
                    if not folds_present:
                        break

                if folds_present:
                    print("runSequence: Existing fold CSVs detected for all folds - skipping split")
                else:
                    if nestedKfold > 1:
                        print("runSequence: Splitting data into nested k-fold: %d outer folds x %d inner folds" % (nestedKfold, kfold))
                    else:
                        print("runSequence: Splitting data into %d folds" % kfold)
                    splitData.splitCsvData(configObj, allDataCsvPath, outDir=outFolder, kFold=kfold, nestedKfold=nestedKfold, debug=debug)

            foldResults = []
        
            # Determine iteration structure based on nested k-fold
            if nestedKfold > 1:
                # Nested k-fold: iterate through outer folds x inner folds
                outer_folds = range(0, nestedKfold)
                inner_folds = range(0, kfold)
            else:
                # Regular k-fold: treat as 1 outer fold with k inner folds
                outer_folds = range(0, 1)
                inner_folds = range(0, kfold)
            
            for nOuterFold in outer_folds:
                for nFold in inner_folds:
                    if nestedKfold > 1:
                        print("runSequence: Outer Fold %d, Inner Fold %d" % (nOuterFold, nFold))
                        outerFoldFolder = os.path.join(outFolder, "outerfold%d" % nOuterFold)
                        if kfold > 1:
                            foldOutFolder = os.path.join(outerFoldFolder, "fold%d" % nFold)
                        else:
                            foldOutFolder = outerFoldFolder
                    elif kfold > 1:
                        print("runSequence: Fold %d" % nFold)
                        foldOutFolder = os.path.join(outFolder, "fold%d" % nFold)
                    else:
                        print("runSequence: No folds - using output folder %s" % outFolder)
                        foldOutFolder = outFolder

                    # CSV files are already generated by splitData, so we can proceed directly to feature extraction
                    testFoldCsvFnamePath = os.path.join(foldOutFolder, testCsvFname)
                    trainFoldCsvFnamePath = os.path.join(foldOutFolder, trainCsvFname)
                
                    # Verify the CSV files exist
                    if not os.path.exists(testFoldCsvFnamePath):
                        print("ERROR: Test CSV file %s does not exist" % testFoldCsvFnamePath)
                        exit(-1)
                    if not os.path.exists(trainFoldCsvFnamePath):
                        print("ERROR: Train CSV file %s does not exist" % trainFoldCsvFnamePath)
                        exit(-1)

                    nSeizure, nNonseizure = calculateFileStats(trainFoldCsvFnamePath)
                    print(f"runSequence: Using training data from {trainFoldCsvFnamePath}, containing {nSeizure} seizure events and {nNonseizure} non-seizure events")

                    # Augment training data
                    trainAugCsvFnamePath = os.path.join(foldOutFolder, trainAugCsvFname)
                    if not os.path.exists(trainAugCsvFnamePath):
                        print("runSequence: Augmenting training data %s" % trainFoldCsvFnamePath)
                        augmentData.augmentSeizureData(configObj, dataDir=foldOutFolder, debug=debug)
                        nSeizure, nNonseizure = calculateFileStats(trainAugCsvFnamePath)
                        print(f"runSequence: Augmented training data saved to {trainAugCsvFnamePath}, containing {nSeizure} seizure events and {nNonseizure} non-seizure events")
                    else:
                        print("runSequence: Training data %s already augmented - skipping" % trainAugCsvFname)

                    # After data augmentation
                    trainAugCsvFnamePath = os.path.join(foldOutFolder, configObj['dataFileNames']['trainAugmentedFileCsv'])
                    trainFeaturesCsvPath = os.path.join(foldOutFolder, configObj['dataFileNames']['trainFeaturesFileCsv'])
                    testFoldCsvFnamePath = os.path.join(foldOutFolder, configObj['dataFileNames']['testDataFileCsv'])
                    testFeaturesCsvPath = os.path.join(foldOutFolder, configObj['dataFileNames']['testFeaturesFileCsv'])

                    # Extract features for training data
                    if not os.path.exists(trainFeaturesCsvPath):
                        print("runSequence: Extracting features for training data")
                        extractFeatures(trainAugCsvFnamePath, trainFeaturesCsvPath, configObj)
                    else:
                        print(f"runSequence: Training features {trainFeaturesCsvPath} already exist - skipping")

                    # Extract features for test data
                    if not os.path.exists(testFeaturesCsvPath):
                        print("runSequence: Extracting features for test data")
                        extractFeatures(testFoldCsvFnamePath, testFeaturesCsvPath, configObj)
                    else:
                        print(f"runSequence: Test features {testFeaturesCsvPath} already exist - skipping")

                    # Generate feature history files if configured and needed
                    addHistoryLength = configObj.get('dataProcessing', {}).get('addFeatureHistoryLength', 0)
                
                    # Check if features list contains only raw acceleration (no calculated features)
                    features = configObj.get('dataProcessing', {}).get('features', [])
                    raw_acc_features = {'acc_magnitude', 'acc_x', 'acc_y', 'acc_z'}
                    only_raw_acc = all(f in raw_acc_features for f in features) if features else False
                
                    skip_history = (addHistoryLength == 0) or only_raw_acc
                
                    if skip_history:
                        print("runSequence: Skipping feature history (addFeatureHistoryLength=0 or only raw acceleration features)")
                        # Use regular feature files directly
                        # Training files are already set correctly above
                    else:
                        trainFeaturesHistoryCsvPath = os.path.join(foldOutFolder, configObj['dataFileNames']['trainFeaturesHistoryFileCsv'])
                        testFeaturesHistoryCsvPath = os.path.join(foldOutFolder, configObj['dataFileNames']['testFeaturesHistoryFileCsv'])
                        if not (os.path.exists(trainFeaturesHistoryCsvPath) and os.path.exists(testFeaturesHistoryCsvPath)):
                            print("runSequence: Generating feature history files")
                            add_feature_history(configObj, foldOutFolder=foldOutFolder)
                        else:
                            print("runSequence: Feature history files already exist - skipping")

                        # Update training to use history files
                        configObj['dataFileNames']['trainFeaturesFileCsv'] = configObj['dataFileNames']['trainFeaturesHistoryFileCsv']
                        configObj['dataFileNames']['testFeaturesFileCsv'] = configObj['dataFileNames']['testFeaturesHistoryFileCsv']

                    # Get framework - check 'framework' field first, fall back to legacy 'modelType'
                    framework = configObj['modelConfig'].get('framework')
                    if framework is None:
                        framework = configObj['modelConfig'].get('modelType', 'tensorflow')
                
                    if framework == "sklearn":
                        print("runSequence: Training sklearn model")
                        import skTrainer
                        import skTester
                        trainingResults = skTrainer.trainModel(configObj, dataDir=foldOutFolder, debug=debug)
                        print("runSequence: Model trained")
                        testResults = skTester.testModel(configObj, dataDir=foldOutFolder, debug=debug)
                        foldResults.append(testResults)
                        
                        # Analyze test results
                        analyze_test_results(foldOutFolder, configObj, debug=debug)
                    elif framework in ["tensorflow", "pytorch"]:
                        import nnTrainer
                        import nnTester
                    
                        # Set random seed based on framework
                        if ('randomSeed' in configObj):
                            seed = configObj['randomSeed']
                            if framework == 'pytorch':
                                print("runSequence: Setting PyTorch random seed to %d" % seed)
                                import torch
                                torch.manual_seed(seed)
                                if torch.cuda.is_available():
                                    torch.cuda.manual_seed_all(seed)
                            else:
                                print("runSequence: Setting TensorFlow random seed to %d" % seed)
                                import tensorflow as tf
                                tf.random.set_seed(seed)
                    
                        print("runSequence: Training %s neural network model" % framework)
                        nnTrainer.trainModel(configObj, dataDir=foldOutFolder, debug=debug)
                        print("runSequence: Testing Model")
                        # Skip .ptl and .pte testing during inner fold evaluation (test_ptl=False, test_pte=False)
                        testResults = nnTester.testModel(configObj, dataDir=foldOutFolder, balanced=False, debug=debug, test_ptl=False, test_pte=False) 
                        foldResults.append(testResults)
                        
                        # Analyze test results
                        analyze_test_results(foldOutFolder, configObj, debug=debug)
                    
                    if nestedKfold > 1:
                        print("runSequence: Finished outer fold %d, inner fold %d, data in folder %s" % (nOuterFold, nFold, foldOutFolder))
                    else:
                        print("runSequence: Finished fold %d, data in folder %s" % (nFold, foldOutFolder))

            # Compute average results across folds
            avgResults = {}
            for key in foldResults[0].keys():
                avgResults[key] = sum(foldResult[key] for foldResult in foldResults) / len(foldResults)
            # calculate standard deviation for each key
            for key in foldResults[0].keys():
                avgResults[key + "_std"] = np.std([foldResult[key] for foldResult in foldResults])


            # Save the results to a file
            kfoldSummaryPath = os.path.join(outFolder, "kfold_summary.txt")
            kfoldSummaryPath = os.path.join(outFolder, "kfold_summary.txt")
            kfoldJsonPath = os.path.join(outFolder, "kfold_summary.json")
            with open(kfoldSummaryPath, 'w') as f:
                f.write("K-Fold Summary (epoch based analysis):\n")
                f.write("|--------|-------------------------------------------------------|-----------------------------------------------|\n")
                f.write( "|        |   Model Results                                       | OSD Algorithm Results                         |\n")
                f.write("| FoldID |   np  |  tn   |  fn   |  fp   |  tp   |  tpr  |  fpr  | tnOsd | fnOsd | fpOsd | tpOsd |tprOsd |fprOsd |\n")
                f.write("|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|\n")
                for nFold, foldResult in enumerate(foldResults):
                    f.write(f"| Fold {nFold} | {foldResult['tp']+foldResult['fn']:5d} | {foldResult['tn']:5d} | {foldResult['fn']:5d} | {foldResult['fp']:5d} | {foldResult['tp']:5d} | {foldResult['tpr']:5.2f} | {foldResult['fpr']:5.2f} | {foldResult['tnOsd']:5d} | {foldResult['fnOsd']:5d} | {foldResult['fpOsd']:5d} | {foldResult['tpOsd']:5d} | {foldResult['tprOsd']:5.2f} | {foldResult['fprOsd']:5.2f} |\n")
                f.write("|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|\n")
                f.write(f"| Avg    |       | {avgResults['tn']:5.0f} | {avgResults['fn']:5.0f} | {avgResults['fp']:5.0f} | {avgResults['tp']:5.0f} | {avgResults['tpr']:5.2f} | {avgResults['fpr']:5.2f} | {avgResults['tnOsd']:5.0f} | {avgResults['fnOsd']:5.0f} | {avgResults['fpOsd']:5.0f} | {avgResults['tpOsd']:5.0f} | {avgResults['tprOsd']:5.2f} | {avgResults['fprOsd']:5.2f} |\n")
                f.write(f"| Std    |       | {avgResults['tn_std']/avgResults['tn']:5.2f} | {avgResults['fn_std']/avgResults['fn']:5.2f} | {avgResults['fp_std']/avgResults['fp']:5.2f} | {avgResults['tp_std']/avgResults['tp']:5.2f} | {avgResults['tpr_std']:5.2f} | {avgResults['fpr_std']:5.2f} | {avgResults['tnOsd_std']/avgResults['tnOsd']:5.2f} | {avgResults['fnOsd_std']/avgResults['fnOsd']:5.2f} | {avgResults['fpOsd_std']/avgResults['fpOsd']:5.2f} | {avgResults['tpOsd_std']/avgResults['tpOsd']:5.2f} | {avgResults['tprOsd_std']:5.2f} | {avgResults['fprOsd_std']:5.2f} |\n")
                f.write("|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|\n")
                f.write("\n\n")
                f.write("K-Fold Summary (event based analysis):\n")
                f.write("|--------|-------------------------------------------------------|-----------------------------------------------|\n")
                f.write( "|        |   Model Results                                       | OSD Algorithm Results                         |\n")
                f.write("| FoldID |   np  |  tn   |  fn   |  fp   |  tp   |  tpr  |  fpr  | tnOsd | fnOsd | fpOsd | tpOsd |tprOsd |fprOsd |\n")
                f.write("|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|\n")
                for nFold, foldResult in enumerate(foldResults):
                    f.write(f"| Fold {nFold} | {foldResult['event_tp']+foldResult['event_fn']:5d} | {foldResult['event_tn']:5d} | {foldResult['event_fn']:5d} | {foldResult['event_fp']:5d} | {foldResult['event_tp']:5d} | {foldResult['event_tpr']:5.2f} | {foldResult['event_fpr']:5.2f} | {foldResult['osd_event_tn']:5d} | {foldResult['osd_event_fn']:5d} | {foldResult['osd_event_fp']:5d} | {foldResult['osd_event_tp']:5d} | {foldResult['osd_event_tpr']:5.2f} | {foldResult['osd_event_fpr']:5.2f} |\n")
                f.write("|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|\n")
                f.write(f"| Avg    |       | {avgResults['event_tn']:5.0f} | {avgResults['event_fn']:5.0f} | {avgResults['event_fp']:5.0f} | {avgResults['event_tp']:5.0f} | {avgResults['event_tpr']:5.2f} | {avgResults['event_fpr']:5.2f} | {avgResults['osd_event_tn']:5.0f} | {avgResults['osd_event_fn']:5.0f} | {avgResults['osd_event_fp']:5.0f} | {avgResults['osd_event_tp']:5.0f} | {avgResults['osd_event_tpr']:5.2f} | {avgResults['osd_event_fpr']:5.2f} |\n")
                f.write(f"| Std    |       | {avgResults['event_tn_std']/avgResults['event_tn']:5.2f} | {avgResults['event_fn_std']/avgResults['event_fn']:5.2f} | {avgResults['event_fp_std']/avgResults['event_fp']:5.2f} | {avgResults['event_tp_std']/avgResults['event_tp']:5.2f} | {avgResults['event_tpr_std']:5.2f} | {avgResults['event_fpr_std']:5.2f} | {avgResults['osd_event_tn_std']/avgResults['osd_event_tn']:5.2f} | {avgResults['osd_event_fn_std']/avgResults['osd_event_fn']:5.2f} | {avgResults['osd_event_fp_std']/avgResults['osd_event_fp']:5.2f} | {avgResults['osd_event_tp_std']/avgResults['osd_event_tp']:5.2f} | {avgResults['osd_event_tpr_std']:5.2f} | {avgResults['osd_event_fpr_std']:5.2f} |\n")
                f.write("|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|\n")

            # Write foldResults as formatted JSON
            with open(kfoldJsonPath, 'w') as jf:
                json.dump(foldResults, jf, indent=2)
            print("K-Fold summary saved to %s" % (kfoldSummaryPath))    
            print("K-Fold JSON summary saved to %s" % (kfoldJsonPath))
            with open(kfoldSummaryPath, 'r') as summary_file:
                print(summary_file.read())
            print("===== End K-Fold Summary =====\n")

            # NESTED K-FOLD: Test on truly independent outer fold test sets
            if nestedKfold > 1:
                print("\n" + "="*80)
                print("NESTED K-FOLD: Testing on Independent Outer Fold Test Sets")
                print("="*80 + "\n")
            
                # Test all outer folds using the reusable function
                # (This function already handles all outer folds in a loop)
                outerFoldResults = test_outer_folds(configObj, kfold, nestedKfold, outFolder, foldResults, args=args, debug=debug)
            
            
                # Compute and save outer fold summary
                if len(outerFoldResults) > 0:
                    print("\n" + "="*80)
                    print("NESTED K-FOLD: Summary of Independent Outer Fold Test Results")
                    print("="*80)
                
                    outerAvgResults = {}
                    for key in outerFoldResults[0].keys():
                        # Skip non-numeric keys (like 'outer_fold', 'best_inner_fold', 'best_model_path', 'test_output_path')
                        if key not in ['outer_fold', 'best_inner_fold', 'best_model_path', 'test_output_path']:
                            try:
                                outerAvgResults[key] = sum(result[key] for result in outerFoldResults) / len(outerFoldResults)
                                outerAvgResults[key + "_std"] = np.std([result[key] for result in outerFoldResults])
                            except (TypeError, KeyError):
                                # Skip keys that can't be averaged (not numeric)
                                pass
                
                    print("  Model Results:")
                    print("    TPR (Sensitivity): %.3f ± %.3f" % (outerAvgResults['tpr'], outerAvgResults['tpr_std']))
                    print("    FPR: %.3f ± %.3f" % (outerAvgResults['fpr'], outerAvgResults['fpr_std']))
                    print("    Event TPR: %.3f ± %.3f" % (outerAvgResults['event_tpr'], outerAvgResults['event_tpr_std']))
                    print("    Event FPR: %.3f ± %.3f" % (outerAvgResults['event_fpr'], outerAvgResults['event_fpr_std']))
                    print("  OSD Algorithm Results (for comparison):")
                    print("    TPR (Sensitivity): %.3f ± %.3f" % (outerAvgResults['tprOsd'], outerAvgResults['tprOsd_std']))
                    print("    FPR: %.3f ± %.3f" % (outerAvgResults['fprOsd'], outerAvgResults['fprOsd_std']))
                    print("    Event TPR: %.3f ± %.3f" % (outerAvgResults['osd_event_tpr'], outerAvgResults['osd_event_tpr_std']))
                    print("    Event FPR: %.3f ± %.3f" % (outerAvgResults['osd_event_fpr'], outerAvgResults['osd_event_fpr_std']))
                
                    # Save outer fold results
                    outerFoldSummaryPath = os.path.join(outFolder, "nested_kfold_outer_summary.txt")
                    outerFoldJsonPath = os.path.join(outFolder, "nested_kfold_outer_summary.json")
                
                    with open(outerFoldSummaryPath, 'w') as f:
                        f.write("NESTED K-FOLD: Independent Outer Fold Test Results\n")
                        f.write("="*80 + "\n\n")
                        f.write("These results are from TRULY INDEPENDENT test sets that were never used during training.\n")
                        f.write("Each outer fold's best inner fold model was tested on its corresponding independent test set.\n\n")
                    
                        # List the best models
                        f.write("BEST MODELS (suitable for production use):\n")
                        f.write("-" * 80 + "\n")
                        for result in outerFoldResults:
                            model_path = result.get('best_model_path', 'unknown')
                            f.write(f"  Outer Fold {result['outer_fold']}: {model_path}\n")
                            f.write(f"    - Best inner fold: {result['best_inner_fold']}\n")
                            f.write(f"    - Test Events: {result['event_tp']+result['event_fn']} seizures, {result['event_tn']+result['event_fp']} non-seizures\n")
                            f.write(f"    - Epoch TPR: {result['tpr']:.3f}, FPR: {result['fpr']:.3f}\n")
                            f.write(f"    - Event TPR: {result['event_tpr']:.3f}, FPR: {result['event_fpr']:.3f}\n")
                            f.write("\n")
                    
                        f.write("\nOuter Fold Results (epoch based analysis):\n")
                        f.write("|-----------|-------|-------------------------------------------------------|-----------------------------------------------|\n")
                        f.write("| Outer     | Best  |   Model Results                                       | OSD Algorithm Results                         |\n")
                        f.write("| Fold ID   | Inner |   np  |  tn   |  fn   |  fp   |  tp   |  tpr  |  fpr  | tnOsd | fnOsd | fpOsd | tpOsd |tprOsd |fprOsd |\n")
                        f.write("|-----------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|\n")
                        for result in outerFoldResults:
                            f.write(f"| Outer {result['outer_fold']:2d}  |   {result['best_inner_fold']:2d}  | {result['tp']+result['fn']:5d} | {result['tn']:5d} | {result['fn']:5d} | {result['fp']:5d} | {result['tp']:5d} | {result['tpr']:5.3f} | {result['fpr']:5.3f} | {result['tnOsd']:5d} | {result['fnOsd']:5d} | {result['fpOsd']:5d} | {result['tpOsd']:5d} | {result['tprOsd']:5.3f} | {result['fprOsd']:5.3f} |\n")
                        f.write("|-----------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|\n")
                        f.write(f"| Average   |       |       | {outerAvgResults['tn']:5.0f} | {outerAvgResults['fn']:5.0f} | {outerAvgResults['fp']:5.0f} | {outerAvgResults['tp']:5.0f} | {outerAvgResults['tpr']:5.3f} | {outerAvgResults['fpr']:5.3f} | {outerAvgResults['tnOsd']:5.0f} | {outerAvgResults['fnOsd']:5.0f} | {outerAvgResults['fpOsd']:5.0f} | {outerAvgResults['tpOsd']:5.0f} | {outerAvgResults['tprOsd']:5.3f} | {outerAvgResults['fprOsd']:5.3f} |\n")
                        f.write(f"| Std Dev   |       |       | {outerAvgResults['tn_std']:5.1f} | {outerAvgResults['fn_std']:5.1f} | {outerAvgResults['fp_std']:5.1f} | {outerAvgResults['tp_std']:5.1f} | {outerAvgResults['tpr_std']:5.3f} | {outerAvgResults['fpr_std']:5.3f} | {outerAvgResults['tnOsd_std']:5.1f} | {outerAvgResults['fnOsd_std']:5.1f} | {outerAvgResults['fpOsd_std']:5.1f} | {outerAvgResults['tpOsd_std']:5.1f} | {outerAvgResults['tprOsd_std']:5.3f} | {outerAvgResults['fprOsd_std']:5.3f} |\n")
                        f.write("|-----------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|\n")
                        f.write("\n\n")
                        f.write("Outer Fold Results (event based analysis):\n")
                        f.write("|-----------|-------|-------------------------------------------------------|-----------------------------------------------|\n")
                        f.write("| Outer     | Best  |   Model Results                                       | OSD Algorithm Results                         |\n")
                        f.write("| Fold ID   | Inner |   np  |  tn   |  fn   |  fp   |  tp   |  tpr  |  fpr  | tnOsd | fnOsd | fpOsd | tpOsd |tprOsd |fprOsd |\n")
                        f.write("|-----------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|\n")
                        for result in outerFoldResults:
                            f.write(f"| Outer {result['outer_fold']:2d}  |   {result['best_inner_fold']:2d}  | {result['event_tp']+result['event_fn']:5d} | {result['event_tn']:5d} | {result['event_fn']:5d} | {result['event_fp']:5d} | {result['event_tp']:5d} | {result['event_tpr']:5.3f} | {result['event_fpr']:5.3f} | {result['osd_event_tn']:5d} | {result['osd_event_fn']:5d} | {result['osd_event_fp']:5d} | {result['osd_event_tp']:5d} | {result['osd_event_tpr']:5.3f} | {result['osd_event_fpr']:5.3f} |\n")
                        f.write("|-----------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|\n")
                        f.write(f"| Average   |       |       | {outerAvgResults['event_tn']:5.0f} | {outerAvgResults['event_fn']:5.0f} | {outerAvgResults['event_fp']:5.0f} | {outerAvgResults['event_tp']:5.0f} | {outerAvgResults['event_tpr']:5.3f} | {outerAvgResults['event_fpr']:5.3f} | {outerAvgResults['osd_event_tn']:5.0f} | {outerAvgResults['osd_event_fn']:5.0f} | {outerAvgResults['osd_event_fp']:5.0f} | {outerAvgResults['osd_event_tp']:5.0f} | {outerAvgResults['osd_event_tpr']:5.3f} | {outerAvgResults['osd_event_fpr']:5.3f} |\n")
                        f.write(f"| Std Dev   |       |       | {outerAvgResults['event_tn_std']:5.1f} | {outerAvgResults['event_fn_std']:5.1f} | {outerAvgResults['event_fp_std']:5.1f} | {outerAvgResults['event_tp_std']:5.1f} | {outerAvgResults['event_tpr_std']:5.3f} | {outerAvgResults['event_fpr_std']:5.3f} | {outerAvgResults['osd_event_tn_std']:5.1f} | {outerAvgResults['osd_event_fn_std']:5.1f} | {outerAvgResults['osd_event_fp_std']:5.1f} | {outerAvgResults['osd_event_tp_std']:5.1f} | {outerAvgResults['osd_event_tpr_std']:5.3f} | {outerAvgResults['osd_event_fpr_std']:5.3f} |\n")
                        f.write("|-----------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|\n")
                        
                        # Add Tonic-Clonic analysis if available
                        if 'tc_count' in outerAvgResults and outerAvgResults['tc_count'] > 0:
                            f.write("\n\nTONIC-CLONIC SEIZURES ANALYSIS (Event Based):\n")
                            f.write("|-----------|-------------------------|-------|\n")
                            f.write("| Outer     | Count | TP  | FN  | TPR   |\n")
                            f.write("|-----------|-------|-----|-----|-------|\n")
                            for result in outerFoldResults:
                                tc_count = result.get('tc_count', 0)
                                tc_tp = result.get('tc_tp', 0)
                                tc_fn = result.get('tc_fn', 0)
                                tc_tpr = result.get('tc_tpr', 0.0)
                                f.write(f"| Outer {result['outer_fold']:2d}  | {tc_count:5d}  | {tc_tp:3d} | {tc_fn:3d} | {tc_tpr:5.3f} |\n")
                            f.write("|-----------|-------|-----|-----|-------|\n")
                            f.write(f"| Average   | {outerAvgResults['tc_count']:5.0f}  | {outerAvgResults['tc_tp']:3.0f} | {outerAvgResults['tc_fn']:3.0f} | {outerAvgResults['tc_tpr']:5.3f} |\n")
                            if 'tc_count_std' in outerAvgResults:
                                f.write(f"| Std Dev   | {outerAvgResults['tc_count_std']:5.1f}  | {outerAvgResults['tc_tp_std']:3.1f} | {outerAvgResults['tc_fn_std']:3.1f} | {outerAvgResults['tc_tpr_std']:5.3f} |\n")
                            f.write("|-----------|-------|-----|-----|-------|\n")
                        
                        f.write("\n\nIMPORTANT: Report these outer fold results in publications as they represent\n")
                        f.write("unbiased estimates of model generalization on truly independent test sets.\n")
                
                    with open(outerFoldJsonPath, 'w') as jf:
                        json.dump(outerFoldResults, jf, indent=2)
                
                    print("\nNested k-fold outer fold summary saved to: %s" % outerFoldSummaryPath)
                    print("Nested k-fold outer fold JSON saved to: %s" % outerFoldJsonPath)
                
                    with open(outerFoldSummaryPath, 'r') as summary_file:
                        print("\n" + summary_file.read())
                    print("="*80 + "\n")


            if (os.path.exists(args['config'])):
                shutil.copy(args['config'], outFolder)

            print("runSequence: Finished training and testing - output in folder %s" % outFolder)    

        if args['test']:
            print("runSequence: Testing model")
            # if we specify an experiment to rerun, use the specified rerun folder, otherwise use the last run
            if (int(args['rerun']) > 0):
                outFolder = getOutputPath(outPath=args['outDir'], rerun=args['rerun'], prefix=modelFname)
            else:
                outFolder = getLatestOutputFolder(outPath=args['outDir'], prefix=modelFname)
            print("runSequence: Using Output to folder %s" % outFolder)
            # Get framework - check 'framework' field first, fall back to legacy 'modelType'
            framework = configObj['modelConfig'].get('framework')
            if framework is None:
                framework = configObj['modelConfig'].get('modelType', 'tensorflow')
            
            if framework == "sklearn":
                import skTester
                # Test the model using sklearn
                if kfold > 1:
                    print("ERROR: K-fold testing not yet supported for sklearn models")
                    exit(-1)
                skTester.testModel(configObj, dataDir=outFolder, debug=debug)
            elif framework in ["tensorflow", "pytorch"]:
                import nnTester
                print("runSequence: Testing %s neural network model" % framework)
                outFolder = getOutputPath(outPath=args['outDir'], rerun=args['rerun'], prefix=configObj['modelConfig']['modelFname'])
                print("runSequence: Testing in folder %s" % outFolder)
                
                # Determine if we should rerun tests based on --rerun parameter
                rerunTests = int(args['rerun']) > 0
                
                # Check if this is nested k-fold structure and run appropriate tests
                if nestedKfold > 1:
                    print("runSequence: Detected nested k-fold structure (nestedKfold=%d, kfold=%d)" % (nestedKfold, kfold))
                    print("runSequence: Running outer fold testing on independent test sets")
                    
                    # Load inner fold results for model selection
                    foldResults = []
                    for nOuterFold in range(0, nestedKfold):
                        outerFoldOutFolder = os.path.join(outFolder, "outerfold%d" % nOuterFold)
                        for nFold in range(0, kfold):
                            if kfold > 1:
                                foldPath = os.path.join(outerFoldOutFolder, "fold%d" % nFold)
                            else:
                                foldPath = outerFoldOutFolder
                            
                            test_results_file = os.path.join(foldPath, "testResults.json")
                            if os.path.exists(test_results_file):
                                with open(test_results_file, 'r') as f:
                                    foldResults.append(json.load(f))
                            else:
                                # If results don't exist, add a placeholder with TPR=0
                                foldResults.append({'tpr': 0.0, 'fpr': 1.0})
                    
                    # Test outer folds using the reusable function
                    outerFoldResults = test_outer_folds(configObj, kfold, nestedKfold, outFolder, foldResults, args=args, debug=debug)
                    
                    # Generate and display summary (same as in training path)
                    if len(outerFoldResults) > 0:
                        print("\n" + "="*80)
                        print("NESTED K-FOLD: Summary of Independent Outer Fold Test Results")
                        print("="*80)
                        
                        outerAvgResults = {}
                        for key in outerFoldResults[0].keys():
                            # Skip non-numeric keys
                            if key not in ['outer_fold', 'best_inner_fold', 'best_model_path', 'test_output_path']:
                                try:
                                    outerAvgResults[key] = sum(result[key] for result in outerFoldResults) / len(outerFoldResults)
                                    outerAvgResults[key + "_std"] = np.std([result[key] for result in outerFoldResults])
                                except (TypeError, KeyError):
                                    pass
                        
                        print("Average across %d outer folds:" % len(outerFoldResults))
                        print("  Model Results:")
                        print("    TPR (Sensitivity): %.3f ± %.3f" % (outerAvgResults['tpr'], outerAvgResults['tpr_std']))
                        print("    FPR: %.3f ± %.3f" % (outerAvgResults['fpr'], outerAvgResults['fpr_std']))
                        print("    Event TPR: %.3f ± %.3f" % (outerAvgResults['event_tpr'], outerAvgResults['event_tpr_std']))
                        print("    Event FPR: %.3f ± %.3f" % (outerAvgResults['event_fpr'], outerAvgResults['event_fpr_std']))
                        print("="*80)
                
                elif kfold > 1:
                    print("runSequence: Running regular k-fold testing with %d folds (rerun=%s)" % (kfold, rerunTests))
                    nnTester.testKFold(configObj, kfold=kfold, dataDir=outFolder, rerun=rerunTests, debug=debug)
                    
                    # Analyze test results for each fold
                    for nFold in range(kfold):
                        fold_dir = os.path.join(outFolder, f"fold{nFold}")
                        if os.path.exists(fold_dir):
                            analyze_test_results(fold_dir, configObj, debug=debug)
                else:
                    print("runSequence: Testing single model")
                    # Test .ptl and .pte models for final single model test (only if --testPtl/--testPte flags are used)
                    test_percent = args.get('testPercent', 100.0)
                    nnTester.testModel(configObj, dataDir=outFolder, balanced=False, debug=debug, test_ptl=args.get('testPtl', False), test_pte=args.get('testPte', False), test_percent=test_percent)
                    
                    # Analyze test results
                    analyze_test_results(outFolder, configObj, debug=debug)
            else:
                print("ERROR: Unsupported framework: %s" % framework)
                exit(-1)
        
        if args.get('testOuter', False):
            print("runSequence: Testing outer fold independent test sets")
            if nestedKfold <= 1:
                print("ERROR: --testOuter requires --nestedKfold > 1")
                exit(-1)
            
            # Get output folder
            if int(args['rerun']) > 0:
                outFolder = getOutputPath(outPath=args['outDir'], rerun=args['rerun'], prefix=modelFname)
            else:
                outFolder = getLatestOutputFolder(outPath=args['outDir'], prefix=modelFname)
            
            print("runSequence: Using output folder %s" % outFolder)
            
            # Get framework
            framework = configObj['modelConfig'].get('framework')
            if framework is None:
                framework = configObj['modelConfig'].get('modelType', 'tensorflow')
            
            # Outer fold testing (this is the same code as in the train section)
            import nnTester
            # Load inner fold results from testResults.json files
            # We need these to select the best model for each outer fold
            foldResults = []
            for nOuterFold in range(0, nestedKfold):
                outerFoldOutFolder = os.path.join(outFolder, "outerfold%d" % nOuterFold)
                for nFold in range(0, kfold):
                    if kfold > 1:
                        foldPath = os.path.join(outerFoldOutFolder, "fold%d" % nFold)
                    else:
                        foldPath = outerFoldOutFolder
                    
                    test_results_file = os.path.join(foldPath, "testResults.json")
                    if os.path.exists(test_results_file):
                        with open(test_results_file, 'r') as f:
                            foldResults.append(json.load(f))
                    else:
                        # If results don't exist, add a placeholder with TPR=0
                        foldResults.append({'tpr': 0.0, 'fpr': 1.0})
            
            # Test outer folds using the reusable function
            outerFoldResults = test_outer_folds(configObj, kfold, nestedKfold, outFolder, foldResults, args=args, debug=debug)
            
            # Generate and display summary (same as in training path)
            if len(outerFoldResults) > 0:
                print("\n" + "="*80)
                print("NESTED K-FOLD: Summary of Independent Outer Fold Test Results")
                print("="*80)
                
                outerAvgResults = {}
                for key in outerFoldResults[0].keys():
                            # Skip non-numeric keys
                            if key not in ['outer_fold', 'best_inner_fold', 'best_model_path', 'test_output_path']:
                                try:
                                    outerAvgResults[key] = sum(result[key] for result in outerFoldResults) / len(outerFoldResults)
                                    outerAvgResults[key + "_std"] = np.std([result[key] for result in outerFoldResults])
                                except (TypeError, KeyError):
                                    # Skip keys that can't be averaged (not numeric)
                                    pass
                print("  Model Results:")
                print("    TPR (Sensitivity): %.3f ± %.3f" % (outerAvgResults['tpr'], outerAvgResults['tpr_std']))
                print("    FPR: %.3f ± %.3f" % (outerAvgResults['fpr'], outerAvgResults['fpr_std']))
                print("    Event TPR: %.3f ± %.3f" % (outerAvgResults['event_tpr'], outerAvgResults['event_tpr_std']))
                print("    Event FPR: %.3f ± %.3f" % (outerAvgResults['event_fpr'], outerAvgResults['event_fpr_std']))
                print("  OSD Algorithm Results (for comparison):")
                print("    TPR (Sensitivity): %.3f ± %.3f" % (outerAvgResults['tprOsd'], outerAvgResults['tprOsd_std']))
                print("    FPR: %.3f ± %.3f" % (outerAvgResults['fprOsd'], outerAvgResults['fprOsd_std']))
                print("    Event TPR: %.3f ± %.3f" % (outerAvgResults['osd_event_tpr'], outerAvgResults['osd_event_tpr_std']))
                print("    Event FPR: %.3f ± %.3f" % (outerAvgResults['osd_event_fpr'], outerAvgResults['osd_event_fpr_std']))
                
                # Save outer fold results
                outerFoldSummaryPath = os.path.join(outFolder, "nested_kfold_outer_summary.txt")
                outerFoldJsonPath = os.path.join(outFolder, "nested_kfold_outer_summary.json")
                
                with open(outerFoldSummaryPath, 'w') as f:
                    f.write("NESTED K-FOLD: Independent Outer Fold Test Results\n")
                    f.write("="*80 + "\n\n")
                    f.write("These results are from TRULY INDEPENDENT test sets that were never used during training.\n")
                    f.write("Each outer fold's best inner fold model was tested on its corresponding independent test set.\n\n")
                    
                    # List the best models
                    f.write("BEST MODELS (suitable for production use):\n")
                    f.write("-" * 80 + "\n")
                    for result in outerFoldResults:
                        model_path = result.get('best_model_path', 'unknown')
                        f.write(f"  Outer Fold {result['outer_fold']}: {model_path}\n")
                        f.write(f"    - Best inner fold: {result['best_inner_fold']}\n")
                        f.write(f"    - Test Events: {result['event_tp']+result['event_fn']} seizures, {result['event_tn']+result['event_fp']} non-seizures\n")
                        f.write(f"    - Epoch TPR: {result['tpr']:.3f}, FPR: {result['fpr']:.3f}\n")
                        f.write(f"    - Event TPR: {result['event_tpr']:.3f}, FPR: {result['event_fpr']:.3f}\n")
                        f.write(f"    - OSD Event TPR: {result['osd_event_tpr']:.3f}, FPR: {result['osd_event_fpr']:.3f}\n")
                        f.write("\n")
                    
                    f.write("\nOuter Fold Results (epoch based analysis):\n")
                    f.write("|-----------|-------|-------------------------------------------------------|-----------------------------------------------|\n")
                    f.write("| Outer     | Best  |   Model Results                                       | OSD Algorithm Results                         |\n")
                    f.write("| Fold ID   | Inner |   np  |  tn   |  fn   |  fp   |  tp   |  tpr  |  fpr  | tnOsd | fnOsd | fpOsd | tpOsd |tprOsd |fprOsd |\n")
                    f.write("|-----------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|\n")
                    for result in outerFoldResults:
                        f.write(f"| Outer {result['outer_fold']:2d}  |   {result['best_inner_fold']:2d}  | {result['tp']+result['fn']:5d} | {result['tn']:5d} | {result['fn']:5d} | {result['fp']:5d} | {result['tp']:5d} | {result['tpr']:5.3f} | {result['fpr']:5.3f} | {result['tnOsd']:5d} | {result['fnOsd']:5d} | {result['fpOsd']:5d} | {result['tpOsd']:5d} | {result['tprOsd']:5.3f} | {result['fprOsd']:5.3f} |\n")
                    f.write("|-----------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|\n")
                    f.write(f"| Average   |       |       | {outerAvgResults['tn']:5.0f} | {outerAvgResults['fn']:5.0f} | {outerAvgResults['fp']:5.0f} | {outerAvgResults['tp']:5.0f} | {outerAvgResults['tpr']:5.3f} | {outerAvgResults['fpr']:5.3f} | {outerAvgResults['tnOsd']:5.0f} | {outerAvgResults['fnOsd']:5.0f} | {outerAvgResults['fpOsd']:5.0f} | {outerAvgResults['tpOsd']:5.0f} | {outerAvgResults['tprOsd']:5.3f} | {outerAvgResults['fprOsd']:5.3f} |\n")
                    f.write(f"| Std Dev   |       |       | {outerAvgResults['tn_std']:5.1f} | {outerAvgResults['fn_std']:5.1f} | {outerAvgResults['fp_std']:5.1f} | {outerAvgResults['tp_std']:5.1f} | {outerAvgResults['tpr_std']:5.3f} | {outerAvgResults['fpr_std']:5.3f} | {outerAvgResults['tnOsd_std']:5.1f} | {outerAvgResults['fnOsd_std']:5.1f} | {outerAvgResults['fpOsd_std']:5.1f} | {outerAvgResults['tpOsd_std']:5.1f} | {outerAvgResults['tprOsd_std']:5.3f} | {outerAvgResults['fprOsd_std']:5.3f} |\n")
                    f.write("|-----------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|\n")
                    f.write("\n\n")
                    f.write("Outer Fold Results (event based analysis):\n")
                    f.write("|-----------|-------|-------------------------------------------------------|-----------------------------------------------|\n")
                    f.write("| Outer     | Best  |   Model Results                                       | OSD Algorithm Results                         |\n")
                    f.write("| Fold ID   | Inner |   np  |  tn   |  fn   |  fp   |  tp   |  tpr  |  fpr  | tnOsd | fnOsd | fpOsd | tpOsd |tprOsd |fprOsd |\n")
                    f.write("|-----------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|\n")
                    for result in outerFoldResults:
                        f.write(f"| Outer {result['outer_fold']:2d}  |   {result['best_inner_fold']:2d}  | {result['event_tp']+result['event_fn']:5d} | {result['event_tn']:5d} | {result['event_fn']:5d} | {result['event_fp']:5d} | {result['event_tp']:5d} | {result['event_tpr']:5.3f} | {result['event_fpr']:5.3f} | {result['osd_event_tn']:5d} | {result['osd_event_fn']:5d} | {result['osd_event_fp']:5d} | {result['osd_event_tp']:5d} | {result['osd_event_tpr']:5.3f} | {result['osd_event_fpr']:5.3f} |\n")
                    f.write("|-----------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|\n")
                    f.write(f"| Average   |       |       | {outerAvgResults['event_tn']:5.0f} | {outerAvgResults['event_fn']:5.0f} | {outerAvgResults['event_fp']:5.0f} | {outerAvgResults['event_tp']:5.0f} | {outerAvgResults['event_tpr']:5.3f} | {outerAvgResults['event_fpr']:5.3f} | {outerAvgResults['osd_event_tn']:5.0f} | {outerAvgResults['osd_event_fn']:5.0f} | {outerAvgResults['osd_event_fp']:5.0f} | {outerAvgResults['osd_event_tp']:5.0f} | {outerAvgResults['osd_event_tpr']:5.3f} | {outerAvgResults['osd_event_fpr']:5.3f} |\n")
                    f.write(f"| Std Dev   |       |       | {outerAvgResults['event_tn_std']:5.1f} | {outerAvgResults['event_fn_std']:5.1f} | {outerAvgResults['event_fp_std']:5.1f} | {outerAvgResults['event_tp_std']:5.1f} | {outerAvgResults['event_tpr_std']:5.3f} | {outerAvgResults['event_fpr_std']:5.3f} | {outerAvgResults['osd_event_tn_std']:5.1f} | {outerAvgResults['osd_event_fn_std']:5.1f} | {outerAvgResults['osd_event_fp_std']:5.1f} | {outerAvgResults['osd_event_tp_std']:5.1f} | {outerAvgResults['osd_event_tpr_std']:5.3f} | {outerAvgResults['osd_event_fpr_std']:5.3f} |\n")
                    f.write("|-----------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|\n")
                    
                    # Add Tonic-Clonic analysis if available
                    if 'tc_count' in outerAvgResults and outerAvgResults['tc_count'] > 0:
                        f.write("\n\nTONIC-CLONIC SEIZURES ANALYSIS (Event Based):\n")
                        f.write("|-----------|-------------------------|-------|\n")
                        f.write("| Outer     | Count | TP  | FN  | TPR   |\n")
                        f.write("|-----------|-------|-----|-----|-------|\n")
                        for result in outerFoldResults:
                            tc_count = result.get('tc_count', 0)
                            tc_tp = result.get('tc_tp', 0)
                            tc_fn = result.get('tc_fn', 0)
                            tc_tpr = result.get('tc_tpr', 0.0)
                            f.write(f"| Outer {result['outer_fold']:2d}  | {tc_count:5d}  | {tc_tp:3d} | {tc_fn:3d} | {tc_tpr:5.3f} |\n")
                        f.write("|-----------|-------|-----|-----|-------|\n")
                        f.write(f"| Average   | {outerAvgResults['tc_count']:5.0f}  | {outerAvgResults['tc_tp']:3.0f} | {outerAvgResults['tc_fn']:3.0f} | {outerAvgResults['tc_tpr']:5.3f} |\n")
                        if 'tc_count_std' in outerAvgResults:
                            f.write(f"| Std Dev   | {outerAvgResults['tc_count_std']:5.1f}  | {outerAvgResults['tc_tp_std']:3.1f} | {outerAvgResults['tc_fn_std']:3.1f} | {outerAvgResults['tc_tpr_std']:5.3f} |\n")
                        f.write("|-----------|-------|-----|-----|-------|\n")
                    
                    f.write("\n\nIMPORTANT: Report these outer fold results in publications as they represent\n")
                    f.write("unbiased estimates of model generalization on truly independent test sets.\n")
                
                with open(outerFoldJsonPath, 'w') as jf:
                    json.dump(outerFoldResults, jf, indent=2)
                
                print("\nNested k-fold outer fold summary saved to: %s" % outerFoldSummaryPath)
                print("Nested k-fold outer fold JSON saved to: %s" % outerFoldJsonPath)
                
                with open(outerFoldSummaryPath, 'r') as summary_file:
                    print("\n" + summary_file.read())
                print("="*80 + "\n")

        # Archive Results
        #import shutil
        #if (os.path.exists(testDataFname)):
    #    shutil.copy(testDataFname, outFolder)
    #if (os.path.exists(trainDataFname)):
    #    shutil.copy(trainDataFname, outFolder)
    #if (os.path.exists(valDataFname)):
    #    shutil.copy(valDataFname, outFolder)
    #if (os.path.exists(testCsvFname)):
    #    shutil.copy(testCsvFname, outFolder)
    #if (os.path.exists(trainCsvFname)):
    #    shutil.copy(trainCsvFname, outFolder)
    #if (os.path.exists(valCsvFname)):
    #    shutil.copy(valCsvFname, outFolder)
    #if (os.path.exists(trainAugCsvFname)):
    #    shutil.copy(trainAugCsvFname, outFolder)
    #if (os.path.exists(testBalCsvFname)):
    #    shutil.copy(testBalCsvFname, outFolder)
    #if (os.path.exists("%s.keras" % configObj['modelFname'])):
    #    shutil.copy("%s.keras" % configObj['modelFname'], outFolder)
        #if (os.path.exists("%s_confusion.png" % configObj['modelFname'])):
        #    shutil.copy("%s_confusion.png" % configObj['modelFname'], outFolder)
        #if (os.path.exists("%s_probabilities.png" % configObj['modelFname'])):
        #    shutil.copy("%s_probabilities.png" % configObj['modelFname'], outFolder)
        #if (os.path.exists("%s_training.png" % configObj['modelFname'])):
        #    shutil.copy("%s_training.png" % configObj['modelFname'], outFolder)
        #if (os.path.exists("%s_training2.png" % configObj['modelFname'])):
        #    shutil.copy("%s_training2.png" % configObj['modelFname'], outFolder)
        #if (os.path.exists("%s_stats.txt" % configObj['modelFname'])):
        #    shutil.copy("%s_stats.txt" % configObj['modelFname'], outFolder)
        #if (os.path.exists(args['config'])):
        #    shutil.copy(args['config'], outFolder)

        print("Finished - output in folder %s" % outFolder)
    
    finally:
        # Clean up logging and restore original stdout/stderr
        cleanup_logging(log_file)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run the Neural Network Training toolchain sequence')
    parser.add_argument('--config', default="nnConfig.json",
                        help='name of json file containing configuration')
    parser.add_argument('--kfold', default=1,
                        help='number of folds for cross-validation')
    parser.add_argument('--nestedKfold', default=1,
                        help='number of outer folds for nested k-fold validation (default 1 = no nesting)')
    parser.add_argument('--rerun', default=0,
                        help='re-run the specified run number.  If 0 then a new run is created.')
    parser.add_argument('--outDir', default="./output",
                        help='folder for training output (stored in sequential numbered folders within this folder)')
    parser.add_argument('--train', action="store_true",
                        help='Train the model')
    parser.add_argument('--test', action="store_true",
                        help='Test the model')
    parser.add_argument('--testOuter', action="store_true",
                        help='Test outer fold independent test sets (nested k-fold only)')
    parser.add_argument('--testPtl', action="store_true",
                        help='Test PyTorch Lite (.ptl) models in addition to standard models (slow, CPU-only)')
    parser.add_argument('--testPte', action="store_true",
                        help='Test ExecuTorch (.pte) models in addition to standard models (edge deployment)')
    parser.add_argument('--testPercent', type=float, default=100.0,
                        help='Percentage of events to include in testing for all model types (1-100, default: 100)')
    parser.add_argument('--pteTestPercent', dest='testPercent', type=float,
                        help='(deprecated) Use --testPercent instead; applies to all models')
    parser.add_argument('--clean', action="store_true",
                        help='Clean up output files before running')
    parser.add_argument('--debug', action="store_true",
                        help='Write debugging information to screen')
    argsNamespace = parser.parse_args()
    args = vars(argsNamespace)
    run_sequence(args)

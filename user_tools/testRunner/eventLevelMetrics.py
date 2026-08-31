"""eventLevelMetrics.py – Event-level statistics for testRunner.

Provides event-level metrics calculation similar to nnTester, with support
for sensitivity analysis treating warnings (alarmState=1) as seizure detections.

Key functions:
- calculate_event_level_metrics: Aggregate datapoint results to event-level
- compare_sensitivity_modes: Calculate metrics for both standard and sensitive modes
- save_event_results_csv: Save event-level results in nnTester-compatible format
"""
import os
import csv
import numpy as np


def calculate_event_level_metrics(results, eventIdsLst, osd, algNames, 
                                   alarm_threshold=2, debug=False):
    """Calculate event-level metrics from datapoint-level alarm states.
    
    Args:
        results: np.ndarray shape [nEvents, nAlgs, nStatus] - count of each alarm state
        eventIdsLst: List of event IDs
        osd: OsdDbConnection object with event metadata
        algNames: List of algorithm names
        alarm_threshold: Alarm state threshold (2=standard, 1=sensitive mode)
        debug: Enable debug output
        
    Returns:
        Dictionary containing:
        - event_predictions: List of dicts with eventId, true_label, predictions per algorithm
        - metrics: Dict with TP/FP/TN/FN counts and TPR/TNR per algorithm
        - event_stats_df_data: List of dicts for CSV export
    """
    nEvents = len(eventIdsLst)
    nAlgs = results.shape[1]
    
    event_predictions = []
    
    # Initialize counters for each algorithm
    NTP = np.zeros(nAlgs)
    NTN = np.zeros(nAlgs)
    NFP = np.zeros(nAlgs)
    NFN = np.zeros(nAlgs)
    
    for eventNo in range(nEvents):
        eventId = eventIdsLst[eventNo]
        eventObj = osd.getEvent(eventId, includeDatapoints=False)
        
        # Ground truth: 1 if seizure, 0 otherwise
        true_label = 1 if eventObj['type'].lower() == 'seizure' else 0
        
        event_data = {
            'eventId': str(eventId),
            'userId': eventObj.get('userId', 'N/A'),
            'type': eventObj['type'],
            'subType': eventObj.get('subType', ''),
            'dataTime': eventObj.get('dataTime', ''),
            'desc': eventObj.get('desc', ''),
            'true_label': true_label,
            'predictions': {},
            'alarm_counts': {}
        }
        
        # Calculate prediction for each algorithm
        for algNo in range(nAlgs):
            algName = algNames[algNo]
            
            # Count datapoints at each alarm state
            alarm_counts = {
                'ok': int(results[eventNo][algNo][0]),
                'warning': int(results[eventNo][algNo][1]),
                'alarm': int(results[eventNo][algNo][2])
            }
            
            # Event is predicted as seizure if ANY datapoint >= threshold
            if alarm_threshold == 2:
                # Standard mode: only ALARM states count
                predicted = 1 if alarm_counts['alarm'] > 0 else 0
            else:
                # Sensitive mode: WARNING or ALARM counts
                predicted = 1 if (alarm_counts['warning'] > 0 or alarm_counts['alarm'] > 0) else 0
            
            event_data['predictions'][algName] = predicted
            event_data['alarm_counts'][algName] = alarm_counts
            
            # Update confusion matrix counts
            if true_label == 1:  # Actual seizure
                if predicted == 1:
                    NTP[algNo] += 1
                else:
                    NFN[algNo] += 1
            else:  # Actual non-seizure
                if predicted == 1:
                    NFP[algNo] += 1
                else:
                    NTN[algNo] += 1
        
        event_predictions.append(event_data)
    
    # Calculate metrics for each algorithm
    metrics = {}
    for algNo in range(nAlgs):
        algName = algNames[algNo]
        
        tp, fp, tn, fn = NTP[algNo], NFP[algNo], NTN[algNo], NFN[algNo]
        
        # Calculate rates
        tpr = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        tnr = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        fpr = 1.0 - tnr
        
        metrics[algName] = {
            'TP': int(tp),
            'FP': int(fp),
            'TN': int(tn),
            'FN': int(fn),
            'TPR': tpr,
            'TNR': tnr,
            'FPR': fpr,
            'sensitivity': tpr,
            'specificity': tnr
        }
        
        if debug:
            print(f"Event-level metrics for {algName} (threshold={alarm_threshold}):")
            print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
            print(f"  TPR={tpr:.3f}, TNR={tnr:.3f}, FPR={fpr:.3f}")
    
    return {
        'event_predictions': event_predictions,
        'metrics': metrics,
        'alarm_threshold': alarm_threshold
    }


def compare_sensitivity_modes(results, eventIdsLst, osd, algNames, debug=False):
    """Calculate event-level metrics for both standard and sensitive modes.
    
    Standard mode: alarmState=2 (ALARM) counts as seizure detection
    Sensitive mode: alarmState>=1 (WARNING or ALARM) counts as seizure detection
    
    Args:
        results: np.ndarray shape [nEvents, nAlgs, nStatus]
        eventIdsLst: List of event IDs
        osd: OsdDbConnection object
        algNames: List of algorithm names
        debug: Enable debug output
        
    Returns:
        Dictionary with 'standard' and 'sensitive' keys containing metrics
    """
    standard_results = calculate_event_level_metrics(
        results, eventIdsLst, osd, algNames, alarm_threshold=2, debug=debug)
    
    sensitive_results = calculate_event_level_metrics(
        results, eventIdsLst, osd, algNames, alarm_threshold=1, debug=debug)
    
    return {
        'standard': standard_results,
        'sensitive': sensitive_results
    }


def save_event_results_csv(outDir, event_predictions, algNames, mode_name='standard'):
    """Save event-level results to CSV file in nnTester-compatible format.
    
    Args:
        outDir: Output directory path
        event_predictions: List of event prediction dicts from calculate_event_level_metrics
        algNames: List of algorithm names
        mode_name: String identifier for the mode (e.g., 'standard', 'sensitive')
    """
    csv_path = os.path.join(outDir, f'eventLevel_{mode_name}.csv')
    
    with open(csv_path, 'w', newline='') as f:
        # Build header
        base_cols = ['EventID', 'UserID', 'Type', 'SubType', 'DataTime', 'TrueLabel']
        alg_cols = [f'{algName}_Pred' for algName in algNames]
        alg_alarm_cols = []
        for algName in algNames:
            alg_alarm_cols.extend([
                f'{algName}_AlarmCount',
                f'{algName}_WarnCount',
                f'{algName}_OkCount'
            ])
        desc_col = ['Description']
        
        header = base_cols + alg_cols + alg_alarm_cols + desc_col
        
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        
        # Write each event
        for event in event_predictions:
            row = {
                'EventID': event['eventId'],
                'UserID': event['userId'],
                'Type': event['type'],
                'SubType': event['subType'],
                'DataTime': event['dataTime'],
                'TrueLabel': event['true_label'],
                'Description': event['desc']
            }
            
            # Add predictions for each algorithm
            for algName in algNames:
                row[f'{algName}_Pred'] = event['predictions'][algName]
                row[f'{algName}_AlarmCount'] = event['alarm_counts'][algName]['alarm']
                row[f'{algName}_WarnCount'] = event['alarm_counts'][algName]['warning']
                row[f'{algName}_OkCount'] = event['alarm_counts'][algName]['ok']
            
            writer.writerow(row)
    
    print(f"Event-level results ({mode_name} mode) saved to {csv_path}")
    return csv_path


def save_sensitivity_comparison(outDir, comparison_results, algNames):
    """Save sensitivity mode comparison to text file.
    
    Args:
        outDir: Output directory
        comparison_results: Dict with 'standard' and 'sensitive' results
        algNames: List of algorithm names
    """
    summary_path = os.path.join(outDir, 'eventLevel_comparison.txt')
    
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EVENT-LEVEL METRICS - SENSITIVITY MODE COMPARISON\n")
        f.write("="*80 + "\n\n")
        
        f.write("Standard Mode: alarmState=2 (ALARM) counts as seizure detection\n")
        f.write("Sensitive Mode: alarmState>=1 (WARNING or ALARM) counts as seizure detection\n\n")
        
        for algName in algNames:
            f.write("-"*80 + "\n")
            f.write(f"Algorithm: {algName}\n")
            f.write("-"*80 + "\n\n")
            
            std_metrics = comparison_results['standard']['metrics'][algName]
            sen_metrics = comparison_results['sensitive']['metrics'][algName]
            
            # Standard mode
            f.write("STANDARD MODE (alarmState=2 only):\n")
            f.write(f"  True Positives (TP):  {std_metrics['TP']}\n")
            f.write(f"  False Positives (FP): {std_metrics['FP']}\n")
            f.write(f"  True Negatives (TN):  {std_metrics['TN']}\n")
            f.write(f"  False Negatives (FN): {std_metrics['FN']}\n")
            f.write(f"  Sensitivity (TPR):    {std_metrics['TPR']:.3f} ({std_metrics['TPR']*100:.1f}%)\n")
            f.write(f"  Specificity (TNR):    {std_metrics['TNR']:.3f} ({std_metrics['TNR']*100:.1f}%)\n")
            f.write(f"  False Alarm Rate:     {std_metrics['FPR']:.3f} ({std_metrics['FPR']*100:.1f}%)\n\n")
            
            # Sensitive mode
            f.write("SENSITIVE MODE (alarmState>=1, warnings count):\n")
            f.write(f"  True Positives (TP):  {sen_metrics['TP']}\n")
            f.write(f"  False Positives (FP): {sen_metrics['FP']}\n")
            f.write(f"  True Negatives (TN):  {sen_metrics['TN']}\n")
            f.write(f"  False Negatives (FN): {sen_metrics['FN']}\n")
            f.write(f"  Sensitivity (TPR):    {sen_metrics['TPR']:.3f} ({sen_metrics['TPR']*100:.1f}%)\n")
            f.write(f"  Specificity (TNR):    {sen_metrics['TNR']:.3f} ({sen_metrics['TNR']*100:.1f}%)\n")
            f.write(f"  False Alarm Rate:     {sen_metrics['FPR']:.3f} ({sen_metrics['FPR']*100:.1f}%)\n\n")
            
            # Comparison
            tpr_diff = sen_metrics['TPR'] - std_metrics['TPR']
            fpr_diff = sen_metrics['FPR'] - std_metrics['FPR']
            
            f.write("SENSITIVITY COMPARISON:\n")
            f.write(f"  TPR Increase: {tpr_diff:+.3f} ({tpr_diff*100:+.1f}%)\n")
            f.write(f"  FPR Increase: {fpr_diff:+.3f} ({fpr_diff*100:+.1f}%)\n")
            f.write(f"  Additional Seizures Detected: {sen_metrics['TP'] - std_metrics['TP']}\n")
            f.write(f"  Additional False Alarms: {sen_metrics['FP'] - std_metrics['FP']}\n\n")
        
        f.write("="*80 + "\n")
    
    print(f"Sensitivity comparison saved to {summary_path}")
    return summary_path


def calculate_metrics_by_seizure_type(event_predictions, algNames, debug=False):
    """Calculate metrics for each seizure type (tonic-clonic, other, aura, etc.).
    
    Args:
        event_predictions: List of event prediction dicts from calculate_event_level_metrics
        algNames: List of algorithm names
        debug: Enable debug output
        
    Returns:
        Dictionary mapping seizure subtype → metrics dict for each algorithm
    """
    # Collect seizure subtypes
    seizure_subtypes = {}
    
    for event in event_predictions:
        if event['true_label'] == 1:  # Only seizure events
            subtype = event.get('subType', '').strip()
            if not subtype:
                subtype = '(no subtype)'
            
            if subtype not in seizure_subtypes:
                seizure_subtypes[subtype] = []
            seizure_subtypes[subtype].append(event)
    
    # Calculate metrics for each subtype
    metrics_by_type = {}
    nAlgs = len(algNames)
    
    for subtype in sorted(seizure_subtypes.keys()):
        events = seizure_subtypes[subtype]
        
        # Initialize counters
        TP = np.zeros(nAlgs)
        FN = np.zeros(nAlgs)
        
        for event in events:
            for algNo, algName in enumerate(algNames):
                predicted = event['predictions'][algName]
                true_label = event['true_label']
                
                if predicted == 1:
                    TP[algNo] += 1
                else:
                    FN[algNo] += 1
        
        # Calculate TPR for this seizure type
        metrics_by_type[subtype] = {
            'count': len(events),
            'metrics': {}
        }
        
        for algNo, algName in enumerate(algNames):
            total = int(TP[algNo] + FN[algNo])
            tpr = (TP[algNo] / total) if total > 0 else 0.0
            
            metrics_by_type[subtype]['metrics'][algName] = {
                'TP': int(TP[algNo]),
                'FN': int(FN[algNo]),
                'total': total,
                'TPR': tpr
            }
            
            if debug:
                print(f"  {algName}: TP={int(TP[algNo])}, FN={int(FN[algNo])}, TPR={tpr:.3f}")
    
    return metrics_by_type


def save_metrics_by_seizure_type(outDir, event_predictions, algNames, mode_name='standard'):
    """Save metrics breakdown by seizure type to a text file.
    
    Args:
        outDir: Output directory
        event_predictions: List of event prediction dicts
        algNames: List of algorithm names
        mode_name: String identifier for the mode (e.g., 'standard', 'sensitive')
    """
    metrics_by_type = calculate_metrics_by_seizure_type(event_predictions, algNames)
    
    summary_path = os.path.join(outDir, f'eventLevel_bySeizureType_{mode_name}.txt')
    
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"EVENT-LEVEL METRICS BY SEIZURE TYPE ({mode_name.upper()} MODE)\n")
        f.write("="*80 + "\n\n")
        
        # First, calculate overall seizure metrics (not counting non-seizures)
        f.write("ALL SEIZURES COMBINED:\n")
        f.write("-"*80 + "\n")
        
        all_seizures = [e for e in event_predictions if e['true_label'] == 1]
        seizure_count = len(all_seizures)
        
        f.write(f"Total seizure events: {seizure_count}\n\n")
        
        for algName in algNames:
            tp_total = sum(1 for e in all_seizures if e['predictions'][algName] == 1)
            tpr_overall = (tp_total / seizure_count) if seizure_count > 0 else 0.0
            
            f.write(f"{algName}:\n")
            f.write(f"  TP: {tp_total}/{seizure_count}\n")
            f.write(f"  TPR (Sensitivity): {tpr_overall:.3f} ({tpr_overall*100:.1f}%)\n\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("BREAKDOWN BY SEIZURE TYPE:\n")
        f.write("="*80 + "\n\n")
        
        # Show metrics for each seizure type
        for subtype in sorted(metrics_by_type.keys()):
            type_data = metrics_by_type[subtype]
            count = type_data['count']
            
            f.write("-"*80 + "\n")
            f.write(f"Seizure Type: {subtype}\n")
            f.write(f"Count: {count} events\n")
            f.write("-"*80 + "\n\n")
            
            for algName in algNames:
                metrics = type_data['metrics'][algName]
                tp = metrics['TP']
                fn = metrics['FN']
                total = metrics['total']
                tpr = metrics['TPR']
                
                f.write(f"{algName}:\n")
                f.write(f"  Detected: {tp}/{total}\n")
                f.write(f"  Missed: {fn}/{total}\n")
                f.write(f"  TPR: {tpr:.3f} ({tpr*100:.1f}%)\n\n")
        
        f.write("="*80 + "\n")
    
    print(f"Seizure type breakdown ({mode_name} mode) saved to {summary_path}")
    return summary_path


def generate_event_level_report(outDir, results, resultsStrArr, eventIdsLst, 
                                osd, algNames, config=None, debug=False):
    """Generate complete event-level analysis with sensitivity comparison.
    
    This is the main entry point called from results.py.
    
    Args:
        outDir: Output directory
        results: np.ndarray shape [nEvents, nAlgs, nStatus]
        resultsStrArr: Results string array (for reference)
        eventIdsLst: List of event IDs
        osd: OsdDbConnection object
        algNames: List of algorithm names
        config: Optional config dict with eventLevelMetrics settings
        debug: Enable debug output
    """
    print("\n" + "="*80)
    print("GENERATING EVENT-LEVEL METRICS")
    print("="*80)
    
    # Parse config
    if config is None:
        config = {}
    
    enabled = config.get('enabled', True)
    compare_modes = config.get('compareSensitivityModes', True)
    treat_warnings_as_seizures = config.get('treatWarningsAsSeizures', False)
    
    if not enabled:
        print("Event-level metrics disabled in configuration")
        return
    
    if compare_modes:
        # Generate both modes and comparison
        print("Calculating metrics for both standard and sensitive modes...")
        comparison = compare_sensitivity_modes(results, eventIdsLst, osd, algNames, debug=debug)
        
        # Save both CSV files
        save_event_results_csv(
            outDir, 
            comparison['standard']['event_predictions'], 
            algNames, 
            mode_name='standard'
        )
        save_event_results_csv(
            outDir, 
            comparison['sensitive']['event_predictions'], 
            algNames, 
            mode_name='sensitive'
        )
        
        # Save comparison summary
        save_sensitivity_comparison(outDir, comparison, algNames)
        
        # Save seizure type breakdown for both modes
        print("\nGenerating seizure type breakdown...")
        save_metrics_by_seizure_type(
            outDir,
            comparison['standard']['event_predictions'],
            algNames,
            mode_name='standard'
        )
        save_metrics_by_seizure_type(
            outDir,
            comparison['sensitive']['event_predictions'],
            algNames,
            mode_name='sensitive'
        )
        
    else:
        # Generate only the requested mode
        threshold = 1 if treat_warnings_as_seizures else 2
        mode_name = 'sensitive' if treat_warnings_as_seizures else 'standard'
        
        print(f"Calculating event-level metrics ({mode_name} mode)...")
        results_dict = calculate_event_level_metrics(
            results, eventIdsLst, osd, algNames, 
            alarm_threshold=threshold, debug=debug
        )
        
        save_event_results_csv(
            outDir, 
            results_dict['event_predictions'], 
            algNames, 
            mode_name=mode_name
        )
        
        # Save seizure type breakdown
        print("\nGenerating seizure type breakdown...")
        save_metrics_by_seizure_type(
            outDir,
            results_dict['event_predictions'],
            algNames,
            mode_name=mode_name
        )
        
        # Print summary to console
        print("\nEvent-Level Metrics Summary:")
        print("-"*80)
        for algName, metrics in results_dict['metrics'].items():
            print(f"{algName}:")
            print(f"  TP={metrics['TP']}, FP={metrics['FP']}, TN={metrics['TN']}, FN={metrics['FN']}")
            print(f"  Sensitivity={metrics['TPR']:.3f}, Specificity={metrics['TNR']:.3f}")
    
    print("="*80)
    print("EVENT-LEVEL METRICS GENERATION COMPLETE")
    print("="*80 + "\n")

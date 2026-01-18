#!/usr/bin/env python3
"""
Post-processing script to analyze nnTester event results CSV files.

Usage:
    python3 analyzeEventResults.py <path_to_event_results.csv> [--output <output_dir>] [--alldata-json <path_to_allData.json>]

This script generates:
    - Analysis of TPR and FAR by user
    - Analysis of TPR by seizure type
    - Analysis of event types/subtypes contributing to false alarms
    - Table of false negative results with details from allData.json
    - Report with figures saved as PDF

The --alldata-json parameter is optional. If not provided, the script will look for allData.json
in the same directory as the event results CSV file.
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_event_results(csv_path):
    """Load event results CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Event results CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} events from {csv_path}")
    print(f"Columns: {list(df.columns)}")
    return df


def load_alldata_json(json_path):
    """Load allData.json file containing event metadata."""
    if not os.path.exists(json_path):
        print(f"Warning: allData.json not found: {json_path}")
        return None
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded allData.json with {len(data)} events")
        return data
    except Exception as e:
        print(f"Error loading allData.json: {e}")
        return None


def extract_false_negatives_details(df, alldata_list):
    """Extract details for false negative results from allData.json.
    
    Returns a DataFrame with columns: EventID, UserID, DateTime, Type, SubType, Description
    from allData.json, enriching the base false negatives data.
    If alldata_list is None, returns empty DataFrame (CSV data will be used as fallback).
    """
    if alldata_list is None:
        return pd.DataFrame()
    
    # Get false negatives from results
    fn_df = df[(df['ActualLabel'] == 1) & (df['ModelPrediction'] == 0)].copy()
    
    if len(fn_df) == 0:
        return pd.DataFrame()
    
    # Create a mapping from event IDs to metadata
    event_metadata = {}
    for event in alldata_list:
        event_id = event.get('id') or (event.get('datapoints', [{}])[0].get('eventId') if event.get('datapoints') else None)
        if event_id:
            # Try to convert to int, but handle cases where it might be a string or mixed
            try:
                event_id_int = int(event_id)
            except (ValueError, TypeError):
                # If conversion fails, try to extract numeric part
                try:
                    event_id_int = int(str(event_id).rstrip('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'))
                except (ValueError, TypeError):
                    continue
            
            event_metadata[event_id_int] = {
                'dataTime': event.get('dataTime', ''),
                'type': event.get('type', ''),
                'subType': event.get('subType', ''),
                'userId': event.get('userId', ''),
                'desc': event.get('desc', '')
            }
    
    # Build false negatives details with enhanced metadata from allData.json
    fn_details = []
    for _, row in fn_df.iterrows():
        try:
            event_id = int(row['EventID']) if pd.notna(row['EventID']) else None
        except (ValueError, TypeError):
            continue
        
        if event_id and event_id in event_metadata:
            metadata = event_metadata[event_id]
            fn_details.append({
                'EventID': event_id,
                'UserID': metadata['userId'],
                'DateTime': metadata['dataTime'],
                'Type': metadata['type'],
                'SubType': metadata['subType'],
                'Description': metadata['desc']
            })
    
    return pd.DataFrame(fn_details)


def analyze_by_user(df, seizure_threshold=3, far_threshold=100):
    """Analyze TPR and FAR by user, grouping users with < seizure_threshold seizures and < far_threshold non-seizures as 'other'."""
    print("\n" + "="*80)
    print("ANALYSIS BY USER")
    print(f"Seizure threshold: {seizure_threshold}, FAR threshold: {far_threshold}")
    print("="*80)
    
    # Get seizure events only
    seizure_df = df[df['ActualLabel'] == 1].copy()
    
    # Count seizures per user
    user_seizure_counts = seizure_df['UserID'].value_counts()
    print(f"\nTotal users with seizures: {len(user_seizure_counts)}")
    print(f"Users with >= {seizure_threshold} seizures: {(user_seizure_counts >= seizure_threshold).sum()}")
    print(f"Users with < {seizure_threshold} seizures: {(user_seizure_counts < seizure_threshold).sum()}")
    
    # Group users with < seizure_threshold seizures as 'other'
    def group_users(user_id):
        if pd.isna(user_id):
            return 'Unknown'
        if user_seizure_counts.get(user_id, 0) < seizure_threshold:
            return 'Other'
        return str(user_id)
    
    seizure_df['UserGroup'] = seizure_df['UserID'].apply(group_users)
    
    # Calculate metrics by user
    user_metrics = []
    for user in sorted(seizure_df['UserGroup'].unique()):
        user_data = seizure_df[seizure_df['UserGroup'] == user]
        tp = (user_data['ModelPrediction'] == 1).sum()
        fn = (user_data['ModelPrediction'] == 0).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        user_metrics.append({
            'User': user,
            'Seizures': len(user_data),
            'TP': tp,
            'FN': fn,
            'TPR': tpr
        })
    
    user_metrics_df = pd.DataFrame(user_metrics).sort_values('TPR', ascending=False)
    print("\nTPR by User:")
    print(user_metrics_df.to_string(index=False))
    
    # Calculate FAR (False Alarm Rate) by user
    non_seizure_df = df[df['ActualLabel'] == 0].copy()
    
    # Count non-seizures per user for FAR grouping
    user_non_seizure_counts = non_seizure_df['UserID'].value_counts()
    print(f"Users with >= {far_threshold} non-seizure events: {(user_non_seizure_counts >= far_threshold).sum()}")
    print(f"Users with < {far_threshold} non-seizure events: {(user_non_seizure_counts < far_threshold).sum()}")
    
    # Group users with < far_threshold non-seizure events as 'other'
    def group_users_far(user_id):
        if pd.isna(user_id):
            return 'Unknown'
        if user_non_seizure_counts.get(user_id, 0) < far_threshold:
            return 'Other'
        return str(user_id)
    
    non_seizure_df['UserGroup'] = non_seizure_df['UserID'].apply(group_users_far)
    
    far_metrics = []
    for user in sorted(non_seizure_df['UserGroup'].unique()):
        user_data = non_seizure_df[non_seizure_df['UserGroup'] == user]
        fp = (user_data['ModelPrediction'] == 1).sum()
        tn = (user_data['ModelPrediction'] == 0).sum()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        far_metrics.append({
            'User': user,
            'NonSeizures': len(user_data),
            'FP': fp,
            'TN': tn,
            'FAR': far
        })
    
    far_metrics_df = pd.DataFrame(far_metrics).sort_values('FAR', ascending=False)
    print("\nFAR (False Alarm Rate) by User:")
    print(far_metrics_df.to_string(index=False))
    
    return seizure_df, user_metrics_df, far_metrics_df


def analyze_by_seizure_type(df):
    """Analyze TPR by seizure type."""
    print("\n" + "="*80)
    print("ANALYSIS BY SEIZURE TYPE")
    print("="*80)
    
    seizure_df = df[df['ActualLabel'] == 1].copy()
    
    # Get all seizure types
    print(f"\nUnique seizure types: {seizure_df['Type'].unique()}")
    print(f"Unique seizure subtypes: {seizure_df['SubType'].unique()}")
    
    # Analyze by subtype
    subtype_metrics = []
    # Filter out NaN values before sorting to avoid TypeError
    subtypes = [st for st in seizure_df['SubType'].unique() if pd.notna(st)]
    for subtype in sorted(subtypes):
        subtype_data = seizure_df[seizure_df['SubType'] == subtype]
        tp = (subtype_data['ModelPrediction'] == 1).sum()
        fn = (subtype_data['ModelPrediction'] == 0).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        subtype_metrics.append({
            'SubType': str(subtype),
            'Count': len(subtype_data),
            'TP': tp,
            'FN': fn,
            'TPR': tpr
        })
    
    subtype_metrics_df = pd.DataFrame(subtype_metrics).sort_values('TPR', ascending=False)
    print("\nTPR by Seizure SubType:")
    print(subtype_metrics_df.to_string(index=False))
    
    # Analyze combined vs Tonic-Clonic
    combined_df = seizure_df[seizure_df['SubType'].str.contains('Tonic-Clonic|Combined', case=False, na=False)]
    tc_tp = (combined_df['ModelPrediction'] == 1).sum()
    tc_fn = (combined_df['ModelPrediction'] == 0).sum()
    tc_tpr = tc_tp / (tc_tp + tc_fn) if (tc_tp + tc_fn) > 0 else 0
    
    print(f"\nTonic-Clonic/Combined Seizures: {len(combined_df)}")
    print(f"  TP: {tc_tp}, FN: {tc_fn}, TPR: {tc_tpr:.3f}")
    
    return seizure_df, subtype_metrics_df


def analyze_tonic_clonic_seizures(df):
    """Analyze TPR specifically for Tonic-Clonic seizures (Type=Seizure, Sub-Type=Tonic-Clonic).
    
    Returns:
        tc_metrics_dict: Dictionary with keys: Count, TP, FN, TPR, FPR, TN, FP
    """
    print("\n" + "="*80)
    print("ANALYSIS OF TONIC-CLONIC SEIZURES ONLY")
    print("="*80)
    
    # Filter to only actual seizures
    seizure_df = df[df['ActualLabel'] == 1].copy()
    
    # Filter to only Tonic-Clonic type (Type=Seizure, Sub-Type=Tonic-Clonic)
    tc_df = seizure_df[(seizure_df['Type'] == 'Seizure') & 
                       (seizure_df['SubType'] == 'Tonic-Clonic')].copy()
    
    tp = (tc_df['ModelPrediction'] == 1).sum()
    fn = (tc_df['ModelPrediction'] == 0).sum()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Also get FP/FN stats from non-seizure events (for false alarm rate context)
    non_seizure_df = df[df['ActualLabel'] == 0].copy()
    fp = (non_seizure_df['ModelPrediction'] == 1).sum()
    tn = (non_seizure_df['ModelPrediction'] == 0).sum()
    
    metrics = {
        'Count': len(tc_df),
        'TP': int(tp),
        'FN': int(fn),
        'TPR': float(tpr),
        'TP_FN_Total': int(tp + fn),
        'FP': int(fp),
        'TN': int(tn),
        'FPR': float(fp / (fp + tn)) if (fp + tn) > 0 else 0
    }
    
    print(f"\nTonic-Clonic Seizures (Type=Seizure, Sub-Type=Tonic-Clonic):")
    print(f"  Total Count: {metrics['Count']}")
    print(f"  TP: {metrics['TP']}, FN: {metrics['FN']}")
    print(f"  TPR (Sensitivity): {metrics['TPR']:.3f}")
    print(f"  False Positives on non-seizure data: {metrics['FP']}")
    print(f"  FPR: {metrics['FPR']:.3f}")
    
    return metrics


def analyze_false_alarms(df):
    """Analyze event types/subtypes contributing to false alarms."""
    print("\n" + "="*80)
    print("ANALYSIS OF FALSE ALARMS")
    print("="*80)
    
    non_seizure_df = df[df['ActualLabel'] == 0].copy()
    false_alarms_df = non_seizure_df[non_seizure_df['ModelPrediction'] == 1].copy()
    
    print(f"\nTotal non-seizure events: {len(non_seizure_df)}")
    print(f"False alarms (predicted as seizure): {len(false_alarms_df)}")
    print(f"Overall FAR: {len(false_alarms_df) / len(non_seizure_df):.3f}" if len(non_seizure_df) > 0 else "N/A")
    
    # Count events per subtype to group small ones as 'other'
    subtype_counts = non_seizure_df['SubType'].value_counts()
    
    # Analyze by type
    if len(false_alarms_df) > 0:
        print("\nFalse Alarms by Type:")
        type_counts = false_alarms_df['Type'].value_counts()
        print(type_counts)
        
        # Analyze by subtype
        print("\nFalse Alarms by SubType:")
        subtype_counts_fa = false_alarms_df['SubType'].value_counts()
        print(subtype_counts_fa)
        
        # Calculate FAR by subtype, grouping small ones as 'other'
        print("\nFAR by SubType (for non-seizure events, grouping < 5 events as 'other'):")
        
        # Create a mapping function
        def group_subtypes(subtype):
            if pd.isna(subtype):
                return 'Unknown'
            if subtype_counts.get(subtype, 0) < 5:
                return 'Other'
            return str(subtype)
        
        # Apply grouping
        non_seizure_df_grouped = non_seizure_df.copy()
        non_seizure_df_grouped['SubType_Grouped'] = non_seizure_df_grouped['SubType'].apply(group_subtypes)
        
        far_by_subtype = []
        for subtype in sorted(non_seizure_df_grouped['SubType_Grouped'].unique()):
            subtype_data = non_seizure_df_grouped[non_seizure_df_grouped['SubType_Grouped'] == subtype]
            fp = (subtype_data['ModelPrediction'] == 1).sum()
            tn = (subtype_data['ModelPrediction'] == 0).sum()
            far = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            far_by_subtype.append({
                'SubType': subtype,
                'Count': len(subtype_data),
                'FP': fp,
                'TN': tn,
                'FAR': far
            })
        
        far_by_subtype_df = pd.DataFrame(far_by_subtype).sort_values('FAR', ascending=False)
        print(far_by_subtype_df.to_string(index=False))
    
    return false_alarms_df, far_by_subtype_df if len(false_alarms_df) > 0 else pd.DataFrame()


def generate_plots(df, seizure_df, user_metrics_df, far_metrics_df, subtype_metrics_df, false_alarms_df, false_negatives_df, fn_enhanced_details_df, output_dir, seizure_threshold=3, far_threshold=100):
    """Generate plots and save to PDF."""
    pdf_path = os.path.join(output_dir, 'event_analysis_report.pdf')
    
    with PdfPages(pdf_path) as pdf:
        # Plot 1: TPR by User
        if len(user_metrics_df) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            users = user_metrics_df['User']
            tprs = user_metrics_df['TPR']
            counts = user_metrics_df['Seizures']
            colors = ['#2ecc71' if tpr >= 0.8 else '#f39c12' if tpr >= 0.6 else '#e74c3c' for tpr in tprs]
            
            ax.barh(users, tprs, color=colors)
            ax.set_xlabel('TPR (True Positive Rate)', fontsize=12, fontweight='bold')
            ax.set_title('TPR by User (Seizure Events)', fontsize=14, fontweight='bold')
            ax.set_xlim(0, 1)
            
            # Add value labels
            for i, (user, tpr, count) in enumerate(zip(users, tprs, counts)):
                ax.text(tpr + 0.02, i, f'{tpr:.2%} (n={count})', va='center', fontsize=10)
            
            ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5, label='80% threshold')
            ax.legend()
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

        # Plot 1b: TPR by User (Tonic-Clonic SubType)
        # Compute user metrics restricted to Tonic-Clonic seizures
        if len(seizure_df) > 0:
            # Filter to Tonic-Clonic subtype (case-insensitive, handle NaN)
            tc_seizure_df = seizure_df[seizure_df['SubType'].str.contains('Tonic-Clonic', case=False, na=False)].copy()
            if len(tc_seizure_df) > 0:
                # Group users with < seizure_threshold total seizures as 'Other' (consistent with overall)
                user_seizure_counts_overall = seizure_df['UserID'].value_counts()
                def group_users_tc(user_id):
                    if pd.isna(user_id):
                        return 'Unknown'
                    if user_seizure_counts_overall.get(user_id, 0) < seizure_threshold:
                        return 'Other'
                    return str(user_id)
                tc_seizure_df['UserGroup'] = tc_seizure_df['UserID'].apply(group_users_tc)

                # Calculate TPR by user for TC only
                tc_user_metrics = []
                for user in sorted(tc_seizure_df['UserGroup'].unique()):
                    user_data = tc_seizure_df[tc_seizure_df['UserGroup'] == user]
                    tp = (user_data['ModelPrediction'] == 1).sum()
                    fn = (user_data['ModelPrediction'] == 0).sum()
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    tc_user_metrics.append({
                        'User': user,
                        'Seizures': len(user_data),
                        'TP': tp,
                        'FN': fn,
                        'TPR': tpr
                    })

                tc_user_metrics_df = pd.DataFrame(tc_user_metrics).sort_values('TPR', ascending=False)

                # Plot TPR by user for TC seizures
                if len(tc_user_metrics_df) > 0:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    users_tc = tc_user_metrics_df['User']
                    tprs_tc = tc_user_metrics_df['TPR']
                    counts_tc = tc_user_metrics_df['Seizures']
                    colors_tc = ['#2ecc71' if tpr >= 0.8 else '#f39c12' if tpr >= 0.6 else '#e74c3c' for tpr in tprs_tc]

                    ax.barh(users_tc, tprs_tc, color=colors_tc)
                    ax.set_xlabel('TPR (True Positive Rate)', fontsize=12, fontweight='bold')
                    ax.set_title('TPR by User (Tonic-Clonic Seizures)', fontsize=14, fontweight='bold')
                    ax.set_xlim(0, 1)

                    # Add value labels
                    for i, (user, tpr, count) in enumerate(zip(users_tc, tprs_tc, counts_tc)):
                        ax.text(tpr + 0.02, i, f'{tpr:.2%} (n={count})', va='center', fontsize=10)

                    ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5, label='80% threshold')
                    ax.legend()
                    plt.tight_layout()
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
        
        # Plot 2: FAR by User
        if len(far_metrics_df) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            users = far_metrics_df['User']
            fars = far_metrics_df['FAR']
            counts = far_metrics_df['NonSeizures']
            colors = ['#e74c3c' if far > 0.05 else '#f39c12' if far > 0.01 else '#2ecc71' for far in fars]
            
            ax.barh(users, fars, color=colors)
            ax.set_xlabel('FAR (False Alarm Rate)', fontsize=12, fontweight='bold')
            ax.set_title('FAR by User (Non-Seizure Events)', fontsize=14, fontweight='bold')
            
            # Add value labels
            for i, (user, far, count) in enumerate(zip(users, fars, counts)):
                ax.text(far + 0.002, i, f'{far:.2%} (n={count})', va='center', fontsize=10)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Plot 3: TPR by Seizure SubType
        if len(subtype_metrics_df) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            subtypes = subtype_metrics_df['SubType']
            tprs = subtype_metrics_df['TPR']
            counts = subtype_metrics_df['Count']
            
            bars = ax.barh(subtypes, tprs, color='steelblue')
            ax.set_xlabel('TPR (True Positive Rate)', fontsize=12, fontweight='bold')
            ax.set_title('TPR by Seizure SubType', fontsize=14, fontweight='bold')
            ax.set_xlim(0, 1)
            
            # Add value labels with counts
            for i, (subtype, tpr, count) in enumerate(zip(subtypes, tprs, counts)):
                ax.text(tpr + 0.02, i, f'{tpr:.2%} (n={count})', va='center', fontsize=10)
            
            ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5, label='80% threshold')
            ax.legend()
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Plot 4: False Alarms by SubType
        if len(false_alarms_df) > 0:
            non_seizure_df_plot = df[df['ActualLabel'] == 0].copy()
            
            # Count events per subtype
            subtype_counts_plot = non_seizure_df_plot['SubType'].value_counts()
            
            # Group subtypes with < 5 events as 'other'
            def group_subtypes_plot(subtype):
                if pd.isna(subtype):
                    return 'Unknown'
                if subtype_counts_plot.get(subtype, 0) < 5:
                    return 'Other'
                return str(subtype)
            
            non_seizure_df_plot['SubType_Grouped'] = non_seizure_df_plot['SubType'].apply(group_subtypes_plot)
            
            # Get FAR by grouped subtype
            subtype_fa_counts = []
            for subtype in sorted(non_seizure_df_plot['SubType_Grouped'].unique()):
                subtype_data = non_seizure_df_plot[non_seizure_df_plot['SubType_Grouped'] == subtype]
                fp = (subtype_data['ModelPrediction'] == 1).sum()
                tn = (subtype_data['ModelPrediction'] == 0).sum()
                far = fp / (fp + tn) if (fp + tn) > 0 else 0
                subtype_fa_counts.append((str(subtype), far, len(subtype_data)))
            
            subtype_fa_counts.sort(key=lambda x: x[1], reverse=True)
            subtypes_fa = [x[0] for x in subtype_fa_counts]
            fars_fa = [x[1] for x in subtype_fa_counts]
            counts_fa = [x[2] for x in subtype_fa_counts]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            colors = ['#e74c3c' if far > 0.05 else '#f39c12' if far > 0.01 else '#2ecc71' for far in fars_fa]
            bars = ax.barh(subtypes_fa, fars_fa, color=colors)
            ax.set_xlabel('FAR (False Alarm Rate)', fontsize=12, fontweight='bold')
            ax.set_title('FAR by Event SubType (Non-Seizure Events)', fontsize=14, fontweight='bold')
            
            # Add value labels
            for i, (subtype, far, count) in enumerate(zip(subtypes_fa, fars_fa, counts_fa)):
                ax.text(far + 0.002, i, f'{far:.2%} (n={count})', va='center', fontsize=10)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Plot 5: Confusion Matrix Summary
        fig, ax = plt.subplots(figsize=(10, 8))
        tp = (df['ModelPrediction'] == 1) & (df['ActualLabel'] == 1)
        fp = (df['ModelPrediction'] == 1) & (df['ActualLabel'] == 0)
        fn = (df['ModelPrediction'] == 0) & (df['ActualLabel'] == 1)
        tn = (df['ModelPrediction'] == 0) & (df['ActualLabel'] == 0)
        
        cm = np.array([[tn.sum(), fp.sum()], [fn.sum(), tp.sum()]])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'},
                   xticklabels=['Predicted Non-Seizure', 'Predicted Seizure'],
                   yticklabels=['Actual Non-Seizure', 'Actual Seizure'])
        ax.set_title('Overall Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 6: False Negatives Table (from CSV data) - comes first because it's usually shorter
        if len(false_negatives_df) > 0:
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.axis('tight')
            ax.axis('off')
            
            # Prepare table data - show first 30 rows
            table_data = false_negatives_df[['EventID', 'UserID', 'Type', 'SubType', 'Description']].copy()
            table_data = table_data.head(30)
            table_data.columns = ['Event ID', 'User ID', 'Type', 'SubType', 'Description']
            
            # Wrap descriptions to prevent them from running off page - using textwrap
            import textwrap
            table_data['Description'] = table_data['Description'].fillna('').apply(
                lambda x: '\n'.join(textwrap.wrap(str(x), width=35)) if len(str(x)) > 0 else ''
            )
            
            # Create table with wrapped text
            table = ax.table(cellText=table_data.values, colLabels=table_data.columns,
                           cellLoc='left', loc='center', bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.8)
            
            # Enable text wrapping by setting cell properties
            for key, cell in table.get_celld().items():
                cell.set_text_props(wrap=True)
            
            # Style header
            for i in range(len(table_data.columns)):
                table[(0, i)].set_facecolor('#70AD47')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Alternate row colors
            for i in range(1, len(table_data) + 1):
                for j in range(len(table_data.columns)):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#E2EFD9')
                    else:
                        table[(i, j)].set_facecolor('#F2F2F2')
            
            title = f'FALSE NEGATIVES (showing 1-{min(30, len(false_negatives_df))} of {len(false_negatives_df)} total)'
            fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Page 7: False Negatives Enhanced Details (from allData.json if available)
        if len(fn_enhanced_details_df) > 0:
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.axis('tight')
            ax.axis('off')
            
            # Prepare table data - show first 30 rows
            table_data = fn_enhanced_details_df[['EventID', 'UserID', 'DateTime', 'Type', 'SubType', 'Description']].copy()
            table_data = table_data.head(30)
            table_data.columns = ['Event ID', 'User ID', 'DateTime', 'Type', 'SubType', 'Description']
            
            # Wrap descriptions to prevent them from running off page
            import textwrap
            table_data['Description'] = table_data['Description'].fillna('').apply(
                lambda x: '\n'.join(textwrap.wrap(str(x), width=30)) if len(str(x)) > 0 else ''
            )
            
            # Create table with wrapped text
            table = ax.table(cellText=table_data.values, colLabels=table_data.columns,
                           cellLoc='left', loc='center', bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1, 1.6)
            
            # Enable text wrapping by setting cell properties
            for key, cell in table.get_celld().items():
                cell.set_text_props(wrap=True)
            
            # Style header
            for i in range(len(table_data.columns)):
                table[(0, i)].set_facecolor('#70AD47')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Alternate row colors
            for i in range(1, len(table_data) + 1):
                for j in range(len(table_data.columns)):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#E2EFD9')
                    else:
                        table[(i, j)].set_facecolor('#F2F2F2')
            
            title = f'FALSE NEGATIVES - ENHANCED DETAILS (showing 1-{min(30, len(fn_enhanced_details_df))} of {len(fn_enhanced_details_df)} total)'
            fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Page 8: False Alarms Table - comes last because it might be very long
        if len(false_alarms_df) > 0:
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.axis('tight')
            ax.axis('off')
            
            # Prepare table data - show first 30 rows
            table_data = false_alarms_df[['EventID', 'UserID', 'Type', 'SubType', 'MaxSeizureProbability']].copy()
            table_data = table_data.head(30)
            table_data.columns = ['Event ID', 'User ID', 'Type', 'SubType', 'Max Seizure Prob']
            
            # Format the probability column
            table_data['Max Seizure Prob'] = table_data['Max Seizure Prob'].apply(lambda x: f'{x:.4f}')
            
            # Create table
            table = ax.table(cellText=table_data.values, colLabels=table_data.columns,
                           cellLoc='left', loc='center', bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            
            # Style header
            for i in range(len(table_data.columns)):
                table[(0, i)].set_facecolor('#4472C4')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Alternate row colors
            for i in range(1, len(table_data) + 1):
                for j in range(len(table_data.columns)):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#E7E6E6')
                    else:
                        table[(i, j)].set_facecolor('#F2F2F2')
            
            title = f'FALSE ALARMS (showing 1-{min(30, len(false_alarms_df))} of {len(false_alarms_df)} total)'
            fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
    
    print(f"\nReport saved to: {pdf_path}")
    return pdf_path


def generate_text_report(df, user_metrics_df, far_metrics_df, subtype_metrics_df, false_alarms_df, false_negatives_df, fn_details_df, output_dir, seizure_threshold=3, far_threshold=100):
    """Generate text report."""
    report_path = os.path.join(output_dir, 'event_analysis_report.txt')
    
    tp_total = ((df['ModelPrediction'] == 1) & (df['ActualLabel'] == 1)).sum()
    fp_total = ((df['ModelPrediction'] == 1) & (df['ActualLabel'] == 0)).sum()
    fn_total = ((df['ModelPrediction'] == 0) & (df['ActualLabel'] == 1)).sum()
    tn_total = ((df['ModelPrediction'] == 0) & (df['ActualLabel'] == 0)).sum()
    
    tpr = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    far = fp_total / (fp_total + tn_total) if (fp_total + tn_total) > 0 else 0
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EVENT ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("OVERALL METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Events: {len(df)}\n")
        f.write(f"Seizure Events: {(df['ActualLabel'] == 1).sum()}\n")
        f.write(f"Non-Seizure Events: {(df['ActualLabel'] == 0).sum()}\n\n")
        f.write(f"True Positives (TP): {tp_total}\n")
        f.write(f"False Positives (FP): {fp_total}\n")
        f.write(f"True Negatives (TN): {tn_total}\n")
        f.write(f"False Negatives (FN): {fn_total}\n\n")
        f.write(f"Overall TPR (Sensitivity): {tpr:.3f}\n")
        f.write(f"Overall FAR (False Alarm Rate): {far:.3f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("TPR BY USER\n")
        f.write("="*80 + "\n")
        f.write(user_metrics_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("="*80 + "\n")
        f.write("FAR BY USER\n")
        f.write("="*80 + "\n")
        f.write(far_metrics_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("="*80 + "\n")
        f.write("TPR BY SEIZURE SUBTYPE\n")
        f.write("="*80 + "\n")
        f.write(subtype_metrics_df.to_string(index=False))
        f.write("\n\n")

        # Add TPR by user specifically for Tonic-Clonic seizures
        seizure_df_local = df[df['ActualLabel'] == 1].copy()
        tc_seizure_df_local = seizure_df_local[seizure_df_local['SubType'].str.contains('Tonic-Clonic', case=False, na=False)].copy()
        if len(tc_seizure_df_local) > 0:
            # Group users consistent with overall rule (<seizure_threshold seizures -> Other)
            user_seizure_counts_overall = seizure_df_local['UserID'].value_counts()
            def group_users_tc_local(user_id):
                if pd.isna(user_id):
                    return 'Unknown'
                if user_seizure_counts_overall.get(user_id, 0) < seizure_threshold:
                    return 'Other'
                return str(user_id)
            tc_seizure_df_local['UserGroup'] = tc_seizure_df_local['UserID'].apply(group_users_tc_local)

            tc_user_metrics_local = []
            for user in sorted(tc_seizure_df_local['UserGroup'].unique()):
                user_data = tc_seizure_df_local[tc_seizure_df_local['UserGroup'] == user]
                tp_u = (user_data['ModelPrediction'] == 1).sum()
                fn_u = (user_data['ModelPrediction'] == 0).sum()
                tpr_u = tp_u / (tp_u + fn_u) if (tp_u + fn_u) > 0 else 0
                tc_user_metrics_local.append({
                    'User': user,
                    'Seizures': len(user_data),
                    'TP': tp_u,
                    'FN': fn_u,
                    'TPR': tpr_u
                })

            tc_user_metrics_df_local = pd.DataFrame(tc_user_metrics_local).sort_values('TPR', ascending=False)

            # Overall TPR for Tonic-Clonic seizures
            tp_tc = (tc_seizure_df_local['ModelPrediction'] == 1).sum()
            fn_tc = (tc_seizure_df_local['ModelPrediction'] == 0).sum()
            tpr_tc_overall = tp_tc / (tp_tc + fn_tc) if (tp_tc + fn_tc) > 0 else 0

            f.write("="*80 + "\n")
            f.write("TPR BY USER (TONIC-CLONIC SUBTYPE)\n")
            f.write("="*80 + "\n")
            f.write(f"Tonic-Clonic seizure events: {len(tc_seizure_df_local)} | Overall TPR: {tpr_tc_overall:.3f}\n\n")
            f.write(tc_user_metrics_df_local.to_string(index=False))
            f.write("\n\n")
        else:
            f.write("="*80 + "\n")
            f.write("TPR BY USER (TONIC-CLONIC SUBTYPE)\n")
            f.write("="*80 + "\n")
            f.write("No Tonic-Clonic seizures present in dataset.\n\n")
        
        # Add FALSE NEGATIVES section first (before false alarms)
        fn_total = ((df['ModelPrediction'] == 0) & (df['ActualLabel'] == 1)).sum()
        f.write("="*80 + "\n")
        f.write(f"FALSE NEGATIVES ({fn_total} total)\n")
        f.write("="*80 + "\n")
        
        if len(fn_details_df) > 0:
            # Format the description column to be more readable (truncate long descriptions)
            fn_display_df = fn_details_df.copy()
            fn_display_df['Description'] = fn_display_df['Description'].fillna('').apply(
                lambda x: (x[:100] + '...') if len(str(x)) > 100 else x
            )
            f.write(fn_display_df[['EventID', 'UserID', 'DateTime', 'Type', 'SubType', 'Description']].to_string(index=False))
            f.write("\n\n")
        elif len(false_negatives_df) > 0:
            # Fall back to CSV-based false negatives data if enhanced details not available
            fn_display_df = false_negatives_df[['EventID', 'UserID', 'Type', 'SubType', 'Description']].copy()
            fn_display_df['Description'] = fn_display_df['Description'].fillna('').apply(
                lambda x: (x[:100] + '...') if len(str(x)) > 100 else x
            )
            f.write(fn_display_df.to_string(index=False))
            f.write("\n\n")
        else:
            f.write("No false negative details available\n\n")
        
        # Add FALSE ALARMS section after false negatives
        if len(false_alarms_df) > 0:
            f.write("="*80 + "\n")
            f.write(f"FALSE ALARMS ({len(false_alarms_df)} total)\n")
            f.write("="*80 + "\n")
            f.write(false_alarms_df[['EventID', 'UserID', 'Type', 'SubType', 'MaxSeizureProbability']].to_string(index=False))
            f.write("\n\n")
    
    print(f"Text report saved to: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description='Analyze nnTester event results CSV files')
    parser.add_argument('csv_file', help='Path to event_results.csv file')
    parser.add_argument('--output', '-o', default=None, help='Output directory (default: same as input)')
    parser.add_argument('--seizure-threshold', type=int, default=3, help='Minimum seizures per user to avoid grouping as "Other" (default: 3)')
    parser.add_argument('--far-threshold', type=int, default=100, help='Minimum non-seizure events per user to avoid grouping as "Other" (default: 100)')
    parser.add_argument('--alldata-json', default=None, help='Path to allData.json for enhanced false negative reporting')
    
    args = parser.parse_args()
    
    # Load data
    df = load_event_results(args.csv_file)
    
    # Determine output directory
    if args.output:
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.dirname(args.csv_file) or '.'
    
    # Try to find allData.json if not specified
    alldata_json_path = args.alldata_json
    if not alldata_json_path:
        # Look for allData.json in the same directory as the CSV
        csv_dir = os.path.dirname(args.csv_file) or '.'
        potential_json = os.path.join(csv_dir, 'allData.json')
        if os.path.exists(potential_json):
            alldata_json_path = potential_json
    
    # Load allData.json and extract false negatives details
    alldata_list = None
    fn_details_df = pd.DataFrame()
    if alldata_json_path:
        alldata_list = load_alldata_json(alldata_json_path)
        fn_details_df = extract_false_negatives_details(df, alldata_list)
        print(f"Extracted details for {len(fn_details_df)} false negatives")
    
    # Extract false negatives from CSV (same source as false positives)
    false_negatives_df = df[(df['ActualLabel'] == 1) & (df['ModelPrediction'] == 0)].copy()
    
    # Run analyses
    seizure_df, user_metrics_df, far_metrics_df = analyze_by_user(df, args.seizure_threshold, args.far_threshold)
    seizure_df_st, subtype_metrics_df = analyze_by_seizure_type(df)
    false_alarms_df, far_by_subtype_df = analyze_false_alarms(df)
    
    # Generate reports
    generate_text_report(df, user_metrics_df, far_metrics_df, subtype_metrics_df, false_alarms_df, false_negatives_df, fn_details_df, output_dir, args.seizure_threshold, args.far_threshold)
    generate_plots(df, seizure_df, user_metrics_df, far_metrics_df, subtype_metrics_df, false_alarms_df, false_negatives_df, fn_details_df, output_dir, args.seizure_threshold, args.far_threshold)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()

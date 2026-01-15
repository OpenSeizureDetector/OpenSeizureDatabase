#!/usr/bin/env python3
"""
Post-processing script to analyze nnTester event results CSV files.

Usage:
    python3 analyzeEventResults.py <path_to_event_results.csv> [--output <output_dir>]

This script generates:
    - Analysis of TPR and FAR by user
    - Analysis of TPR by seizure type
    - Analysis of event types/subtypes contributing to false alarms
    - Report with figures saved as PDF
"""

import argparse
import os
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


def analyze_by_user(df):
    """Analyze TPR and FAR by user, grouping users with < 5 seizures as 'other'."""
    print("\n" + "="*80)
    print("ANALYSIS BY USER")
    print("="*80)
    
    # Get seizure events only
    seizure_df = df[df['ActualLabel'] == 1].copy()
    
    # Count seizures per user
    user_seizure_counts = seizure_df['UserID'].value_counts()
    print(f"\nTotal users with seizures: {len(user_seizure_counts)}")
    print(f"Users with >= 3 seizures: {(user_seizure_counts >= 3).sum()}")
    print(f"Users with < 3 seizures: {(user_seizure_counts < 3).sum()}")
    
    # Group users with < 3 seizures as 'other'
    def group_users(user_id):
        if pd.isna(user_id):
            return 'Unknown'
        if user_seizure_counts.get(user_id, 0) < 3:
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
    non_seizure_df['UserGroup'] = non_seizure_df['UserID'].apply(group_users)
    
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


def generate_plots(df, seizure_df, user_metrics_df, far_metrics_df, subtype_metrics_df, false_alarms_df, output_dir):
    """Generate plots and save to PDF."""
    pdf_path = os.path.join(output_dir, 'event_analysis_report.pdf')
    
    with PdfPages(pdf_path) as pdf:
        # Plot 1: TPR by User
        if len(user_metrics_df) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            users = user_metrics_df['User']
            tprs = user_metrics_df['TPR']
            colors = ['#2ecc71' if tpr >= 0.8 else '#f39c12' if tpr >= 0.6 else '#e74c3c' for tpr in tprs]
            
            ax.barh(users, tprs, color=colors)
            ax.set_xlabel('TPR (True Positive Rate)', fontsize=12, fontweight='bold')
            ax.set_title('TPR by User (Seizure Events)', fontsize=14, fontweight='bold')
            ax.set_xlim(0, 1)
            
            # Add value labels
            for i, (user, tpr) in enumerate(zip(users, tprs)):
                ax.text(tpr + 0.02, i, f'{tpr:.2%}', va='center', fontsize=10)
            
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
            colors = ['#e74c3c' if far > 0.05 else '#f39c12' if far > 0.01 else '#2ecc71' for far in fars]
            
            ax.barh(users, fars, color=colors)
            ax.set_xlabel('FAR (False Alarm Rate)', fontsize=12, fontweight='bold')
            ax.set_title('FAR by User (Non-Seizure Events)', fontsize=14, fontweight='bold')
            
            # Add value labels
            for i, (user, far) in enumerate(zip(users, fars)):
                ax.text(far + 0.002, i, f'{far:.2%}', va='center', fontsize=10)
            
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
    
    print(f"\nReport saved to: {pdf_path}")
    return pdf_path


def generate_text_report(df, user_metrics_df, far_metrics_df, subtype_metrics_df, false_alarms_df, output_dir):
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
    
    args = parser.parse_args()
    
    # Load data
    df = load_event_results(args.csv_file)
    
    # Determine output directory
    if args.output:
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.dirname(args.csv_file) or '.'
    
    # Run analyses
    seizure_df, user_metrics_df, far_metrics_df = analyze_by_user(df)
    seizure_df_st, subtype_metrics_df = analyze_by_seizure_type(df)
    false_alarms_df, far_by_subtype_df = analyze_false_alarms(df)
    
    # Generate reports
    generate_text_report(df, user_metrics_df, far_metrics_df, subtype_metrics_df, false_alarms_df, output_dir)
    generate_plots(df, seizure_df, user_metrics_df, far_metrics_df, subtype_metrics_df, false_alarms_df, output_dir)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()

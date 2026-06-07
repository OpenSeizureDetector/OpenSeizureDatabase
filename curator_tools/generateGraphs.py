#!/usr/bin/env python3
"""
generateGraphs.py
Generate summary graphs from OSDB JSON event files.

This script reads OSDB JSON files and produces a set of summary graphs:
1. Total counts of seizures, false alarms, and NDA events
2. Bar chart of seizure events per user (grouping users with < threshold seizures)
3. Cumulative seizures per user per month (grouping users with < threshold seizures)

The script can be run standalone with JSON files provided on the command line,
or integrated into the makeOsdDb.py workflow.

Graham Jones 2026
Licence: GPL v3 or later.
"""

import sys
import os
import json
import argparse
from datetime import datetime
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Make the libosd folder accessible in the search path.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def load_events_from_json(json_files, debug=False):
    """
    Load events from JSON files.
    
    Parameters:
    json_files - list of JSON file paths
    debug - print debug information
    
    Returns:
    list of event dictionaries
    """
    all_events = []
    
    for json_file in json_files:
        if not os.path.exists(json_file):
            print(f"Warning: File {json_file} not found, skipping...")
            continue
            
        try:
            with open(json_file, 'r') as f:
                events = json.load(f)
                if isinstance(events, list):
                    all_events.extend(events)
                else:
                    all_events.append(events)
            if debug:
                print(f"Loaded {len(events)} events from {json_file}")
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    return all_events


def categorize_events(all_events):
    """
    Categorize events by type.
    
    Parameters:
    all_events - list of event dictionaries
    
    Returns:
    dict with keys for 'seizures', 'false_alarms', 'nda', categorized by event type/subtype
    """
    seizures = []
    false_alarms = []
    nda = []
    
    for event in all_events:
        event_type = event.get('type', 'unknown').lower()
        
        if event_type == 'seizure':
            seizures.append(event)
        elif event_type == 'false alarm':
            false_alarms.append(event)
        elif event_type in ['nda', 'normal daily activity']:
            nda.append(event)
    
    return {
        'seizures': seizures,
        'false_alarms': false_alarms,
        'nda': nda
    }


def create_summary_stats(categorized_events):
    """
    Create summary statistics.
    
    Parameters:
    categorized_events - dict of categorized events
    
    Returns:
    dict with summary counts
    """
    return {
        'total_seizures': len(categorized_events['seizures']),
        'total_false_alarms': len(categorized_events['false_alarms']),
        'total_nda': len(categorized_events['nda']),
        'total_events': sum(len(v) for v in categorized_events.values())
    }


def create_events_by_user_chart(seizures, output_dir, threshold=5, debug=False):
    """
    Create a bar chart of seizure events per user.
    Users with fewer than 'threshold' seizures are grouped as 'Other'.
    
    Parameters:
    seizures - list of seizure events
    output_dir - directory for output
    threshold - minimum number of seizures to have individual bar
    debug - print debug information
    """
    # Count seizures per user
    user_counts = defaultdict(int)
    for event in seizures:
        user_id = event.get('userId', 'Unknown')
        user_counts[user_id] += 1
    
    # Separate users above and below threshold
    above_threshold = {}
    below_threshold_count = 0
    
    for user_id, count in sorted(user_counts.items(), key=lambda x: x[1], reverse=True):
        if count >= threshold:
            above_threshold[user_id] = count
        else:
            below_threshold_count += count
    
    if below_threshold_count > 0:
        above_threshold['Other'] = below_threshold_count
    
    if debug:
        print(f"Events by user: {above_threshold}")
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    users = list(above_threshold.keys())
    counts = list(above_threshold.values())
    
    # Use numeric x-positions to avoid matplotlib type issues
    x_pos = range(len(users))
    ax.bar(x_pos, counts, color='steelblue', edgecolor='navy')
    ax.set_xlabel('User ID', fontsize=12)
    ax.set_ylabel('Number of Seizure Events', fontsize=12)
    ax.set_title(f'Seizure Events per User (Users with <{threshold} seizures grouped as "Other")', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    # Set user names as x-tick labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(users)
    
    # Rotate x-axis labels if many users
    if len(users) > 10:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'seizures_by_user.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved chart: {output_path}")
    plt.close()


def create_cumulative_seizures_per_month(seizures, output_dir, threshold=5, debug=False):
    """
    Create a line chart of cumulative seizures per user per month.
    Users with fewer than 'threshold' total seizures are grouped as 'Other'.
    
    Parameters:
    seizures - list of seizure events
    output_dir - directory for output
    threshold - minimum number of seizures to have individual line
    debug - print debug information
    """
    # Count total seizures per user
    user_totals = defaultdict(int)
    for event in seizures:
        user_id = event.get('userId', 'Unknown')
        user_totals[user_id] += 1
    
    # Create monthly data
    monthly_data = defaultdict(lambda: defaultdict(int))
    
    for event in seizures:
        user_id = event.get('userId', 'Unknown')
        data_time_str = event.get('dataTime', '')
        
        # Determine if user should be included individually
        if user_totals[user_id] < threshold:
            user_id = 'Other'
        
        # Parse date - handle multiple formats
        try:
            # Try ISO 8601 format first (2024-01-02T12:00:00Z)
            try:
                dt = datetime.fromisoformat(data_time_str.replace('Z', '+00:00'))
            except (ValueError, TypeError):
                # Try DD-MM-YYYY HH:MM:SS format
                dt = datetime.strptime(data_time_str, '%d-%m-%Y %H:%M:%S')
            
            month_key = dt.strftime('%Y-%m')
            monthly_data[user_id][month_key] += 1
        except Exception as e:
            if debug:
                print(f"Error parsing date '{data_time_str}': {e}")
            continue
    
    # Get users with >= threshold seizures
    above_threshold_users = [user_id for user_id, count in user_totals.items() if count >= threshold]
    if 'Other' in monthly_data:
        above_threshold_users.append('Other')
    
    if debug:
        print(f"Users in cumulative chart: {above_threshold_users}")
    
    # Create cumulative chart
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Get all month keys
    all_months = sorted(set(month for user_data in monthly_data.values() for month in user_data.keys()))
    
    if not all_months:
        print("No data to plot for cumulative seizures")
        plt.close()
        return
    
    # Plot each user with distinct color/marker for accessibility
    colors = plt.cm.tab20(range(len(above_threshold_users)))
    markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', 'h', '+']
    for idx, user_id in enumerate(above_threshold_users):
        if user_id not in monthly_data:
            continue
            
        cumulative = []
        current_sum = 0
        for month in all_months:
            current_sum += monthly_data[user_id].get(month, 0)
            cumulative.append(current_sum)
        
        marker = markers[idx % len(markers)]
        ax.plot(
            all_months,
            cumulative,
            marker=marker,
            label=user_id,
            color=colors[idx],
            linewidth=2,
            markersize=6,
        )
    
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Cumulative Number of Seizures', fontsize=12)
    ax.set_title(f'Cumulative Seizures per User per Month (Users with <{threshold} seizures grouped as "Other")', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'cumulative_seizures_per_month.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved chart: {output_path}")
    plt.close()


def create_summary_stats_chart(stats, output_dir):
    """
    Create a simple bar chart of summary statistics.
    
    Parameters:
    stats - dict with summary statistics
    output_dir - directory for output
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    categories = ['Seizures', 'False Alarms', 'NDA Events']
    values = [
        stats['total_seizures'],
        stats['total_false_alarms'],
        stats['total_nda']
    ]
    colors = ['#ff7f0e', '#d62728', '#2ca02c']
    
    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Number of Events', fontsize=12)
    ax.set_title('OSDB Summary Statistics', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'summary_statistics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved chart: {output_path}")
    plt.close()


def generate_all_graphs(json_files, output_dir, threshold=5, debug=False):
    """
    Generate all summary graphs.
    
    Parameters:
    json_files - list of JSON file paths
    output_dir - directory for output graphs
    threshold - minimum number of events for individual user bar/line
    debug - print debug information
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading events from {len(json_files)} file(s)...")
    all_events = load_events_from_json(json_files, debug=debug)
    print(f"Loaded {len(all_events)} total events")
    
    if not all_events:
        print("Error: No events loaded from files")
        return False
    
    print("\nCategorizing events...")
    categorized = categorize_events(all_events)
    
    print("\nGenerating summary statistics...")
    stats = create_summary_stats(categorized)
    print(f"  Seizures: {stats['total_seizures']}")
    print(f"  False Alarms: {stats['total_false_alarms']}")
    print(f"  NDA Events: {stats['total_nda']}")
    print(f"  Total Events: {stats['total_events']}")
    
    print("\nGenerating graphs...")
    print("  - Summary statistics chart...")
    create_summary_stats_chart(stats, output_dir)
    
    if categorized['seizures']:
        print("  - Seizures by user bar chart...")
        create_events_by_user_chart(categorized['seizures'], output_dir, threshold=threshold, debug=debug)
        
        print("  - Cumulative seizures per user per month chart...")
        create_cumulative_seizures_per_month(categorized['seizures'], output_dir, threshold=threshold, debug=debug)
    else:
        print("  - No seizure events to plot")
    
    print(f"\nGraphs saved to: {output_dir}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate summary graphs from OSDB JSON files')
    parser.add_argument('json_files', nargs='+',
                        help='JSON files to process')
    parser.add_argument('--output', '-o', default='output',
                        help='Output directory for graphs (default: output)')
    parser.add_argument('--threshold', '-t', type=int, default=5,
                        help='Minimum number of events for individual user display (default: 5)')
    parser.add_argument('--debug', action='store_true',
                        help='Print debug information')
    
    args = parser.parse_args()
    
    if not args.json_files:
        parser.print_help()
        sys.exit(1)
    
    success = generate_all_graphs(args.json_files, args.output, threshold=args.threshold, debug=args.debug)
    sys.exit(0 if success else 1)

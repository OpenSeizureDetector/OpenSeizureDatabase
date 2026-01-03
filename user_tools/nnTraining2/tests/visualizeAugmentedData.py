#!/usr/bin/env python3
"""
Visualize augmented accelerometer data from OSDB CSV files.

This script reads CSV files produced by augmentData.py and generates plots
showing acceleration magnitude and x,y,z components for each event.

Features:
- Generates graphs for seizure events only (default) or all events
- Optionally overlays augmented data with base events for comparison
- Saves plots as PNG files with descriptive names
- Configurable output directory
"""

import argparse
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_event_id(event_id):
    """
    Parse event ID to determine if it's a base event or augmented.
    
    Returns:
        tuple: (base_id, augmentation_suffix)
        e.g., "24077" -> ("24077", None)
              "24077-1" -> ("24077", "1")
              "24077-1-2" -> ("24077", "1-2")
    """
    parts = str(event_id).split('-', 1)
    if len(parts) == 1:
        return (str(event_id), None)
    else:
        return (parts[0], parts[1])


def get_acceleration_data(event_df):
    """
    Extract acceleration data from all rows of an event.
    
    Args:
        event_df: DataFrame containing all rows for a single event (sorted by dataTime)
    
    Returns:
        dict with keys: 'magnitude', 'x', 'y', 'z', 'time'
    """
    # Magnitude data: M000 to M124
    mag_cols = [f'M{i:03d}' for i in range(125)]
    
    # X, Y, Z acceleration data: X000-X124, Y000-Y124, Z000-Z124
    x_cols = [f'X{i:03d}' for i in range(125)]
    y_cols = [f'Y{i:03d}' for i in range(125)]
    z_cols = [f'Z{i:03d}' for i in range(125)]
    
    # Concatenate all rows for this event
    magnitude_list = []
    x_acc_list = []
    y_acc_list = []
    z_acc_list = []
    
    for _, row in event_df.iterrows():
        # Get magnitude data for this datapoint
        mag_data = row[mag_cols].values.astype(float)
        magnitude_list.append(mag_data)
        
        # Try to get X, Y, Z data (may not exist in all datasets)
        try:
            x_data = row[x_cols].values.astype(float)
            y_data = row[y_cols].values.astype(float)
            z_data = row[z_cols].values.astype(float)
            x_acc_list.append(x_data)
            y_acc_list.append(y_data)
            z_acc_list.append(z_data)
        except (KeyError, ValueError):
            # If X, Y, Z columns don't exist, use zeros
            x_acc_list.append(np.zeros_like(mag_data))
            y_acc_list.append(np.zeros_like(mag_data))
            z_acc_list.append(np.zeros_like(mag_data))
    
    # Concatenate all datapoints into single arrays
    magnitude = np.concatenate(magnitude_list)
    x_acc = np.concatenate(x_acc_list)
    y_acc = np.concatenate(y_acc_list)
    z_acc = np.concatenate(z_acc_list)
    
    # Time array (assuming 125 samples per datapoint at 25Hz = 5 seconds per datapoint)
    total_samples = len(magnitude)
    total_time = (total_samples / 125) * 5  # 5 seconds per 125 samples
    time = np.linspace(0, total_time, total_samples)
    
    return {
        'magnitude': magnitude,
        'x': x_acc,
        'y': y_acc,
        'z': z_acc,
        'time': time
    }


def plot_event(event_data, event_id, is_seizure, output_dir, include_augmented=False):
    """
    Create and save plots for an event.
    
    Args:
        event_data: Dictionary with base event and optional augmented events
        event_id: Base event ID
        is_seizure: Boolean indicating if this is a seizure event
        output_dir: Directory to save plots
        include_augmented: Whether to overlay augmented data
    """
    base_data = event_data['base']
    augmented_data = event_data.get('augmented', [])
    
    # Create figure with 2 subplots (magnitude and x,y,z)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Magnitude
    # Draw augmented data first as faint dots
    if include_augmented and augmented_data:
        colors = plt.cm.rainbow(np.linspace(0, 1, len(augmented_data)))
        for idx, (aug_id, aug_data) in enumerate(augmented_data):
            ax1.scatter(aug_data['time'], aug_data['magnitude'], 
                       s=10, label=f'Augmented {aug_id}', 
                       color=colors[idx], alpha=0.15)
    
    # Draw base event on top with bright color and lines
    ax1.plot(base_data['time'], base_data['magnitude'], 
             linewidth=2.5, label='Base Event', color='darkblue', alpha=1.0, zorder=10)
    
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('Acceleration Magnitude (mg)', fontsize=12)
    ax1.set_title(f'Acceleration Magnitude - Event {event_id}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Plot 2: X, Y, Z components
    # Draw augmented data first as faint scatter dots
    if include_augmented and augmented_data:
        for idx, (aug_id, aug_data) in enumerate(augmented_data):
            # Use muted colors for augmented data
            ax2.scatter(aug_data['time'], aug_data['x'], 
                       s=8, color='darkred', alpha=0.1)
            ax2.scatter(aug_data['time'], aug_data['y'], 
                       s=8, color='darkgreen', alpha=0.1)
            ax2.scatter(aug_data['time'], aug_data['z'], 
                       s=8, color='darkblue', alpha=0.1)
    
    # Draw base event on top with bright colors
    ax2.plot(base_data['time'], base_data['x'], 
             linewidth=2.5, label='X (Base)', color='red', alpha=1.0, zorder=10)
    ax2.plot(base_data['time'], base_data['y'], 
             linewidth=2.5, label='Y (Base)', color='green', alpha=1.0, zorder=10)
    ax2.plot(base_data['time'], base_data['z'], 
             linewidth=2.5, label='Z (Base)', color='blue', alpha=1.0, zorder=10)
    
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Acceleration (mg)', fontsize=12)
    ax2.set_title(f'X, Y, Z Components - Event {event_id}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    
    # Add event type to figure
    event_type = 'seizure' if is_seizure else 'non-seizure'
    fig.suptitle(f'Event {event_id} ({event_type.upper()})', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    seizure_tag = '_seizure' if is_seizure else ''
    filename = f'event_{event_id}{seizure_tag}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filepath


def process_csv(csv_file, output_dir, all_events=False, include_augmented=False):
    """
    Process CSV file and generate plots for each event.
    
    Args:
        csv_file: Path to input CSV file
        output_dir: Directory to save plots
        all_events: If True, plot all events; if False, only seizures
        include_augmented: If True, overlay augmented data on base events
    """
    print(f"Reading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()[:10]}...")  # Show first 10 columns
    
    # Check required columns
    required_cols = ['eventId', 'type']
    for col in required_cols:
        if col not in df.columns:
            print(f"ERROR: Required column '{col}' not found in CSV file")
            sys.exit(1)
    
    # Group by event ID
    print("\nGrouping events by eventId and sorting by dataTime...")
    
    # Parse event IDs to find base events and their augmentations
    df['base_id'] = df['eventId'].apply(lambda x: parse_event_id(x)[0])
    df['aug_suffix'] = df['eventId'].apply(lambda x: parse_event_id(x)[1])
    
    # Sort by dataTime if the column exists
    if 'dataTime' in df.columns:
        # Convert dataTime to datetime for proper sorting
        try:
            df['dataTime_dt'] = pd.to_datetime(df['dataTime'], dayfirst=True)
            df = df.sort_values(['eventId', 'dataTime_dt'])
        except:
            # If conversion fails, try sorting as string
            df = df.sort_values(['eventId', 'dataTime'])
    
    # Group by eventId to get all rows for each event
    grouped_events = df.groupby('eventId')
    
    # Organize events
    events_dict = {}
    
    for event_id, event_df in grouped_events:
        base_id, aug_suffix = parse_event_id(event_id)
        is_seizure = event_df['type'].iloc[0] == 1
        
        # Skip non-seizure events if not plotting all events
        if not all_events and not is_seizure:
            continue
        
        # Initialize base event entry if needed
        if base_id not in events_dict:
            events_dict[base_id] = {
                'base': None,
                'augmented': [],
                'is_seizure': is_seizure
            }
        
        # Get acceleration data for all rows of this event
        acc_data = get_acceleration_data(event_df)
        
        # Classify as base or augmented
        if aug_suffix is None:
            events_dict[base_id]['base'] = acc_data
        else:
            events_dict[base_id]['augmented'].append((aug_suffix, acc_data))
    
    print(f"\nFound {len(events_dict)} base events to process")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Generate plots
    print("\nGenerating plots...")
    count = 0
    for base_id, event_info in events_dict.items():
        # Skip if base event data is missing
        if event_info['base'] is None:
            print(f"Warning: No base data for event {base_id}, skipping")
            continue
        
        # Plot the event
        filepath = plot_event(
            event_info, 
            base_id, 
            event_info['is_seizure'],
            output_dir,
            include_augmented
        )
        count += 1
        
        if count % 10 == 0:
            print(f"  Processed {count}/{len(events_dict)} events...")
    
    print(f"\nCompleted! Generated {count} plots in {output_dir}")
    
    # Print summary
    seizure_count = sum(1 for e in events_dict.values() if e['is_seizure'])
    non_seizure_count = len(events_dict) - seizure_count
    print(f"\nSummary:")
    print(f"  Seizure events: {seizure_count}")
    print(f"  Non-seizure events: {non_seizure_count}")
    print(f"  Total events plotted: {count}")
    
    if include_augmented:
        aug_count = sum(len(e['augmented']) for e in events_dict.values())
        print(f"  Augmented variants included: {aug_count}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize augmented accelerometer data from OSDB CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot only seizure events from CSV file
  %(prog)s data.csv
  
  # Plot all events (including non-seizures)
  %(prog)s data.csv --all-events
  
  # Include augmented data overlays
  %(prog)s data.csv --include-augmented
  
  # Specify output directory
  %(prog)s data.csv --output-dir ./plots
  
  # Combine options
  %(prog)s data.csv --all-events --include-augmented --output-dir ./all_plots
        """
    )
    
    parser.add_argument(
        'csv_file',
        help='Path to the CSV file produced by augmentData.py'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        default='visualization_output',
        help='Directory to save output plots (default: visualization_output)'
    )
    
    parser.add_argument(
        '-a', '--all-events',
        action='store_true',
        help='Plot all events, not just seizure events'
    )
    
    parser.add_argument(
        '-i', '--include-augmented',
        action='store_true',
        help='Include augmented data on the same axes as base events for comparison'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.csv_file):
        print(f"ERROR: CSV file not found: {args.csv_file}")
        sys.exit(1)
    
    # Process the CSV file
    process_csv(
        args.csv_file,
        args.output_dir,
        args.all_events,
        args.include_augmented
    )


if __name__ == '__main__':
    main()

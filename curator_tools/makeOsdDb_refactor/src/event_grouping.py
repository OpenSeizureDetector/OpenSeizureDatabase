#!/usr/bin/env python3
"""
event_grouping.py - Phase 1 & 2 features
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            for item in self.iterable:
                yield item


def parse_time_delta(time_str: str) -> timedelta:
    time_str = time_str.lower().strip()
    if time_str.endswith('s'):
        return timedelta(seconds=int(time_str[:-1]))
    elif time_str.endswith('min'):
        return timedelta(minutes=int(time_str[:-3]))
    elif time_str.endswith('h'):
        return timedelta(hours=int(time_str[:-1]))
    else:
        return timedelta(minutes=int(time_str))


def concatenate_datapoints(events: List[Dict[str, Any]],
                           remove_duplicates: bool = True,
                           time_tolerance_ms: int = 100) -> List[Dict]:
    if not events:
        return []
    
    all_datapoints = []
    for event in events:
        datapoints = event.get('datapoints', [])
        if datapoints:
            all_datapoints.extend(datapoints)
    
    if not all_datapoints:
        return []
    
    def get_time(dp):
        """Get time value from datapoint, handling strings and different field names."""
        for field in ['time', 'dataTime', 't']:
            if field in dp:
                value = dp[field]
                # If it's a string, try to parse it to timestamp
                if isinstance(value, str):
                    try:
                        from dateutil import parser as dateutil_parser
                        dt = dateutil_parser.parse(value, dayfirst=True)
                        return dt.timestamp() * 1000  # Convert to milliseconds
                    except:
                        # If parsing fails, return 0 to put at beginning
                        return 0
                return value
        return 0
    
    all_datapoints.sort(key=get_time)
    
    if not remove_duplicates:
        return all_datapoints
    
    deduplicated = []
    last_time = None
    
    for dp in all_datapoints:
        current_time = get_time(dp)
        if last_time is not None:
            time_diff = abs(current_time - last_time)
            if time_diff < time_tolerance_ms:
                continue
        deduplicated.append(dp)
        last_time = current_time
    
    return deduplicated


def merge_grouped_events(group: List[Dict[str, Any]],
                        selected_event: Dict[str, Any],
                        concatenate_datapoints_flag: bool = True,
                        update_desc: bool = True) -> Dict[str, Any]:
    if not concatenate_datapoints_flag or len(group) <= 1:
        return selected_event
    
    merged = selected_event.copy()
    concatenated_datapoints = concatenate_datapoints(group, remove_duplicates=True)
    merged['datapoints'] = concatenated_datapoints
    merged['_merged_from_event_ids'] = [e['id'] for e in group]
    merged['_merged_event_count'] = len(group)
    merged['_merged_datapoint_count'] = len(concatenated_datapoints)
    
    # Update desc field to note merged events (only for events from existing data)
    if update_desc and len(group) > 1:
        selected_id = selected_event.get('id')
        merged_ids = [e['id'] for e in group if e['id'] != selected_id]
        
        if merged_ids:
            current_desc = merged.get('desc', '') or ''
            
            # Convert all IDs to strings before sorting (handles mixed str/int IDs)
            merged_ids_str = ', '.join(sorted(str(eid) for eid in merged_ids))
            merge_note = f"Includes data from merged event(s): {merged_ids_str}"
            
            # Check if this exact merge note already exists to avoid duplicates
            if merge_note not in current_desc:
                if current_desc and not current_desc.endswith('.'):
                    current_desc += '.'
                if current_desc:
                    current_desc += ' '
                
                merged['desc'] = current_desc + merge_note
            else:
                # Note already exists, keep desc as is
                merged['desc'] = current_desc
    
    return merged


def group_events_by_proximity(events: List[Dict[str, Any]],
                              time_threshold: str = '3min',
                              debug: bool = False) -> List[List[Dict]]:
    if not events:
        return []
    
    threshold_delta = parse_time_delta(time_threshold)
    threshold_seconds = threshold_delta.total_seconds()
    
    df = pd.DataFrame(events)
    df['dataTime'] = pd.to_datetime(df['dataTime'], format='mixed', utc=True)
    df = df.sort_values(['userId', 'type', 'dataTime']).reset_index(drop=True)
    
    groups = []
    current_group = []
    current_user = None
    current_type = None
    last_time = None
    
    for idx, row in df.iterrows():
        event = row.to_dict()
        user_id = event['userId']
        event_type = event['type']
        event_time = event['dataTime']
        
        if user_id != current_user or event_type != current_type:
            if current_group:
                groups.append(current_group)
            current_group = [event]
            current_user = user_id
            current_type = event_type
            last_time = event_time
            continue
        
        time_diff = (event_time - last_time).total_seconds()
        
        if time_diff <= threshold_seconds:
            current_group.append(event)
        else:
            groups.append(current_group)
            current_group = [event]
        
        last_time = event_time
    
    if current_group:
        groups.append(current_group)
    
    return groups


def select_best_event_from_group(group: List[Dict[str, Any]],
                                 strategy: str = 'alarm_first',
                                 debug: bool = False) -> Dict[str, Any]:
    if not group:
        return None
    if len(group) == 1:
        return group[0]
    
    # CRITICAL: Always prefer events from existing published database
    # Check if any events are marked as existing (from published database)
    existing_events = [e for e in group if e.get('_is_existing_event', False)]
    
    if existing_events:
        # If we have existing events in the group, only consider those for selection
        # This preserves published event IDs
        group_to_select_from = existing_events
        if debug:
            print(f"Group has {len(existing_events)} existing event(s) out of {len(group)} total - prioritizing existing")
    else:
        group_to_select_from = group
    
    if strategy == 'alarm_first':
        def sort_key(event):
            # Prioritize existing events first, then alarm state, description, and time
            is_existing = 0 if event.get('_is_existing_event', False) else 1
            alarm_state = event.get('osdAlarmState', 0)
            has_desc = 1 if event.get('desc', '').strip() else 0
            datatime = pd.to_datetime(event['dataTime'], format='mixed', utc=True)
            return (is_existing, -alarm_state, -has_desc, datatime)
        sorted_group = sorted(group, key=sort_key)
        return sorted_group[0]
    elif strategy == 'most_datapoints':
        # Still prioritize existing events
        if existing_events:
            return max(existing_events, key=lambda e: len(e.get('datapoints', [])))
        return max(group, key=lambda e: len(e.get('datapoints', [])))
    elif strategy == 'last':
        if existing_events:
            return max(existing_events, key=lambda e: pd.to_datetime(e['dataTime'], format='mixed', utc=True))
        return max(group, key=lambda e: pd.to_datetime(e['dataTime'], format='mixed', utc=True))
    else:
        if existing_events:
            return min(existing_events, key=lambda e: pd.to_datetime(e['dataTime'], format='mixed', utc=True))
        return min(group, key=lambda e: pd.to_datetime(e['dataTime'], format='mixed', utc=True))


def apply_sliding_window_grouping(events: List[Dict[str, Any]],
                                  time_threshold: str = '3min',
                                  selection_strategy: str = 'alarm_first',
                                  concatenate_datapoints_flag: bool = True,
                                  debug: bool = False,
                                  show_progress: bool = False,
                                  exclude_event_types: List[str] = None,
                                  update_desc: bool = True) -> Tuple[List[Dict], Dict]:
    if not events:
        return [], {'total_groups': 0, 'total_input_events': 0, 'discarded_events': []}
    
    # Separate excluded event types (e.g., NDA events)
    excluded_types = exclude_event_types or []
    if excluded_types:
        excluded_events = [e for e in events if e.get('type', '').lower() in [t.lower() for t in excluded_types]]
        events_to_group = [e for e in events if e.get('type', '').lower() not in [t.lower() for t in excluded_types]]
        if excluded_events:
            print(f"  Skipping grouping for {len(excluded_events)} events of type(s): {', '.join(excluded_types)}")
    else:
        excluded_events = []
        events_to_group = events
    
    if not events_to_group:
        return excluded_events, {'total_groups': 0, 'total_input_events': len(events), 'discarded_events': [], 'excluded_events': len(excluded_events)}
    
    # Track existing events for validation
    existing_event_ids = {e['id'] for e in events if e.get('_is_existing_event', False)}
    if existing_event_ids:
        print(f"  Tracking {len(existing_event_ids)} existing published events for preservation...")
    
    groups = group_events_by_proximity(events_to_group, time_threshold, debug)
    
    unique_events = []
    discarded_events = []
    total_datapoints_before = 0
    total_datapoints_after = 0
    
    iterator = tqdm(groups, desc="Selecting from groups", unit="group", disable=not show_progress) if show_progress else groups
    
    for group in iterator:
        # Safely count datapoints - handle cases where datapoints might be a number instead of list
        for e in group:
            dps = e.get('datapoints', [])
            if isinstance(dps, list):
                total_datapoints_before += len(dps)
        
        selected = select_best_event_from_group(group, selection_strategy, debug)
        
        if concatenate_datapoints_flag and len(group) > 1:
            merged = merge_grouped_events(group, selected, concatenate_datapoints_flag=True, update_desc=update_desc)
            unique_events.append(merged)
            # Safely count datapoints
            dps = merged.get('datapoints', [])
            if isinstance(dps, list):
                total_datapoints_after += len(dps)
        else:
            unique_events.append(selected)
            # Safely count datapoints
            dps = selected.get('datapoints', [])
            if isinstance(dps, list):
                total_datapoints_after += len(dps)
        
        discarded = [e['id'] for e in group if e['id'] != selected['id']]
        discarded_events.extend(discarded)
    
    # VALIDATION: Ensure all existing events are preserved
    lost_existing_events = []
    if existing_event_ids:
        preserved_ids = set()
        
        for event in unique_events:
            # Check if this event's ID was from existing database
            if event['id'] in existing_event_ids:
                preserved_ids.add(event['id'])
            
            # Check if existing events were merged into this one
            merged_from = event.get('_merged_from_event_ids', [])
            
            # Normalize merged_from to always be a list (handle legacy formats)
            if merged_from is None:
                merged_from = []
            elif not isinstance(merged_from, list):
                # Handle case where it's a single value (float, int, etc.)
                merged_from = [merged_from]
            
            for merged_id in merged_from:
                if merged_id in existing_event_ids:
                    preserved_ids.add(merged_id)
        
        lost_existing_events = list(existing_event_ids - preserved_ids)
        
        if lost_existing_events:
            print(f"\n⚠️  WARNING: {len(lost_existing_events)} existing published events were LOST during grouping!")
            print(f"  Lost event IDs: {lost_existing_events[:20]}")  # Show first 20
            print(f"  This should NOT happen - existing events must be preserved!")
        else:
            print(f"  ✓ All {len(existing_event_ids)} existing events preserved")
    
    grouping_info = {
        'total_groups': len(groups),
        'total_input_events': len(events),
        'total_output_events': len(unique_events) + len(excluded_events),
        'excluded_events': len(excluded_events),
        'discarded_events': discarded_events,
        'events_per_group': [len(g) for g in groups],
        'time_threshold': time_threshold,
        'selection_strategy': selection_strategy,
        'concatenate_datapoints': concatenate_datapoints_flag,
        'total_datapoints_before': total_datapoints_before,
        'total_datapoints_after': total_datapoints_after,
        'existing_events_tracked': len(existing_event_ids),
        'existing_events_preserved': len(existing_event_ids) - len(lost_existing_events),
        'lost_existing_events': lost_existing_events
    }
    
    # Combine grouped events with excluded events (e.g., NDA events)
    final_events = unique_events + excluded_events
    
    return final_events, grouping_info

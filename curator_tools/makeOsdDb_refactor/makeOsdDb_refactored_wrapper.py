#!/usr/bin/env python3
"""
makeOsdDb_refactored_wrapper.py

Wrapper script that replicates the original makeOsdDb.py behavior but uses
the refactored processing modules from Phase 1-5.

This script:
1. Downloads event lists from the web API (using original libosd code)
2. Filters and groups events (using original logic)
3. Downloads detailed event data (using original or refactored downloader)
4. Processes events with refactored modules (grouping, deduplication, validation)
5. Saves to JSON files in the specified directory
6. Optionally generates CSV index files from JSON files
7. Optionally generates summary graphs of the database content

Usage:
    # Basic usage (download and process events)
    python3 makeOsdDb_refactored_wrapper.py --config ../osdb.cfg --osdb-dir /path/to/osdb
    
    # With index file generation
    python3 makeOsdDb_refactored_wrapper.py --config ../osdb.cfg --osdb-dir /path/to/osdb --generate-index
    
    # With summary graph generation
    python3 makeOsdDb_refactored_wrapper.py --config ../osdb.cfg --osdb-dir /path/to/osdb --generate-graphs
    
    # With both index and graph generation
    python3 makeOsdDb_refactored_wrapper.py --config ../osdb.cfg --osdb-dir /path/to/osdb --generate-index --generate-graphs
    
    # Custom graph output directory
    python3 makeOsdDb_refactored_wrapper.py --config ../osdb.cfg --osdb-dir /path/to/osdb --generate-graphs --graph-output /path/to/graphs
"""

import sys
import os
import argparse
import json
from pathlib import Path
from io import StringIO
import pandas as pd

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Import original libosd modules for web API access
import libosd.webApiConnection
import libosd.osdDbConnection
import libosd.configUtils

# Import refactored modules for processing
from event_validation import validate_events_batch, print_validation_summary
from event_grouping import apply_sliding_window_grouping
from event_deduplication import remove_duplicate_events
from datetime_normalization import normalize_events_batch
from osdb_sqlite import OsdWorkingDb
from database_utils import backup_database

# Import modules for index and graph generation
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import generateGraphs


def extractJsonVal(row, elem, debug=False):
    """Extract the value of element 'elem' from the JSON string
    in the 'dataJSON' element of dictionary 'row', and return the value,
    or None on error.
    
    (Copied from original makeOsdDb.py)
    """
    if (debug): print("extractJsonVal(): row=",row)
    dataJSON = row['dataJSON']
    if (dataJSON is not None):
        if (debug): print("extractJsonVal(): dataJSON=",dataJSON,
                          "length=",len(dataJSON))
    else:
        if (debug): print("extractJsonVal(): dataJSON is None")
    if (dataJSON is not None and dataJSON != ''):
        if (debug): print("extractJsonVal(): dataJSON=",dataJSON)
        dataObj = json.loads(dataJSON)
        if (elem in dataObj.keys()):
            elemVal = dataObj[elem]
        else:
            elemVal = None
    else:
        elemVal = None
    return(elemVal)


def getUniqueEventsListsFromServer(configFname="osdb.cfg",
                         outFile="listEvents",
                         start=None,
                         end=None,
                         outDir='.',
                         debug=False):
    """
    Obtain a list of all of the events in the data sharing database.
    Group them by userId and time so that all events within a given
    time period are assumed to be part of the same event to avoid
    duplication.
    
    Returns a tuple containing lists of unique EventIDs for:
      - all Seizures,
      - tonic clonic seizures,
      - false alarms,
      - unknown events,
      - falls,
      - NDA events (normal daily activities)
    
    (Based on original makeOsdDb.py logic)
    """
    # Create empty dataframes for the different classes of events
    allUniqueEventsDf = pd.DataFrame()
    tcUniqueEventsDf = pd.DataFrame()
    allSeizureUniqueEventsDf = pd.DataFrame()
    falseAlarmUniqueEventsDf = pd.DataFrame()
    unknownUniqueEventsDf = pd.DataFrame()
    fallUniqueEventsDf = pd.DataFrame()
    ndaUniqueEventsDf = pd.DataFrame()

    cfgObj = libosd.configUtils.loadConfig(configFname)
    osd = libosd.webApiConnection.WebApiConnection(cfg=cfgObj['credentialsFname'],
                                                   download=True,
                                                   debug=debug)
    eventLst = osd.getEvents(userId=None, includeDatapoints=False)
    print("Loaded %d Raw Events" % len(eventLst))

    # Read the event list into a pandas data frame.
    df = pd.read_json(StringIO(json.dumps(eventLst)))
    df['dataTime'] = pd.to_datetime(df['dataTime'])
    # Add some extra metadata to the event records.
    df['phoneAppVersion'] = df.apply(lambda row: extractJsonVal(row,'phoneAppVersion', debug=False), axis = 1)
    df['dataSource'] = df.apply(lambda row: extractJsonVal(row,'dataSourceName', debug=False), axis = 1)
    df['watchAppVersion'] = df.apply(lambda row: extractJsonVal(row,'watchSdVersion', debug=False), axis = 1)
    # drop the dataJSON column because we do not need it.
    df=df.drop('dataJSON', axis=1)

    # Filter out warnings (unless they are tagged as a seizure) and tests.
    if not cfgObj['includeWarnings']:
        print("Filtering out warnings (unless they are associated with a seizure or a fall event)")
        df=df.query("type=='Seizure' or type=='Fall' or osdAlarmState!=1")

    # Filter by date
    if (start is not None):
        startDateTime = pd.to_datetime(start, utc=True)
        dateQueryStr = 'dataTime >= "%s"' % startDateTime
        print("Applying Date Query: %s" % dateQueryStr)
        df = df.query(dateQueryStr)

    # Filter by end date
    if (end is not None):
        endDateTime = pd.to_datetime(end, utc=True)
        dateQueryStr = 'dataTime <= "%s"' % endDateTime
        print("Applying Date Query: %s" % dateQueryStr)
        df = df.query(dateQueryStr)

    # Filter out 'Test' data
    print("Filtering out events described as 'test'")
    df=df.query("not(desc.str.lower().str.contains('test'))")

    # Filter by data source
    excludeDataSources = cfgObj.get('excludeDataSources', None)
    if excludeDataSources:
        print("Filtering out data sources: %s" % excludeDataSources)
        df = df[~df['dataSource'].isin(excludeDataSources)]
        print("%d events remaining after data source exclusion" % len(df.index))

    includeDataSources = cfgObj.get('includeDataSources', None)
    if includeDataSources:
        print("Including only data sources: %s" % includeDataSources)
        df = df[df['dataSource'].isin(includeDataSources)]
        print("%d events remaining after data source inclusion filter" % len(df.index))

    # Group the data by userID and time period
    print("Grouping into periods of %s" % cfgObj['groupingPeriod'])
    groupedDf=df.groupby(['userId','type',pd.Grouper(
        key='dataTime',
        freq=cfgObj['groupingPeriod'])])

    # Loop through the grouped data to select best event from each group
    columnList = ['id', 'userId', 'dataTime', 'type', 'subType', 'osdAlarmState',
                  'dataSource', 'phoneAppVersion', 'watchAppVersion', 'desc']

    for groupParts, group in groupedDf:
        userId, eventType, dataTime = groupParts
        if (debug): print("\nStarting New Group....")
        
        # Select ALARM rows first
        alarmRows=group[group.osdAlarmState==2]
        if len(alarmRows.index)>0:
            outputRows = alarmRows
        else:
            # No alarm rows - select tagged rows
            taggedRows=group[group.type!='Unknown']
            if len(taggedRows.index)>0:
                outputRows = taggedRows
            else:
                # No tagged rows - use all rows
                outputRows = group

        # Now prioritize by whether there is a description
        if (debug): print("Selecting rows with desc")
        descRows = outputRows[outputRows.desc.str.len() > 1]
        if len(descRows.index)>0:
            if (debug): print("descRows:", len(descRows.index))
            outputRows = descRows
        else:
            if (debug): print("No rows with desc")

        # Keep the first row as representative
        outputRow = outputRows.head(1)
        
        # Add to appropriate dataframe based on event type
        if eventType == 'Seizure':
            subType = outputRow['subType'].iloc[0]
            if pd.isna(subType) or subType == '':
                subType = 'Unknown'
            
            if 'tonic' in str(subType).lower() or 'clonic' in str(subType).lower():
                tcUniqueEventsDf = pd.concat([tcUniqueEventsDf, outputRow])
            allSeizureUniqueEventsDf = pd.concat([allSeizureUniqueEventsDf, outputRow])
        elif eventType == 'Fall':
            fallUniqueEventsDf = pd.concat([fallUniqueEventsDf, outputRow])
        elif eventType == 'False Alarm':
            falseAlarmUniqueEventsDf = pd.concat([falseAlarmUniqueEventsDf, outputRow])
        elif eventType == 'NDA':
            ndaUniqueEventsDf = pd.concat([ndaUniqueEventsDf, outputRow])
        else:
            unknownUniqueEventsDf = pd.concat([unknownUniqueEventsDf, outputRow])

    print("\n=== Event Summary ===")
    print("All Seizures: %d events" % len(allSeizureUniqueEventsDf.index))
    print("Tonic-Clonic Seizures: %d events" % len(tcUniqueEventsDf.index))
    print("Fall Events: %d events" % len(fallUniqueEventsDf.index))
    print("False Alarms: %d events" % len(falseAlarmUniqueEventsDf.index))
    print("Unknown Events: %d events" % len(unknownUniqueEventsDf.index))
    print("NDA Events: %d events" % len(ndaUniqueEventsDf.index))

    # Convert dataframes to lists of event IDs
    seizureEventsLst = allSeizureUniqueEventsDf['id'].tolist() if not allSeizureUniqueEventsDf.empty else []
    tcEventsLst = tcUniqueEventsDf['id'].tolist() if not tcUniqueEventsDf.empty else []
    falseAlarmEventsLst = falseAlarmUniqueEventsDf['id'].tolist() if not falseAlarmUniqueEventsDf.empty else []
    unknownEventsLst = unknownUniqueEventsDf['id'].tolist() if not unknownUniqueEventsDf.empty else []
    fallEventsLst = fallUniqueEventsDf['id'].tolist() if not fallUniqueEventsDf.empty else []
    ndaEventsLst = ndaUniqueEventsDf['id'].tolist() if not ndaUniqueEventsDf.empty else []

    return (seizureEventsLst, tcEventsLst, falseAlarmEventsLst, 
            unknownEventsLst, fallEventsLst, ndaEventsLst)


def downloadAndProcessEvents(eventIdsList, configFname, debug=False):
    """
    Download detailed event data for a list of event IDs.
    Returns list of event dictionaries.
    """
    cfgObj = libosd.configUtils.loadConfig(configFname)
    osd = libosd.webApiConnection.WebApiConnection(cfg=cfgObj['credentialsFname'],
                                                   download=True,
                                                   debug=debug)
    
    # Filter out invalid events
    invalidEvents = cfgObj.get('invalidEvents', [])
    eventIdsList = [eid for eid in eventIdsList if eid not in invalidEvents]
    
    print(f"Downloading {len(eventIdsList)} events...")
    events = []
    for i, eventId in enumerate(eventIdsList):
        # Print progress every 10 events, or on first event
        if (i % 10 == 0) or (i == 0):
            print(f"Retrieved {i}/{len(eventIdsList)} events", flush=True)
        try:
            event = osd.getEvent(eventId, includeDatapoints=True)
            if event:
                events.append(event)
        except Exception as e:
            print(f"Error downloading event {eventId}: {e}", flush=True)
            continue
    
    print(f"Successfully downloaded {len(events)} events")
    return events


def loadExistingEvents(fname, debug=False):
    """
    Load existing events from a JSON file if it exists.
    Returns a list of events, or empty list if file doesn't exist.
    """
    if not os.path.exists(fname):
        print(f"No existing file found: {fname}")
        return []
    
    try:
        with open(fname, 'r') as f:
            events = json.load(f)
        print(f"Loaded {len(events)} existing events from {fname}")
        return events
    except Exception as e:
        print(f"Error loading existing events from {fname}: {e}")
        return []


def loadExistingEventsFromDb(db_path, event_type, debug=False):
    """
    Load existing events from SQLite database for a specific event type.
    Returns a list of events, or empty list if database doesn't exist.
    
    Parameters:
    -----------
    db_path : str
        Path to SQLite database
    event_type : str
        Event type filter (e.g., 'tcSeizures', 'allSeizures', 'fallEvents')
    debug : bool
        Enable debug output
    
    Returns:
    --------
    list : List of event dictionaries
    """
    if not os.path.exists(db_path):
        print(f"No existing database found: {db_path}")
        return []
    
    try:
        db = OsdWorkingDb(db_path)
        
        # Map event_type to database query filters
        # For now, load all events and filter by type if needed
        # In future, could add type filter to database query
        all_events = db.get_events(include_datapoints=True)
        
        # Filter by event type if needed
        # This is a simplified approach - in production might want more sophisticated filtering
        if event_type == 'tcSeizures':
            # Tonic-clonic seizures
            events = [e for e in all_events 
                     if e.get('type') == 'Seizure' and 
                     ('tonic' in str(e.get('subType', '')).lower() or 
                      'clonic' in str(e.get('subType', '')).lower())]
        elif event_type == 'allSeizures':
            events = [e for e in all_events if e.get('type') == 'Seizure']
        elif event_type == 'fallEvents':
            events = [e for e in all_events if e.get('type') == 'Fall']
        elif event_type == 'falseAlarms':
            events = [e for e in all_events if e.get('type') == 'False Alarm']
        elif event_type == 'ndaEvents':
            events = [e for e in all_events if e.get('type') == 'NDA']
        else:
            events = all_events
        
        print(f"Loaded {len(events)} existing {event_type} events from database")
        return events
    except Exception as e:
        print(f"Error loading existing events from database {db_path}: {e}")
        return []


def getNewEventIds(eventIdsList, existingEvents, debug=False):
    """
    Identify which event IDs from eventIdsList are not already in existingEvents.
    Returns list of new event IDs.
    
    Note: Normalizes IDs to strings for comparison to handle mixed int/str types.
    Also checks for events that were merged (tracked in description field and/or merged_from_events).
    """
    import re
    import json
    
    # Normalize existing IDs to strings for comparison
    existing_ids = {str(event.get('id')) for event in existingEvents if 'id' in event}
    
    # Also extract merged event IDs from multiple sources
    merged_ids = set()
    for event in existingEvents:
        # Check merged_from_events field (JSON array)
        merged_from = event.get('merged_from_events')
        if merged_from:
            if isinstance(merged_from, str):
                try:
                    merged_list = json.loads(merged_from)
                    merged_ids.update(str(mid) for mid in merged_list)
                except:
                    pass
            elif isinstance(merged_from, list):
                merged_ids.update(str(mid) for mid in merged_from)
        
        # Also check description field (legacy/backup)
        desc = event.get('desc', '')
        if desc and 'merged event' in desc.lower():
            # Extract IDs from "Includes data from merged event(s): 123, 456, 789"
            match = re.search(r'merged event\(s\): ([\d, ]+)', desc)
            if match:
                id_str = match.group(1)
                # Split by comma and strip whitespace
                for mid in id_str.split(','):
                    merged_ids.add(mid.strip())
    
    # Combine both sets
    all_existing_ids = existing_ids | merged_ids
    
    # Normalize new IDs to strings for comparison, but return original type
    new_ids = []
    for eid in eventIdsList:
        if str(eid) not in all_existing_ids:
            new_ids.append(eid)
    
    if debug:
        print(f"Debug: Primary event IDs: {len(existing_ids)}")
        print(f"Debug: Merged event IDs: {len(merged_ids)}")
        print(f"Debug: Total existing IDs: {len(all_existing_ids)}")
        print(f"Debug: Sample merged IDs: {sorted(list(merged_ids))[:5]}")
    
    print(f"Existing events: {len(existing_ids)}, Merged events: {len(merged_ids)}, New events to download: {len(new_ids)}")
    return new_ids


def filterEventsByDataSources(events, excludeDataSources=None, includeDataSources=None, debug=False):
    """
    Filter events by data source name.
    Matches behavior of original makeOsdDb.py's removeEventsByDataSources().
    
    Parameters:
    -----------
    events : list
        List of event dictionaries
    excludeDataSources : list or None
        List of dataSourceName values to exclude (e.g., ['Phone', 'AndroidWear'])
    includeDataSources : list or None
        List of dataSourceName values to include exclusively
    debug : bool
        Print debug information
    
    Returns:
    --------
    list : Filtered events
    """
    if not excludeDataSources and not includeDataSources:
        return events
    
    filtered_events = events
    
    # Apply exclude filter
    if excludeDataSources:
        before_count = len(filtered_events)
        filtered_events = [
            e for e in filtered_events 
            if e.get('dataSourceName') not in excludeDataSources
        ]
        removed = before_count - len(filtered_events)
        if removed > 0:
            print(f"Filtered out {removed} events with excluded data sources {excludeDataSources}")
    
    # Apply include filter
    if includeDataSources:
        before_count = len(filtered_events)
        filtered_events = [
            e for e in filtered_events 
            if e.get('dataSourceName') in includeDataSources
        ]
        removed = before_count - len(filtered_events)
        if removed > 0:
            print(f"Filtered out {removed} events not in included data sources {includeDataSources}")
    
    return filtered_events


def saveEventsToDatabase(eventIdsList, event_type, db_path, configFname, debug=False):
    """
    Download new events and merge with existing events in SQLite database.
    Uses refactored processing modules.
    
    Parameters:
    -----------
    eventIdsList : list
        List of event IDs to process
    event_type : str
        Event type filter (e.g., 'tcSeizures', 'allSeizures', 'fallEvents', etc.)
    db_path : str
        Path to SQLite database
    configFname : str
        Path to configuration file
    debug : bool
        Enable debug output
    """
    print(f"\n=== Processing {event_type} ===")
    
    # Load existing events from database
    existing_events = loadExistingEventsFromDb(db_path, event_type, debug)
    
    # Identify which events are new
    new_event_ids = getNewEventIds(eventIdsList, existing_events, debug)
    
    if not new_event_ids:
        print(f"No new events to add to {event_type}")
        print(f"Database already contains {len(existing_events)} events for this type")
        return
    
    # Download new events
    new_events = downloadAndProcessEvents(new_event_ids, configFname, debug)
    
    if not new_events:
        print(f"No new events downloaded for {event_type}")
        return
    
    print(f"\n--- Applying Refactored Processing to New Events ---")
    
    # Phase 1: Validate events
    print("\n[1/4] Validating new events...")
    valid_events, validation_report = validate_events_batch(new_events)
    print(f"Valid: {validation_report['valid']}, Skipped: {validation_report['skipped']}")
    if validation_report['skipped'] > 0:
        print("Skipped events:")
        for skipped in validation_report['skipped_events'][:10]:  # Show first 10
            event_id = skipped['event'].get('id', 'unknown')
            reason = skipped['reason']
            print(f"  Event {event_id}: {reason}")
    
    # Phase 2: Normalize datetimes
    print("\n[2/4] Normalizing datetime formats...")
    normalized_events, normalization_stats = normalize_events_batch(valid_events)
    print(f"Normalized {normalization_stats['events_normalized']} events")
    
    # Phase 3: Remove duplicates
    print("\n[3/4] Removing duplicates...")
    deduplicated_events, dedup_info = remove_duplicate_events(
        normalized_events,
        method='hash',
        keep='first'
    )
    print(f"Removed {dedup_info['duplicates_removed']} duplicates")
    print(f"Kept {dedup_info['total_output']} unique events")
    
    # Phase 4: Merge new events with existing events
    print(f"\n[4/4] Merging {len(deduplicated_events)} new events with {len(existing_events)} existing events...")
    
    # CRITICAL: Mark existing events to preserve their IDs during grouping
    for event in existing_events:
        event['_is_existing_event'] = True
    
    # New events are NOT marked (or explicitly marked as False)
    for event in deduplicated_events:
        event['_is_existing_event'] = False
    
    # Combine all events
    all_events = existing_events + deduplicated_events
    print(f"Combined total: {len(all_events)} events")
    print(f"  Existing (published): {len(existing_events)} events")
    print(f"  New (downloaded): {len(deduplicated_events)} events")
    
    # Remove duplicates across all events (in case of overlaps)
    all_events, dedup_info2 = remove_duplicate_events(
        all_events,
        method='hash',
        keep='first'
    )
    if dedup_info2['duplicates_removed'] > 0:
        print(f"Removed {dedup_info2['duplicates_removed']} duplicate events from combined set")
    
    # Filter by data sources (both excludeDataSources and includeDataSources)
    cfgObj = libosd.configUtils.loadConfig(configFname)
    excludeDataSources = cfgObj.get('excludeDataSources', None)
    includeDataSources = cfgObj.get('includeDataSources', None)
    if includeDataSources == []:  # Empty list means no filter
        includeDataSources = None
    
    if excludeDataSources or includeDataSources:
        print(f"\nFiltering by data sources...")
        all_events = filterEventsByDataSources(
            all_events,
            excludeDataSources=excludeDataSources,
            includeDataSources=includeDataSources,
            debug=debug
        )
        print(f"After data source filtering: {len(all_events)} events")
    
    # Apply sliding window grouping to ALL events (both old and new)
    # NDA events are NOT grouped as they are expected to be contiguous
    print("\n[5/5] Applying sliding window grouping to complete dataset (3min threshold)...")
    print("  Note: NDA events are excluded from grouping (expected to be contiguous)")
    final_events, grouping_info = apply_sliding_window_grouping(
        all_events,
        time_threshold='3min',
        selection_strategy='alarm_first',
        concatenate_datapoints_flag=True,
        exclude_event_types=['nda'],
        update_desc=True
    )
    print(f"Grouped {len(all_events)} events into {len(final_events)} final events")
    groups_merged = len(all_events) - len(final_events) - grouping_info.get('excluded_events', 0)
    if groups_merged > 0:
        print(f"Merged {groups_merged} event groups")
    if grouping_info.get('excluded_events', 0) > 0:
        print(f"Excluded {grouping_info['excluded_events']} NDA events from grouping")
    
    # Load config for skipElements and invalidEvents
    cfgObj = libosd.configUtils.loadConfig(configFname)
    skipElements = cfgObj.get('skipElements', [])
    invalidEvents = cfgObj.get('invalidEvents', [])
    
    # Filter out invalid events
    if invalidEvents:
        print(f"\nFiltering out {len(invalidEvents)} invalid event IDs")
        final_events = [e for e in final_events if e.get('id') not in invalidEvents]
        print(f"After filtering: {len(final_events)} events")
    
    # Remove skipElements from events
    if skipElements:
        print(f"\nRemoving skip elements: {skipElements}")
        for event in final_events:
            for elem in skipElements:
                event.pop(elem, None)
            # Also remove from datapoints (only if datapoints is a list)
            if 'datapoints' in event and isinstance(event['datapoints'], list):
                for dp in event['datapoints']:
                    if isinstance(dp, dict):  # Only process if datapoint is a dict
                        for elem in skipElements:
                            dp.pop(elem, None)
    
    # Save to SQLite database
    print(f"\nSaving {len(final_events)} events to database: {db_path}")
    
    # Clean up internal markers before saving
    for event in final_events:
        # Remove internal tracking fields
        event.pop('_is_existing_event', None)
    
    # Convert any pandas Timestamps to strings for database serialization
    for event in final_events:
        if 'dataTime' in event and hasattr(event['dataTime'], 'isoformat'):
            event['dataTime'] = event['dataTime'].isoformat()
        if 'datapoints' in event and isinstance(event['datapoints'], list):
            for dp in event['datapoints']:
                if isinstance(dp, dict):
                    if 'dataTime' in dp and hasattr(dp['dataTime'], 'isoformat'):
                        dp['dataTime'] = dp['dataTime'].isoformat()
                    if 'time' in dp and hasattr(dp['time'], 'isoformat'):
                        dp['time'] = dp['time'].isoformat()
    
    # Open database and save events
    db = OsdWorkingDb(db_path)
    db.add_events(final_events)
    
    print(f"✓ Saved {len(final_events)} events to database")
    print(f"  Event type: {event_type}")
    print(f"  Database: {db_path}")


def generateIndexFiles(osdb_dir, groupingPeriod, debug=False):
    """
    Generate CSV index files from JSON event files.
    Matches functionality of makeIndex.py.
    
    Parameters:
    osdb_dir - directory containing JSON files
    groupingPeriod - grouping period string (e.g., '3min')
    debug - print debug information
    """
    print("\n" + "="*70)
    print("Generating Index Files")
    print("="*70)
    
    # List of JSON files to process
    json_files = [
        f"osdb_{groupingPeriod}_tcSeizures.json",
        f"osdb_{groupingPeriod}_allSeizures.json",
        f"osdb_{groupingPeriod}_fallEvents.json",
        f"osdb_{groupingPeriod}_falseAlarms.json",
        f"osdb_{groupingPeriod}_ndaEvents.json",
    ]
    
    for json_fname in json_files:
        json_path = os.path.join(osdb_dir, json_fname)
        
        if not os.path.exists(json_path):
            if debug:
                print(f"Skipping {json_fname} (file not found)")
            continue
        
        # Generate output CSV filename
        csv_fname = json_fname.replace('.json', '.csv')
        csv_path = os.path.join(osdb_dir, csv_fname)
        
        try:
            if debug:
                print(f"Processing {json_fname}...")
            
            # Load events using OsdDbConnection
            osd = libosd.osdDbConnection.OsdDbConnection(cacheDir=None, debug=debug)
            osd.loadDbFile(json_path, useCacheDir=False)
            
            # Save index file
            osd.saveIndexFile(csv_path, useCacheDir=False)
            
            print(f"✓ Generated index: {csv_fname}")
            
        except Exception as e:
            print(f"Error generating index for {json_fname}: {e}")
            continue
    
    print("="*70)


def generateSummaryGraphs(osdb_dir, groupingPeriod, output_dir=None, threshold=5, debug=False):
    """
    Generate summary graphs from JSON event files.
    Uses the generateGraphs module.
    
    Parameters:
    osdb_dir - directory containing JSON files
    groupingPeriod - grouping period string (e.g., '3min')
    output_dir - output directory for graphs (default: osdb_dir/output)
    threshold - minimum number of events for individual user display
    debug - print debug information
    """
    print("\n" + "="*70)
    print("Generating Summary Graphs")
    print("="*70)
    
    # Default output directory
    if output_dir is None:
        output_dir = os.path.join(osdb_dir, 'output')
    
    # List of JSON files to process (for graph generation, typically all event types)
    json_files = []
    json_file_patterns = [
        f"osdb_{groupingPeriod}_tcSeizures.json",
        f"osdb_{groupingPeriod}_allSeizures.json",
        f"osdb_{groupingPeriod}_fallEvents.json",
        f"osdb_{groupingPeriod}_falseAlarms.json",
        f"osdb_{groupingPeriod}_ndaEvents.json",
    ]
    
    for json_fname in json_file_patterns:
        json_path = os.path.join(osdb_dir, json_fname)
        if os.path.exists(json_path):
            json_files.append(json_path)
    
    if not json_files:
        print("No JSON files found for graph generation")
        return False
    
    if debug:
        print(f"Processing files: {json_files}")
    
    # Generate graphs using the generateGraphs module
    success = generateGraphs.generate_all_graphs(
        json_files,
        output_dir,
        threshold=threshold,
        debug=debug
    )
    
    if success:
        print(f"✓ Graphs saved to: {output_dir}")
    else:
        print("Graph generation failed")
    
    print("="*70)
    return success


def publishDatabaseToJson(db_path, osdb_dir, groupingPeriod, configFname, debug=False):
    """
    Export events from SQLite database to JSON files for publication.
    Creates separate JSON files for each event type.
    
    Parameters:
    -----------
    db_path : str
        Path to SQLite database
    osdb_dir : str
        Output directory for JSON files
    groupingPeriod : str
        Grouping period string (e.g., '3min')
    configFname : str
        Path to configuration file
    debug : bool
        Enable debug output
    """
    print("\n" + "="*70)
    print("Publishing Database to JSON Files")
    print("="*70)
    print(f"Database: {db_path}")
    print(f"Output directory: {osdb_dir}")
    print("="*70)
    
    if not os.path.exists(db_path):
        print(f"Error: Database not found: {db_path}")
        return False
    
    # Create output directory if needed
    os.makedirs(osdb_dir, exist_ok=True)
    
    # Load configuration
    cfgObj = libosd.configUtils.loadConfig(configFname)
    skipElements = cfgObj.get('skipElements', [])
    invalidEvents = cfgObj.get('invalidEvents', [])
    excludeDataSources = cfgObj.get('excludeDataSources', None)
    includeDataSources = cfgObj.get('includeDataSources', None)
    if includeDataSources == []:
        includeDataSources = None
    
    # Load all events from database
    db = OsdWorkingDb(db_path)
    all_events = db.get_events(include_datapoints=True)
    print(f"Loaded {len(all_events)} total events from database")
    
    # Filter by data sources
    if excludeDataSources or includeDataSources:
        print(f"Applying data source filtering...")
        all_events = filterEventsByDataSources(
            all_events,
            excludeDataSources=excludeDataSources,
            includeDataSources=includeDataSources,
            debug=debug
        )
        print(f"After data source filtering: {len(all_events)} events")
    
    # Filter out invalid events
    if invalidEvents:
        print(f"Filtering out {len(invalidEvents)} invalid event IDs")
        all_events = [e for e in all_events if e.get('id') not in invalidEvents]
        print(f"After filtering: {len(all_events)} events")
    
    # Remove skipElements from events
    if skipElements:
        print(f"Removing skip elements: {skipElements}")
        for event in all_events:
            for elem in skipElements:
                event.pop(elem, None)
            # Also remove from datapoints
            if 'datapoints' in event and isinstance(event['datapoints'], list):
                for dp in event['datapoints']:
                    if isinstance(dp, dict):
                        for elem in skipElements:
                            dp.pop(elem, None)
    
    # Split events by type and save to separate JSON files
    event_categories = {
        'tcSeizures': [],
        'allSeizures': [],
        'fallEvents': [],
        'falseAlarms': [],
        'ndaEvents': [],
    }
    
    for event in all_events:
        event_type = event.get('type', 'Unknown')
        sub_type = event.get('subType', '')
        
        if event_type == 'Seizure':
            # Check if tonic-clonic
            if 'tonic' in str(sub_type).lower() or 'clonic' in str(sub_type).lower():
                event_categories['tcSeizures'].append(event)
            # All seizures
            event_categories['allSeizures'].append(event)
        elif event_type == 'Fall':
            event_categories['fallEvents'].append(event)
        elif event_type == 'False Alarm':
            event_categories['falseAlarms'].append(event)
        elif event_type == 'NDA':
            event_categories['ndaEvents'].append(event)
    
    # Save each category to a JSON file
    for category, events in event_categories.items():
        if not events:
            print(f"No events for category: {category}")
            continue
        
        fname = os.path.join(osdb_dir, f"osdb_{groupingPeriod}_{category}.json")
        
        with open(fname, 'w') as f:
            json.dump(events, f, indent=2)
        
        print(f"✓ Saved {len(events)} events to {fname}")
    
    print("="*70)
    print("✓ Publication Complete!")
    print("="*70)
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Create an anonymised database of unique seizure-like events (using refactored processing with SQLite)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update database with new events from server
  python3 makeOsdDb_refactored_wrapper.py --config ../osdb.cfg --osdb-dir /path/to/osdb
  
  # Specify custom database path
  python3 makeOsdDb_refactored_wrapper.py --config ../osdb.cfg --osdb-dir /path/to/osdb --database /path/to/custom.db
  
  # Publish database to JSON files
  python3 makeOsdDb_refactored_wrapper.py --config ../osdb.cfg --osdb-dir /path/to/osdb --publish
  
  # Full workflow: update + publish + generate index + graphs
  python3 makeOsdDb_refactored_wrapper.py --config ../osdb.cfg --osdb-dir /path/to/osdb --publish --generate-index --generate-graphs
        """
    )
    
    parser.add_argument('--config', default="../osdb.cfg",
                        help='Path to osdb.cfg configuration file')
    parser.add_argument('--osdb-dir', required=True,
                        help='Output directory for OSDB files')
    parser.add_argument('--database', default=None,
                        help='Path to SQLite database (default: {osdb-dir}/osdb_working.db)')
    parser.add_argument('--publish', action='store_true',
                        help='Publish database to JSON files (instead of updating database)')
    parser.add_argument('--start', default=None,
                        help="Start date for saving data (yyyy-mm-dd format)")
    parser.add_argument('--end', default=None,
                        help="End date for saving data (yyyy-mm-dd format)")
    parser.add_argument('--debug', action='store_true',
                        help="Write debugging information to screen")
    parser.add_argument('--generate-index', action='store_true',
                        help="Generate CSV index files from JSON files after publishing")
    parser.add_argument('--generate-graphs', action='store_true',
                        help="Generate summary graphs from JSON files after publishing")
    parser.add_argument('--graph-output', default=None,
                        help="Output directory for graphs (default: osdb-dir/output)")
    parser.add_argument('--graph-threshold', type=int, default=5,
                        help="Minimum number of events for individual user display in graphs (default: 5)")
    
    args = parser.parse_args()
    
    # Load config to get database path and other settings
    cfgObj = libosd.configUtils.loadConfig(args.config)
    if cfgObj is None:
        print(f"ERROR: Failed to load configuration file: {args.config}")
        print(f"Please check that the file exists and is readable.")
        sys.exit(1)
    
    groupingPeriod = cfgObj.get('groupingPeriod', '3min')
    
    # Determine database path (CLI argument takes precedence over config file)
    if args.database is None:
        # Try to get from config file
        config_db_path = cfgObj.get('databasePath', None)
        if config_db_path:
            args.database = config_db_path
        else:
            # Default to osdb_working.db in output directory
            args.database = os.path.join(args.osdb_dir, 'osdb_working.db')
    
    print("="*70)
    print("makeOsdDb Refactored Wrapper (SQLite)")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"Output directory: {args.osdb_dir}")
    print(f"Database: {args.database}")
    if args.publish:
        print(f"Mode: PUBLISH (export database to JSON)")
    else:
        print(f"Mode: UPDATE (download and process events to database)")
    print(f"Start date: {args.start or '(no filter - all events)'}")
    print(f"End date: {args.end or '(no filter - all events)'}")
    print("="*70)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.osdb_dir, exist_ok=True)
    
    # Create database directory if it doesn't exist
    db_dir = os.path.dirname(args.database)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    
    # Publish mode: export database to JSON files
    if args.publish:
        success = publishDatabaseToJson(
            args.database,
            args.osdb_dir,
            groupingPeriod,
            args.config,
            debug=args.debug
        )
        
        if not success:
            print("Error: Publication failed")
            sys.exit(1)
        
        # Post-processing: Generate index files if requested
        if args.generate_index:
            generateIndexFiles(args.osdb_dir, groupingPeriod, debug=args.debug)
        
        # Post-processing: Generate summary graphs if requested
        if args.generate_graphs:
            generateSummaryGraphs(
                args.osdb_dir,
                groupingPeriod,
                output_dir=args.graph_output,
                threshold=args.graph_threshold,
                debug=args.debug
            )
        
        print("\n" + "="*70)
        print("✓ All Operations Complete!")
        print("="*70)
        return
    
    # Update mode: download and process events to database
    print("\n=== Step 1: Fetching and Filtering Events from Server ===")
    (seizureEventsLst, tcEventsLst, falseAlarmEventsLst,
     unknownEventsLst, fallEventsLst, ndaEventsLst) = \
        getUniqueEventsListsFromServer(
            args.config,
            start=args.start,
            end=args.end,
            outDir=args.osdb_dir,
            debug=args.debug
        )
    
    # Process and save each category to database
    print("\n" + "="*70)
    print("Step 2: Downloading and Processing Events to Database")
    print("="*70)
    
    # Tonic-Clonic Seizures
    if tcEventsLst:
        saveEventsToDatabase(tcEventsLst, 'tcSeizures', args.database, args.config, debug=args.debug)
    
    # All Seizures
    if seizureEventsLst:
        saveEventsToDatabase(seizureEventsLst, 'allSeizures', args.database, args.config, debug=args.debug)
    
    # Fall Events
    if fallEventsLst:
        saveEventsToDatabase(fallEventsLst, 'fallEvents', args.database, args.config, debug=args.debug)
    
    # False Alarms
    if falseAlarmEventsLst:
        saveEventsToDatabase(falseAlarmEventsLst, 'falseAlarms', args.database, args.config, debug=args.debug)
    
    # NDA Events
    if ndaEventsLst:
        saveEventsToDatabase(ndaEventsLst, 'ndaEvents', args.database, args.config, debug=args.debug)
    
    print("\n" + "="*70)
    print("✓ Database Update Complete!")
    print("="*70)
    print(f"Database: {args.database}")
    print("To publish to JSON files, run:")
    print(f"  python3 {os.path.basename(__file__)} --config {args.config} --osdb-dir {args.osdb_dir} --publish")
    print("="*70)


if __name__ == '__main__':
    main()

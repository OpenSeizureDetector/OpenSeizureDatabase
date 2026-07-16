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

Usage:
    python3 makeOsdDb_refactored_wrapper.py --config ../osdb.cfg --osdb-dir /path/to/osdb
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
        if (i % 100 == 0):
            print(f"Retrieved {i}/{len(eventIdsList)} events")
        try:
            event = osd.getEvent(eventId, includeDatapoints=True)
            if event:
                events.append(event)
        except Exception as e:
            print(f"Error downloading event {eventId}: {e}")
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


def getNewEventIds(eventIdsList, existingEvents, debug=False):
    """
    Identify which event IDs from eventIdsList are not already in existingEvents.
    Returns list of new event IDs.
    """
    existing_ids = {event.get('id') for event in existingEvents if 'id' in event}
    new_ids = [eid for eid in eventIdsList if eid not in existing_ids]
    
    print(f"Existing events: {len(existing_ids)}, New events to download: {len(new_ids)}")
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


def saveEventsAsJson(eventIdsList, fname, configFname, debug=False):
    """
    Download new events and merge with existing events in JSON file.
    Uses refactored processing modules.
    """
    print(f"\n=== Processing {fname} ===")
    
    # Load existing events from file
    existing_events = loadExistingEvents(fname, debug)
    
    # Identify which events are new
    new_event_ids = getNewEventIds(eventIdsList, existing_events, debug)
    
    if not new_event_ids:
        print(f"No new events to add to {fname}")
        # Still need to filter existing events by config
        cfgObj = libosd.configUtils.loadConfig(configFname)
        invalidEvents = cfgObj.get('invalidEvents', [])
        excludeDataSources = cfgObj.get('excludeDataSources', None)
        includeDataSources = cfgObj.get('includeDataSources', None)
        if includeDataSources == []:
            includeDataSources = None
        
        # Filter by data sources
        if excludeDataSources or includeDataSources:
            print(f"Applying data source filtering...")
            existing_events = filterEventsByDataSources(
                existing_events,
                excludeDataSources=excludeDataSources,
                includeDataSources=includeDataSources,
                debug=debug
            )
        
        # Filter by invalid event IDs
        filtered_events = [e for e in existing_events if e.get('id') not in invalidEvents]
        
        if len(filtered_events) < len(existing_events):
            print(f"Filtered out {len(existing_events) - len(filtered_events)} invalid events")
            with open(fname, 'w') as f:
                json.dump(filtered_events, f, indent=2)
            print(f"✓ Updated {fname} (removed invalid events)")
        return
    
    # Download new events
    new_events = downloadAndProcessEvents(new_event_ids, configFname, debug)
    # Download new events
    new_events = downloadAndProcessEvents(new_event_ids, configFname, debug)
    
    if not new_events:
        print(f"No new events downloaded for {fname}")
        # Just save existing events (possibly filtered)
        cfgObj = libosd.configUtils.loadConfig(configFname)
        invalidEvents = cfgObj.get('invalidEvents', [])
        excludeDataSources = cfgObj.get('excludeDataSources', None)
        includeDataSources = cfgObj.get('includeDataSources', None)
        if includeDataSources == []:
            includeDataSources = None
        
        # Filter by data sources
        if excludeDataSources or includeDataSources:
            print(f"Applying data source filtering...")
            existing_events = filterEventsByDataSources(
                existing_events,
                excludeDataSources=excludeDataSources,
                includeDataSources=includeDataSources,
                debug=debug
            )
        
        # Filter by invalid event IDs
        filtered_events = [e for e in existing_events if e.get('id') not in invalidEvents]
        
        with open(fname, 'w') as f:
            json.dump(filtered_events, f, indent=2)
        print(f"✓ Updated {fname}")
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
    
    # Save to JSON
    print(f"\nSaving {len(final_events)} events to {fname}")
    
    # Clean up internal markers before saving
    for event in final_events:
        # Remove internal tracking fields
        event.pop('_is_existing_event', None)
    
    # Convert any pandas Timestamps to strings for JSON serialization
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
    
    with open(fname, 'w') as f:
        json.dump(final_events, f, indent=2)
    
    print(f"✓ Saved {fname}")


def main():
    parser = argparse.ArgumentParser(
        description='Create an anonymised database of unique seizure-like events (using refactored processing)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--config', default="../osdb.cfg",
                        help='Path to osdb.cfg configuration file')
    parser.add_argument('--osdb-dir', required=True,
                        help='Output directory for OSDB files')
    parser.add_argument('--start', default=None,
                        help="Start date for saving data (yyyy-mm-dd format)")
    parser.add_argument('--end', default=None,
                        help="End date for saving data (yyyy-mm-dd format)")
    parser.add_argument('--debug', action='store_true',
                        help="Write debugging information to screen")
    
    args = parser.parse_args()
    
    print("="*70)
    print("makeOsdDb Refactored Wrapper")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"Output directory: {args.osdb_dir}")
    print(f"Start date: {args.start or 'None'}")
    print(f"End date: {args.end or 'None'}")
    print("="*70)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.osdb_dir, exist_ok=True)
    
    # Get unique events lists from server
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
    
    # Load config to get grouping period
    cfgObj = libosd.configUtils.loadConfig(args.config)
    groupingPeriod = cfgObj.get('groupingPeriod', '3min')
    
    # Process and save each category
    print("\n" + "="*70)
    print("Step 2: Downloading and Processing Events")
    print("="*70)
    
    # Tonic-Clonic Seizures
    if tcEventsLst:
        fname = os.path.join(args.osdb_dir, f"osdb_{groupingPeriod}_tcSeizures.json")
        saveEventsAsJson(tcEventsLst, fname, args.config, debug=args.debug)
    
    # All Seizures
    if seizureEventsLst:
        fname = os.path.join(args.osdb_dir, f"osdb_{groupingPeriod}_allSeizures.json")
        saveEventsAsJson(seizureEventsLst, fname, args.config, debug=args.debug)
    
    # Fall Events
    if fallEventsLst:
        fname = os.path.join(args.osdb_dir, f"osdb_{groupingPeriod}_fallEvents.json")
        saveEventsAsJson(fallEventsLst, fname, args.config, debug=args.debug)
    
    # False Alarms
    if falseAlarmEventsLst:
        fname = os.path.join(args.osdb_dir, f"osdb_{groupingPeriod}_falseAlarms.json")
        saveEventsAsJson(falseAlarmEventsLst, fname, args.config, debug=args.debug)
    
    # NDA Events
    if ndaEventsLst:
        fname = os.path.join(args.osdb_dir, f"osdb_{groupingPeriod}_ndaEvents.json")
        saveEventsAsJson(ndaEventsLst, fname, args.config, debug=args.debug)
    
    print("\n" + "="*70)
    print("✓ Complete!")
    print("="*70)


if __name__ == '__main__':
    main()

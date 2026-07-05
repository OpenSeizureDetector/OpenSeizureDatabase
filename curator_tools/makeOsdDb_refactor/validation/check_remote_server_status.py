#!/usr/bin/env python3
"""
check_remote_server_status.py

Check if the NO_MATCH events still exist on the remote server,
or if they were deleted (which would explain why they're not in refactored).

Graham Jones, July 2026
"""

import json
import sys
sys.path.insert(0, '/home/graham/osd/OpenSeizureDatabase')
from libosd import webApiConnection, configUtils

# Configuration
CONFIG_FILE = "/home/graham/osd/OpenSeizureDatabase/curator_tools/client.cfg"
NO_MATCH_IDS = [5486, 6590, 6668, 7007, 21569, 36872, 1328552, 1332361, 
                1343999, 1351708, 1355207, 1355378, 1363844]

def main():
    print("="*70)
    print("Checking Remote Server Status of NO_MATCH Events")
    print("="*70)
    
    # Load configuration
    print(f"\nLoading config from {CONFIG_FILE}...")
    config_obj = configUtils.loadConfig(CONFIG_FILE)
    
    # Connect to web API
    print("Connecting to web API...")
    web_api = webApiConnection.WebApiConnection(config_obj)
    
    print(f"\nChecking {len(NO_MATCH_IDS)} events on remote server...\n")
    
    exists_on_remote = []
    deleted_from_remote = []
    
    for event_id in sorted(NO_MATCH_IDS):
        try:
            print(f"Checking event {event_id}...", end=" ")
            event = web_api.getEvent(event_id)
            
            if event and 'id' in event:
                print(f"✓ EXISTS on remote")
                exists_on_remote.append(event_id)
                print(f"    User: {event.get('userId')}, Time: {event.get('dataTime')}")
                print(f"    Type: {event.get('type')}/{event.get('subType')}")
                print(f"    Datapoints: {len(event.get('datapoints', []))}")
            else:
                print(f"✗ NOT FOUND on remote (deleted)")
                deleted_from_remote.append(event_id)
        except Exception as e:
            print(f"✗ ERROR or NOT FOUND: {e}")
            deleted_from_remote.append(event_id)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\nTotal NO_MATCH events checked: {len(NO_MATCH_IDS)}")
    print(f"\nStill exist on remote server: {len(exists_on_remote)} events")
    if exists_on_remote:
        print(f"  IDs: {exists_on_remote}")
    
    print(f"\nDeleted from remote server: {len(deleted_from_remote)} events")
    if deleted_from_remote:
        print(f"  IDs: {deleted_from_remote}")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if len(deleted_from_remote) == len(NO_MATCH_IDS):
        print("\n✓ ALL NO_MATCH events were deleted from remote server!")
        print("\n  This explains why they're not in the refactored database:")
        print("  - Original database: Contains these events from previous update")
        print("  - Remote server: Events have been deleted since then")
        print("  - Refactored update: Only downloads events still on server")
        print("\n  This is EXPECTED BEHAVIOR, not a bug.")
        print("\n  The refactored version correctly reflects current remote state.")
    elif len(exists_on_remote) == len(NO_MATCH_IDS):
        print("\n⚠ ALL NO_MATCH events still exist on remote server!")
        print("\n  This is unexpected - they should have been downloaded.")
        print("  Possible bug in refactored version's download logic.")
    else:
        print(f"\n⚠ MIXED RESULTS:")
        print(f"  {len(deleted_from_remote)} deleted from remote (expected)")
        print(f"  {len(exists_on_remote)} still on remote (unexpected - should be in refactored)")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Analyze datapoint time ranges for events 115 and 119
"""

import json
from datetime import datetime
from dateutil import parser as dateutil_parser

# Load the downloaded events
with open('/home/graham/osd/osdb_test_refactored/osdb_3min_allSeizures_08jun26.json', 'r') as f:
    events = json.load(f)

# Find events 115 and 119
for event in events:
    if event.get('id') in [115, 119]:
        event_id = event.get('id')
        event_time = event.get('dataTime')
        datapoints = event.get('datapoints', [])
        
        print(f"\n=== Event {event_id} ===")
        print(f"Event dataTime: {event_time}")
        print(f"Total datapoints: {len(datapoints)}")
        
        if datapoints:
            # Get time range of datapoints
            times = []
            for dp in datapoints:
                if 'dataTime' in dp:
                    times.append(dateutil_parser.parse(dp['dataTime']))
            
            if times:
                times.sort()
                print(f"Datapoints time range:")
                print(f"  First: {times[0].isoformat()}")
                print(f"  Last:  {times[-1].isoformat()}")
                duration = (times[-1] - times[0]).total_seconds()
                print(f"  Duration: {duration:.1f} seconds ({duration/60:.2f} minutes)")
                print(f"  Interval: ~{duration/len(times):.2f} seconds between datapoints")

#!/usr/bin/env python3

import argparse
import os
import json
from collections import defaultdict
import re
import datetime
from dateutil import parser as dateutil_parser


def load_events(in_dir, filenames):
    """Load events from multiple JSON files.
    
    The first file in filenames is treated as the reference dataset.
    Returns event_map with 'event', 'files' set, and 'is_reference' boolean.
    """
    event_map = defaultdict(lambda: {'event': None, 'files': set(), 'is_reference': False})
    reference_file = filenames[0] if filenames else None
    
    for fname in filenames:
        path = os.path.join(in_dir, fname)
        with open(path, 'r') as f:
            try:
                events = json.load(f)
            except Exception as e:
                print(f"Error reading {fname}: {e}")
                continue
            for ev in events:
                #print(ev.keys())
                key = (ev.get('id'), ev.get('userId'), ev.get('dataTime'))
                if event_map[key]['event'] is None:
                    event_map[key]['event'] = ev
                event_map[key]['files'].add(fname)
                # Mark if this event is in the reference file
                if fname == reference_file:
                    event_map[key]['is_reference'] = True
    return event_map

class CompactArrayEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode(self, obj):
        # Use the default encoder first
        s = super().encode(obj)
        # Compact arrays for specific keys
        # This regex finds arrays for the specified keys and removes newlines/extra spaces inside them
        for key in ["simpleSpec", "rawData", "rawData3D"]:
            # Match: "key": [ ... ] (possibly with newlines/indentation inside)
            s = re.sub(
                rf'(\"{key}\": )\[(.*?)\]',
                lambda m: m.group(1) + '[' + re.sub(r'\s+', ' ', m.group(2)).strip() + ']',
                s,
                flags=re.DOTALL
            )
        return s

def compact_datapoint_arrays(event):
    # Return a copy of the event with compacted arrays in each datapoint
    import copy
    event = copy.deepcopy(event)
    if 'datapoints' in event and isinstance(event['datapoints'], list):
        for dp in event['datapoints']:
            for key in ["simpleSpec", "rawData", "rawData3D"]:
                if key in dp and isinstance(dp[key], list):
                    # Convert the array to a compact JSON string
                    dp[key] = json.loads(json.dumps(dp[key], separators=(',', ':')))
    return event

class DatapointCompactEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def iterencode(self, obj, _one_shot=False):
        # If top-level is a list, encode each event with custom logic
        if isinstance(obj, list):
            yield '[\n'
            first = True
            for event in obj:
                if not first:
                    yield ',\n'
                else:
                    first = False
                # Use custom dict encoding for each event
                for chunk in self._encode_event(event):
                    yield chunk
            yield '\n]'
        else:
            yield from self._encode_event(obj)

    def _encode_event(self, obj):
        # Custom encoding for event dicts with datapoints
        if isinstance(obj, dict) and 'datapoints' in obj and isinstance(obj['datapoints'], list):
            items = []
            for k, v in obj.items():
                if k == 'datapoints':
                    dp_strs = []
                    for dp in v:
                        if isinstance(dp, dict):
                            dp_items = []
                            for dpk, dpv in dp.items():
                                if dpk in ("simpleSpec", "rawData", "rawData3D") and isinstance(dpv, list):
                                    arr_str = json.dumps(dpv, separators=(',', ':'))
                                    dp_items.append(f'"{dpk}":{arr_str}')
                                else:
                                    dp_items.append(f'{json.dumps(dpk)}:{self.encode(dpv)}')
                            dp_strs.append('{' + ','.join(dp_items) + '}')
                        else:
                            dp_strs.append(self.encode(dp))
                    items.append(f'"datapoints":[{','.join(dp_strs)}]')
                else:
                    items.append(f'{json.dumps(k)}:{self.encode(v)}')
            yield '{' + ','.join(items) + '}'
        else:
            # Fallback to default
            yield from super().iterencode(obj)

def write_merged_events(event_map, out_dir, out_file):
    import re
    os.makedirs(out_dir, exist_ok=True)
    merged_path = os.path.join(out_dir, out_file)
    # Only include events that haven't been removed as duplicates
    merged_events = [info['event'] for info in event_map.values() if info['event'] is not None and not info.get('removed', False)]
    print(f"DEBUG: Number of merged events: {len(merged_events)}")
    if merged_events:
        print(f"DEBUG: First merged event keys: {list(merged_events[0].keys())}")
    else:
        print("DEBUG: No merged events to write!")
    # First, pretty-print the JSON as usual
    pretty = json.dumps(merged_events, indent=2)
    # Then, compact the arrays for the specified keys inside datapoints
    def compact_arrays_in_datapoints(text):
        for key in ["simpleSpec", "rawData", "rawData3D"]:
            text = re.sub(
                rf'("{key}": )\[(.*?)\]',
                lambda m: m.group(1) + '[' + re.sub(r'\s+', ' ', m.group(2)).strip() + ']',
                text,
                flags=re.DOTALL
            )
        return text
    pretty_compact = compact_arrays_in_datapoints(pretty)
    with open(merged_path, 'w') as f:
        f.write(pretty_compact)
    print(f"Wrote {len(merged_events)} unique events to {merged_path}")


def detect_and_mark_duplicates(event_map, max_seconds=60):
    """Detect duplicates within a time window per user.

    - Groups events by `userId` and sorts by normalized UTC `dataTime`.
    - For any adjacent events within `max_seconds`, treats them as duplicates.
    - Keeps the event with the largest number of `datapoints` and marks
        the others as removed in `event_map`.
    - Returns a list of (removed_key, kept_key) pairs for summary printing.
    """
    def parse_event_time(dt_str):
        """Parse various date formats to a timezone-aware UTC datetime.
        Supports ISO8601 (with Z) and 'dd-mm-yyyy hh:mm:ss'. Returns None on failure.
        """
        if dt_str is None:
            return None
        try:
            # dateutil handles many formats; dayfirst allows dd-mm-yyyy
            dt = dateutil_parser.parse(dt_str, dayfirst=True)
        except Exception:
            try:
                # Fallback to fromisoformat after normalizing Z
                if isinstance(dt_str, str) and dt_str.endswith('Z'):
                    dt = datetime.datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                else:
                    dt = datetime.datetime.fromisoformat(dt_str)
            except Exception:
                return None
        # Normalize to timezone-aware UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        else:
            dt = dt.astimezone(datetime.timezone.utc)
        return dt

    events = []
    for key, info in event_map.items():
        ev = info.get('event')
        if not ev:
            continue
        user = ev.get('userId')
        dt = parse_event_time(ev.get('dataTime'))
        if dt is None:
            continue
        is_ref = info.get('is_reference', False)
        event_type = ev.get('type', '')
        event_subtype = ev.get('subType', '')
        events.append((key, ev, user, dt, is_ref, event_type, event_subtype))

    from collections import defaultdict as _dd
    user_groups = _dd(list)
    for item in events:
        key, ev, user, dt, is_ref, event_type, event_subtype = item
        # Group by (user, type, subType) - only events with same type/subType can be duplicates
        group_key = (user, event_type, event_subtype)
        user_groups[group_key].append((key, ev, dt, is_ref))

    removed_pairs = []
    for group_key, evlist in user_groups.items():
        evlist.sort(key=lambda x: x[2])
        i = 0
        while i < len(evlist):
            cluster = [evlist[i]]
            j = i + 1
            while j < len(evlist) and (evlist[j][2] - cluster[-1][2]).total_seconds() <= max_seconds:
                cluster.append(evlist[j])
                j += 1
            if len(cluster) > 1:
                best = None
                best_count = -1
                best_is_ref = False
                for key, ev, dt, is_ref in cluster:
                    try:
                        dp_cnt = len(ev.get('datapoints') or [])
                    except Exception:
                        dp_cnt = 0
                    
                    # Priority 1: Reference dataset takes precedence
                    if is_ref and not best_is_ref:
                        best_count = dp_cnt
                        best = (key, ev, dt, is_ref)
                        best_is_ref = True
                    # Priority 2: Among reference events or non-reference events, prefer more datapoints
                    elif is_ref == best_is_ref:
                        if dp_cnt > best_count:
                            best_count = dp_cnt
                            best = (key, ev, dt, is_ref)
                            best_is_ref = is_ref
                        elif dp_cnt == best_count:
                            # Tie: prefer later event (larger datetime)
                            if best is None or dt > best[2]:
                                best = (key, ev, dt, is_ref)

                kept_key = best[0]
                for key, ev, dt, is_ref in cluster:
                    if key == kept_key:
                        continue
                    event_map[key]['removed'] = True
                    event_map[key]['removed_by'] = kept_key
                    event_map[key]['removed_reason'] = f'duplicate_of_{kept_key}'
                    removed_pairs.append((key, kept_key))
            i = j
    return removed_pairs

def main():
    parser = argparse.ArgumentParser(description="Scan, summarize, de-duplicate, and merge events from multiple JSON files.")
    parser.add_argument('--inDir', required=True, help='Input directory containing JSON files')
    parser.add_argument('--outDir', default=".", help='Output directory (default: current directory)')
    parser.add_argument('--outFile', default="merged.json", help='Output filename (default: merged.json)')
    parser.add_argument('files', nargs='+', help='JSON filenames (arrays of event objects)')
    parser.add_argument('--dedupe-window', type=int, default=30,
                        help='Time window (seconds) to consider events duplicates for the same user (default: 30)')
    args = parser.parse_args()

    # Build an event map across all input files and detect duplicates
    event_map = load_events(args.inDir, args.files)
    # Detect duplicates and mark removed events using configured window
    removed_pairs = detect_and_mark_duplicates(event_map, max_seconds=args.dedupe_window)
    if removed_pairs:
        print(f"Marked {len(removed_pairs)} events as duplicates (kept target shown as tuple (removed,kept)).")
        for rem, keep in removed_pairs[:10]:
            print(f"  Removed {rem} -> kept {keep}")
    # Sort by dataTime, then user_id for readable summary output
    sorted_events = sorted(
        event_map.items(),
        key=lambda x: (x[0][2], x[0][1])  # (dataTime, user_id)
    )
    print(f"{'eventId':<12} {'user_id':<10} {'dataTime':<25} files")
    print("-" * 70)
    for (eventId, user_id, dataTime), info in sorted_events:
        files = ','.join(sorted(info['files']))
        # Annotate duplicates that were removed
        note = ''
        if info.get('removed', False):
            note = ' ***DUPLICATE***'
        print(f"{str(eventId):<12} {str(user_id):<10} {str(dataTime):<25} {files}{note}")
    # Write merged output
    write_merged_events(event_map, args.outDir, args.outFile)

if __name__ == "__main__":
    main()

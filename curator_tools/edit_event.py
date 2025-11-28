#!/usr/bin/env python3

import argparse
import json
import os
import sys
import shutil
from datetime import datetime


def load_events_with_format(path):
    """Load events and detect file format (array vs NDJSON).

    Returns (events_list, format), where format is 'array' or 'ndjson'.
    """
    with open(path, 'r') as f:
        first = f.read(1)
        if not first:
            return [], 'array'
        f.seek(0)
        if first == '[':
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError('Expected a JSON array of events')
            return data, 'array'
        else:
            events = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(ev, dict):
                    events.append(ev)
            return events, 'ndjson'


def write_events(path, events, fmt):
    """Write events in the specified format back to path."""
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w') as f:
        if fmt == 'array':
            json.dump(events, f, indent=2)
        else:
            for ev in events:
                f.write(json.dumps(ev, separators=(',', ':')) + '\n')
    os.replace(tmp_path, path)


def find_event_index(events, event_id):
    """Find index of event by id or eventId, matching as strings."""
    target = str(event_id)
    for i, ev in enumerate(events):
        if str(ev.get('id')) == target or str(ev.get('eventId')) == target:
            return i
    return -1


def main():
    parser = argparse.ArgumentParser(
        description='Delete or update a single event in a JSON file (array or NDJSON).'
    )
    parser.add_argument('input', help='Input JSON file (array or NDJSON)')
    parser.add_argument('--id', dest='ids', nargs='+', required=True, help='One or more Event IDs (matches id or eventId field)')
    parser.add_argument('--delete', action='store_true', help='Delete the specified event')
    parser.add_argument('--type', dest='etype', help='Update event type')
    parser.add_argument('--subType', dest='subtype', help='Update event subType')
    parser.add_argument('--userId', dest='userid', help='Update event userId')
    parser.add_argument('--desc', dest='desc', help='Update event description')
    parser.add_argument('-y', '--yes', action='store_true', help='Do not prompt for confirmation')

    args = parser.parse_args()

    in_path = args.input
    events, fmt = load_events_with_format(in_path)
    if not events:
        print('No events loaded from input file.')
        sys.exit(1)

    # Resolve all target event indices
    target_ids = [str(x) for x in args.ids]
    id_to_index = {}
    missing = []
    for tid in target_ids:
        idx = find_event_index(events, tid)
        if idx < 0:
            missing.append(tid)
        else:
            id_to_index[tid] = idx
    if missing:
        print('The following event IDs were not found:', ', '.join(missing))
        sys.exit(1)

    # Validate intent
    updates = {}
    if args.etype is not None:
        updates['type'] = args.etype
    if args.subtype is not None:
        updates['subType'] = args.subtype
    if args.userid is not None:
        updates['userId'] = args.userid
    if args.desc is not None:
        updates['desc'] = args.desc

    if not args.delete and not updates:
        print('Nothing to do: specify --delete or one/more update flags (--type, --subType, --userId, --desc)')
        sys.exit(1)
    if args.delete and updates:
        print('Ambiguous request: cannot combine --delete with update flags. Choose one operation.')
        sys.exit(1)

    # Show plan and prompt for confirmation
    print('Planned operation:')
    print(f"  File: {in_path}")
    print(f"  Target Event IDs: {', '.join(target_ids)}")
    if args.delete:
        print('  Action: DELETE')
        print('  Event summaries (selected fields):')
        for tid in target_ids:
            ev = events[id_to_index[tid]]
            sample = {k: ev.get(k) for k in ('id', 'eventId', 'type', 'subType', 'userId', 'desc', 'dataTime')}
            print('   ', json.dumps(sample, ensure_ascii=False))
    else:
        print('  Action: UPDATE')
        print('  Changes to apply:')
        print('   ', json.dumps(updates, ensure_ascii=False))

    if not args.yes:
        confirm = input("Type 'yes' to confirm: ").strip().lower()
        if confirm not in ('yes', 'y'):
            print('Aborted by user.')
            sys.exit(0)

    # Backup original
    ts = datetime.now().strftime('%Y%m%d%H%M%S')
    backup_path = in_path + '.' + ts
    shutil.copy2(in_path, backup_path)
    print(f'Backup saved to {backup_path}')

    # Apply
    if args.delete:
        # Collect deleted events and delete in reverse index order to avoid shifting
        removed_events = []
        for tid, idx in sorted(id_to_index.items(), key=lambda x: x[1], reverse=True):
            removed_events.append(events[idx])
            del events[idx]
        # Write removed events to a timestamped file for later relocation
        removed_path = f"{in_path}_removed_{ts}.json"
        with open(removed_path, 'w') as rf:
            json.dump(removed_events, rf, indent=2)
        print(f'Removed events saved to {removed_path}')
    else:
        for tid, idx in id_to_index.items():
            events[idx].update(updates)

    write_events(in_path, events, fmt)
    print('Operation completed successfully.')


if __name__ == '__main__':
    main()

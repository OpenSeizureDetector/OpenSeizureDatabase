#!/usr/bin/env python3

import argparse
import json
import os


def iter_events_from_file(path):
    """Yield event dicts from a JSON file.

    Supports:
    - Standard JSON array: [ {...}, {...}, ... ]
    - NDJSON: one JSON object per line
    """
    with open(path, 'r') as f:
        first = f.read(1)
        if not first:
            return
        f.seek(0)
        if first == '[':
            data = json.load(f)
            if isinstance(data, list):
                for ev in data:
                    if isinstance(ev, dict):
                        yield ev
            else:
                raise ValueError("Input JSON must be an array of events or NDJSON")
        else:
            # NDJSON
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(ev, dict):
                    yield ev


def main():
    parser = argparse.ArgumentParser(description="Strip events with datasource=='Phone'.")
    parser.add_argument('input', help='Input JSON file (array or NDJSON)')
    parser.add_argument('-o', '--out', default='stripped.json', help='Output JSON file (default: stripped.json)')
    args = parser.parse_args()

    in_path = args.input
    out_path = args.out

    kept = []
    removed = 0
    for ev in iter_events_from_file(in_path):
        if ev.get('dataSourceName') == 'Phone':
            removed += 1
            continue
        kept.append(ev)

    # Write as a compact JSON array
    with open(out_path, 'w') as f:
        json.dump(kept, f, indent=2)

    print(f"Read {len(kept)+removed} events from {in_path}")
    print(f"Removed {removed} Phone datasource events")
    print(f"Wrote {len(kept)} events to {out_path}")


if __name__ == '__main__':
    main()

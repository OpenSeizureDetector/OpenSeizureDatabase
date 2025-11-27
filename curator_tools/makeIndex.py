#!/usr/bin/env python3

import argparse
import os
import sys

# Ensure repository modules are importable
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import libosd.osdDbConnection as osdConn


def make_index(input_json: str, output_csv: str | None = None, debug: bool = False) -> str:
    """Read a JSON file of events and write an index CSV similar to makeOsdDb.py.

    - input_json: path to input JSON file (array of event objects)
    - output_csv: optional explicit output CSV path; if None, uses input root + .csv
    - returns the path to the written CSV
    """
    if not os.path.exists(input_json):
        raise FileNotFoundError(f"Input file does not exist: {input_json}")

    if output_csv is None:
        root, _ = os.path.splitext(input_json)
        output_csv = f"{root}.csv"

    if debug:
        print(f"makeIndex: reading events from {input_json}")

    osd = osdConn.OsdDbConnection(cacheDir=None, debug=debug)
    # Load the specified JSON file directly (no cache dir)
    osd.loadDbFile(input_json, useCacheDir=False)

    if debug:
        print(f"makeIndex: writing index to {output_csv}")

    # Mirror makeOsdDb.py: save index CSV with selected columns and sorted by dataTime
    osd.saveIndexFile(output_csv, useCacheDir=False)

    if debug:
        print(f"makeIndex: index written: {output_csv}")

    return output_csv


def parse_args():
    p = argparse.ArgumentParser(description="Create an index CSV from an OSDB events JSON file.")
    p.add_argument("input", help="Path to input JSON file (array of event objects)")
    p.add_argument("--out", dest="out", default=None, help="Output CSV path (default: <input_root>.csv)")
    p.add_argument("--debug", action="store_true", help="Print debug information")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        out_path = make_index(args.input, args.out, args.debug)
        print(f"Index saved to {out_path}")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

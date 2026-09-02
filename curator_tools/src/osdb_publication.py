#!/usr/bin/env python3
"""
osdb_publication.py - Multi-Format OSDB Publication

Phase 5 implementation: Export OSDB data in multiple formats optimized for
different use cases.

Supported Formats:
- JSON: Original format (backward compatible)
- JSON.GZ: Compressed JSON (60-80% size reduction)
- Parquet: Columnar format for ML/analysis (4-10x smaller)
- CSV: Summary/index file (event metadata only)

Usage:
    from osdb_publication import OsdbPublisher
    
    publisher = OsdbPublisher()
    
    # Publish in all formats
    publisher.publish_all_formats(
        events,
        output_prefix='osdb_3min_allSeizures',
        formats=['json', 'json.gz', 'parquet', 'csv']
    )
"""

import json
import gzip
import csv
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys


class OsdbPublisher:
    """Multi-format publisher for OSDB data."""
    
    def __init__(self, debug: bool = False):
        """
        Initialize publisher.
        
        Args:
            debug: Enable debug output
        """
        self.debug = debug
    
    def publish_json(
        self,
        events: List[Dict[str, Any]],
        output_path: str,
        pretty: bool = True
    ) -> Dict[str, Any]:
        """
        Publish events as JSON.
        
        Args:
            events: List of event dictionaries
            output_path: Output file path
            pretty: Pretty-print JSON
            
        Returns:
            Publication statistics
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            if pretty:
                json.dump(events, f, indent=2)
            else:
                json.dump(events, f)
        
        size_bytes = Path(output_path).stat().st_size
        
        stats = {
            'format': 'json',
            'output_file': output_path,
            'event_count': len(events),
            'size_bytes': size_bytes,
            'size_mb': size_bytes / (1024 * 1024)
        }
        
        if self.debug:
            print(f"Published JSON: {output_path} ({stats['size_mb']:.2f} MB)")
        
        return stats
    
    def publish_json_gz(
        self,
        events: List[Dict[str, Any]],
        output_path: str,
        compression_level: int = 6
    ) -> Dict[str, Any]:
        """
        Publish events as compressed JSON.
        
        Args:
            events: List of event dictionaries
            output_path: Output file path (.json.gz)
            compression_level: gzip compression level (1-9)
            
        Returns:
            Publication statistics
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        json_str = json.dumps(events, indent=2)
        
        with gzip.open(output_path, 'wt', compresslevel=compression_level) as f:
            f.write(json_str)
        
        compressed_size = Path(output_path).stat().st_size
        original_size = len(json_str.encode('utf-8'))
        compression_ratio = (1 - compressed_size / original_size) * 100
        
        stats = {
            'format': 'json.gz',
            'output_file': output_path,
            'event_count': len(events),
            'size_bytes': compressed_size,
            'size_mb': compressed_size / (1024 * 1024),
            'original_size_bytes': original_size,
            'compression_ratio_percent': compression_ratio
        }
        
        if self.debug:
            print(f"Published JSON.GZ: {output_path} ({stats['size_mb']:.2f} MB, {compression_ratio:.1f}% reduction)")
        
        return stats
    
    def publish_parquet(
        self,
        events: List[Dict[str, Any]],
        output_path: str,
        flatten_datapoints: bool = True
    ) -> Dict[str, Any]:
        """
        Publish events as Apache Parquet format.
        
        Args:
            events: List of event dictionaries
            output_path: Output file path (.parquet)
            flatten_datapoints: Create one row per datapoint (recommended)
            
        Returns:
            Publication statistics
        """
        try:
            import pandas as pd
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError(
                "Parquet export requires pandas and pyarrow:\n"
                "  pip install pandas pyarrow"
            )
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        if flatten_datapoints:
            # Create one row per datapoint (recommended for analysis)
            records = []
            for event in events:
                base_record = {
                    'event_id': event.get('id'),
                    'userId': event.get('userId'),
                    'event_type': event.get('type'),
                    'event_subtype': event.get('subType'),
                    'event_desc': event.get('desc'),
                    'event_dataTime': event.get('dataTime'),
                    'event_dataTimeEnd': event.get('dataTimeEnd'),
                    'osdAlarmState': event.get('osdAlarmState'),
                    'dataSourceName': event.get('dataSourceName'),
                    'merged_from_events': json.dumps(event.get('merged_from_events')) if event.get('merged_from_events') else None,
                    'merged_event_count': event.get('merged_event_count', 1)
                }
                
                datapoints = event.get('datapoints', [])
                if datapoints:
                    for dp in datapoints:
                        record = base_record.copy()
                        record.update({
                            'datapoint_dataTime': dp.get('dataTime'),
                            'datapoint_alarmState': dp.get('alarmState'),
                            'datapoint_hr': dp.get('hr'),
                            'datapoint_o2Sat': dp.get('o2Sat'),
                            'has_rawData': 1 if 'rawData' in dp else 0,
                            'has_rawData3D': 1 if 'rawData3D' in dp else 0
                        })
                        records.append(record)
                else:
                    # Event with no datapoints
                    records.append(base_record)
            
            df = pd.DataFrame(records)
            row_count = len(records)
        else:
            # One row per event (keeps datapoints as JSON)
            records = []
            for event in events:
                record = event.copy()
                # Convert datapoints to JSON string
                if 'datapoints' in record:
                    record['datapoints'] = json.dumps(record['datapoints'])
                if 'merged_from_events' in record and record['merged_from_events']:
                    record['merged_from_events'] = json.dumps(record['merged_from_events'])
                records.append(record)
            
            df = pd.DataFrame(records)
            row_count = len(events)
        
        # Write Parquet file
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_path, compression='snappy')
        
        size_bytes = Path(output_path).stat().st_size
        
        stats = {
            'format': 'parquet',
            'output_file': output_path,
            'event_count': len(events),
            'row_count': row_count,
            'size_bytes': size_bytes,
            'size_mb': size_bytes / (1024 * 1024),
            'flattened': flatten_datapoints
        }
        
        if self.debug:
            print(f"Published Parquet: {output_path} ({stats['size_mb']:.2f} MB, {row_count} rows)")
        
        return stats
    
    def publish_csv(
        self,
        events: List[Dict[str, Any]],
        output_path: str,
        include_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Publish event metadata as CSV (index file).
        Does not include datapoints (use Parquet for that).
        
        Args:
            events: List of event dictionaries
            output_path: Output file path (.csv)
            include_fields: Fields to include (default: all metadata fields)
            
        Returns:
            Publication statistics
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        if include_fields is None:
            # Default fields for CSV index
            include_fields = [
                'id', 'userId', 'dataTime', 'dataTimeEnd', 'type', 'subType',
                'desc', 'osdAlarmState', 'dataSourceName', 'merged_event_count',
                'duration_seconds'
            ]
        
        with open(output_path, 'w', newline='') as f:
            # Get all possible fields from events
            all_fields = set()
            for event in events:
                all_fields.update(event.keys())
            
            # Filter to included fields that exist
            fields = [f for f in include_fields if f in all_fields]
            
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            
            for event in events:
                # Skip datapoints
                row = {k: v for k, v in event.items() if k != 'datapoints'}
                # Convert lists to JSON
                for key in row:
                    if isinstance(row[key], (list, dict)):
                        row[key] = json.dumps(row[key])
                writer.writerow(row)
        
        size_bytes = Path(output_path).stat().st_size
        
        stats = {
            'format': 'csv',
            'output_file': output_path,
            'event_count': len(events),
            'size_bytes': size_bytes,
            'size_mb': size_bytes / (1024 * 1024),
            'fields': fields
        }
        
        if self.debug:
            print(f"Published CSV: {output_path} ({stats['size_mb']:.2f} MB)")
        
        return stats
    
    def publish_all_formats(
        self,
        events: List[Dict[str, Any]],
        output_prefix: str,
        formats: Optional[List[str]] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Publish events in multiple formats.
        
        Args:
            events: List of event dictionaries
            output_prefix: Prefix for output files (e.g., 'osdb_3min_allSeizures')
            formats: List of formats to publish (default: all)
            output_dir: Output directory (default: current directory)
            
        Returns:
            Dictionary mapping format to publication statistics
        """
        if formats is None:
            formats = ['json', 'json.gz', 'parquet', 'csv']
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path('.')
        
        results = {}
        
        print(f"\n{'='*70}")
        print(f"Publishing OSDB Data: {output_prefix}")
        print(f"{'='*70}")
        print(f"Events to publish: {len(events)}")
        print(f"Formats: {', '.join(formats)}\n")
        
        if 'json' in formats:
            output_path = output_dir / f"{output_prefix}.json"
            results['json'] = self.publish_json(events, str(output_path))
            print(f"  ✓ JSON: {results['json']['size_mb']:.2f} MB")
        
        if 'json.gz' in formats:
            output_path = output_dir / f"{output_prefix}.json.gz"
            results['json.gz'] = self.publish_json_gz(events, str(output_path))
            print(f"  ✓ JSON.GZ: {results['json.gz']['size_mb']:.2f} MB "
                  f"({results['json.gz']['compression_ratio_percent']:.1f}% smaller)")
        
        if 'parquet' in formats:
            try:
                output_path = output_dir / f"{output_prefix}.parquet"
                results['parquet'] = self.publish_parquet(events, str(output_path))
                print(f"  ✓ Parquet: {results['parquet']['size_mb']:.2f} MB "
                      f"({results['parquet']['row_count']:,} rows)")
            except ImportError as e:
                print(f"  ⚠ Parquet: Skipped ({e})")
        
        if 'csv' in formats:
            output_path = output_dir / f"{output_prefix}.csv"
            results['csv'] = self.publish_csv(events, str(output_path))
            print(f"  ✓ CSV: {results['csv']['size_mb']:.2f} MB (metadata only)")
        
        # Summary
        print(f"\n{'='*70}")
        print("Publication Summary")
        print(f"{'='*70}")
        
        total_size = sum(r['size_bytes'] for r in results.values())
        print(f"Total size: {total_size / (1024 * 1024):.2f} MB across {len(results)} formats")
        
        if 'json' in results and 'parquet' in results:
            reduction = (1 - results['parquet']['size_bytes'] / results['json']['size_bytes']) * 100
            print(f"Parquet vs JSON: {reduction:.1f}% smaller")
        
        return results


def main():
    """CLI for publishing OSDB data."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Multi-format OSDB publication tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Publish in all formats
  python3 osdb_publication.py --input events.json --output-prefix osdb_3min_allSeizures
  
  # Publish only JSON and Parquet
  python3 osdb_publication.py --input events.json --formats json parquet -o output/
  
  # Compare format sizes
  python3 osdb_publication.py --input events.json --formats json json.gz parquet --compare
        """
    )
    
    parser.add_argument('--input', '-i', required=True, help='Input JSON file')
    parser.add_argument('--output-prefix', '-p', help='Output file prefix (default: input basename)')
    parser.add_argument('--output-dir', '-o', help='Output directory (default: current)')
    parser.add_argument('--formats', nargs='+', 
                       choices=['json', 'json.gz', 'parquet', 'csv'],
                       help='Formats to publish (default: all)')
    parser.add_argument('--compare', action='store_true', help='Show format comparison')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    # Load input
    print(f"Loading {args.input}...")
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'events' in data:
        events = data['events']
    else:
        events = data
    
    print(f"Loaded {len(events)} events")
    
    # Determine output prefix
    if args.output_prefix:
        output_prefix = args.output_prefix
    else:
        output_prefix = Path(args.input).stem
    
    # Publish
    publisher = OsdbPublisher(debug=args.debug)
    results = publisher.publish_all_formats(
        events,
        output_prefix,
        formats=args.formats,
        output_dir=args.output_dir
    )
    
    # Comparison table
    if args.compare and len(results) > 1:
        print(f"\n{'='*70}")
        print("Format Comparison")
        print(f"{'='*70}")
        print(f"{'Format':<15} {'Size (MB)':<15} {'Reduction':<15}")
        print("-" * 70)
        
        baseline = results.get('json', results[list(results.keys())[0]])
        baseline_size = baseline['size_bytes']
        
        for fmt, stats in sorted(results.items(), key=lambda x: x[1]['size_bytes']):
            size_mb = stats['size_mb']
            reduction = (1 - stats['size_bytes'] / baseline_size) * 100
            print(f"{fmt:<15} {size_mb:<15.2f} {reduction:>6.1f}%")


if __name__ == '__main__':
    main()

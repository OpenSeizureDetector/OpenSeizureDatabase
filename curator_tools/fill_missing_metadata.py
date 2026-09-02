#!/usr/bin/env python3
"""
fill_missing_metadata.py

Wrapper script to fill missing metadata in the database.

This is a convenience wrapper that makes it easy to run the metadata filler.

Usage:
    # Dry run (see what would be updated)
    python3 fill_missing_metadata.py --db /home/graham/osd/osdb/osdb_working.db \
        --json-dir /home/graham/osd/osdb --dry-run
    
    # Actually fill in missing metadata
    python3 fill_missing_metadata.py --db /home/graham/osd/osdb/osdb_working.db \
        --json-dir /home/graham/osd/osdb
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the actual script
from fill_missing_metadata import main

if __name__ == '__main__':
    main()

#!/bin/bash
# Example usage scripts for OSDB graph generation

# Activate the virtual environment
source /home/graham/pyEnvs/osdb/bin/activate
cd /home/graham/osd/OpenSeizureDatabase/curator_tools

echo "=========================================="
echo "OSDB Graph Generation - Usage Examples"
echo "=========================================="
echo ""

# Example 1: Generate graphs from all seizure-related files
echo "Example 1: Generate graphs from seizure files"
echo "Command:"
echo "  python makeOsdDb.py graphs osdb_3min_allSeizures.json osdb_3min_tcSeizures.json \\"
echo "    --output ./seizure_graphs --threshold 5"
echo ""
echo "This creates three graphs:"
echo "  - summary_statistics.png: Total seizure counts"
echo "  - seizures_by_user.png: Bar chart of seizures per user"
echo "  - cumulative_seizures_per_month.png: Monthly trend chart"
echo ""

# Example 2: Generate graphs from all event types
echo "Example 2: Generate graphs from all event types"
echo "Command:"
echo "  python makeOsdDb.py graphs \\"
echo "    osdb_3min_allSeizures.json \\"
echo "    osdb_3min_falseAlarms.json \\"
echo "    osdb_3min_ndaEvents.json \\"
echo "    --output ./all_events_graphs --threshold 3"
echo ""
echo "This combines all event types in the summary statistics."
echo ""

# Example 3: Standalone usage with custom threshold
echo "Example 3: Standalone usage with lower threshold"
echo "Command:"
echo "  python generateGraphs.py data.json --output . --threshold 2"
echo ""
echo "Users with fewer than 2 events will be grouped as 'Other'."
echo ""

# Example 4: Debug mode for troubleshooting
echo "Example 4: Debug mode with detailed output"
echo "Command:"
echo "  python generateGraphs.py seizures.json --output reports --debug"
echo ""
echo "Shows detailed processing information and user grouping details."
echo ""

# Example 5: Process all JSON files in a directory
echo "Example 5: Process all JSON files in current directory"
echo "Command:"
echo "  python generateGraphs.py *.json --output ./summary"
echo ""
echo "Processes all JSON files matching the pattern."
echo ""

# Example 6: Check help for all options
echo "Example 6: View all available options"
echo "Command (makeOsdDb.py):"
echo "  python makeOsdDb.py graphs --help"
echo ""
echo "Command (standalone):"
echo "  python generateGraphs.py --help"
echo ""

echo "=========================================="
echo "For more information, see GRAPHS_README.md"
echo "=========================================="

#!/bin/bash
# Example usage of makeOsdDb_refactored_wrapper.py with all features
# This script demonstrates the complete workflow including index and graph generation

# Configuration
CONFIG_FILE="../osdb.cfg"
OSDB_DIR="/home/graham/osd/osdb"
GRAPH_OUTPUT="/home/graham/osd/osdb/graphs"

echo "======================================================================"
echo "makeOsdDb Refactored Wrapper - Complete Workflow Example"
echo "======================================================================"
echo ""
echo "This example demonstrates:"
echo "  1. Downloading and processing events from the web API"
echo "  2. Generating CSV index files"
echo "  3. Generating summary graphs"
echo ""
echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Output directory: $OSDB_DIR"
echo "  Graph output: $GRAPH_OUTPUT"
echo ""
echo "======================================================================"
echo ""

# Run the refactored wrapper with all features enabled
python3 makeOsdDb_refactored_wrapper.py \
    --config "$CONFIG_FILE" \
    --osdb-dir "$OSDB_DIR" \
    --generate-index \
    --generate-graphs \
    --graph-output "$GRAPH_OUTPUT" \
    --graph-threshold 5

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✓ Workflow completed successfully!"
    echo "======================================================================"
    echo ""
    echo "Generated files:"
    echo "  JSON event files: $OSDB_DIR/osdb_3min_*.json"
    echo "  CSV index files: $OSDB_DIR/osdb_3min_*.csv"
    echo "  Summary graphs: $GRAPH_OUTPUT/*.png"
    echo ""
else
    echo ""
    echo "======================================================================"
    echo "✗ Workflow failed - check error messages above"
    echo "======================================================================"
    echo ""
    exit 1
fi

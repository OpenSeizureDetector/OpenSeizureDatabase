#!/bin/bash
# batch_convert_to_pte.sh - Batch convert models to ExecuTorch format
#
# Usage:
#   ./batch_convert_to_pte.sh                    # Convert all .ptl in current dir
#   ./batch_convert_to_pte.sh /path/to/models    # Convert all .ptl in specified dir
#   ./batch_convert_to_pte.sh --from-pt          # Convert all .pt directly

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERT_PTL2PTE="$SCRIPT_DIR/convertPtl2Pte.py"
CONVERT_PT2PTE="$SCRIPT_DIR/convertPt2Pte.py"

# Parse arguments
FROM_PT=false
TARGET_DIR="."

while [[ $# -gt 0 ]]; do
    case $1 in
        --from-pt)
            FROM_PT=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS] [DIRECTORY]"
            echo ""
            echo "Batch convert PyTorch models to ExecuTorch (.pte) format"
            echo ""
            echo "Options:"
            echo "  --from-pt       Convert from .pt files directly (default: from .ptl)"
            echo "  --help, -h      Show this help message"
            echo ""
            echo "Arguments:"
            echo "  DIRECTORY       Directory containing models (default: current directory)"
            echo ""
            echo "Examples:"
            echo "  $0                           # Convert all .ptl in current dir"
            echo "  $0 /path/to/models           # Convert all .ptl in specified dir"
            echo "  $0 --from-pt                 # Convert all .pt in current dir"
            echo "  $0 --from-pt output/fold_1   # Convert all .pt in output/fold_1"
            exit 0
            ;;
        *)
            TARGET_DIR="$1"
            shift
            ;;
    esac
done

# Check if directory exists
if [[ ! -d "$TARGET_DIR" ]]; then
    echo "Error: Directory not found: $TARGET_DIR"
    exit 1
fi

# Check if conversion scripts exist
if [[ ! -f "$CONVERT_PTL2PTE" ]]; then
    echo "Error: convertPtl2Pte.py not found at: $CONVERT_PTL2PTE"
    exit 1
fi

if [[ "$FROM_PT" == true ]] && [[ ! -f "$CONVERT_PT2PTE" ]]; then
    echo "Error: convertPt2Pte.py not found at: $CONVERT_PT2PTE"
    exit 1
fi

# Determine which files to convert
if [[ "$FROM_PT" == true ]]; then
    PATTERN="*.pt"
    CONVERTER="$CONVERT_PT2PTE"
    EXTENSION=".pt"
    echo "Converting .pt files to .pte in: $TARGET_DIR"
else
    PATTERN="*.ptl"
    CONVERTER="$CONVERT_PTL2PTE"
    EXTENSION=".ptl"
    echo "Converting .ptl files to .pte in: $TARGET_DIR"
fi

# Find and convert files
COUNT=0
FAILED=0
SKIPPED=0

shopt -s nullglob  # Handle case when no files match
for file in "$TARGET_DIR"/$PATTERN; do
    # Skip if already has .pte version
    pte_file="${file%$EXTENSION}.pte"
    if [[ -f "$pte_file" ]]; then
        echo "  ⊘ Skipping (already exists): $(basename "$file")"
        ((SKIPPED++))
        continue
    fi
    
    echo ""
    echo "Converting: $(basename "$file")"
    
    if python3 "$CONVERTER" "$file" -q; then
        echo "  ✓ Success: $(basename "$pte_file")"
        ((COUNT++))
    else
        echo "  ✗ Failed: $(basename "$file")"
        ((FAILED++))
    fi
done

# Summary
echo ""
echo "="*60
echo "Conversion Summary"
echo "="*60
echo "  Successfully converted: $COUNT"
echo "  Failed:                 $FAILED"
echo "  Skipped (already exist): $SKIPPED"
echo ""

if [[ $COUNT -eq 0 ]] && [[ $FAILED -eq 0 ]] && [[ $SKIPPED -eq 0 ]]; then
    echo "No $EXTENSION files found in $TARGET_DIR"
    exit 0
elif [[ $FAILED -gt 0 ]]; then
    echo "⚠ Some conversions failed. Check output above for details."
    exit 1
else
    echo "✓ All conversions completed successfully!"
    exit 0
fi

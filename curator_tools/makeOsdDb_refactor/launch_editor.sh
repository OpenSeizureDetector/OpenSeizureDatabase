#!/bin/bash
# Launcher script for OSDB Event Editor

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if PyQt5 is installed
if ! python3 -c "import PyQt5" 2>/dev/null; then
    echo "Error: PyQt5 is not installed"
    echo "Install with: pip install -r requirements.txt"
    exit 1
fi

# Check if matplotlib is installed
if ! python3 -c "import matplotlib" 2>/dev/null; then
    echo "Error: matplotlib is not installed"
    echo "Install with: pip install -r requirements.txt"
    exit 1
fi

# Launch the editor
cd "$SCRIPT_DIR"
python3 event_editor.py "$@"

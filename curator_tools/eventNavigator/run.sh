#!/bin/bash

# Simple script to run the OSDB Event Navigator

echo "Starting OSDB Event Navigator..."
echo "================================"

# Check if Python is available
if ! command -v python3 &> /dev/null
then
    echo "Error: Python3 is not installed or not in PATH"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "Error: main.py not found. Please run this script from the eventNavigator directory."
    exit 1
fi

# Run the application
python3 main.py
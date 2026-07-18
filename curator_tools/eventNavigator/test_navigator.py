#!/usr/bin/env python3
"""
Test script for the OSDB Event Navigator
"""

import os
import sys
import tkinter as tk
from tkinter import messagebox

# Add the current directory to the path so we can import our module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import EventNavigatorGUI

def test_gui():
    """Test that the GUI can be created without errors"""
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        # Create the GUI
        app = EventNavigatorGUI(root)
        
        # Test that we can set up the basic structure
        print("GUI created successfully")
        print("Database folder variable:", app.db_folder_var.get())
        print("Event count variable:", app.event_count_var.get())
        print("Event index variable:", app.event_index_var.get())
        
        root.destroy()
        return True
        
    except Exception as e:
        print(f"Error creating GUI: {e}")
        return False

if __name__ == "__main__":
    print("Testing OSDB Event Navigator...")
    success = test_gui()
    if success:
        print("Test passed!")
    else:
        print("Test failed!")
        sys.exit(1)
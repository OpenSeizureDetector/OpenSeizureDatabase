#!/usr/bin/env python3
"""
Event Navigator GUI for OpenSeizureDatabase
This tool allows users to select a database folder and navigate through events,
displaying metadata for each event.
"""

import os
import sys
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import argparse
import shutil
import datetime

# Add the libosd directory to the path so we can import OsdDbConnection
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from libosd.osdDbConnection import OsdDbConnection

class EventNavigatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("OSDB Event Navigator")
        self.root.geometry("1000x700")
        
        # Database connection
        self.db_connection = None
        self.events = []
        self.current_event_index = 0
        self.original_events = []  # Keep track of original events for backup
        self.events_modified = False  # Track if events have been modified
        
        # Create the UI
        self.create_widgets()
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Database selection section
        db_frame = ttk.LabelFrame(main_frame, text="Database Selection", padding="5")
        db_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        db_frame.columnconfigure(1, weight=1)
        
        ttk.Label(db_frame, text="Database Folder:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        
        self.db_folder_var = tk.StringVar()
        db_entry = ttk.Entry(db_frame, textvariable=self.db_folder_var, width=50)
        db_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        
        browse_btn = ttk.Button(db_frame, text="Browse", command=self.browse_database)
        browse_btn.grid(row=0, column=2, padx=(0, 5))
        
        load_btn = ttk.Button(db_frame, text="Load Database", command=self.load_database)
        load_btn.grid(row=0, column=3)
        
        # Event navigation section
        nav_frame = ttk.LabelFrame(main_frame, text="Event Navigation", padding="5")
        nav_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        nav_frame.columnconfigure(1, weight=1)
        
        ttk.Label(nav_frame, text="Event:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        
        self.event_index_var = tk.StringVar()
        event_index_entry = ttk.Entry(nav_frame, textvariable=self.event_index_var, width=10)
        event_index_entry.grid(row=0, column=1, sticky=tk.W, padx=(0, 5))
        
        self.event_count_var = tk.StringVar()
        ttk.Label(nav_frame, textvariable=self.event_count_var).grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        
        prev_btn = ttk.Button(nav_frame, text="Previous", command=self.previous_event)
        prev_btn.grid(row=0, column=3, padx=(0, 5))
        
        next_btn = ttk.Button(nav_frame, text="Next", command=self.next_event)
        next_btn.grid(row=0, column=4, padx=(0, 5))
        
        # Edit section
        edit_frame = ttk.LabelFrame(main_frame, text="Edit Event", padding="5")
        edit_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        edit_frame.columnconfigure(1, weight=1)
        edit_frame.columnconfigure(3, weight=1)
        
        # Type edit
        ttk.Label(edit_frame, text="Type:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.type_var = tk.StringVar()
        type_entry = ttk.Entry(edit_frame, textvariable=self.type_var, width=15)
        type_entry.grid(row=0, column=1, sticky=tk.W, padx=(0, 10))
        
        # SubType edit
        ttk.Label(edit_frame, text="SubType:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.subtype_var = tk.StringVar()
        subtype_entry = ttk.Entry(edit_frame, textvariable=self.subtype_var, width=15)
        subtype_entry.grid(row=0, column=3, sticky=tk.W, padx=(0, 10))
        
        # Description edit
        ttk.Label(edit_frame, text="Description:").grid(row=0, column=4, sticky=tk.W, padx=(0, 5))
        self.desc_var = tk.StringVar()
        desc_entry = ttk.Entry(edit_frame, textvariable=self.desc_var, width=30)
        desc_entry.grid(row=0, column=5, sticky=tk.W, padx=(0, 10))
        
        # Save button
        save_btn = ttk.Button(edit_frame, text="Save Changes", command=self.save_changes)
        save_btn.grid(row=0, column=6, padx=(0, 5))
        
        # Event details section
        details_frame = ttk.LabelFrame(main_frame, text="Event Details", padding="5")
        details_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        details_frame.columnconfigure(0, weight=1)
        details_frame.rowconfigure(0, weight=1)
        
        # Create a text widget for displaying event details
        self.details_text = tk.Text(details_frame, wrap=tk.WORD, height=15)
        self.details_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add scrollbar to text widget
        scrollbar = ttk.Scrollbar(details_frame, orient=tk.VERTICAL, command=self.details_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.details_text.configure(yscrollcommand=scrollbar.set)
        
        # Graph buttons section
        graph_frame = ttk.LabelFrame(main_frame, text="Graph Generation", padding="5")
        graph_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(graph_frame, text="Generate Acceleration Vector Magnitude Graph", 
                  command=self.generate_acceleration_graph).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(graph_frame, text="Generate Heart Rate Graph", 
                  command=self.generate_heart_rate_graph).grid(row=0, column=1, padx=(0, 5))
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
    def browse_database(self):
        """Open a dialog to browse for database folder"""
        folder_selected = filedialog.askdirectory(
            title="Select OSDB Database Folder",
            initialdir=self.db_folder_var.get() or os.path.expanduser("~")
        )
        if folder_selected:
            self.db_folder_var.set(folder_selected)
            
    def load_database(self):
        """Load database from selected folder"""
        db_folder = self.db_folder_var.get()
        if not db_folder:
            messagebox.showwarning("Warning", "Please select a database folder first.")
            return
            
        try:
            # Create database connection
            self.db_connection = OsdDbConnection(cacheDir=db_folder, debug=False)
            
            # Try to load a sample file to verify the database
            # We'll look for common OSDB JSON files
            db_files = []
            for file in os.listdir(db_folder):
                if file.endswith('.json') and 'osdb' in file.lower():
                    db_files.append(file)
            
            if not db_files:
                # If no osdb files found, try to load any json file
                for file in os.listdir(db_folder):
                    if file.endswith('.json'):
                        db_files.append(file)
            
            if not db_files:
                messagebox.showerror("Error", "No JSON database files found in the selected folder.")
                return
                
            # Load the first database file found
            first_file = db_files[0]
            self.db_connection.loadDbFile(first_file, useCacheDir=True)
            
            self.events = self.db_connection.getAllEvents(includeDatapoints=False)
            self.original_events = [event.copy() for event in self.events]  # Create a copy for backup
            self.current_event_index = 0
            self.events_modified = False
            
            self.event_count_var.set(f"of {len(self.events)} events")
            self.event_index_var.set(str(self.current_event_index + 1))
            
            self.display_event_details()
            self.status_var.set(f"Loaded database with {len(self.events)} events from {first_file}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load database: {str(e)}")
            self.status_var.set("Error loading database")
            
    def display_event_details(self):
        """Display the current event details"""
        if not self.events or self.current_event_index >= len(self.events):
            self.details_text.delete(1.0, tk.END)
            self.details_text.insert(tk.END, "No events available or invalid event index.")
            return
            
        event = self.events[self.current_event_index]
        details = f"Event ID: {event.get('id', 'N/A')}\n"
        details += f"Data Time: {event.get('dataTime', 'N/A')}\n"
        details += f"User ID: {event.get('userId', 'N/A')}\n"
        details += f"Type: {event.get('type', 'N/A')}\n"
        details += f"SubType: {event.get('subType', 'N/A')}\n"
        details += f"DataSource: {event.get('dataSourceName', 'N/A')}\n"
        details += f"Phone App Version: {event.get('phoneAppVersion', 'N/A')}\n"
        details += f"Watch SD Version: {event.get('watchSdVersion', 'N/A')}\n"
        details += f"Has 3D Data: {event.get('has3dData', 'N/A')}\n"
        details += f"Has HR Data: {event.get('hasHrData', 'N/A')}\n"
        details += f"Has O2 Sat Data: {event.get('hasO2SatData', 'N/A')}\n"
        details += f"Description: {event.get('desc', 'N/A')}\n"
        details += f"Alarm State: {event.get('osdAlarmState', 'N/A')}\n"
        
        # Add any additional metadata that might be in the dataJSON field
        if 'dataJSON' in event:
            try:
                data_json = json.loads(event['dataJSON'])
                details += "\nAdditional Data:\n"
                for key, value in data_json.items():
                    details += f"  {key}: {value}\n"
            except:
                details += "\nAdditional Data (JSON parsing failed)\n"
        
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(tk.END, details)
        
        # Update the edit fields with current event data
        self.type_var.set(event.get('type', ''))
        self.subtype_var.set(event.get('subType', ''))
        self.desc_var.set(event.get('desc', ''))
        
    def previous_event(self):
        """Navigate to the previous event"""
        if self.events and self.current_event_index > 0:
            self.current_event_index -= 1
            self.event_index_var.set(str(self.current_event_index + 1))
            self.display_event_details()
            self.status_var.set(f"Showing event {self.current_event_index + 1} of {len(self.events)}")
            
    def next_event(self):
        """Navigate to the next event"""
        if self.events and self.current_event_index < len(self.events) - 1:
            self.current_event_index += 1
            self.event_index_var.set(str(self.current_event_index + 1))
            self.display_event_details()
            self.status_var.set(f"Showing event {self.current_event_index + 1} of {len(self.events)}")
            
    def save_changes(self):
        """Save changes to the current event"""
        if not self.events:
            messagebox.showwarning("Warning", "No events loaded. Please load a database first.")
            return
            
        # Get the current event
        event = self.events[self.current_event_index]
        
        # Update the event with edited values
        event['type'] = self.type_var.get()
        event['subType'] = self.subtype_var.get()
        event['desc'] = self.desc_var.get()
        
        # Mark as modified
        self.events_modified = True
        
        self.status_var.set("Changes saved to memory. Click Save to write to file.")
        
    def backup_file(self, file_path):
        """Create a backup of the original file"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            # Create backup in a dedicated backup folder to avoid interfering with database loading
            backup_dir = os.path.join(os.path.dirname(file_path), "backups")
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            backup_path = os.path.join(backup_dir, f"{os.path.basename(file_path)}.{timestamp}")
            shutil.copy2(file_path, backup_path)
            return backup_path
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create backup: {str(e)}")
            return None
            
    def save_to_file(self, db_file_path):
        """Save events back to the JSON file with backup"""
        try:
            # Create backup
            backup_path = self.backup_file(db_file_path)
            if not backup_path:
                return False
                
            # Write updated events back to file
            with open(db_file_path, 'w') as f:
                json.dump(self.events, f, indent=2)
                
            self.status_var.set(f"Saved to {db_file_path}. Backup created: {backup_path}")
            self.events_modified = False
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save to file: {str(e)}")
            self.status_var.set("Error saving to file")
            return False
            
    def on_closing(self):
        """Handle window closing with confirmation if there are unsaved changes"""
        if self.events_modified:
            result = messagebox.askyesnocancel(
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save before exiting?"
            )
            if result is True:  # Yes, save
                # Find the database file that was loaded
                db_folder = self.db_folder_var.get()
                if db_folder:
                    db_files = []
                    for file in os.listdir(db_folder):
                        if file.endswith('.json') and 'osdb' in file.lower():
                            db_files.append(file)
                    if not db_files:
                        for file in os.listdir(db_folder):
                            if file.endswith('.json'):
                                db_files.append(file)
                    if db_files:
                        db_file = db_files[0]
                        db_file_path = os.path.join(db_folder, db_file)
                        if self.save_to_file(db_file_path):
                            self.root.destroy()
                        else:
                            return  # Don't close if save failed
                    else:
                        messagebox.showerror("Error", "Could not find database file to save changes.")
                        return
                else:
                    messagebox.showerror("Error", "No database folder selected.")
                    return
            elif result is False:  # No, don't save
                self.root.destroy()
            # If result is None (Cancel), don't close the window
        else:
            self.root.destroy()
            
    def generate_acceleration_graph(self):
        """Placeholder for acceleration graph generation"""
        if not self.events:
            messagebox.showwarning("Warning", "No events loaded. Please load a database first.")
            return
            
        event = self.events[self.current_event_index]
        messagebox.showinfo("Graph Generation", 
                           f"Generating acceleration vector magnitude graph for event {event.get('id', 'N/A')}\n"
                           "This is a placeholder - actual implementation will be added later.")
        
    def generate_heart_rate_graph(self):
        """Placeholder for heart rate graph generation"""
        if not self.events:
            messagebox.showwarning("Warning", "No events loaded. Please load a database first.")
            return
            
        event = self.events[self.current_event_index]
        messagebox.showinfo("Graph Generation", 
                           f"Generating heart rate graph for event {event.get('id', 'N/A')}\n"
                           "This is a placeholder - actual implementation will be added later.")

def main():
    root = tk.Tk()
    app = EventNavigatorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

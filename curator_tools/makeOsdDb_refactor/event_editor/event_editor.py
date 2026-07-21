#!/usr/bin/env python3
"""
Event Editor GUI - Qt5-based event viewer and editor for OSDB SQLite database

Features:
- Open database file (dialog or command line)
- Filter by event type and subtype
- Navigate through events (forward, back, item selector)
- Edit event fields: type, subType, desc, seizureTimes
- Display graphs: acceleration magnitude and heart rate
- Show seizureTimes markers on graphs
- Prompt to save/discard changes before navigation or exit
"""

import sys
import os
import json
import sqlite3
import argparse
from typing import Optional, List, Dict, Any

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QTextEdit, QComboBox, QPushButton, QSpinBox,
    QFileDialog, QMessageBox, QGroupBox, QGridLayout, QScrollArea, QDialog,
    QDateEdit
)
from PyQt5.QtCore import Qt, pyqtSignal, QDate
from PyQt5.QtGui import QDoubleValidator, QKeySequence

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Add parent src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class DatabaseManager:
    """Handles database operations for event editing."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        
    def get_event_types(self) -> List[str]:
        """Get unique event types from database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT type FROM events WHERE type IS NOT NULL ORDER BY type")
        return [row[0] for row in cursor.fetchall()]
    
    def get_event_subtypes(self, event_type: Optional[str] = None) -> List[str]:
        """Get unique event subtypes, optionally filtered by type."""
        cursor = self.conn.cursor()
        if event_type:
            cursor.execute(
                "SELECT DISTINCT subType FROM events WHERE subType IS NOT NULL AND type = ? ORDER BY subType",
                (event_type,)
            )
        else:
            cursor.execute("SELECT DISTINCT subType FROM events WHERE subType IS NOT NULL ORDER BY subType")
        return [row[0] for row in cursor.fetchall()]
    
    def get_user_ids(self, event_type: Optional[str] = None, event_subtype: Optional[str] = None) -> List[int]:
        """Get unique user IDs, optionally filtered by type and subtype."""
        cursor = self.conn.cursor()
        query = "SELECT DISTINCT userId FROM events WHERE userId IS NOT NULL"
        params = []
        
        if event_type:
            query += " AND type = ?"
            params.append(event_type)
        
        if event_subtype:
            query += " AND subType = ?"
            params.append(event_subtype)
        
        query += " ORDER BY userId"
        cursor.execute(query, params)
        return [row[0] for row in cursor.fetchall()]
    
    def get_filtered_events(
        self, 
        event_type: Optional[str] = None, 
        event_subtype: Optional[str] = None,
        user_id: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        desc_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get events matching filters."""
        cursor = self.conn.cursor()
        query = "SELECT id, type, subType, userId, dataTime, desc, datapoint_count FROM events WHERE 1=1"
        params = []
        
        if event_type:
            query += " AND type = ?"
            params.append(event_type)
        
        if event_subtype:
            query += " AND subType = ?"
            params.append(event_subtype)
        
        if user_id is not None:
            query += " AND userId = ?"
            params.append(user_id)
        
        if start_date:
            query += " AND dataTime >= ?"
            params.append(start_date)
        
        if end_date:
            # Add one day to end_date to include events on that day
            query += " AND dataTime < ?"
            params.append(end_date)
        
        if desc_filter:
            query += " AND desc LIKE ? COLLATE NOCASE"
            params.append(desc_filter)
        
        query += " ORDER BY dataTime"
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def get_event_details(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get complete event details including metadata."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM events WHERE id = ?", (event_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        event = dict(row)
        
        # Parse metadata JSON
        if event['metadata']:
            try:
                metadata = json.loads(event['metadata'])
                event.update(metadata)
            except json.JSONDecodeError:
                pass
        
        # Parse seizureTimes from dedicated column (takes precedence over metadata)
        if event.get('seizureTimes'):
            try:
                event['seizureTimes'] = json.loads(event['seizureTimes'])
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Get datapoints
        cursor.execute(
            "SELECT * FROM datapoints WHERE event_id = ? ORDER BY dataTime",
            (event_id,)
        )
        datapoints = []
        for dp_row in cursor.fetchall():
            dp = dict(dp_row)
            # Parse JSON fields
            for field in ['rawData', 'rawData3D']:
                if dp.get(field):
                    try:
                        dp[field] = json.loads(dp[field])
                    except json.JSONDecodeError:
                        dp[field] = None
            datapoints.append(dp)
        
        event['datapoints'] = datapoints
        return event
    
    def update_event(
        self, 
        event_id: str, 
        event_type: str, 
        subtype: str, 
        description: str,
        seizure_times: Optional[List[float]] = None
    ) -> bool:
        """Update event fields in database."""
        try:
            cursor = self.conn.cursor()
            
            # Get current metadata
            cursor.execute("SELECT metadata FROM events WHERE id = ?", (event_id,))
            row = cursor.fetchone()
            if not row:
                return False
            
            # Parse existing metadata
            metadata = {}
            if row['metadata']:
                try:
                    metadata = json.loads(row['metadata'])
                except json.JSONDecodeError:
                    pass
            
            # Update metadata with description
            metadata['desc'] = description
            
            # Prepare seizureTimes for dedicated column
            seizure_times_json = None
            if seizure_times is not None:
                seizure_times_json = json.dumps(seizure_times)
            
            # Update database (seizureTimes in dedicated column, not metadata)
            cursor.execute(
                """UPDATE events 
                   SET type = ?, subType = ?, desc = ?, metadata = ?, seizureTimes = ?
                   WHERE id = ?""",
                (event_type, subtype, description, json.dumps(metadata), seizure_times_json, event_id)
            )
            
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error updating event: {e}")
            self.conn.rollback()
            return False
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


class EventEditor(QMainWindow):
    """Main window for event editing GUI."""
    
    def __init__(self, db_path: Optional[str] = None):
        super().__init__()
        
        self.db_manager: Optional[DatabaseManager] = None
        self.current_events: List[Dict[str, Any]] = []
        self.current_index: int = 0
        self.current_event: Optional[Dict[str, Any]] = None
        self.has_unsaved_changes: bool = False
        
        self.init_ui()
        
        if db_path:
            self.open_database(db_path)
    
    def init_ui(self):
        """Initialize user interface."""
        self.setWindowTitle("OSDB Event Editor")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Status label for when no database is loaded
        self.no_db_label = QLabel("No database loaded. Use File → Open Database to begin.")
        self.no_db_label.setAlignment(Qt.AlignCenter)
        self.no_db_label.setStyleSheet("QLabel { font-size: 14px; color: gray; padding: 20px; }")
        main_layout.addWidget(self.no_db_label)
        
        # Main content widget (will be hidden when no database)
        self.main_content_widget = QWidget()
        content_layout = QVBoxLayout(self.main_content_widget)
        
        # Filter section with vertical label/control stacking
        filter_group = QGroupBox("Filters")
        filter_layout = QHBoxLayout()
        
        # Event Type filter
        type_layout = QVBoxLayout()
        type_label = QLabel("Event Type:")
        type_label.setAlignment(Qt.AlignBottom)
        type_layout.addWidget(type_label)
        self.type_combo = QComboBox()
        self.type_combo.addItem("All Types", None)
        self.type_combo.setMinimumWidth(150)
        self.type_combo.currentIndexChanged.connect(self.on_type_changed)
        type_layout.addWidget(self.type_combo)
        filter_layout.addLayout(type_layout)
        
        # Sub-Type filter
        subtype_layout = QVBoxLayout()
        subtype_label = QLabel("Sub-Type:")
        subtype_label.setAlignment(Qt.AlignBottom)
        subtype_layout.addWidget(subtype_label)
        self.subtype_combo = QComboBox()
        self.subtype_combo.addItem("All Sub-Types", None)
        self.subtype_combo.setMinimumWidth(150)
        self.subtype_combo.currentIndexChanged.connect(self.on_subtype_changed)
        subtype_layout.addWidget(self.subtype_combo)
        filter_layout.addLayout(subtype_layout)
        
        # User ID filter
        user_layout = QVBoxLayout()
        user_label = QLabel("User ID:")
        user_label.setAlignment(Qt.AlignBottom)
        user_layout.addWidget(user_label)
        self.user_combo = QComboBox()
        self.user_combo.addItem("All Users", None)
        self.user_combo.setMinimumWidth(120)
        self.user_combo.currentIndexChanged.connect(self.apply_filters)
        user_layout.addWidget(self.user_combo)
        filter_layout.addLayout(user_layout)
        
        # Start Date filter
        start_date_layout = QVBoxLayout()
        start_date_label = QLabel("Start Date:")
        start_date_label.setAlignment(Qt.AlignBottom)
        start_date_layout.addWidget(start_date_label)
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.start_date_edit.setDate(QDate(2000, 1, 1))
        self.start_date_edit.setMinimumWidth(120)
        self.start_date_edit.setSpecialValueText("No limit")
        self.start_date_edit.dateChanged.connect(self.apply_filters)
        start_date_layout.addWidget(self.start_date_edit)
        filter_layout.addLayout(start_date_layout)
        
        # End Date filter
        end_date_layout = QVBoxLayout()
        end_date_label = QLabel("End Date:")
        end_date_label.setAlignment(Qt.AlignBottom)
        end_date_layout.addWidget(end_date_label)
        self.end_date_edit = QDateEdit()
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.end_date_edit.setDate(QDate.currentDate().addDays(1))
        self.end_date_edit.setMinimumWidth(120)
        self.end_date_edit.setSpecialValueText("No limit")
        self.end_date_edit.dateChanged.connect(self.apply_filters)
        end_date_layout.addWidget(self.end_date_edit)
        filter_layout.addLayout(end_date_layout)
        
        # Description text filter
        desc_layout = QVBoxLayout()
        desc_label = QLabel("Description:")
        desc_label.setAlignment(Qt.AlignBottom)
        desc_layout.addWidget(desc_label)
        self.desc_filter_edit = QLineEdit()
        self.desc_filter_edit.setPlaceholderText("Filter by text...")
        self.desc_filter_edit.setToolTip(
            "Filter events by description text.\n\n"
            "Wildcards:\n"
            "  % = any characters (e.g., %seizure% finds 'tonic clonic seizure')\n"
            "  _ = single character\n\n"
            "Examples:\n"
            "  %seizure% = contains 'seizure'\n"
            "  tonic% = starts with 'tonic'\n"
            "  %clonic = ends with 'clonic'\n\n"
            "Search is case-insensitive. Leave empty to show all."
        )
        self.desc_filter_edit.setMinimumWidth(150)
        self.desc_filter_edit.returnPressed.connect(self.apply_filters)
        desc_layout.addWidget(self.desc_filter_edit)
        filter_layout.addLayout(desc_layout)
        
        # Apply and Clear buttons
        button_layout = QVBoxLayout()
        button_layout.addWidget(QLabel(" "))  # Spacer to align with other labels
        button_sublayout = QHBoxLayout()
        apply_filter_btn = QPushButton("🔍 Apply")
        apply_filter_btn.setToolTip("Apply selected filters and reload event list")
        apply_filter_btn.clicked.connect(self.apply_filters)
        apply_filter_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 5px 10px; }")
        button_sublayout.addWidget(apply_filter_btn)
        clear_filter_btn = QPushButton("Clear")
        clear_filter_btn.setToolTip("Clear all filters")
        clear_filter_btn.clicked.connect(self.clear_filters)
        button_sublayout.addWidget(clear_filter_btn)
        button_layout.addLayout(button_sublayout)
        filter_layout.addLayout(button_layout)
        
        filter_layout.addStretch()  # Push everything to the left
        
        filter_group.setLayout(filter_layout)
        content_layout.addWidget(filter_group)
        
        # Navigation section with improved layout
        nav_group = QGroupBox("Navigation")
        nav_layout = QHBoxLayout()
        
        # Previous button with keyboard shortcut
        self.prev_btn = QPushButton("⬅ Previous")
        self.prev_btn.setShortcut("Left")
        self.prev_btn.setToolTip("Previous event (Left Arrow)")
        self.prev_btn.clicked.connect(self.previous_event)
        self.prev_btn.setEnabled(False)
        self.prev_btn.setMinimumWidth(100)
        nav_layout.addWidget(self.prev_btn)
        
        # Event position indicator with better visual hierarchy
        nav_layout.addWidget(QLabel("Event Position:"))
        self.event_index_spin = QSpinBox()
        self.event_index_spin.setMinimum(1)
        self.event_index_spin.setMaximum(1)
        self.event_index_spin.setMinimumWidth(70)
        self.event_index_spin.setToolTip("Current event position in filtered list")
        self.event_index_spin.valueChanged.connect(self.goto_event_by_position)
        nav_layout.addWidget(self.event_index_spin)
        
        self.event_count_label = QLabel("of 0")
        self.event_count_label.setMinimumWidth(50)
        nav_layout.addWidget(self.event_count_label)
        
        # Next button with keyboard shortcut
        self.next_btn = QPushButton("Next ➡")
        self.next_btn.setShortcut("Right")
        self.next_btn.setToolTip("Next event (Right Arrow)")
        self.next_btn.clicked.connect(self.next_event)
        self.next_btn.setEnabled(False)
        self.next_btn.setMinimumWidth(100)
        nav_layout.addWidget(self.next_btn)
        
        # Separator
        nav_layout.addWidget(QLabel("  |  "))
        
        # Jump to event ID feature
        nav_layout.addWidget(QLabel("Jump to Event ID:"))
        self.event_id_input = QLineEdit()
        self.event_id_input.setPlaceholderText("Enter event ID...")
        self.event_id_input.setToolTip("Enter an event ID and press Enter to jump to it")
        self.event_id_input.setMaximumWidth(150)
        self.event_id_input.returnPressed.connect(self.jump_to_event_id)
        nav_layout.addWidget(self.event_id_input)
        
        jump_btn = QPushButton("Go")
        jump_btn.setToolTip("Jump to the specified event ID")
        jump_btn.clicked.connect(self.jump_to_event_id)
        jump_btn.setMaximumWidth(50)
        nav_layout.addWidget(jump_btn)
        
        nav_layout.addStretch()
        
        nav_group.setLayout(nav_layout)
        content_layout.addWidget(nav_group)
        
        # Event details and graphs in horizontal split
        event_content_layout = QHBoxLayout()
        
        # Left side: Event details and editing
        details_widget = QWidget()
        details_layout = QVBoxLayout(details_widget)
        details_layout.setContentsMargins(0, 0, 0, 0)
        
        # Event info (read-only)
        info_group = QGroupBox("Event Information")
        info_layout = QGridLayout()
        
        info_layout.addWidget(QLabel("Event ID:"), 0, 0)
        self.event_id_label = QLabel("-")
        info_layout.addWidget(self.event_id_label, 0, 1)
        
        info_layout.addWidget(QLabel("Date/Time:"), 1, 0)
        self.datetime_label = QLabel("-")
        info_layout.addWidget(self.datetime_label, 1, 1)
        
        info_layout.addWidget(QLabel("User ID:"), 2, 0)
        self.user_id_label = QLabel("-")
        info_layout.addWidget(self.user_id_label, 2, 1)
        
        info_layout.addWidget(QLabel("Datapoints:"), 3, 0)
        self.datapoint_count_label = QLabel("-")
        info_layout.addWidget(self.datapoint_count_label, 3, 1)
        
        # Show Details button
        info_layout.addWidget(QLabel(""), 4, 0)  # Spacer row
        self.show_details_btn = QPushButton("📋 Show Details")
        self.show_details_btn.clicked.connect(self.show_event_details_dialog)
        self.show_details_btn.setToolTip("Show all event metadata in a popup")
        info_layout.addWidget(self.show_details_btn, 5, 0, 1, 2)
        
        info_group.setLayout(info_layout)
        details_layout.addWidget(info_group)
        
        # Editable fields
        edit_group = QGroupBox("Edit Event")
        edit_layout = QGridLayout()
        
        edit_layout.addWidget(QLabel("Type:"), 0, 0)
        self.type_edit = QComboBox()
        self.type_edit.setEditable(True)
        self.type_edit.currentTextChanged.connect(self.mark_changed)
        edit_layout.addWidget(self.type_edit, 0, 1, 1, 2)
        
        edit_layout.addWidget(QLabel("Sub-Type:"), 1, 0)
        self.subtype_edit = QComboBox()
        self.subtype_edit.setEditable(True)
        self.subtype_edit.currentTextChanged.connect(self.mark_changed)
        edit_layout.addWidget(self.subtype_edit, 1, 1, 1, 2)
        
        edit_layout.addWidget(QLabel("Description:"), 2, 0)
        self.desc_edit = QTextEdit()
        self.desc_edit.setMaximumHeight(80)
        self.desc_edit.textChanged.connect(self.mark_changed)
        edit_layout.addWidget(self.desc_edit, 2, 1, 1, 2)
        
        edit_group.setLayout(edit_layout)
        details_layout.addWidget(edit_group)
        
        # Seizure times editor (start and end) - visibility controlled by event type
        self.seizure_group = QGroupBox("Seizure Times (seconds from event start)")
        seizure_layout = QGridLayout()
        
        # Start time
        seizure_layout.addWidget(QLabel("Start Time:"), 0, 0)
        self.seizure_start_edit = QLineEdit("0.0")
        self.seizure_start_edit.setValidator(QDoubleValidator(-999999.0, 999999.0, 1))
        self.seizure_start_edit.textChanged.connect(self.mark_changed)
        self.seizure_start_edit.textChanged.connect(self.plot_event_data)
        seizure_layout.addWidget(self.seizure_start_edit, 0, 1)
        
        start_minus_btn = QPushButton("-5s")
        start_minus_btn.setMaximumWidth(50)
        start_minus_btn.clicked.connect(lambda: self.adjust_seizure_time('start', -5.0))
        seizure_layout.addWidget(start_minus_btn, 0, 2)
        
        start_plus_btn = QPushButton("+5s")
        start_plus_btn.setMaximumWidth(50)
        start_plus_btn.clicked.connect(lambda: self.adjust_seizure_time('start', 5.0))
        seizure_layout.addWidget(start_plus_btn, 0, 3)
        
        # End time
        seizure_layout.addWidget(QLabel("End Time:"), 1, 0)
        self.seizure_end_edit = QLineEdit("0.0")
        self.seizure_end_edit.setValidator(QDoubleValidator(-999999.0, 999999.0, 1))
        self.seizure_end_edit.textChanged.connect(self.mark_changed)
        self.seizure_end_edit.textChanged.connect(self.plot_event_data)
        seizure_layout.addWidget(self.seizure_end_edit, 1, 1)
        
        end_minus_btn = QPushButton("-5s")
        end_minus_btn.setMaximumWidth(50)
        end_minus_btn.clicked.connect(lambda: self.adjust_seizure_time('end', -5.0))
        seizure_layout.addWidget(end_minus_btn, 1, 2)
        
        end_plus_btn = QPushButton("+5s")
        end_plus_btn.setMaximumWidth(50)
        end_plus_btn.clicked.connect(lambda: self.adjust_seizure_time('end', 5.0))
        seizure_layout.addWidget(end_plus_btn, 1, 3)
        
        # Info label
        info_label = QLabel("(Times relative to event dataTime)")
        info_label.setStyleSheet("color: gray; font-size: 10px;")
        seizure_layout.addWidget(info_label, 2, 0, 1, 4)
        
        self.seizure_group.setLayout(seizure_layout)
        details_layout.addWidget(self.seizure_group)
        
        # Save/Revert buttons
        button_layout = QHBoxLayout()
        
        self.save_btn = QPushButton("💾 Save Changes")
        self.save_btn.clicked.connect(self.save_changes)
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        button_layout.addWidget(self.save_btn)
        
        self.revert_btn = QPushButton("↶ Revert")
        self.revert_btn.clicked.connect(self.revert_changes)
        self.revert_btn.setEnabled(False)
        button_layout.addWidget(self.revert_btn)
        
        details_layout.addLayout(button_layout)
        details_layout.addStretch()
        
        event_content_layout.addWidget(details_widget, 1)
        
        # Right side: Graphs
        graphs_widget = QWidget()
        graphs_layout = QVBoxLayout(graphs_widget)
        graphs_layout.setContentsMargins(0, 0, 0, 0)
        
        # Matplotlib figure for graphs
        self.figure = Figure(figsize=(8, 10))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        graphs_layout.addWidget(self.toolbar)
        graphs_layout.addWidget(self.canvas)
        
        event_content_layout.addWidget(graphs_widget, 2)
        
        content_layout.addLayout(event_content_layout, 1)
        
        # Add main content widget to main layout and initially hide it
        main_layout.addWidget(self.main_content_widget)
        self.main_content_widget.setVisible(False)
        
        # Status bar
        self.statusBar().showMessage("No database loaded")
    
    def create_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        # Open Database action
        open_action = file_menu.addAction("&Open Database...")
        open_action.setShortcut(QKeySequence.Open)
        open_action.setStatusTip("Open an OSDB SQLite database file")
        open_action.triggered.connect(self.open_database_dialog)
        
        # Close Database action
        self.close_db_action = file_menu.addAction("&Close Database")
        self.close_db_action.setShortcut("Ctrl+W")
        self.close_db_action.setStatusTip("Close the current database")
        self.close_db_action.triggered.connect(self.close_database)
        self.close_db_action.setEnabled(False)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = file_menu.addAction("E&xit")
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.close)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        
        # Mark as Deleted action
        self.mark_deleted_action = edit_menu.addAction("Mark as &Deleted")
        self.mark_deleted_action.setShortcut("Ctrl+D")
        self.mark_deleted_action.setStatusTip("Mark current event type as 'Deleted' (recommended over actual deletion)")
        self.mark_deleted_action.triggered.connect(self.mark_event_deleted)
        self.mark_deleted_action.setEnabled(False)
        
        # Mark as Unknown action
        self.mark_unknown_action = edit_menu.addAction("Mark as &Unknown")
        self.mark_unknown_action.setShortcut("Ctrl+U")
        self.mark_unknown_action.setStatusTip("Mark current event type as 'Unknown'")
        self.mark_unknown_action.triggered.connect(self.mark_event_unknown)
        self.mark_unknown_action.setEnabled(False)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        # Show Details action
        self.show_details_action = view_menu.addAction("Show Event &Details")
        self.show_details_action.setShortcut("Ctrl+I")
        self.show_details_action.setStatusTip("Show all event metadata in a popup")
        self.show_details_action.triggered.connect(self.show_event_details_dialog)
        self.show_details_action.setEnabled(False)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        # About action
        about_action = help_menu.addAction("&About")
        about_action.setStatusTip("About OSDB Event Editor")
        about_action.triggered.connect(self.show_about_dialog)
        
    def show_about_dialog(self):
        """Show the About dialog."""
        QMessageBox.about(
            self,
            "About OSDB Event Editor",
            "<h3>OSDB Event Editor</h3>"
            "<p>A Qt5-based GUI for viewing and editing events in OSDB SQLite databases.</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>Filter events by type, subtype, and user ID</li>"
            "<li>Edit event metadata and seizure times</li>"
            "<li>Visualize acceleration and heart rate data</li>"
            "<li>Navigate through events with keyboard shortcuts</li>"
            "</ul>"
            "<p><b>Keyboard Shortcuts:</b></p>"
            "<ul>"
            "<li>Left/Right Arrow: Navigate events</li>"
            "<li>Ctrl+O: Open database</li>"
            "<li>Ctrl+W: Close database</li>"
            "<li>Ctrl+D: Mark as Deleted</li>"
            "<li>Ctrl+U: Mark as Unknown</li>"
            "<li>Ctrl+I: Show event details</li>"
            "</ul>"
            "<p><b>Text Search:</b></p>"
            "<ul>"
            "<li>Use % as wildcard for any characters (e.g., %seizure%)</li>"
            "<li>Use _ as wildcard for single character</li>"
            "<li>Search is case-insensitive</li>"
            "<li>Leave empty to show all events</li>"
            "</ul>"
        )
    
    def mark_event_deleted(self):
        """Mark current event type as 'Deleted'."""
        if not self.current_event:
            return
        
        reply = QMessageBox.question(
            self,
            "Mark as Deleted",
            "This will change the event type to 'Deleted', which will exclude it from exports.\n\n"
            "This is the recommended approach instead of permanently deleting records.\n\n"
            "Do you want to continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.type_edit.setCurrentText("Deleted")
            self.mark_changed()
            self.statusBar().showMessage("Event type changed to 'Deleted'. Remember to save changes.", 5000)
    
    def mark_event_unknown(self):
        """Mark current event type as 'Unknown'."""
        if not self.current_event:
            return
        
        reply = QMessageBox.question(
            self,
            "Mark as Unknown",
            "This will change the event type to 'Unknown'.\n\n"
            "Do you want to continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.type_edit.setCurrentText("Unknown")
            self.mark_changed()
            self.statusBar().showMessage("Event type changed to 'Unknown'. Remember to save changes.", 5000)
    
    def show_event_details_dialog(self):
        """Show a dialog with all event metadata."""
        if not self.current_event:
            QMessageBox.information(self, "No Event", "No event is currently loaded.")
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Event Details: {self.current_event.get('id', 'Unknown')}")
        dialog.setMinimumSize(600, 500)
        
        layout = QVBoxLayout(dialog)
        
        # Create text area for metadata
        details_text = QTextEdit()
        details_text.setReadOnly(True)
        details_text.setStyleSheet("QTextEdit { font-family: monospace; }")
        
        # Build metadata text (exclude rawData and rawData3d)
        metadata_lines = []
        metadata_lines.append("=" * 60)
        metadata_lines.append(f"EVENT METADATA: {self.current_event.get('id', 'Unknown')}")
        metadata_lines.append("=" * 60)
        metadata_lines.append("")
        
        # Display all fields except rawData and rawData3d
        for key, value in sorted(self.current_event.items()):
            if key in ['rawData', 'rawData3D', 'rawData3d', 'datapoints']:
                # Skip large data fields
                if key == 'datapoints':
                    metadata_lines.append(f"{key}: [{len(value)} datapoints]")
                else:
                    metadata_lines.append(f"{key}: [Data excluded from display]")
            else:
                # Pretty print JSON-like data
                if isinstance(value, (dict, list)):
                    import json
                    try:
                        value_str = json.dumps(value, indent=2)
                        metadata_lines.append(f"{key}:")
                        for line in value_str.split('\n'):
                            metadata_lines.append(f"  {line}")
                    except:
                        metadata_lines.append(f"{key}: {value}")
                else:
                    metadata_lines.append(f"{key}: {value}")
            metadata_lines.append("")
        
        details_text.setPlainText('\n'.join(metadata_lines))
        layout.addWidget(details_text)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec_()

    
    def open_database_dialog(self):
        """Open file dialog to select database."""
        if self.has_unsaved_changes:
            reply = self.confirm_discard_changes()
            if not reply:
                return
        
        db_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open OSDB Database",
            "",
            "SQLite Database (*.db);;All Files (*)"
        )
        
        if db_path:
            self.open_database(db_path)
    
    def open_database(self, db_path: str):
        """Open database and load initial data."""
        try:
            if self.db_manager:
                self.db_manager.close()
            
            self.db_manager = DatabaseManager(db_path)
            
            # Show main content and hide "no database" message
            self.no_db_label.setVisible(False)
            self.main_content_widget.setVisible(True)
            
            # Enable menu actions
            self.close_db_action.setEnabled(True)
            self.mark_deleted_action.setEnabled(True)
            self.mark_unknown_action.setEnabled(True)
            self.show_details_action.setEnabled(True)
            
            # Update window title
            self.setWindowTitle(f"OSDB Event Editor - {os.path.basename(db_path)}")
            
            # Populate filter dropdowns
            self.populate_filters()
            
            # Load all events initially
            self.apply_filters()
            
            self.statusBar().showMessage(f"Loaded database: {os.path.basename(db_path)}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open database:\n{e}")
    
    def close_database(self):
        """Close the current database."""
        if self.has_unsaved_changes:
            reply = self.confirm_discard_changes()
            if not reply:
                return
        
        if self.db_manager:
            self.db_manager.close()
            self.db_manager = None
        
        # Hide main content and show "no database" message
        self.main_content_widget.setVisible(False)
        self.no_db_label.setVisible(True)
        
        # Disable menu actions
        self.close_db_action.setEnabled(False)
        self.mark_deleted_action.setEnabled(False)
        self.mark_unknown_action.setEnabled(False)
        self.show_details_action.setEnabled(False)
        
        # Reset window title
        self.setWindowTitle("OSDB Event Editor")
        
        # Clear current state
        self.current_events = []
        self.current_index = 0
        self.current_event = None
        self.has_unsaved_changes = False
        
        self.statusBar().showMessage("Database closed")

    
    def populate_filters(self):
        """Populate filter combo boxes with unique values."""
        if not self.db_manager:
            return
        
        # Save current selections
        current_type = self.type_combo.currentData()
        current_subtype = self.subtype_combo.currentData()
        current_user = self.user_combo.currentData()
        
        # Populate types
        self.type_combo.clear()
        self.type_combo.addItem("All Types", None)
        for event_type in self.db_manager.get_event_types():
            self.type_combo.addItem(event_type, event_type)
        
        # Restore selection
        if current_type:
            index = self.type_combo.findData(current_type)
            if index >= 0:
                self.type_combo.setCurrentIndex(index)
        
        # Populate subtypes and users
        self.populate_subtypes()
        self.populate_users()
    
    def populate_subtypes(self):
        """Populate subtype combo based on selected type."""
        if not self.db_manager:
            return
        
        selected_type = self.type_combo.currentData()
        
        self.subtype_combo.clear()
        self.subtype_combo.addItem("All Sub-Types", None)
        
        for subtype in self.db_manager.get_event_subtypes(selected_type):
            self.subtype_combo.addItem(subtype, subtype)
    
    def populate_users(self):
        """Populate user combo based on selected type and subtype."""
        if not self.db_manager:
            return
        
        selected_type = self.type_combo.currentData()
        selected_subtype = self.subtype_combo.currentData()
        
        self.user_combo.clear()
        self.user_combo.addItem("All Users", None)
        
        for user_id in self.db_manager.get_user_ids(selected_type, selected_subtype):
            self.user_combo.addItem(str(user_id), user_id)
    
    def on_type_changed(self):
        """Handle type filter change."""
        self.populate_subtypes()
        self.populate_users()
    
    def on_subtype_changed(self):
        """Handle subtype filter change."""
        self.populate_users()
    
    def clear_filters(self):
        """Clear all filters and reset to defaults."""
        self.type_combo.setCurrentIndex(0)
        self.subtype_combo.setCurrentIndex(0)
        self.user_combo.setCurrentIndex(0)
        self.start_date_edit.setDate(QDate(2000, 1, 1))
        self.end_date_edit.setDate(QDate.currentDate().addDays(1))
        self.desc_filter_edit.clear()
        self.apply_filters()
    
    def apply_filters(self):
        """Apply filters and reload event list."""
        if not self.db_manager:
            return
        
        if self.has_unsaved_changes:
            reply = self.confirm_discard_changes()
            if not reply:
                return
        
        event_type = self.type_combo.currentData()
        event_subtype = self.subtype_combo.currentData()
        user_id = self.user_combo.currentData()
        
        # Date range filters
        start_date = None
        end_date = None
        if self.start_date_edit.date() > QDate(2000, 1, 1):
            start_date = self.start_date_edit.date().toString("yyyy-MM-dd")
        if self.end_date_edit.date() < QDate.currentDate().addDays(1):
            end_date = self.end_date_edit.date().addDays(1).toString("yyyy-MM-dd")
        
        # Description text filter
        desc_filter = self.desc_filter_edit.text().strip()
        if not desc_filter:
            desc_filter = None
        
        self.current_events = self.db_manager.get_filtered_events(
            event_type, event_subtype, user_id, start_date, end_date, desc_filter
        )
        self.current_index = 0
        
        # Update navigation
        event_count = len(self.current_events)
        self.event_count_label.setText(f"of {event_count}")
        self.event_index_spin.setMaximum(max(1, event_count))
        
        if event_count > 0:
            self.event_index_spin.setValue(1)
            self.load_event(0)
            self.update_navigation_buttons()
        else:
            self.clear_event_display()
            self.statusBar().showMessage("No events match filter criteria")
    
    def update_navigation_buttons(self):
        """Update navigation button states."""
        has_events = len(self.current_events) > 0
        self.prev_btn.setEnabled(has_events and self.current_index > 0)
        self.next_btn.setEnabled(has_events and self.current_index < len(self.current_events) - 1)
        self.event_index_spin.setEnabled(has_events)
    
    def previous_event(self):
        """Navigate to previous event."""
        if self.current_index > 0:
            if self.has_unsaved_changes:
                reply = self.confirm_discard_changes()
                if not reply:
                    return
            
            self.current_index -= 1
            self.event_index_spin.setValue(self.current_index + 1)
            self.load_event(self.current_index)
    
    def next_event(self):
        """Navigate to next event."""
        if self.current_index < len(self.current_events) - 1:
            if self.has_unsaved_changes:
                reply = self.confirm_discard_changes()
                if not reply:
                    return
            
            self.current_index += 1
            self.event_index_spin.setValue(self.current_index + 1)
            self.load_event(self.current_index)
    
    def goto_event_by_position(self, value: int):
        """Go to specific event by position (1-based index)."""
        new_index = value - 1
        if 0 <= new_index < len(self.current_events) and new_index != self.current_index:
            if self.has_unsaved_changes:
                reply = self.confirm_discard_changes()
                if not reply:
                    # Revert spinbox to current index
                    self.event_index_spin.blockSignals(True)
                    self.event_index_spin.setValue(self.current_index + 1)
                    self.event_index_spin.blockSignals(False)
                    return
            
            self.current_index = new_index
            self.load_event(self.current_index)
    
    def jump_to_event_id(self):
        """Jump to a specific event by its ID."""
        if not self.db_manager or not self.current_events:
            QMessageBox.warning(self, "No Events", "No events loaded. Please apply filters first.")
            return
        
        event_id = self.event_id_input.text().strip()
        if not event_id:
            return
        
        # Search for event in current filtered list
        for i, event in enumerate(self.current_events):
            if str(event['id']) == event_id:
                if self.has_unsaved_changes:
                    reply = self.confirm_discard_changes()
                    if not reply:
                        return
                
                self.current_index = i
                self.event_index_spin.setValue(i + 1)
                self.load_event(i)
                self.event_id_input.clear()
                self.statusBar().showMessage(f"Jumped to event {event_id}", 3000)
                return
        
        # Event not found in current filter
        QMessageBox.information(
            self,
            "Event Not Found",
            f"Event ID '{event_id}' not found in current filtered list.\n\n"
            f"Try adjusting filters or check the event ID."
        )
        self.event_id_input.selectAll()
    
    def load_event(self, index: int):
        """Load event at specified index."""
        if not self.current_events or index < 0 or index >= len(self.current_events):
            return
        
        event_summary = self.current_events[index]
        event_id = event_summary['id']
        
        # Load full event details
        self.current_event = self.db_manager.get_event_details(event_id)
        
        if not self.current_event:
            QMessageBox.warning(self, "Error", f"Failed to load event {event_id}")
            return
        
        # Display event details
        self.display_event()
        self.has_unsaved_changes = False
        self.update_save_buttons()
        self.update_navigation_buttons()
        
        self.statusBar().showMessage(
            f"Event {index + 1}/{len(self.current_events)}: {event_id}"
        )
    
    def display_event(self):
        """Display current event in UI."""
        if not self.current_event:
            return
        
        event = self.current_event
        
        # Display read-only info
        self.event_id_label.setText(str(event.get('id', '-')))
        self.datetime_label.setText(str(event.get('dataTime', '-')))
        self.user_id_label.setText(str(event.get('userId', '-')))
        self.datapoint_count_label.setText(str(event.get('datapoint_count', 0)))
        
        # Populate editable fields
        self.type_edit.blockSignals(True)
        self.type_edit.clear()
        for event_type in self.db_manager.get_event_types():
            self.type_edit.addItem(event_type)
        self.type_edit.setCurrentText(event.get('type', ''))
        self.type_edit.currentTextChanged.connect(self.on_event_type_changed)
        self.type_edit.blockSignals(False)
        
        # Update seizure times visibility based on event type
        self.update_seizure_times_visibility()
        
        self.subtype_edit.blockSignals(True)
        self.subtype_edit.clear()
        for subtype in self.db_manager.get_event_subtypes():
            self.subtype_edit.addItem(subtype)
        self.subtype_edit.setCurrentText(event.get('subType', ''))
        self.subtype_edit.blockSignals(False)
        
        self.desc_edit.blockSignals(True)
        self.desc_edit.setPlainText(event.get('desc', ''))
        self.desc_edit.blockSignals(False)
        
        # Display seizure times (handle None) - expect exactly 2 values [start, end]
        seizure_times = event.get('seizureTimes', [0.0, 0.0])
        if seizure_times is None or not isinstance(seizure_times, list):
            seizure_times = [0.0, 0.0]
        elif len(seizure_times) < 2:
            seizure_times = seizure_times + [0.0] * (2 - len(seizure_times))
        elif len(seizure_times) > 2:
            seizure_times = seizure_times[:2]  # Take only first 2
        
        self.display_seizure_times(seizure_times)
        
        # Plot graphs
        self.plot_event_data()
    
    def update_seizure_times_visibility(self):
        """Show/hide seizure times section based on event type."""
        if not self.current_event:
            return
        
        event_type = self.type_edit.currentText()
        is_seizure = event_type.lower() == 'seizure'
        self.seizure_group.setVisible(is_seizure)
    
    def on_event_type_changed(self):
        """Handle event type change in editor."""
        self.update_seizure_times_visibility()
        self.mark_changed()
        self.plot_event_data()
    
    def display_seizure_times(self, seizure_times: List[float]):
        """Display seizure times (start and end) in UI."""
        # Ensure we have exactly 2 values
        if len(seizure_times) < 2:
            seizure_times = seizure_times + [0.0] * (2 - len(seizure_times))
        
        # Block signals to avoid triggering mark_changed during initialization
        self.seizure_start_edit.blockSignals(True)
        self.seizure_end_edit.blockSignals(True)
        
        self.seizure_start_edit.setText(f"{seizure_times[0]:.1f}")
        self.seizure_end_edit.setText(f"{seizure_times[1]:.1f}")
        
        self.seizure_start_edit.blockSignals(False)
        self.seizure_end_edit.blockSignals(False)
    
    def adjust_seizure_time(self, which: str, adjustment: float):
        """Adjust seizure time by given amount."""
        if which == 'start':
            try:
                current = float(self.seizure_start_edit.text())
            except ValueError:
                current = 0.0
            new_val = current + adjustment
            self.seizure_start_edit.setText(f"{new_val:.1f}")
        elif which == 'end':
            try:
                current = float(self.seizure_end_edit.text())
            except ValueError:
                current = 0.0
            new_val = current + adjustment
            self.seizure_end_edit.setText(f"{new_val:.1f}")
    
    def get_current_seizure_times(self) -> List[float]:
        """Get current seizure times from UI - returns [start, end]."""
        try:
            start = float(self.seizure_start_edit.text())
        except (ValueError, AttributeError):
            start = 0.0
        
        try:
            end = float(self.seizure_end_edit.text())
        except (ValueError, AttributeError):
            end = 0.0
        
        return [start, end]
    
    def plot_event_data(self):
        """Plot acceleration magnitude and heart rate graphs."""
        if not self.current_event:
            self.figure.clear()
            self.canvas.draw()
            return
        
        datapoints = self.current_event.get('datapoints', [])
        if not datapoints:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No datapoints available', 
                   ha='center', va='center', fontsize=14)
            self.canvas.draw()
            return
        
        # Get event dataTime as reference point
        from datetime import datetime
        try:
            event_dt = datetime.fromisoformat(self.current_event['dataTime'].replace('Z', '+00:00'))
        except (ValueError, KeyError, AttributeError):
            # Fallback: use index-based time if parsing fails
            event_dt = None
        
        # Extract data
        raw_data_list = []
        hr_list = []
        time_points = []  # Time in seconds relative to event dataTime
        
        for i, dp in enumerate(datapoints):
            # Calculate time relative to event dataTime
            if event_dt:
                try:
                    dp_dt = datetime.fromisoformat(dp['dataTime'].replace('Z', '+00:00'))
                    time_sec = (dp_dt - event_dt).total_seconds()
                except (ValueError, KeyError, AttributeError):
                    time_sec = i * 5  # Fallback
            else:
                time_sec = i * 5  # Fallback: assume 5s intervals
            
            time_points.append(time_sec)
            
            # Extract rawData (acceleration magnitude)
            raw_data = dp.get('rawData')
            if raw_data and isinstance(raw_data, list):
                raw_data_list.extend(raw_data)
            else:
                raw_data_list.extend([0] * 125)  # Placeholder
            
            # Extract heart rate
            hr = dp.get('hr', 0)
            hr_list.append(hr if hr else 0)
        
        # Create time axes
        if raw_data_list and time_points:
            # 125 samples per 5 seconds = 25 Hz
            # Create time array for raw data samples
            start_time = time_points[0]
            end_time = time_points[-1] + 5  # Last datapoint + 5 seconds
            raw_time = np.linspace(start_time, end_time, len(raw_data_list))
        else:
            raw_time = []
        
        hr_time = np.array(time_points)
        
        # Get seizure times for markers (only if event type is Seizure)
        event_type = self.type_edit.currentText() if self.current_event else ''
        show_seizure_markers = event_type.lower() == 'seizure'
        seizure_times = self.get_current_seizure_times() if show_seizure_markers else None
        
        # Clear and create subplots
        self.figure.clear()
        
        if raw_data_list:
            ax1 = self.figure.add_subplot(211)
            ax1.plot(raw_time, raw_data_list, 'b-', alpha=0.7, linewidth=0.5)
            ax1.set_title('Acceleration Magnitude (rawData)')
            ax1.set_xlabel('Time (seconds)')
            ax1.set_ylabel('Acceleration (milli-g)')
            ax1.grid(True, alpha=0.3)
            
            # Add seizure time markers (start and end) with shaded region - only for seizure events
            if show_seizure_markers and seizure_times and len(seizure_times) >= 2:
                start_time = seizure_times[0]
                end_time = seizure_times[1]
                
                # Shade the seizure region
                if start_time < end_time:
                    ax1.axvspan(start_time, end_time, alpha=0.2, color='red', label='Seizure Period')
                
                # Add vertical lines at start and end
                ax1.axvline(x=start_time, color='red', linestyle='--', linewidth=2, alpha=0.8)
                ax1.axvline(x=end_time, color='red', linestyle='--', linewidth=2, alpha=0.8)
                
                # Add labels
                y_pos = ax1.get_ylim()[1] * 0.95
                ax1.text(start_time, y_pos, f'Start: {start_time:.1f}s',
                        rotation=90, va='top', ha='right', color='red', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                ax1.text(end_time, y_pos, f'End: {end_time:.1f}s',
                        rotation=90, va='top', ha='right', color='red', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                
                ax1.legend(loc='upper left')
        
        if hr_list and any(hr > 0 for hr in hr_list):
            ax2 = self.figure.add_subplot(212)
            ax2.plot(hr_time, hr_list, 'r-', marker='o', linewidth=2, markersize=4)
            ax2.set_title('Heart Rate')
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Heart Rate (bpm)')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(bottom=0)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def mark_changed(self):
        """Mark that event has unsaved changes."""
        self.has_unsaved_changes = True
        self.update_save_buttons()
    
    def update_save_buttons(self):
        """Update save/revert button states."""
        self.save_btn.setEnabled(self.has_unsaved_changes)
        self.revert_btn.setEnabled(self.has_unsaved_changes)
    
    def save_changes(self):
        """Save changes to database."""
        if not self.current_event or not self.db_manager:
            return
        
        event_id = self.current_event['id']
        event_type = self.type_edit.currentText()
        subtype = self.subtype_edit.currentText()
        description = self.desc_edit.toPlainText()
        seizure_times = self.get_current_seizure_times()
        
        success = self.db_manager.update_event(
            event_id, event_type, subtype, description, seizure_times
        )
        
        if success:
            self.has_unsaved_changes = False
            self.update_save_buttons()
            
            # Update current event cache
            self.current_event['type'] = event_type
            self.current_event['subType'] = subtype
            self.current_event['desc'] = description
            self.current_event['seizureTimes'] = seizure_times
            
            # Update summary in list
            if 0 <= self.current_index < len(self.current_events):
                self.current_events[self.current_index]['type'] = event_type
                self.current_events[self.current_index]['subType'] = subtype
            
            QMessageBox.information(self, "Success", "Changes saved successfully")
            self.statusBar().showMessage("Changes saved", 3000)
        else:
            QMessageBox.critical(self, "Error", "Failed to save changes to database")
    
    def revert_changes(self):
        """Revert changes and reload event."""
        if self.current_event:
            self.display_event()
            self.has_unsaved_changes = False
            self.update_save_buttons()
            self.statusBar().showMessage("Changes reverted", 3000)
    
    def confirm_discard_changes(self) -> bool:
        """Prompt user to save or discard changes. Returns True if OK to proceed."""
        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            "You have unsaved changes. Do you want to save them?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
        )
        
        if reply == QMessageBox.Save:
            self.save_changes()
            return True
        elif reply == QMessageBox.Discard:
            self.has_unsaved_changes = False
            return True
        else:  # Cancel
            return False
    
    def clear_event_display(self):
        """Clear event display."""
        self.event_id_label.setText("-")
        self.datetime_label.setText("-")
        self.user_id_label.setText("-")
        self.datapoint_count_label.setText("-")
        self.type_edit.clear()
        self.subtype_edit.clear()
        self.desc_edit.clear()
        
        # Clear seizure times
        self.seizure_start_edit.setText("0.0")
        self.seizure_end_edit.setText("0.0")
        
        # Clear graphs
        self.figure.clear()
        self.canvas.draw()
    
    def closeEvent(self, event):
        """Handle window close event."""
        if self.has_unsaved_changes:
            reply = self.confirm_discard_changes()
            if not reply:
                event.ignore()
                return
        
        if self.db_manager:
            self.db_manager.close()
        
        event.accept()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="OSDB Event Editor GUI")
    parser.add_argument('--db', type=str, help='Path to SQLite database file')
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    editor = EventEditor(db_path=args.db)
    editor.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

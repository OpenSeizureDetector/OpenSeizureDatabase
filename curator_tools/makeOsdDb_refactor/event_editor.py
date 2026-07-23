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
import argparse
from typing import Optional, List, Dict, Any
from datetime import datetime

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QTextEdit, QComboBox, QPushButton, QSpinBox,
    QFileDialog, QMessageBox, QGroupBox, QGridLayout, QScrollArea, QDialog,
    QDateEdit, QListWidget, QAbstractItemView, QProgressDialog, QInputDialog
)
from PyQt5.QtCore import Qt, pyqtSignal, QDate
from PyQt5.QtGui import QDoubleValidator, QKeySequence, QCursor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Add src directory to path (shared with makeOsdDb_refactored_wrapper.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import OsdWorkingDb from src
from osdb_sqlite import OsdWorkingDb

# Import modules for publishing (JSON files needed for publication)
import libosd.osdDbConnection
import libosd.configUtils


class EventEditor(QMainWindow):
    """Main window for event editing GUI."""
    
    # Event types and their corresponding sub-types from http://osdapi.org.uk/static/eventTypes.json
    EVENT_TYPES_MAP = {
        "Seizure": ["Tonic-Clonic", "Aura", "Other", "Simulation"],
        "Fall": ["Stumble", "Controlled", "Uncontrolled"],
        "Other Medical Issue": ["Fever", "Cold", "Other"],
        "False Alarm": [
            "Brushing Hair/Teeth", "Computer Games", "Walking/Running/Cycling",
            "Motor Vehicle", "Pushing Pram/Wheelchair/Lawn Mower", "Sorting/Knitting",
            "Talking/Standing Still", "Typing/Hand Tools", "Cooking/Washing/Cleaning",
            "Other", "Unknown"
        ],
        "Unknown": ["Unknown"],
        "Testing": ["Testing"]
    }
    
    def __init__(self, db_path: Optional[str] = None):
        super().__init__()
        
        self.db_manager: Optional[OsdWorkingDb] = None
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
        
        # Event Type filter (multi-select)
        type_layout = QVBoxLayout()
        type_label = QLabel("Event Type:")
        type_label.setAlignment(Qt.AlignTop)
        type_layout.addWidget(type_label)
        self.type_list = QListWidget()
        self.type_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.type_list.setMinimumWidth(150)
        self.type_list.setMaximumHeight(80)
        self.type_list.setToolTip("Hold Ctrl/Cmd to select multiple types")
        # Prevent 'current item' highlighting when item is not selected
        self.type_list.setStyleSheet("QListWidget::item:!selected { background: transparent; }")
        self.type_list.itemSelectionChanged.connect(self.on_type_selection_changed)
        type_layout.addWidget(self.type_list)
        filter_layout.addLayout(type_layout)
        
        # Sub-Type filter (multi-select)
        subtype_layout = QVBoxLayout()
        subtype_label = QLabel("Sub-Type:")
        subtype_label.setAlignment(Qt.AlignTop)
        subtype_layout.addWidget(subtype_label)
        self.subtype_list = QListWidget()
        self.subtype_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.subtype_list.setMinimumWidth(150)
        self.subtype_list.setMaximumHeight(80)
        self.subtype_list.setToolTip("Hold Ctrl/Cmd to select multiple sub-types (filtered by selected types)")
        # Prevent 'current item' highlighting when item is not selected
        self.subtype_list.setStyleSheet("QListWidget::item:!selected { background: transparent; }")
        subtype_layout.addWidget(self.subtype_list)
        filter_layout.addLayout(subtype_layout)
        
        # User ID filter (multi-select)
        user_layout = QVBoxLayout()
        user_label = QLabel("User ID:")
        user_label.setAlignment(Qt.AlignTop)
        user_layout.addWidget(user_label)
        self.user_list = QListWidget()
        self.user_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.user_list.setMinimumWidth(120)
        self.user_list.setMaximumHeight(80)
        self.user_list.setToolTip("Hold Ctrl/Cmd to select multiple users")
        # Prevent 'current item' highlighting when item is not selected
        self.user_list.setStyleSheet("QListWidget::item:!selected { background: transparent; }")
        user_layout.addWidget(self.user_list)
        filter_layout.addLayout(user_layout)
        
        # Start Date filter
        start_date_layout = QVBoxLayout()
        start_date_label = QLabel("Start Date:")
        start_date_label.setAlignment(Qt.AlignTop)
        start_date_layout.addWidget(start_date_label)
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.start_date_edit.setDate(QDate(2000, 1, 1))
        self.start_date_edit.setMinimumWidth(120)
        self.start_date_edit.setSpecialValueText("No limit")
        # Don't auto-trigger apply to avoid unsaved changes dialog
        start_date_layout.addWidget(self.start_date_edit)
        start_date_layout.addStretch()  # Push content to top
        filter_layout.addLayout(start_date_layout)
        
        # End Date filter
        end_date_layout = QVBoxLayout()
        end_date_label = QLabel("End Date:")
        end_date_label.setAlignment(Qt.AlignTop)
        end_date_layout.addWidget(end_date_label)
        self.end_date_edit = QDateEdit()
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.end_date_edit.setDate(QDate.currentDate().addDays(1))
        self.end_date_edit.setMinimumWidth(120)
        self.end_date_edit.setSpecialValueText("No limit")
        # Don't auto-trigger apply to avoid unsaved changes dialog
        end_date_layout.addWidget(self.end_date_edit)
        end_date_layout.addStretch()  # Push content to top
        filter_layout.addLayout(end_date_layout)
        
        # Description text filter
        desc_layout = QVBoxLayout()
        desc_label = QLabel("Description:")
        desc_label.setAlignment(Qt.AlignTop)
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
        # Don't auto-trigger apply on text change to avoid unsaved changes dialog
        self.desc_filter_edit.returnPressed.connect(self.apply_filters)
        desc_layout.addWidget(self.desc_filter_edit)
        desc_layout.addStretch()  # Push content to top
        filter_layout.addLayout(desc_layout)
        
        # Apply and Clear buttons
        button_layout = QVBoxLayout()
        button_spacer = QLabel(" ")
        button_spacer.setAlignment(Qt.AlignTop)
        button_layout.addWidget(button_spacer)  # Spacer to align with other labels
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
        button_layout.addStretch()  # Push content to top
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
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        
        # Generate Graphs action
        self.generate_graphs_action = tools_menu.addAction("Generate &Graphs...")
        self.generate_graphs_action.setStatusTip("Generate summary graphs from database and save to folder")
        self.generate_graphs_action.triggered.connect(self.generate_graphs_from_db)
        self.generate_graphs_action.setEnabled(False)
        
        # Generate Index action
        self.generate_index_action = tools_menu.addAction("Generate &Index...")
        self.generate_index_action.setStatusTip("Generate CSV index files from database and save to folder")
        self.generate_index_action.triggered.connect(self.generate_index_from_db)
        self.generate_index_action.setEnabled(False)
        
        tools_menu.addSeparator()
        
        # Publish Database action
        self.publish_database_action = tools_menu.addAction("&Publish Database...")
        self.publish_database_action.setStatusTip("Export database to JSON files with index and graphs")
        self.publish_database_action.triggered.connect(self.publish_database)
        self.publish_database_action.setEnabled(False)
        
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
            "<li>Filter events by type, subtype, user ID, date range, and description</li>"
            "<li>Multi-select filtering (hold Ctrl/Cmd to select multiple values)</li>"
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

    def generate_graphs_from_db(self):
        """Generate summary graphs directly from database (efficient, no temp files)."""
        if not self.db_manager:
            QMessageBox.warning(self, "No Database", "No database is currently loaded.")
            return
        
        # Get database path
        db_path = self.db_manager.db_path
        
        print("Opening file dialog for graph output directory...")
        
        # Use text input to avoid Qt filesystem scanning hang with large files
        # Even non-native dialogs scan/stat files which causes hangs
        default_dir = os.path.join(os.path.dirname(db_path), "output")
        
        output_dir, ok = QInputDialog.getText(
            self,
            "Output Directory for Graphs",
            "Enter output directory path:\n(Use Tab to autocomplete in terminal, or paste full path)",
            QLineEdit.Normal,
            default_dir
        )
        
        print(f"Selected directory: {output_dir}")
        
        if not ok or not output_dir:
            print("User cancelled directory selection")
            return  # User cancelled
        
        # Expand ~ and validate path
        output_dir = os.path.expanduser(output_dir.strip())
        
        # Create progress dialog
        progress = QProgressDialog("Initializing...", "Cancel", 0, 5, self)
        progress.setWindowTitle("Generating Graphs")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.show()  # Force show immediately
        progress.setValue(0)
        QApplication.processEvents()  # Force GUI update
        
        try:
            # Force matplotlib to use non-interactive backend
            import matplotlib
            matplotlib.use('Agg', force=True)
            import matplotlib.pyplot as plt
            plt.ioff()  # Turn off interactive mode
            
            # Change cursor to busy
            QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
            
            # Load database
            progress.setLabelText("Loading events from database...")
            progress.setValue(0)
            QApplication.processEvents()
            QApplication.processEvents()  # Double process to ensure update
            if progress.wasCanceled():
                QApplication.restoreOverrideCursor()
                progress.close()
                return
            
            db = OsdWorkingDb(db_path, debug=False)
            all_events = db.get_events(include_datapoints=False)
            progress.setValue(1)
            QApplication.processEvents()
            
            if not all_events:
                QApplication.restoreOverrideCursor()
                progress.close()
                QMessageBox.warning(self, "No Events", "No events found in database to generate graphs.")
                return
            
            # Categorize events
            progress.setLabelText(f"Categorizing {len(all_events)} events...")
            progress.setValue(1)
            QApplication.processEvents()
            QApplication.processEvents()
            if progress.wasCanceled():
                db.close()
                QApplication.restoreOverrideCursor()
                progress.close()
                return
            
            import generateGraphs
            categorized = generateGraphs.categorize_events(all_events)
            stats = generateGraphs.create_summary_stats(categorized)
            progress.setValue(2)
            QApplication.processEvents()
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate summary statistics chart
            progress.setLabelText("Generating summary statistics chart...")
            progress.setValue(2)
            QApplication.processEvents()
            QApplication.processEvents()
            if progress.wasCanceled():
                db.close()
                QApplication.restoreOverrideCursor()
                progress.close()
                return
            
            generateGraphs.create_summary_stats_chart(stats, output_dir)
            progress.setValue(3)
            QApplication.processEvents()
            
            # Generate events by user chart
            progress.setLabelText("Generating events by user chart...")
            progress.setValue(3)
            QApplication.processEvents()
            QApplication.processEvents()
            if progress.wasCanceled():
                db.close()
                QApplication.restoreOverrideCursor()
                progress.close()
                return
            
            generateGraphs.create_events_by_user_chart(
                categorized['seizures'],
                output_dir,
                threshold=5,
                debug=False
            )
            progress.setValue(4)
            QApplication.processEvents()
            
            # Generate cumulative seizures chart
            progress.setLabelText("Generating cumulative seizures chart...")
            progress.setValue(4)
            QApplication.processEvents()
            QApplication.processEvents()
            if progress.wasCanceled():
                db.close()
                QApplication.restoreOverrideCursor()
                progress.close()
                return
            
            generateGraphs.create_cumulative_seizures_per_month(
                categorized['seizures'],
                output_dir,
                threshold=5,
                debug=False
            )
            progress.setValue(5)
            QApplication.processEvents()
            
            # Close database
            db.close()
            
            # Restore cursor
            QApplication.restoreOverrideCursor()
            
            # Close progress dialog
            progress.close()
            
            QMessageBox.information(
                self,
                "Success",
                f"Graphs have been generated successfully.\n\nOutput directory: {output_dir}"
            )
            self.statusBar().showMessage(f"Graphs saved to {output_dir}", 5000)
                
        except Exception as e:
            # Restore cursor and close progress
            QApplication.restoreOverrideCursor()
            if 'progress' in locals():
                progress.close()
            if 'db' in locals():
                db.close()
            
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while generating graphs:\n{e}"
            )
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage("Error generating graphs", 5000)
    
    def generate_index_from_db(self):
        """Generate CSV index files directly from database (efficient, uses temp JSON only during processing)."""
        if not self.db_manager:
            QMessageBox.warning(self, "No Database", "No database is currently loaded.")
            return
        
        # Get text input to avoid Qt filesystem scanning hang with large files
        default_dir = os.path.join(os.path.dirname(db_path), "output")
        
        output_dir, ok = QInputDialog.getText(
            self,
            "Output Directory for Index Files",
            "Enter output directory path:\n(Use Tab to autocomplete in terminal, or paste full path)",
            QLineEdit.Normal,
            default_dir
        )
        
        print(f"Selected directory: {output_dir}")
        
        if not ok or not output_dir:
            print("User cancelled directory selection")
            return  # User cancelled
        
        # Expand ~ and validate path
        output_dir = os.path.expanduser(output_dir.strip())
        print(f"Selected directory: {output_dir}")
        
        if not output_dir:
            print("User cancelled directory selection")
            return  # User cancelled
        
        # Create progress dialog
        progress = QProgressDialog("Processing events...", "Cancel", 0, 5, self)
        progress = QProgressDialog("Preparing to generate index files...", "Cancel", 0, 6, self)
        progress.setWindowTitle("Generating Index Files")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        
        try:
            # Change cursor to busy
            QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
            
            # Load database and get events
            progress.setLabelText("Loading events from database...")
            QApplication.processEvents()
            if progress.wasCanceled():
                return
            
            db = OsdWorkingDb(db_path, debug=False)
            progress.setValue(1)
            
            # Process each category
            categories = {
                'allSeizures': ('All Seizures', lambda e: e.get('type') == 'Seizure'),
                'tcSeizures': ('Tonic-Clonic Seizures', lambda e: (
                    e.get('type') == 'Seizure' and 
                    ('tonic' in str(e.get('subType', '')).lower() or 
                     'clonic' in str(e.get('subType', '')).lower())
                )),
                'fallEvents': ('Fall Events', lambda e: e.get('type') == 'Fall'),
                'falseAlarms': ('False Alarms', lambda e: e.get('type') == 'False Alarm'),
                'ndaEvents': ('NDA Events', lambda e: e.get('type') == 'NDA'),
            }
            
            os.makedirs(output_dir, exist_ok=True)
            generated_count = 0
            step = 1
            
            all_events = db.get_events(include_datapoints=True)
            
            for category, (label, filter_func) in categories.items():
                step += 1
                progress.setValue(step)
                progress.setLabelText(f"Generating index for {label}...")
                QApplication.processEvents()
                if progress.wasCanceled():
                    break
                
                # Filter events
                category_events = [e for e in all_events if filter_func(e)]
                
                if not category_events:
                    continue
                
                # Create temp JSON and generate CSV
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_json:
                    json.dump(category_events, tmp_json, indent=2)
                    tmp_json_path = tmp_json.name
                
                try:
                    csv_filename = f"osdb_3min_{category}.csv"
                    csv_path = os.path.join(output_dir, csv_filename)
                    
                    osd = libosd.osdDbConnection.OsdDbConnection(cacheDir=None, debug=False)
                    osd.loadDbFile(tmp_json_path, useCacheDir=False)
                    osd.saveIndexFile(csv_path, useCacheDir=False)
                    
                    generated_count += 1
                except Exception as e:
                    print(f"Warning: Failed to generate index for {category}: {e}")
                finally:
                    if os.path.exists(tmp_json_path):
                        os.remove(tmp_json_path)
            
            # Close database
            db.close()
            
            # Restore cursor and close progress
            QApplication.restoreOverrideCursor()
            progress.close()
            
            if generated_count > 0:
                QMessageBox.information(
                    self,
                    "Success",
                    f"Generated {generated_count} index file(s) successfully.\n\nOutput directory: {output_dir}"
                )
                self.statusBar().showMessage(f"Index files saved to {output_dir}", 5000)
            else:
                QMessageBox.warning(self, "No Events", "No events found in database to generate indexes.")
                self.statusBar().showMessage("No events to process", 5000)
                
        except Exception as e:
            # Restore cursor and close progress
            QApplication.restoreOverrideCursor()
            if progress:
                progress.close()
            
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while generating index files:\n{e}"
            )
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage("Error generating index files", 5000)
    
    def publish_database(self):
        """Publish database to JSON files with index and graphs."""
        if not self.db_manager:
            QMessageBox.warning(self, "No Database", "No database is currently loaded.")
            return
        
        # Get text input to avoid Qt filesystem scanning hang with large files
        default_dir = os.path.join(os.path.dirname(db_path), "output")
        
        output_dir, ok = QInputDialog.getText(
            self,
            "Output Directory for Publication",
            "Enter output directory path:\n(Use Tab to autocomplete in terminal, or paste full path)",
            QLineEdit.Normal,
            default_dir
        )
        
        print(f"Selected directory: {output_dir}")
        
        if not ok or not output_dir:
            print("User cancelled directory selection")
            return  # User cancelled
        
        # Expand ~ and validate path
        output_dir = os.path.expanduser(output_dir.strip())
        print(f"Selected directory: {output_dir}")
        
        if not output_dir:
            print("User cancelled directory selection")
            return  # User cancelled
        
        # Create progress dialog (12 steps total: 5 JSON exports + 5 CSV indexes + 2 for graphs)
        progress = QProgressDialog("Preparing publication...", "Cancel", 0, 12, self)
        progress.setWindowTitle("Publishing Database")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        
        try:
            # Change cursor to busy
            QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
            
            # Get default grouping period
            grouping_period = '3min'
            
            # Export all event categories to JSON
            event_categories = ['allSeizures', 'tcSeizures', 'fallEvents', 'falseAlarms', 'ndaEvents']
            json_files = []
            
            step = 0
            for category in event_categories:
                step += 1
                progress.setValue(step)
                progress.setLabelText(f"Exporting {category} to JSON...")
                QApplication.processEvents()
                if progress.wasCanceled():
                    return
                
                json_file = os.path.join(output_dir, f"osdb_{grouping_period}_{category}.json")
                
                # Get events for this category
                events = self._get_events_by_category(category)
                
                if not events:
                    continue
                
                # Write JSON file
                with open(json_file, 'w') as f:
                    json.dump(events, f, indent=2)
                json_files.append(json_file)
            
            if not json_files:
                QApplication.restoreOverrideCursor()
                progress.close()
                QMessageBox.warning(self, "No Events", "No events found in database to publish.")
                return
            
            # Generate CSV indexes
            csv_count = 0
            for idx, json_file in enumerate(json_files):
                step += 1
                progress.setValue(step)
                progress.setLabelText(f"Generating CSV index {idx+1}/{len(json_files)}...")
                QApplication.processEvents()
                if progress.wasCanceled():
                    return
                
                csv_file = json_file.replace('.json', '.csv')
                try:
                    osd = libosd.osdDbConnection.OsdDbConnection(cacheDir=None, debug=False)
                    osd.loadDbFile(json_file, useCacheDir=False)
                    osd.saveIndexFile(csv_file, useCacheDir=False)
                    csv_count += 1
                except Exception as e:
                    print(f"Warning: Failed to generate index for {json_file}: {e}")
            
            # Generate graphs (inline to show progress)
            progress.setValue(10)
            progress.setLabelText("Loading events for graphs...")
            QApplication.processEvents()
            if progress.wasCanceled():
                return
            
            import generateGraphs
            db = OsdWorkingDb(db_path, debug=False)
            all_events = db.get_events(include_datapoints=False)
            categorized = generateGraphs.categorize_events(all_events)
            stats = generateGraphs.create_summary_stats(categorized)
            
            progress.setValue(11)
            progress.setLabelText("Generating summary graphs...")
            QApplication.processEvents()
            if progress.wasCanceled():
                return
            
            generateGraphs.create_summary_stats_chart(stats, output_dir)
            generateGraphs.create_events_by_user_chart(categorized['seizures'], output_dir, threshold=5, debug=False)
            generateGraphs.create_cumulative_seizures_per_month(categorized['seizures'], output_dir, threshold=5, debug=False)
            
            db.close()
            graph_success = True
            
            progress.setValue(12)
            
            # Restore cursor and close progress
            QApplication.restoreOverrideCursor()
            progress.close()
            
            # Show summary
            summary = (
                f"Database published successfully!\n\n"
                f"Output directory: {output_dir}\n\n"
                f"Generated:\n"
                f"  • {len(json_files)} JSON file(s)\n"
                f"  • {csv_count} CSV index file(s)\n"
                f"  • Summary graphs: {'✓' if graph_success else '✗'}"
            )
            
            QMessageBox.information(self, "Publication Complete", summary)
            self.statusBar().showMessage(f"Database published to {output_dir}", 5000)
                
        except Exception as e:
            # Restore cursor and close progress
            QApplication.restoreOverrideCursor()
            if progress:
                progress.close()
            
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while publishing database:\n{e}"
            )
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage("Error publishing database", 5000)
    
    def _get_events_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get events from database for a specific category.
        
        Args:
            category: Event category ('allSeizures', 'tcSeizures', 'fallEvents', 'falseAlarms', 'ndaEvents')
        
        Returns:
            List of events with datapoints
        """
        if not self.db_manager:
            return []
        
        # Get all events with datapoints
        all_events = self.db_manager.get_events(include_datapoints=True)
        
        # Filter by category
        filtered_events = []
        for event in all_events:
            event_type = event.get('type', 'Unknown')
            sub_type = event.get('subType', '')
            
            if category == 'tcSeizures':
                # Tonic-clonic seizures
                if event_type == 'Seizure' and ('tonic' in str(sub_type).lower() or 'clonic' in str(sub_type).lower()):
                    filtered_events.append(event)
            elif category == 'allSeizures':
                # All seizures
                if event_type == 'Seizure':
                    filtered_events.append(event)
            elif category == 'fallEvents':
                # Falls
                if event_type == 'Fall':
                    filtered_events.append(event)
            elif category == 'falseAlarms':
                # False alarms
                if event_type == 'False Alarm':
                    filtered_events.append(event)
            elif category == 'ndaEvents':
                # NDA events
                if event_type == 'NDA':
                    filtered_events.append(event)
        
        return filtered_events
    
    
    def open_database_dialog(self):
        """Open file dialog to select database."""
        if self.has_unsaved_changes:
            reply = self.confirm_discard_changes()
            if not reply:
                return

        # Use QFileDialog.getOpenFileName with DontUseNativeDialog option
        # This can help prevent crashes on systems with very large files in the directory
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        options |= QFileDialog.DontUseCustomDirectoryIcons
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open OSDB SQLite Database",
            "",  # Start in the current directory or last used directory
            "SQLite Databases (*.db *.sqlite *.sqlite3);;All Files (*)",
            options=options
        )
        if file_path:
            self.open_database(file_path)
    
    def open_database(self, db_path: str):
        """Open database and load initial data."""
        try:
            if self.db_manager:
                self.db_manager.close()
            
            self.db_manager = OsdWorkingDb(db_path)
            
            # Show main content and hide "no database" message
            self.no_db_label.setVisible(False)
            self.main_content_widget.setVisible(True)
            
            # Enable menu actions
            self.close_db_action.setEnabled(True)
            self.mark_deleted_action.setEnabled(True)
            self.mark_unknown_action.setEnabled(True)
            self.show_details_action.setEnabled(True)
            self.generate_graphs_action.setEnabled(True)
            self.generate_index_action.setEnabled(True)
            self.publish_database_action.setEnabled(True)
            
            # Update window title
            self.setWindowTitle(f"OSDB Event Editor - {os.path.basename(db_path)}")
            
            # Populate filter dropdowns (blocks signals internally)
            self.populate_filters()
            
            # Load all events initially (no unsaved changes at this point)
            self.has_unsaved_changes = False
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
        self.generate_graphs_action.setEnabled(False)
        self.generate_index_action.setEnabled(False)
        self.publish_database_action.setEnabled(False)
        
        # Reset window title
        self.setWindowTitle("OSDB Event Editor")
        
        # Clear current state
        self.current_events = []
        self.current_index = 0
        self.current_event = None
        self.has_unsaved_changes = False
        
        self.statusBar().showMessage("Database closed")

    
    def populate_filters(self):
        """Populate filter lists with unique values."""
        if not self.db_manager:
            return
        
        # Block signals to prevent triggering apply_filters during population
        self.type_list.blockSignals(True)
        self.subtype_list.blockSignals(True)
        self.user_list.blockSignals(True)
        
        # Populate types
        self.type_list.clear()
        for event_type in self.db_manager.get_event_types():
            self.type_list.addItem(event_type)
        
        # Populate subtypes
        self.subtype_list.clear()
        for subtype in self.db_manager.get_event_subtypes():
            self.subtype_list.addItem(subtype)
        
        # Populate users
        self.user_list.clear()
        for user_id in self.db_manager.get_user_ids():
            self.user_list.addItem(str(user_id))
        
        # Select "Seizure" by default if it exists
        for i in range(self.type_list.count()):
            item = self.type_list.item(i)
            if item.text() == "Seizure":
                item.setSelected(True)
                break
        
        # Unblock signals
        self.type_list.blockSignals(False)
        self.subtype_list.blockSignals(False)
        self.user_list.blockSignals(False)
        
        # Update subtypes based on selected types (will filter to Seizure subtypes if Seizure was found)
        self.update_subtype_list()
    
    def on_type_selection_changed(self):
        """Handle type selection change - update subtype list accordingly."""
        self.update_subtype_list()
    
    def update_subtype_list(self):
        """Update subtype list to only show subtypes for selected event types."""
        if not self.db_manager:
            return
        
        # Get currently selected types
        selected_types = [item.text() for item in self.type_list.selectedItems()]
        
        # Remember currently selected subtypes
        selected_subtypes = [item.text() for item in self.subtype_list.selectedItems()]
        
        # Block signals during update
        self.subtype_list.blockSignals(True)
        self.subtype_list.clear()
        
        # Get subtypes for selected types (or all types if none selected)
        if selected_types:
            # Get subtypes that belong to any of the selected types
            all_subtypes = set()
            for event_type in selected_types:
                subtypes = self.db_manager.get_event_subtypes(event_type)
                all_subtypes.update(subtypes)
            
            # Add subtypes in sorted order
            for subtype in sorted(all_subtypes):
                self.subtype_list.addItem(subtype)
        else:
            # No types selected - show all subtypes
            for subtype in self.db_manager.get_event_subtypes():
                self.subtype_list.addItem(subtype)
        
        # Restore selections where possible
        for i in range(self.subtype_list.count()):
            item = self.subtype_list.item(i)
            if item.text() in selected_subtypes:
                item.setSelected(True)
        
        # Clear current item highlighting if nothing is selected
        if not selected_subtypes:
            self.subtype_list.setCurrentRow(-1)
            self.subtype_list.clearFocus()
        
        # Unblock signals
        self.subtype_list.blockSignals(False)
    
    def clear_filters(self):
        """Clear all filters and reset to defaults."""
        # Block signals to prevent triggering during clear
        self.type_list.blockSignals(True)
        self.subtype_list.blockSignals(True)
        self.user_list.blockSignals(True)
        self.start_date_edit.blockSignals(True)
        self.end_date_edit.blockSignals(True)
        self.desc_filter_edit.blockSignals(True)
        
        self.type_list.clearSelection()
        self.subtype_list.clearSelection()
        self.user_list.clearSelection()
        self.start_date_edit.setDate(QDate(2000, 1, 1))
        self.end_date_edit.setDate(QDate.currentDate().addDays(1))
        self.desc_filter_edit.clear()
        
        # Clear current item highlighting
        self.type_list.setCurrentRow(-1)
        self.subtype_list.setCurrentRow(-1)
        self.user_list.setCurrentRow(-1)
        
        # Clear focus to prevent highlighting
        self.type_list.clearFocus()
        self.subtype_list.clearFocus()
        self.user_list.clearFocus()
        
        # Unblock signals
        self.type_list.blockSignals(False)
        self.subtype_list.blockSignals(False)
        self.user_list.blockSignals(False)
        self.start_date_edit.blockSignals(False)
        self.end_date_edit.blockSignals(False)
        self.desc_filter_edit.blockSignals(False)
        
        # Update subtype list to show all subtypes
        self.update_subtype_list()
        
        self.apply_filters()
    
    def apply_filters(self):
        """Apply filters and reload event list."""
        if not self.db_manager:
            return
        
        # Only check for unsaved changes if we actually have a current event loaded
        if self.has_unsaved_changes and self.current_event:
            reply = self.confirm_discard_changes()
            if not reply:
                return
        
        # Get selected types (empty list means all types)
        event_types = [item.text() for item in self.type_list.selectedItems()]
        if not event_types:
            event_types = None
        
        # Get selected subtypes
        event_subtypes = [item.text() for item in self.subtype_list.selectedItems()]
        if not event_subtypes:
            event_subtypes = None
        
        # Get selected user IDs
        user_ids = None
        selected_users = [item.text() for item in self.user_list.selectedItems()]
        if selected_users:
            try:
                user_ids = [int(uid) for uid in selected_users]
            except ValueError:
                pass
        
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
            event_types, event_subtypes, user_ids, start_date, end_date, desc_filter
        )
        self.current_index = 0
        
        # Update navigation
        event_count = len(self.current_events)
        self.event_count_label.setText(f"of {event_count}")
        
        if event_count > 0:
            self.event_index_spin.setMaximum(event_count)
            self.event_index_spin.setMinimum(1)
            self.event_index_spin.setValue(1)
            self.load_event(0)
            self.update_navigation_buttons()
            # Clear unsaved changes flag after loading filtered events
            self.has_unsaved_changes = False
            self.update_save_buttons()
            self.statusBar().showMessage(f"Found {event_count} event(s) matching filters")
        else:
            # No events found - disable navigation and show warning
            self.event_index_spin.setMaximum(1)
            self.event_index_spin.setMinimum(0)
            self.event_index_spin.setValue(0)
            self.clear_event_display()
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            self.event_index_spin.setEnabled(False)
            self.statusBar().showMessage("⚠ No events match filter criteria", 5000)
            QMessageBox.information(
                self,
                "No Results",
                "No events found matching the current filter criteria.\n\n"
                "Try adjusting or clearing your filters."
            )
    
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
        
        # Populate subtypes based on the selected event type
        event_type = event.get('type', '')
        event_subtype = event.get('subType', '')
        
        self.subtype_edit.blockSignals(True)
        self.subtype_edit.clear()
        
        # Get subtypes for this event type from the predefined map
        subtypes = self.EVENT_TYPES_MAP.get(event_type, [])
        
        if subtypes:
            # Add predefined subtypes
            for subtype in subtypes:
                self.subtype_edit.addItem(subtype)
            
            # If the event's subtype is not in the predefined list, add it
            if event_subtype and event_subtype not in subtypes:
                self.subtype_edit.addItem(event_subtype)
        else:
            # If no predefined subtypes, fall back to all subtypes from database
            for subtype in self.db_manager.get_event_subtypes():
                self.subtype_edit.addItem(subtype)
        
        self.subtype_edit.setCurrentText(event_subtype)
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
        # Update subtype options based on selected type
        selected_type = self.type_edit.currentText()
        current_subtype = self.subtype_edit.currentText()
        
        # Block signals to avoid unnecessary mark_changed calls during update
        self.subtype_edit.blockSignals(True)
        self.subtype_edit.clear()
        
        # Get subtypes for the selected type
        subtypes = self.EVENT_TYPES_MAP.get(selected_type, [])
        
        if subtypes:
            # Add the predefined subtypes for this type
            for subtype in subtypes:
                self.subtype_edit.addItem(subtype)
            
            # Try to preserve the current subtype if it's valid for the new type
            if current_subtype in subtypes:
                self.subtype_edit.setCurrentText(current_subtype)
            else:
                # Set to first item if current subtype is not valid
                self.subtype_edit.setCurrentIndex(0)
        else:
            # If no predefined subtypes, allow free-form entry
            if current_subtype:
                self.subtype_edit.addItem(current_subtype)
                self.subtype_edit.setCurrentText(current_subtype)
        
        self.subtype_edit.blockSignals(False)
        
        # Update visibility and mark as changed
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
        raw_time_list = []
        hr_list = []
        hr_time_list = []
        
        # Calculate time for each sample correctly:
        # Each datapoint has 125 samples at 25 Hz (5 seconds total)
        # Each sample time = datapoint_time + (sample_index / 25)
        # This matches the dataSummariser implementation
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
            
            # Extract rawData (acceleration magnitude)
            raw_data = dp.get('rawData')
            if raw_data and isinstance(raw_data, list):
                raw_data_list.extend(raw_data)
                # Calculate time for each sample: datapoint_time + sample_offset
                # 125 samples per 5 seconds = 25 Hz, so each sample is 1/25 second apart
                for n in range(len(raw_data)):
                    raw_time_list.append(time_sec + n / 25.0)
            else:
                # Placeholder
                raw_data_list.extend([0] * 125)
                for n in range(125):
                    raw_time_list.append(time_sec + n / 25.0)
            
            # Extract heart rate (one value per datapoint)
            hr = dp.get('hr', 0)
            hr_list.append(hr if hr else 0)
            hr_time_list.append(time_sec)
        
        # Convert to numpy arrays
        raw_time = np.array(raw_time_list) if raw_time_list else np.array([])
        hr_time = np.array(hr_time_list) if hr_time_list else np.array([])
        
        # Get seizure times for markers (only if event type is Seizure)
        event_type = self.type_edit.currentText() if self.current_event else ''
        show_seizure_markers = event_type.lower() == 'seizure'
        seizure_times = self.get_current_seizure_times() if show_seizure_markers else None
        
        # Clear and create subplots
        self.figure.clear()
        
        if raw_data_list and len(raw_time) > 0:
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

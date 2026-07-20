# OSDB Event Editor

A Qt5-based graphical user interface for viewing and editing events in the OSDB SQLite database.

## Features

- **Database Selection**: Open database via file dialog or command line argument
- **Smart Filtering**: Filter events by type and subtype to focus on specific event categories
- **Easy Navigation**: Step through events with forward/back buttons or jump to specific event numbers
- **Inline Editing**: Modify event type, subtype, description, and seizure times (start/end)
- **Seizure Time Controls**: Adjust seizure start and end times with convenient +/-5s buttons
- **Visual Data Display**: 
  - Acceleration magnitude graph from rawData
  - Heart rate graph over time
  - Seizure time markers overlaid on acceleration graph
- **Change Management**: Prompted to save or discard changes before navigation or exit
- **Event Information**: View event ID, date/time, user ID, and datapoint count

## Requirements

```bash
pip install PyQt5 matplotlib numpy
```

## Usage

### Launch with File Dialog

```bash
python3 event_editor.py
```

Then use "Open Database..." button to select your database file.

### Launch with Database Path

```bash
python3 event_editor.py --db /home/graham/osd/osdb/osdb_working.db
```

## Quick Start

1. **Open Database**: Click "Open Database..." or specify `--db` on command line
2. **Apply Filters** (optional): 
   - Select event type (e.g., "Seizure")
   - Select subtype (e.g., "Tonic-Clonic") - user list updates
   - Select specific user ID - only events from that user
   - Click "🔍 Apply Filters" to load matching events
3. **Navigate**: 
   - Use ⬅ Previous/Next ➡ buttons (or Left/Right arrow keys)
   - Or use event position spinner
   - Or jump directly to an event by entering its ID in "Jump to Event ID" field
4. **Edit Event**: Modify any of the editable fields:
   - Type (dropdown with existing types)
   - Sub-Type (dropdown with existing subtypes)
   - Description (text area)
   - Seizure Times (start and end time in seconds from event dataTime, can be negative)
5. **Adjust Seizure Times**: 
   - Click "+5s" or "-5s" to increment/decrement start or end time by 5 seconds
   - Or type exact values directly into the fields
   - Times are relative to event dataTime (negative = before dataTime, positive = after)
   - Example: -80.0 means 80 seconds before the event's dataTime
6. **Save Changes**: Click "💾 Save Changes" when ready
7. **Revert**: Click "↶ Revert" to discard changes and reload original data

## Interface Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ Database: [path]                          [Open Database...]    │
├─────────────────────────────────────────────────────────────────┤
│ Filters                                                          │
│  Event Type: [All Types ▼]      Sub-Type: [All Sub-Types ▼]    │
│  User ID: [All Users ▼]         [🔍 Apply Filters]              │
├─────────────────────────────────────────────────────────────────┤
│ Navigation                                                       │
│  [⬅ Previous] Event Position: [1] of 100 [Next ➡]  |           │
│                                Jump to Event ID: [____] [Go]    │
├──────────────────┬──────────────────────────────────────────────┤
│ Event Info       │                                              │
│  ID: 12345       │      Acceleration Magnitude Graph            │
│  Date: ...       │      (with seizure time markers)             │
│  User: 42        │                                              │
│  Datapoints: 50  │                                              │
├──────────────────┤                                              │
│ Edit Event       │                                              │
│  Type: [...]     │──────────────────────────────────────────────┤
│  SubType: [...]  │                                              │
│  Description:    │      Heart Rate Graph                        │
│  [text area]     │                                              │
├──────────────────┤                                              │
│ Seizure Times    │                                              │
│  Start: [10.5]   │                                              │
│    [-5s][+5s]    │                                              │
│  End: [25.3]     │                                              │
│    [-5s][+5s]    │                                              │
├──────────────────┤                                              │
│ [💾 Save]        │                                              │
│ [↶ Revert]       │                                              │
└──────────────────┴──────────────────────────────────────────────┘
```

## Data Visualization

### Acceleration Magnitude Graph
- Plots `rawData` values (acceleration magnitude in milli-g)
- Sample rate: 125 samples/second (125 samples per 5-second datapoint)
- **X-axis**: Time in seconds relative to event dataTime (negative = before event, positive = after)
- Red shaded region shows seizure period (start to end time)
- Red dashed vertical lines mark seizure start and end times
- Time labels show exact start and end times in seconds
- Markers are updated in real-time as seizure times are adjusted

### Heart Rate Graph  
- Plots `hr` values over time
- Sample interval: 5 seconds (one value per datapoint)
- **X-axis**: Time in seconds relative to event dataTime
- Empty if no heart rate data available

## Keyboard Shortcuts

- **Left Arrow**: Previous event
- **Right Arrow**: Next event
- **Enter** (in Jump field): Jump to specified event ID
- **Ctrl+O**: Open database (planned)
- **Ctrl+S**: Save changes (planned)

## Database Schema

The editor works with the OSDB SQLite schema:

- **events table**: Main event records (id, type, subType, desc, userId, dataTime, metadata)
- **datapoints table**: Time-series data for each event (hr, o2Sat, rawData, etc.)

Editable fields stored in:
- Direct columns: `type`, `subType`, `desc`
- Dedicated column: `seizureTimes` (JSON array with exactly 2 float values: [start_time, end_time] in seconds relative to event dataTime, can be negative)

## Change Tracking

The application tracks unsaved changes and prompts you to save or discard when:
- Navigating to a different event (Previous/Next/Index change)
- Applying new filters
- Closing the application

Changes are **not automatically saved** - you must explicitly click "💾 Save Changes".

## Technical Details

- **Framework**: PyQt5 for GUI, Matplotlib for graphing
- **Database**: Direct SQLite3 access (no ORM)
- **Data Extraction**: Parses JSON fields (rawData, metadata) from database
- **Graph Rendering**: Embedded Matplotlib canvas with navigation toolbar
- **Safe Updates**: Database updates wrapped in try-except with rollback

## Troubleshooting

**Problem**: Application won't start
- Ensure PyQt5 is installed: `pip install PyQt5 matplotlib numpy`

**Problem**: Database won't open
- Verify database file exists and is a valid SQLite database
- Check file permissions (read/write access required)

**Problem**: Graphs not displaying
- Verify events have datapoints with `rawData` or `hr` fields
- Check that datapoints have valid JSON in rawData field

**Problem**: Changes not saving
- Check console for error messages
- Verify database file is not locked by another process
- Ensure you have write permissions to database file

## Future Enhancements

- Keyboard shortcuts for navigation and save
- Bulk edit mode for multiple events
- Export selected events to JSON
- Undo/redo functionality
- Search/filter by date range
- Display rawData3D (3-axis accelerometer data)
- O2Sat graph overlay
- Zoom/pan synchronization between graphs

## Related Tools

- **manage_events.py**: Command-line event management tool
- **init_database.py**: Import JSON files to SQLite
- **database_utils.py**: Database backup and validation utilities

## License

See main project LICENSE file.

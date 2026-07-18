# OSDB Event Navigator

A GUI tool for navigating through events in the OpenSeizureDatabase (OSDB).

## Features

- Select and load OSDB database folders
- Navigate through events in the database
- Display event metadata
- Generate graphs for events (placeholder functionality)
- Event editing capabilities:
  - Edit Type, SubType, and Description fields
  - Save changes to memory
  - Save changes back to JSON file with automatic backup
- Exit confirmation when unsaved changes exist

## Requirements

- Python 3.x
- tkinter (usually included with Python)
- OpenSeizureDatabase files

## Usage

Run the tool with:

```bash
python main.py
```

1. Use the "Browse" button to select an OSDB database folder
2. Click "Load Database" to load the events
3. Use the "Previous" and "Next" buttons to navigate through events
4. View event metadata in the text area
5. Use the graph generation buttons to create visualizations (placeholder functionality)
6. Edit Type, SubType, and Description fields in the edit section
7. Click "Save Changes" to save to memory
8. Click "Save" to write changes back to the JSON file (with automatic backup)
9. A backup of the original file will be created automatically in a dedicated "backups" folder with a timestamp

## Database Structure

The tool expects OSDB database files to be in JSON format. It will automatically detect and load the first JSON file it finds in the selected folder.

## Event Metadata Display

The tool displays the following event metadata:
- Event ID
- Data Time
- User ID
- Type and SubType
- DataSource
- Phone App Version
- Watch SD Version
- Data availability flags (3D, HR, O2 Sat)
- Description
- Alarm State
- Additional data from dataJSON field

## Editing Workflow

1. Load a database file
2. Navigate to the event you want to edit
3. Modify the Type, SubType, or Description fields
4. Click "Save Changes" to save to memory
5. Click "Save" to write changes back to the JSON file
6. A backup of the original file will be created automatically with a timestamp

## Future Enhancements

- Full graph generation functionality for acceleration vector magnitude and heart rate
- Event filtering and search capabilities
- Export functionality for event data
- More detailed event visualization
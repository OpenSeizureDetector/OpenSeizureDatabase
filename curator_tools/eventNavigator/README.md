# OSDB Event Navigator

A GUI tool for navigating through events in the OpenSeizureDatabase (OSDB).

## Features

- Select and load OSDB database folders
- Navigate through events in the database
- Display event metadata
- Generate graphs for events (placeholder functionality)

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

## Future Enhancements

- Full graph generation functionality for acceleration vector magnitude and heart rate
- Event filtering and search capabilities
- Export functionality for event data
- More detailed event visualization
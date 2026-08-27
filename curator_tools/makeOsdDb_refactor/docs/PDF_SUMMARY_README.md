# PDF Summary Report Generator

Generate professional PDF summaries of OSDB events with acceleration and heart rate graphs.

## Features

- **Flexible Filtering**: Query events by date range, user ID, type, subType, and description text
- **A4 Page Layout**: Configurable events per page (1-20) with responsive column sizing
- **Embedded Graphs**: Each event row includes acceleration magnitude and heart rate graphs
- **Event Metadata**: Display event ID, type, subType, dateTime, userId, and description
- **Dual Interface**: Use via CLI or programmatically from Python
- **Integration**: Built into event_editor.py Tools menu for easy access

## Installation

The PDF generator requires `reportlab`:

```bash
pip install reportlab
```

Other dependencies (numpy, matplotlib, etc.) are already in the requirements.txt.

## CLI Usage

### Basic Usage

```bash
python3 generate_pdf_summary.py database.db output.pdf
```

### With Filters

```bash
# Only Seizures
python3 generate_pdf_summary.py database.db output.pdf --types Seizure

# Multiple types
python3 generate_pdf_summary.py database.db output.pdf --types Seizure Fall

# Specific subtype
python3 generate_pdf_summary.py database.db output.pdf --types Seizure --subtypes "Tonic-Clonic"

# Specific users
python3 generate_pdf_summary.py database.db output.pdf --user-ids 123 456

# Date range
python3 generate_pdf_summary.py database.db output.pdf \
  --start-date 2022-01-01 \
  --end-date 2022-12-31

# Description search (SQL LIKE pattern)
python3 generate_pdf_summary.py database.db output.pdf \
  --description "%seizure%"

# Combined filters
python3 generate_pdf_summary.py database.db output.pdf \
  --types Seizure \
  --user-ids 123 \
  --start-date 2022-06-01 \
  --end-date 2022-06-30 \
  --description "%tonic%"
```

### Customization

```bash
# Different events per page
python3 generate_pdf_summary.py database.db output.pdf --events-per-page 12

# Enable debug output
python3 generate_pdf_summary.py database.db output.pdf --debug
```

## Python API

```python
from generate_pdf_summary import EventsPdfGenerator

# Create generator
generator = EventsPdfGenerator(
    db_path='database.db',
    output_pdf='output.pdf',
    events_per_page=8,
    debug=False
)

# Generate PDF
success = generator.generate(
    event_types=['Seizure'],
    event_subtypes=['Tonic-Clonic'],
    user_ids=[123, 456],
    start_date='2022-01-01',
    end_date='2022-12-31',
    desc_filter='%seizure%'
)

if success:
    print("PDF generated successfully!")
else:
    print("No events matched the filters")
```

## GUI Integration (event_editor.py)

The PDF generator is integrated into event_editor.py under **Tools → Generate PDF Summary**.

### Workflow

1. Open database in event_editor
2. Apply filters using the filter controls
3. Go to **Tools → Generate PDF Summary**
4. Specify output PDF file path
5. Select events per page (1-20)
6. PDF is generated with the currently filtered events

### Filter Options in GUI

- **Event Type**: Multi-select (Seizure, Fall, False Alarm, Unknown, etc.)
- **Sub-Type**: Filtered based on selected types
- **User ID**: Multi-select from available user IDs
- **Start Date**: Date range start (ISO format)
- **End Date**: Date range end (ISO format)
- **Description**: Text search with SQL LIKE wildcards

## Filter Syntax

### Description/Text Filters

The description filter uses SQL LIKE syntax:

- `%seizure%` - Contains "seizure" (case-insensitive)
- `tonic%` - Starts with "tonic"
- `%clonic` - Ends with "clonic"
- `%t_nic%` - Contains "t", any single character, "nic"
- `%%` - Literal % character

Searches are case-insensitive.

## PDF Layout

### Page Format
- **Size**: A4 (210mm × 297mm)
- **Margins**: 10mm left/right, 18mm top, 10mm bottom
- **Usable Area**: 190mm × 269mm

### Per-Page Layout
- **Header**: Title, page number, total pages
- **Events**: User-configurable (1-20 per page)
- **Columns**: 
  - Column 1 (25%): Event metadata
  - Column 2 (37.5%): Acceleration magnitude graph
  - Column 3 (37.5%): Heart rate graph

### Row Height Calculation
```
Row height = Usable height / Events per page
```

For example:
- 8 events/page: ~90mm per row
- 12 events/page: ~60mm per row
- 16 events/page: ~45mm per row

## Output Format

The generated PDF includes:

**Per Event Row:**
- Event ID
- Event Type
- Event SubType
- DateTime
- User ID
- Description (truncated if >100 chars)
- Acceleration magnitude graph (time vs. acceleration)
- Heart rate graph (time vs. BPM)

**Per Page:**
- Header with title and page numbers
- Multiple event rows (configurable)

## Testing

### Create Test Database

```bash
python3 create_test_db.py
```

This creates `test_osdb.db` with 5 sample events.

### Generate Test PDF

```bash
# All events, 3 per page
python3 generate_pdf_summary.py test_osdb.db test_output.pdf --events-per-page 3

# Filtered: only seizures
python3 generate_pdf_summary.py test_osdb.db test_seizures.pdf --types Seizure

# With description filter
python3 generate_pdf_summary.py test_osdb.db test_filtered.pdf --description "%event%1%"
```

## Example Workflow

```bash
# 1. Generate summary of all tonic-clonic seizures from June 2022
python3 generate_pdf_summary.py osdb_working.db seizures_june_2022.pdf \
  --types Seizure \
  --subtypes "Tonic-Clonic" \
  --start-date 2022-06-01 \
  --end-date 2022-06-30 \
  --events-per-page 10

# 2. Generate summary for a specific user
python3 generate_pdf_summary.py osdb_working.db user_123_summary.pdf \
  --user-ids 123 \
  --events-per-page 12

# 3. Generate summary of suspicious patterns
python3 generate_pdf_summary.py osdb_working.db suspicious_events.pdf \
  --description "%warning%" \
  --events-per-page 6
```

## Performance Considerations

- **Large Datasets**: For >1000 events, consider filtering by date range or user
- **Graph Rendering**: Graphs are generated on-the-fly; PDF generation is typically fast
- **Memory**: All filtered events are loaded into memory; very large datasets (>5000 events) may require additional RAM
- **Optimization**: Use narrow date ranges and specific filters for faster PDF generation

## Troubleshooting

### Module Not Found Errors

```
ModuleNotFoundError: No module named 'reportlab'
```

**Solution**: Install reportlab in your environment:
```bash
pip install reportlab
```

### No Events Found

- Check filter criteria are correct
- Ensure date format is ISO (YYYY-MM-DD)
- Use `%` wildcards for text searches

### PDF File Not Created

- Check output directory exists and is writable
- Ensure output path doesn't contain invalid characters
- Check disk space

## Architecture

### Class: EventsPdfGenerator

```python
EventsPdfGenerator(db_path, output_pdf, events_per_page=8, debug=False)
```

**Methods:**
- `generate(event_types, event_subtypes, user_ids, start_date, end_date, desc_filter)` - Generate PDF
- `_create_pdf(events)` - Create PDF document
- `_create_page_header(page_num, total_pages)` - Create header table
- `_create_event_row(event)` - Create event row with graphs
- `_create_metadata_cell(event)` - Format event metadata
- `_generate_acceleration_plot(event)` - Render acceleration graph
- `_generate_hr_plot(event)` - Render heart rate graph

### Data Flow

1. CLI/API parses filter parameters
2. Database query returns filtered events
3. For each event:
   - Load full event details including datapoints
   - Render acceleration and HR graphs
   - Format metadata
4. Arrange events in rows
5. Group rows into pages with headers
6. Generate PDF document

## Dependencies

- **sqlite3**: Database access (built-in)
- **numpy**: Numerical arrays (for graph data)
- **matplotlib**: Graph rendering
- **reportlab**: PDF generation
- **PyQt5**: GUI integration (event_editor.py only)

## Notes

- All times are relative to event dataTime
- Heart rate is one value per datapoint
- Acceleration data contains 125 samples per datapoint at 25Hz
- Graphs are automatically scaled to fit available space
- Description text is truncated at 100 characters for display

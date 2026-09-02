#!/usr/bin/env python3
"""
PDF Summary Report Generator - OSDB Events Summary

Generates a PDF summary of events from the OSDB SQLite database with graphs.
- Queries database using filter criteria (date range, user ID, type, subType, description)
- Creates A4 pages with user-selectable events per page (default: 8)
- Each event displays as a row with 3 columns:
  1. Event metadata (ID, type, subType, dateTime, userId, description)
  2. Acceleration magnitude graph
  3. Heart rate graph
- Includes page header with title, page number, and total pages
- Can be called via CLI or imported as a module

Usage (CLI):
    python generate_pdf_summary.py database.db output.pdf \\
        --types "Seizure" --subtypes "Tonic-Clonic" \\
        --user-ids 123 456 \\
        --start-date 2022-01-01 --end-date 2022-12-31 \\
        --description "%seizure%" \\
        --events-per-page 8

Usage (Python):
    from generate_pdf_summary import EventsPdfGenerator
    
    generator = EventsPdfGenerator('database.db', 'output.pdf', events_per_page=8)
    generator.generate(
        event_types=['Seizure'],
        event_subtypes=['Tonic-Clonic'],
        user_ids=[123, 456],
        start_date='2022-01-01',
        end_date='2022-12-31',
        desc_filter='%seizure%'
    )
"""

import sys
import os
import argparse
import json
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import io

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, PageBreak, Paragraph, Image
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from osdb_sqlite import OsdWorkingDb


class EventsPdfGenerator:
    """Generate PDF summaries of OSDB events with graphs."""
    
    def __init__(self, db_path: str, output_pdf: str, events_per_page: int = 8, debug: bool = False):
        """
        Initialize PDF generator.
        
        Args:
            db_path: Path to SQLite database file
            output_pdf: Path to output PDF file
            events_per_page: Number of events per page (determines height per event)
            debug: Enable debug output
        """
        self.db_path = db_path
        self.output_pdf = output_pdf
        self.events_per_page = max(1, min(events_per_page, 20))  # Clamp between 1-20
        self.debug = debug
        self.db = None
        
        # PDF layout constants (A4)
        self.page_width, self.page_height = A4
        self.margin_top = 18 * mm
        self.margin_bottom = 10 * mm
        self.margin_left = 10 * mm
        self.margin_right = 10 * mm
        
        # Usable space per event
        self.usable_width = self.page_width - self.margin_left - self.margin_right
        self.usable_height = self.page_height - self.margin_top - self.margin_bottom
        self.event_row_height = self.usable_height / self.events_per_page
        
        # Column widths (event info, acceleration graph, heart rate graph)
        self.col1_width = self.usable_width * 0.25
        self.col2_width = self.usable_width * 0.375
        self.col3_width = self.usable_width * 0.375
        
        if self.debug:
            print(f"PDF Layout:")
            print(f"  Page size: {self.page_width/mm:.0f}mm x {self.page_height/mm:.0f}mm")
            print(f"  Usable area: {self.usable_width/mm:.0f}mm x {self.usable_height/mm:.0f}mm")
            print(f"  Events per page: {self.events_per_page}")
            print(f"  Event row height: {self.event_row_height/mm:.1f}mm")
            print(f"  Column widths: {self.col1_width/mm:.1f}mm, {self.col2_width/mm:.1f}mm, {self.col3_width/mm:.1f}mm")
    
    def generate(
        self,
        event_types: Optional[List[str]] = None,
        event_subtypes: Optional[List[str]] = None,
        user_ids: Optional[List[int]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        desc_filter: Optional[str] = None
    ) -> bool:
        """
        Generate PDF summary with filtered events.
        
        Args:
            event_types: List of event types to include (e.g., ['Seizure'])
            event_subtypes: List of event subtypes to include (e.g., ['Tonic-Clonic'])
            user_ids: List of user IDs to include
            start_date: Start date filter (ISO format YYYY-MM-DD)
            end_date: End date filter (ISO format YYYY-MM-DD)
            desc_filter: Description wildcard filter (SQL LIKE pattern)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Open database
            self.db = OsdWorkingDb(self.db_path, debug=self.debug)
            
            if self.debug:
                print(f"Querying database with filters:")
                print(f"  Types: {event_types}")
                print(f"  SubTypes: {event_subtypes}")
                print(f"  Users: {user_ids}")
                print(f"  Date range: {start_date} to {end_date}")
                print(f"  Description filter: {desc_filter}")
            
            # Query filtered events
            events = self.db.get_filtered_events(
                event_types=event_types,
                event_subtypes=event_subtypes,
                user_ids=user_ids,
                start_date=start_date,
                end_date=end_date,
                desc_filter=desc_filter
            )
            
            if not events:
                print("No events found matching the specified filters.")
                self.db.close()
                return False
            
            print(f"Found {len(events)} events matching filters.")
            
            # Create PDF
            self._create_pdf(events)
            
            print(f"PDF summary generated successfully: {self.output_pdf}")
            self.db.close()
            return True
            
        except Exception as e:
            print(f"Error generating PDF: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            if self.db:
                self.db.close()
            return False
    
    def _create_pdf(self, events: List[Dict[str, Any]]):
        """Create the PDF document with events."""
        # Create PDF canvas with custom header/footer
        doc = SimpleDocTemplate(
            self.output_pdf,
            pagesize=A4,
            topMargin=self.margin_top,
            bottomMargin=self.margin_bottom,
            leftMargin=self.margin_left,
            rightMargin=self.margin_right
        )
        
        # Build document content
        story = []
        
        # Calculate total pages
        total_pages = (len(events) + self.events_per_page - 1) // self.events_per_page
        
        # Create event rows grouped by page
        for page_num in range(total_pages):
            # Add page header
            header_table = self._create_page_header(page_num + 1, total_pages)
            story.append(header_table)
            story.append(Spacer(1, 3 * mm))
            
            # Get events for this page
            start_idx = page_num * self.events_per_page
            end_idx = min(start_idx + self.events_per_page, len(events))
            page_events = events[start_idx:end_idx]
            
            # Add event rows
            for event in page_events:
                event_table = self._create_event_row(event)
                story.append(event_table)
            
            # Add page break (except for last page)
            if page_num < total_pages - 1:
                story.append(PageBreak())
        
        # Build PDF
        doc.build(story, onFirstPage=self._add_header, onLaterPages=self._add_header)
    
    def _create_page_header(self, page_num: int, total_pages: int) -> Table:
        """Create page header table."""
        title_style = ParagraphStyle(
            'CustomTitle',
            fontSize=12,
            textColor=colors.HexColor('#1f4788'),
            spaceAfter=2,
            alignment=1,  # Center
            fontName='Helvetica-Bold'
        )
        
        # Create header cells
        header_data = [
            [
                Paragraph("Open Seizure Database Events Summary", title_style),
                Paragraph(f"Page {page_num} of {total_pages}", title_style)
            ]
        ]
        
        header_table = Table(
            header_data,
            colWidths=[self.usable_width * 0.7, self.usable_width * 0.3]
        )
        header_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('LINEBELOW', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        
        return header_table
    
    def _add_header(self, canvas_obj, doc):
        """Add header to each page."""
        # This is handled by _create_page_header in story
        pass
    
    def _create_event_row(self, event: Dict[str, Any]) -> Table:
        """Create a single event row with metadata and graphs."""
        event_id = event.get('id', '-')
        
        # Load full event details for datapoints
        full_event = self.db.get_event_details(event_id)
        
        # Column 1: Event metadata
        col1_content = self._create_metadata_cell(full_event)
        
        # Column 2: Acceleration magnitude graph
        col2_content = self._create_acceleration_graph(full_event)
        
        # Column 3: Heart rate graph
        col3_content = self._create_hr_graph(full_event)
        
        # Create table
        row_data = [[col1_content, col2_content, col3_content]]
        row_table = Table(
            row_data,
            colWidths=[self.col1_width, self.col2_width, self.col3_width],
            rowHeights=[self.event_row_height - 2 * mm]  # Leave some margin
        )
        
        row_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 3),
            ('RIGHTPADDING', (0, 0), (-1, -1), 3),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('BORDER', (0, 0), (-1, -1), 1, colors.grey),
            ('LINEABOVE', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ]))
        
        return row_table
    
    def _create_metadata_cell(self, event: Optional[Dict[str, Any]]) -> Paragraph:
        """Create event metadata cell content."""
        if not event:
            return Paragraph("Event not found", getSampleStyleSheet()['Normal'])
        
        style = ParagraphStyle(
            'EventMeta',
            fontSize=8,
            leading=10,
            fontName='Helvetica',
            textColor=colors.black,
            spaceAfter=2
        )
        
        # Format metadata
        event_id = event.get('id', '-')
        event_type = event.get('type', '-')
        subtype = event.get('subType', '-')
        datetime_str = event.get('dataTime', '-')
        user_id = event.get('userId', '-')
        desc = event.get('desc', '')
        
        # Truncate long descriptions
        if desc and len(desc) > 100:
            desc = desc[:97] + '...'
        
        metadata_text = (
            f"<b>ID:</b> {event_id}<br/>"
            f"<b>Type:</b> {event_type}<br/>"
            f"<b>SubType:</b> {subtype}<br/>"
            f"<b>Time:</b> {datetime_str}<br/>"
            f"<b>User:</b> {user_id}<br/>"
            f"<b>Desc:</b> {desc}"
        )
        
        return Paragraph(metadata_text, style)
    
    def _create_acceleration_graph(self, event: Optional[Dict[str, Any]]) -> Paragraph:
        """Create acceleration magnitude graph."""
        if not event:
            return Paragraph("No data", getSampleStyleSheet()['Normal'])
        
        datapoints = event.get('datapoints', [])
        if not datapoints:
            return Paragraph("No data", getSampleStyleSheet()['Normal'])
        
        try:
            # Generate graph
            img_data = self._generate_acceleration_plot(event)
            if img_data:
                # Create Image element
                img = Image(img_data, width=self.col2_width - 6 * mm, height=self.event_row_height - 6 * mm)
                
                # Create table with image
                img_table = Table(
                    [[img]],
                    colWidths=[self.col2_width - 6 * mm],
                    rowHeights=[self.event_row_height - 6 * mm]
                )
                img_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (0, 0), 'CENTER'),
                    ('VALIGN', (0, 0), (0, 0), 'MIDDLE'),
                ]))
                return img_table
        except Exception as e:
            if self.debug:
                print(f"Error generating acceleration graph: {e}")
        
        return Paragraph("Graph error", getSampleStyleSheet()['Normal'])
    
    def _create_hr_graph(self, event: Optional[Dict[str, Any]]) -> Paragraph:
        """Create heart rate graph."""
        if not event:
            return Paragraph("No data", getSampleStyleSheet()['Normal'])
        
        datapoints = event.get('datapoints', [])
        if not datapoints:
            return Paragraph("No data", getSampleStyleSheet()['Normal'])
        
        try:
            # Generate graph
            img_data = self._generate_hr_plot(event)
            if img_data:
                # Create Image element
                img = Image(img_data, width=self.col3_width - 6 * mm, height=self.event_row_height - 6 * mm)
                
                # Create table with image
                img_table = Table(
                    [[img]],
                    colWidths=[self.col3_width - 6 * mm],
                    rowHeights=[self.event_row_height - 6 * mm]
                )
                img_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (0, 0), 'CENTER'),
                    ('VALIGN', (0, 0), (0, 0), 'MIDDLE'),
                ]))
                return img_table
        except Exception as e:
            if self.debug:
                print(f"Error generating HR graph: {e}")
        
        return Paragraph("Graph error", getSampleStyleSheet()['Normal'])
    
    def _generate_acceleration_plot(self, event: Dict[str, Any]) -> Optional[io.BytesIO]:
        """Generate acceleration magnitude plot as PNG bytes."""
        datapoints = event.get('datapoints', [])
        if not datapoints:
            return None
        
        try:
            from datetime import datetime as dt
            
            # Extract data
            raw_data_list = []
            raw_time_list = []
            
            # Get event time reference
            try:
                event_datetime = dt.fromisoformat(event['dataTime'].replace('Z', '+00:00'))
            except (ValueError, KeyError, AttributeError):
                event_datetime = None
            
            for i, dp in enumerate(datapoints):
                raw_data = dp.get('rawData')
                if raw_data and isinstance(raw_data, list):
                    raw_data_list.extend(raw_data)
                    
                    # Calculate time
                    if event_datetime:
                        try:
                            dp_datetime = dt.fromisoformat(dp['dataTime'].replace('Z', '+00:00'))
                            time_sec = (dp_datetime - event_datetime).total_seconds()
                        except (ValueError, KeyError, AttributeError):
                            time_sec = i * 5
                    else:
                        time_sec = i * 5
                    
                    # Add time for each sample
                    for n in range(len(raw_data)):
                        raw_time_list.append(time_sec + n / 25.0)
            
            if not raw_data_list:
                return None
            
            # Create plot
            fig = Figure(figsize=(2.5, 1.8), dpi=100, tight_layout=True)
            ax = fig.add_subplot(111)
            
            ax.plot(raw_time_list, raw_data_list, 'b-', linewidth=0.5, alpha=0.7)
            ax.set_xlabel('Time (s)', fontsize=7)
            ax.set_ylabel('Acc (mG)', fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=6)
            
            # Convert to bytes
            canvas = FigureCanvas(fig)
            img_buffer = io.BytesIO()
            canvas.print_png(img_buffer)
            img_buffer.seek(0)
            plt.close(fig)
            
            return img_buffer
            
        except Exception as e:
            if self.debug:
                print(f"Error in acceleration plot: {e}")
            return None
    
    def _generate_hr_plot(self, event: Dict[str, Any]) -> Optional[io.BytesIO]:
        """Generate heart rate plot as PNG bytes."""
        datapoints = event.get('datapoints', [])
        if not datapoints:
            return None
        
        try:
            from datetime import datetime as dt
            
            # Extract HR data
            hr_list = []
            hr_time_list = []
            
            # Get event time reference
            try:
                event_datetime = dt.fromisoformat(event['dataTime'].replace('Z', '+00:00'))
            except (ValueError, KeyError, AttributeError):
                event_datetime = None
            
            for i, dp in enumerate(datapoints):
                hr = dp.get('hr', 0)
                if hr and hr > 0:
                    hr_list.append(hr)
                    
                    # Calculate time
                    if event_datetime:
                        try:
                            dp_datetime = dt.fromisoformat(dp['dataTime'].replace('Z', '+00:00'))
                            time_sec = (dp_datetime - event_datetime).total_seconds()
                        except (ValueError, KeyError, AttributeError):
                            time_sec = i * 5
                    else:
                        time_sec = i * 5
                    
                    hr_time_list.append(time_sec)
            
            if not hr_list or all(hr == 0 for hr in hr_list):
                return None
            
            # Create plot
            fig = Figure(figsize=(2.5, 1.8), dpi=100, tight_layout=True)
            ax = fig.add_subplot(111)
            
            ax.plot(hr_time_list, hr_list, 'r-', marker='o', linewidth=1, markersize=3)
            ax.set_xlabel('Time (s)', fontsize=7)
            ax.set_ylabel('HR (bpm)', fontsize=7)
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=6)
            
            # Convert to bytes
            canvas = FigureCanvas(fig)
            img_buffer = io.BytesIO()
            canvas.print_png(img_buffer)
            img_buffer.seek(0)
            plt.close(fig)
            
            return img_buffer
            
        except Exception as e:
            if self.debug:
                print(f"Error in HR plot: {e}")
            return None


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Generate PDF summary of OSDB events with graphs'
    )
    
    parser.add_argument('database', help='Path to SQLite database file')
    parser.add_argument('output', help='Path to output PDF file')
    parser.add_argument('--types', nargs='+', default=None,
                       help='Event types to include (e.g., Seizure Fall)')
    parser.add_argument('--subtypes', nargs='+', default=None,
                       help='Event subtypes to include (e.g., "Tonic-Clonic")')
    parser.add_argument('--user-ids', type=int, nargs='+', default=None,
                       help='User IDs to include')
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--description', type=str, default=None,
                       help='Description filter (SQL LIKE pattern, e.g., "%%seizure%%")')
    parser.add_argument('--events-per-page', type=int, default=8,
                       help='Number of events per page (1-20, default: 8)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.database):
        print(f"Error: Database file not found: {args.database}", file=sys.stderr)
        sys.exit(1)
    
    # Create generator and generate PDF
    generator = EventsPdfGenerator(
        args.database,
        args.output,
        events_per_page=args.events_per_page,
        debug=args.debug
    )
    
    success = generator.generate(
        event_types=args.types,
        event_subtypes=args.subtypes,
        user_ids=args.user_ids,
        start_date=args.start_date,
        end_date=args.end_date,
        desc_filter=args.description
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

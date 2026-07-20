# Installation Guide for OSDB Event Editor

## Dependencies

The Event Editor requires the following Python packages:
- **PyQt5** - GUI framework
- **matplotlib** - Graphing library
- **numpy** - Numerical computing

## Installation Options

### Option 1: pip install (Recommended)

```bash
cd event_editor
pip install -r requirements.txt
```

Or install manually:
```bash
pip install PyQt5 matplotlib numpy
```

### Option 2: System Package Manager (Linux)

#### Ubuntu/Debian:
```bash
sudo apt-get install python3-pyqt5 python3-matplotlib python3-numpy
```

#### Fedora:
```bash
sudo dnf install python3-qt5 python3-matplotlib python3-numpy
```

#### Arch Linux:
```bash
sudo pacman -S python-pyqt5 python-matplotlib python-numpy
```

### Option 3: Virtual Environment (Isolated)

```bash
# Create virtual environment
python3 -m venv venv_event_editor

# Activate it
source venv_event_editor/bin/activate  # Linux/Mac
# or
venv_event_editor\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run editor
python3 event_editor.py --db /path/to/database.db

# When done, deactivate
deactivate
```

## Verify Installation

```bash
python3 -c "import PyQt5, matplotlib, numpy; print('All dependencies installed!')"
```

## Launch the Editor

After installation:

```bash
# From event_editor directory
./launch_editor.sh --db /home/graham/osd/osdb/osdb_working.db

# Or directly
python3 event_editor.py --db /home/graham/osd/osdb/osdb_working.db
```

## Troubleshooting

### Import Error: No module named 'PyQt5'
- Run: `pip install PyQt5`
- Or use system package manager (see above)

### Qt platform plugin error
- Install: `sudo apt-get install libxcb-xinerama0`

### Matplotlib backend error
- Install: `sudo apt-get install python3-tk`

### Permission denied when running scripts
- Make executable: `chmod +x event_editor.py launch_editor.sh`

## System Requirements

- **Python**: 3.7 or later
- **Operating System**: Linux, macOS, or Windows
- **Display**: X11/Wayland display server (Linux) or equivalent
- **Memory**: 512 MB RAM minimum, 1 GB recommended
- **Disk Space**: ~100 MB for dependencies

## Notes

- The editor requires read/write access to the database file
- Large datasets (>1000 events) may load slowly on first filter application
- Graphs are rendered in-memory; very large events (>10,000 datapoints) may be slow

## Next Steps

Once installed, see [README.md](README.md) for usage instructions.

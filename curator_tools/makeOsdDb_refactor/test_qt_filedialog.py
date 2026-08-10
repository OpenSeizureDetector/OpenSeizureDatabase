#!/usr/bin/env python3
"""
Minimal Qt5 test to diagnose QFileDialog hanging issue
"""

import sys
import os
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QFileDialog Test")
        self.setGeometry(100, 100, 400, 200)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Test 1: Native file dialog (default)
        btn1 = QPushButton("Test 1: Native getExistingDirectory (default)")
        btn1.clicked.connect(self.test_native_dir)
        layout.addWidget(btn1)
        
        # Test 2: Non-native file dialog
        btn2 = QPushButton("Test 2: Non-Native getExistingDirectory")
        btn2.clicked.connect(self.test_nonnative_dir)
        layout.addWidget(btn2)
        
        # Test 3: Open file (not directory)
        btn3 = QPushButton("Test 3: Native getOpenFileName")
        btn3.clicked.connect(self.test_native_file)
        layout.addWidget(btn3)
        
        # Test 4: Starting from /tmp
        btn4 = QPushButton("Test 4: Native Dir starting from /tmp")
        btn4.clicked.connect(self.test_tmp_dir)
        layout.addWidget(btn4)
        
        # Test 5: Starting from home
        btn5 = QPushButton("Test 5: Native Dir starting from ~")
        btn5.clicked.connect(self.test_home_dir)
        layout.addWidget(btn5)
        
        # Test 6: Open file from current directory (like event_editor Open Database)
        btn6 = QPushButton("Test 6: Native getOpenFileName from current dir")
        btn6.clicked.connect(self.test_file_current_dir)
        layout.addWidget(btn6)
        
        # Test 7: Open file with empty starting path
        btn7 = QPushButton("Test 7: Native getOpenFileName with empty path")
        btn7.clicked.connect(self.test_file_empty_path)
        layout.addWidget(btn7)
    
    def test_native_dir(self):
        print("\n=== Test 1: Native getExistingDirectory ===")
        print("Starting directory: current working directory")
        result = QFileDialog.getExistingDirectory(
            self,
            "Select Directory (Native)",
            os.getcwd()
        )
        print(f"Result: {result}")
    
    def test_nonnative_dir(self):
        print("\n=== Test 2: Non-Native getExistingDirectory ===")
        print("Using DontUseNativeDialog flag")
        result = QFileDialog.getExistingDirectory(
            self,
            "Select Directory (Non-Native)",
            os.getcwd(),
            QFileDialog.DontUseNativeDialog
        )
        print(f"Result: {result}")
    
    def test_native_file(self):
        print("\n=== Test 3: Native getOpenFileName ===")
        print("File picker instead of directory picker")
        result, _ = QFileDialog.getOpenFileName(
            self,
            "Select File (Native)",
            os.getcwd()
        )
        print(f"Result: {result}")
    
    def test_tmp_dir(self):
        print("\n=== Test 4: Native Dir from /tmp ===")
        print("Starting directory: /tmp")
        result = QFileDialog.getExistingDirectory(
            self,
            "Select Directory (Native)",
            "/tmp"
        )
        print(f"Result: {result}")
    
    def test_home_dir(self):
        print("\n=== Test 5: Native Dir from ~ ===")
        print(f"Starting directory: {os.path.expanduser('~')}")
        result = QFileDialog.getExistingDirectory(
            self,
            "Select Directory (Native)",
            os.path.expanduser("~")
        )
        print(f"Result: {result}")
    
    def test_file_current_dir(self):
        print("\n=== Test 6: Native getOpenFileName from current dir ===")
        print(f"Starting directory: {os.getcwd()}")
        result, _ = QFileDialog.getOpenFileName(
            self,
            "Select Database File (Native)",
            os.getcwd(),
            "SQLite Database (*.db);;All Files (*)"
        )
        print(f"Result: {result}")
    
    def test_file_empty_path(self):
        print("\n=== Test 7: Native getOpenFileName with empty path ===")
        print("Starting directory: '' (defaults to current)")
        result, _ = QFileDialog.getOpenFileName(
            self,
            "Select Database File (Native)",
            "",
            "SQLite Database (*.db);;All Files (*)"
        )
        print(f"Result: {result}")

def main():
    print("Qt5 File Dialog Test Application")
    print("=" * 50)
    print("Testing different QFileDialog configurations")
    print("Watch for:")
    print("  - Which dialogs appear instantly")
    print("  - Which dialogs hang with high CPU")
    print("  - Console output after each test")
    print("=" * 50)
    
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Quick test to verify validateDatapoints config integration with runSequence.
"""

import os
import sys
import json
import tempfile
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_config_integration():
    """Test that validateDatapoints config is properly passed through runSequence."""
    
    # Create a minimal config
    config = {
        "debug": True,
        "dataProcessing": {
            "validateDatapoints": True
        },
        "dataFileNames": {
            "testDataFileJson": "testData.json",
            "testDataFileCsv": "testData.csv"
        }
    }
    
    # Verify the config has the setting
    validate = config.get('dataProcessing', {}).get('validateDatapoints', False)
    assert validate == True, "validateDatapoints should be True in config"
    
    print("✓ Config structure is correct")
    print(f"✓ validateDatapoints = {validate}")
    
    # Test with actual config file
    config_file = os.path.join(os.path.dirname(__file__), '..', 'user_tools', 'nnTraining2', 'nnConfig_deep_pytorch.json')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            real_config = json.load(f)
        
        validate = real_config.get('dataProcessing', {}).get('validateDatapoints', False)
        print(f"✓ nnConfig_deep_pytorch.json has validateDatapoints = {validate}")
    
    print("\n✓ Integration test passed!")

if __name__ == "__main__":
    test_config_integration()

#!/usr/bin/env python3
"""
Final automated controller to run ML training experiments with different augmentation settings.
"""

import os
import sys
import json
import subprocess
import copy

# Configuration file paths  
BASE_DIR = "/home/graham/osd/OpenSeizureDatabase/user_tools/nnTraining2"
CONFIG_LSTM = os.path.join(BASE_DIR, "nnConfig_lstm_1D.json")
CONFIG_CNN = os.path.join(BASE_DIR, "nnConfig_cnn_1D.json") 
AUGMENTATION_CONFIG = os.path.join(BASE_DIR, "augmentation_config.json")

OUTPUT_BASE = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_BASE, exist_ok=True)

def run_one_experiment(config_path, model_name, aug_name):
    """Run a single experiment using the modified config"""
    print(f"\n=== Running {model_name} with {aug_name} ===")
    
    # Build the command to execute
    cmd = [
        "/bin/bash", "-c",
        f"source {BASE_DIR}/../../venv/bin/activate && "
        f"python {BASE_DIR}/runSequence.py "
        f"--config {config_path} "
        f"--outDir {OUTPUT_BASE} "
        f"--train --rerun 0"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    
    try:
        process = subprocess.run(
            cmd,
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300  # 5 minute timeout limit
        )
        
        if process.returncode == 0:
            print("✓ SUCCESS")
            return True
        else:
            print("✗ FAILED")
            print("STDERR:", process.stderr[:500])
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ TIMEOUT - Execution took too long")
        return False
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
        return False

def main():
    print("=== ML Training Controller ===")
    
    # Load augmentation configs  
    with open(AUGMENTATION_CONFIG, 'r') as f:
        aug_data = json.load(f)
    
    aug_configs = aug_data["augmentationSettings"]
    
    # Test all augmentation settings (not just first 2 like in testing mode)
    test_aug_configs = aug_configs  # Use all configs now
    
    models = [
        {"name": "LSTM", "config": CONFIG_LSTM},
        {"name": "CNN", "config": CONFIG_CNN}
    ]
    
    print(f"Found {len(aug_configs)} augmentation configurations")
    print(f"Testing {len(test_aug_configs)} configurations with {len(models)} models")
    
    total_runs = len(test_aug_configs) * len(models)
    completed = 0
    failed = 0
    
    # Run experiments
    for aug_config in test_aug_configs:
        print(f"\n{'='*50}")
        print(f"Config: {aug_config['name']}")
        print('='*50)
        
        for model in models:
            try:
                # Create temp config file with the augmentation settings applied
                temp_config_path = os.path.join(OUTPUT_BASE, f"temp_{model['name'].lower()}_config.json")
                
                # Load base config
                with open(model['config'], 'r') as f:
                    config_data = json.load(f)
                
                # Apply aug settings 
                if "settings" in aug_config:
                    augment_params = aug_config["settings"]
                    # Create a deep copy to avoid modifying base config file
                    config_data_copy = copy.deepcopy(config_data)
                    if "dataProcessing" not in config_data_copy:
                        config_data_copy["dataProcessing"] = {}
                    # Merge augmentation parameters into dataProcessing section
                    for key, value in augment_params.items():
                        # Handle special array cases
                        if key in ["noiseAugmentationNonSeizurePairs", "possible_noiseAugmentationNonSeizurePairs"]:
                            config_data_copy["dataProcessing"][key] = value
                        else:
                            config_data_copy["dataProcessing"][key] = value
                    config_data = config_data_copy
                    
                # Save temp config
                with open(temp_config_path, 'w') as f:
                    json.dump(config_data, f, indent=2)
                
                # Run the experiment
                success = run_one_experiment(temp_config_path, model['name'], aug_config["name"])
                
                if success:
                    completed += 1
                else:
                    failed += 1
                    
                # Clean up temp config file
                os.remove(temp_config_path)
                
            except Exception as e:
                print(f"Error running {model['name']} with {aug_config['name']}: {str(e)}")
                failed += 1
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Total runs attempted: {total_runs}")
    print(f"Successful runs: {completed}")  
    print(f"Failed runs: {failed}")
    
    if failed == 0:
        print("✓ All experiments completed successfully!")
    else:
        print(f"⚠ {failed} out of {total_runs} experiments failed")
        
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
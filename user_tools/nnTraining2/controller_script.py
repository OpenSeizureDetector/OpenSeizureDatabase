#!/usr/bin/env python3
"""
Final automated controller to run ML training experiments with different augmentation settings.
"""

import os
import sys
import json
import subprocess
import copy
import shutil

# Configuration file paths  
BASE_DIR = "/home/graham/osd/OpenSeizureDatabase/user_tools/nnTraining2"
CONFIG_LSTM = os.path.join(BASE_DIR, "nnConfig_lstm_1D.json")
CONFIG_CNN = os.path.join(BASE_DIR, "nnConfig_cnn_1D.json") 
AUGMENTATION_CONFIG = os.path.join(BASE_DIR, "augmentation_config.json")

OUTPUT_BASE = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_BASE, exist_ok=True)

def get_next_folder_number(base_path, model_name):
    """Get next available folder number for a given model"""
    model_path = os.path.join(base_path, model_name)
    if not os.path.exists(model_path):
        return 1
    
    # Find the highest numbered folder
    folders = [f for f in os.listdir(model_path) if f.isdigit()]
    if not folders:
        return 1
    return max(int(f) for f in folders) + 1

def copy_precomputed_files(src_folder, dest_folder):
    """Copy pre-computed data files between runs"""
    # Files to copy (these are the ones that don't need regeneration)
    files_to_copy = ['allData.csv', 'trainData.csv', 'valData.csv', 'testData.csv']
    
    # Copy all the files if they exist in source
    for filename in files_to_copy:
        src_file = os.path.join(src_folder, filename)
        dest_file = os.path.join(dest_folder, filename)
        if os.path.exists(src_file):
            shutil.copy2(src_file, dest_file)
    
    # For trainDataAugmented.csv, copy it only if it exists 
    # This makes sense as a safety measure, though the real tracking would require state between executions
    #src_aug_file = os.path.join(src_folder, 'trainDataAugmented.csv')
    #dest_aug_file = os.path.join(dest_folder, 'trainDataAugmented.csv')
    #if os.path.exists(src_aug_file):
    #    shutil.copy2(src_aug_file, dest_aug_file)

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
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            timeout=3600  # 1 hour timeout limit
        )
        
        if process.returncode == 0:
            print("✓ SUCCESS")
            return True
        else:
            print("✗ FAILED")
            print("Output:", process.stdout[-500:] if len(process.stdout) > 500 else process.stdout)
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
    
    # Run experiments - will let runSequence.py manage the folder creation automatically
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

#!/usr/bin/env python3
"""
Automated controller to run ML training experiments with different augmentation settings.

Efficiency strategy:
  - For each experiment, determine which output files can be reused from the previous run
  - Pre-populate the new output folder with reusable files
  - Use runSequence --rerun <N> to run in the pre-populated folder
  - runSequence skips regeneration of existing files (allData.csv, trainData.csv, etc.)
"""

import os
import sys
import json
import subprocess
import copy
import shutil
import argparse
from datetime import datetime

# Configuration file paths  
BASE_DIR = "/home/graham/osd/OpenSeizureDatabase/user_tools/nnTraining2"
CONFIG_LSTM = os.path.join(BASE_DIR, "nnConfig_lstm_1D.json")
CONFIG_CNN = os.path.join(BASE_DIR, "nnConfig_cnn_1D.json") 
AUGMENTATION_CONFIG = os.path.join(BASE_DIR, "augmentation_config.json")

OUTPUT_BASE = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_BASE, exist_ok=True)

# Data files that depend ONLY on data selection config (never change between augmentation experiments)
DATA_SELECTION_FILES = ['allData.json', 'allData.csv', 'trainData.csv', 'testData.csv', 'valData.csv']

# Augmentation settings keys that affect trainDataAugmented.csv
AUGMENTATION_SETTING_KEYS = [
    'noiseAugmentation', 'noiseAugmentationFactor', 'noiseAugmentationValue',
    'phaseAugmentation', 'phaseAugmentationStep',
    'userAugmentation', 'userAugmentationThreshold',
    'oversample', 'undersample',
    'sampleRateAugmentation', 'sampleRateAugmentationFactors',
    'noiseAugmentationNonSeizure', 'noiseAugmentationNonSeizureFactor',
    'noiseAugmentationNonSeizureValue', 'noiseAugmentationNonSeizurePairs',
]

# Data selection keys that determine if CSV files can be reused across models
DATA_SELECTION_KEYS = [
    'startTime', 'endTime', 'channels',
    'dataSource', 'dataFormat',
    'excludeSeizureTypes', 'includeSeizureTypes',
    'testPatientIds', 'valPatientIds',
    'trainRatio', 'testRatio', 'valRatio',
    'dp2vectorMethod', 'dp2vectorParams',
]


def get_model_fname(config_path):
    """Extract modelFname from a config file (used as output folder prefix)."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['modelConfig']['modelFname']


def get_next_folder_number(base_path, model_fname):
    """Get next available folder number for a given modelFname prefix."""
    model_path = os.path.join(base_path, model_fname)
    if not os.path.exists(model_path):
        return 1
    
    folders = [f for f in os.listdir(model_path) if f.isdigit()]
    if not folders:
        return 1
    return max(int(f) for f in folders) + 1


def aug_settings_match(prev_config, curr_config):
    """Check if augmentation settings are identical between two configs.
    
    Returns True if trainDataAugmented.csv can be reused.
    """
    prev_proc = prev_config.get('dataProcessing', {})
    curr_proc = curr_config.get('dataProcessing', {})
    
    for key in AUGMENTATION_SETTING_KEYS:
        prev_val = prev_proc.get(key)
        curr_val = curr_proc.get(key)
        # Use JSON comparison to handle lists/dicts properly
        if json.dumps(prev_val, sort_keys=True) != json.dumps(curr_val, sort_keys=True):
            return False
    return True


def data_selection_match(prev_config, curr_config):
    """Check if data selection settings are identical between two configs.
    
    Returns True if data selection CSVs (allData, train, test, val) can be reused.
    """
    prev_proc = prev_config.get('dataProcessing', {})
    curr_proc = curr_config.get('dataProcessing', {})
    
    for key in DATA_SELECTION_KEYS:
        prev_val = prev_proc.get(key)
        curr_val = curr_proc.get(key)
        if json.dumps(prev_val, sort_keys=True) != json.dumps(curr_val, sort_keys=True):
            return False
    return True


def find_reusable_source(config_data, output_base):
    """Find the best source folder for data selection files from any completed experiment.
    
    Searches across ALL completed experiments (both models) for one with matching
    data selection parameters. Returns the source folder path or None.
    
    Args:
        config_data: The config dict for the current experiment
        output_base: Base output directory to search
    
    Returns:
        (src_folder, description) or (None, None)
    """
    # Scan all model subfolders in output_base
    if not os.path.exists(output_base):
        return None, None
    
    # Collect all experiment folders across all models, sorted by number (newest first)
    candidates = []
    for model_dir in sorted(os.listdir(output_base)):
        model_path = os.path.join(output_base, model_dir)
        if not os.path.isdir(model_path):
            continue
        for folder_name in sorted(os.listdir(model_path), key=lambda x: int(x) if x.isdigit() else 0, reverse=True):
            folder_path = os.path.join(model_path, folder_name)
            if not os.path.isdir(folder_path):
                continue
            # Check if this folder has data selection files
            has_data = any(os.path.exists(os.path.join(folder_path, f)) 
                          for f in ['allData.json', 'trainData.csv'])
            if has_data:
                candidates.append((folder_path, model_dir, folder_name))
    
    # Check each candidate for matching data selection settings
    for folder_path, model_dir, folder_name in candidates:
        # Try to load the config from the folder (saved as temp_<model>_config.json)
        saved_config = None
        for fname in os.listdir(folder_path):
            if fname.startswith('temp_') and fname.endswith('_config.json'):
                saved_config = os.path.join(folder_path, fname)
                break
        if saved_config is None:
            continue
        
        try:
            with open(saved_config, 'r') as f:
                prev_config = json.load(f)
            if data_selection_match(prev_config, config_data):
                return folder_path, f"{model_dir}/{folder_name}"
        except (json.JSONDecodeError, KeyError):
            continue
    
    return None, None


def copy_reusable_files(src_folder, dest_folder, copy_augmented=False):
    """Copy data files that can be reused from a previous experiment run.
    
    Args:
        src_folder: Source output folder from previous run
        dest_folder: Destination folder for new run
        copy_augmented: If True, also copy trainDataAugmented.csv
    
    Returns:
        dict: Mapping of filename -> source description for README
    """
    files_copied = {}
    
    # Ensure destination folder exists
    os.makedirs(dest_folder, exist_ok=True)
    
    for filename in DATA_SELECTION_FILES:
        src_file = os.path.join(src_folder, filename)
        dest_file = os.path.join(dest_folder, filename)
        if os.path.exists(src_file):
            shutil.copy2(src_file, dest_file)
            files_copied[filename] = os.path.basename(src_folder)
    
    if copy_augmented:
        aug_file = 'trainDataAugmented.csv'
        src_file = os.path.join(src_folder, aug_file)
        dest_file = os.path.join(dest_folder, aug_file)
        if os.path.exists(src_file):
            shutil.copy2(src_file, dest_file)
            files_copied[aug_file] = os.path.basename(src_folder)
    
    return files_copied


def run_one_experiment(config_path, model_name, aug_name, rerun=0):
    """Run a single experiment using the modified config.
    
    Args:
        config_path: Path to the config JSON file
        model_name: Name of the model (e.g. "LSTM", "CNN")
        aug_name: Name of the augmentation config
        rerun: Folder number to use (--rerun parameter). 0 = create new folder.
    """
    print(f"\n=== Running {model_name} with {aug_name} (rerun={rerun}) ===")
    
    cmd = [
        "/bin/bash", "-c",
        f"source {BASE_DIR}/../../venv/bin/activate && "
        f"python {BASE_DIR}/runSequence.py "
        f"--config {config_path} "
        f"--outDir {OUTPUT_BASE} "
        f"--train --rerun {rerun}"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    
    try:
        process = subprocess.run(
            cmd,
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=36000
        )
        
        if process.returncode == 0:
            print("SUCCESS")
            return True
        else:
            print("FAILED")
            print("Output:", process.stdout[-500:] if len(process.stdout) > 500 else process.stdout)
            return False
            
    except subprocess.TimeoutExpired:
        print("TIMEOUT - Execution took too long")
        return False
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False


def generate_readme(experiment_log, output_base):
    """Generate a README.md summarising all experiments and file reuse.
    
    Args:
        experiment_log: List of dicts with experiment details
        output_base: Base output directory
    """
    readme_path = os.path.join(output_base, "README.md")
    
    lines = []
    lines.append("# ML Training Experiments - Summary\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"Total experiments: {len(experiment_log)}\n")
    lines.append("")
    
    lines.append("## File Reuse Strategy\n")
    lines.append("Files that are **unchanged** between augmentation experiments are copied to new")
    lines.append("output folders to avoid re-running expensive data processing steps:\n")
    lines.append("| File | Depends on | Reused when |")
    lines.append("|------|-----------|-------------|")
    lines.append("| allData.json | Data selection config | Always (same base config) |")
    lines.append("| allData.csv | Data selection config | Always (same base config) |")
    lines.append("| trainData.csv | Split config | Always (same split settings) |")
    lines.append("| testData.csv | Split config | Always (same split settings) |")
    lines.append("| valData.csv | Split config | Always (same split settings) |")
    lines.append("| trainDataAugmented.csv | Augmentation settings | Only when augmentation settings match |")
    lines.append("")
    
    lines.append("## Experiment Log\n")
    
    for i, exp in enumerate(experiment_log, 1):
        lines.append(f"### Experiment {i}: {exp['model']} / {exp['aug_name']}")
        lines.append("")
        lines.append(f"- **Output folder:** `{exp['folder']}`")
        lines.append(f"- **Config:** `{exp['config']}`")
        lines.append(f"- **Rerun parameter:** {exp['rerun']}")
        lines.append(f"- **Status:** {'SUCCESS' if exp['success'] else 'FAILED'}")
        lines.append(f"- **Duration:** {exp.get('duration', 'N/A')}")
        
        if exp['files_copied']:
            lines.append(f"- **Files reused from previous run:**")
            for fname, src_folder in exp['files_copied'].items():
                lines.append(f"  - `{fname}` (from `{src_folder}`)")
        else:
            lines.append(f"- **Files reused:** None (first experiment for this model)")
        
        if exp.get('aug_diff'):
            lines.append(f"- **Augmentation changes from previous:**")
            for change in exp['aug_diff']:
                lines.append(f"  - {change}")
        
        lines.append("")
    
    lines.append("## Output Folder Structure\n")
    lines.append("```")
    lines.append("output/")
    
    # Group by model_fname (the actual folder name from modelConfig.modelFname)
    models_seen = {}
    for exp in experiment_log:
        mname = exp.get('model_fname', exp['model'])
        if mname not in models_seen:
            models_seen[mname] = []
        models_seen[mname].append(exp)
    
    for mname, exps in models_seen.items():
        lines.append(f"  {mname}/")
        for exp in exps:
            folder_name = os.path.basename(exp['folder'])
            lines.append(f"    {folder_name}/  # {exp['aug_name']}")
    lines.append("```")
    lines.append("")
    
    with open(readme_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"\nREADME.md written to {readme_path}")


def main(augmentation_config=None):
    """
    Main loop - reads configuration files and loops through experiments,
    saving results to output folders.
    
    Args:
        augmentation_config: Path to augmentation config JSON file.
            If None, uses the default AUGMENTATION_CONFIG.
    """
    
    # Resolve augmentation config path
    if augmentation_config is None:
        augmentation_config = AUGMENTATION_CONFIG
    
    print("=== ML Training Controller (Optimised) ===")
    print(f"Strategy: Pre-populate output folders with reusable files, use --rerun")
    print(f"Augmentation config: {augmentation_config}\n")
    
    with open(augmentation_config, 'r') as f:
        aug_data = json.load(f)
    
    aug_configs = aug_data["augmentationSettings"]
    
    models = [
        {"name": "LSTM", "config": CONFIG_LSTM},
        {"name": "CNN", "config": CONFIG_CNN}
    ]
    
    print(f"Found {len(aug_configs)} augmentation configurations")
    print(f"Testing with {len(models)} models")
    
    total_runs = len(aug_configs) * len(models)
    completed = 0
    failed = 0
    
    # Track state per model for file reuse
    # model_state[model_name] = {
    #   'folder': '/path/to/last/output/folder',
    #   'config': <last config dict>,
    #   'next_folder_num': int
    # }
    model_state = {}
    
    experiment_log = []
    
    for aug_config in aug_configs:
        print(f"\n{'='*60}")
        print(f"Augmentation config: {aug_config['name']}")
        print('='*60)
        
        for model in models:
            model_name = model['name']
            start_time = datetime.now()
            
            try:
                # Load base config and apply augmentation settings
                with open(model['config'], 'r') as f:
                    config_data = json.load(f)
                
                if "settings" in aug_config:
                    augment_params = aug_config["settings"]
                    config_data_copy = copy.deepcopy(config_data)
                    if "dataProcessing" not in config_data_copy:
                        config_data_copy["dataProcessing"] = {}
                    for key, value in augment_params.items():
                        config_data_copy["dataProcessing"][key] = value
                    config_data = config_data_copy
                
                # Save temp config
                temp_config_path = os.path.join(OUTPUT_BASE, f"temp_{model_name.lower()}_config.json")
                with open(temp_config_path, 'w') as f:
                    json.dump(config_data, f, indent=2)
                
                # Determine output folder and file reuse
                model_fname = get_model_fname(model['config'])
                next_num = get_next_folder_number(OUTPUT_BASE, model_fname)
                output_folder = os.path.join(OUTPUT_BASE, model_fname, str(next_num))
                
                files_copied = {}
                aug_diff = []
                rerun_param = next_num
                
                if model_name in model_state:
                    prev = model_state[model_name]
                    prev_folder = prev['folder']
                    prev_config = prev['config']
                    
                    # Check if augmentation settings changed
                    if aug_settings_match(prev_config, config_data):
                        # Augmentation unchanged: reuse all data files + augmented
                        print(f"  Augmentation settings UNCHANGED from previous run")
                        print(f"  Copying all data files from {os.path.basename(prev_folder)}")
                        files_copied = copy_reusable_files(
                            prev_folder, output_folder, copy_augmented=True
                        )
                    else:
                        # Augmentation changed: reuse only data selection files
                        print(f"  Augmentation settings CHANGED from previous run")
                        print(f"  Copying data selection files from {os.path.basename(prev_folder)}")
                        files_copied = copy_reusable_files(
                            prev_folder, output_folder, copy_augmented=False
                        )
                        
                        # Record what changed
                        prev_proc = prev_config.get('dataProcessing', {})
                        curr_proc = config_data.get('dataProcessing', {})
                        for key in AUGMENTATION_SETTING_KEYS:
                            prev_val = prev_proc.get(key)
                            curr_val = curr_proc.get(key)
                            if json.dumps(prev_val, sort_keys=True) != json.dumps(curr_val, sort_keys=True):
                                aug_diff.append(f"{key}: {prev_val} -> {curr_val}")
                    
                    print(f"  Will run with --rerun {rerun_param} in pre-populated folder")
                else:
                    # First experiment for this model - check cross-model reuse
                    src_folder, src_desc = find_reusable_source(config_data, OUTPUT_BASE)
                    if src_folder:
                        print(f"  First experiment for {model_name} - reusing data files from {src_desc} (other model)")
                        files_copied = copy_reusable_files(src_folder, output_folder, copy_augmented=False)
                    else:
                        print(f"  First experiment for {model_name} - no files to reuse")
                        os.makedirs(output_folder, exist_ok=True)
                        print(f"  Created output folder: {output_folder}")
                
                # Run the experiment
                success = run_one_experiment(temp_config_path, model_name, aug_config["name"], rerun=rerun_param)
                
                duration = datetime.now() - start_time
                
                # Update model state for next experiment
                actual_output_folder = os.path.join(OUTPUT_BASE, model_fname, str(next_num))
                model_state[model_name] = {
                    'folder': actual_output_folder,
                    'config': copy.deepcopy(config_data),
                    'next_folder_num': next_num + 1
                }
                
                # Record experiment
                experiment_log.append({
                    'model': model_name,
                    'model_fname': model_fname,
                    'aug_name': aug_config["name"],
                    'folder': actual_output_folder,
                    'config': temp_config_path,
                    'config_data': copy.deepcopy(config_data),
                    'rerun': rerun_param,
                    'success': success,
                    'files_copied': files_copied,
                    'aug_diff': aug_diff,
                    'duration': str(duration).split('.')[0]
                })
                
                if success:
                    completed += 1
                else:
                    failed += 1
                    
                os.remove(temp_config_path)
                
            except Exception as e:
                print(f"Error running {model_name} with {aug_config['name']}: {str(e)}")
                import traceback
                traceback.print_exc()
                failed += 1
                
                experiment_log.append({
                    'model': model_name,
                    'model_fname': model_fname,
                    'aug_name': aug_config["name"],
                    'folder': output_folder if 'output_folder' in dir() else 'N/A',
                    'config': temp_config_path if 'temp_config_path' in dir() else 'N/A',
                    'rerun': 0,
                    'success': False,
                    'files_copied': {},
                    'aug_diff': [],
                    'duration': str(datetime.now() - start_time).split('.')[0]
                })
    
    # Generate README
    generate_readme(experiment_log, OUTPUT_BASE)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total runs attempted: {total_runs}")
    print(f"Successful runs: {completed}")  
    print(f"Failed runs: {failed}")
    
    if failed == 0:
        print("All experiments completed successfully!")
    else:
        print(f"{failed} out of {total_runs} experiments failed")
        
    return failed == 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run ML training experiments with different augmentation settings.'
    )
    parser.add_argument(
        '--augConfig', default=None,
        help='Path to augmentation config JSON file (default: augmentation_config.json in script directory)'
    )
    args = parser.parse_args()
    success = main(augmentation_config=args.augConfig)
    sys.exit(0 if success else 1)

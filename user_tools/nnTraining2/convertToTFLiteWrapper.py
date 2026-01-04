#!/usr/bin/env python3

"""
nnTrainer TensorFlow Lite conversion wrapper.

Integrates model-to-TFLite conversion into the nnTrainer toolchain.
Converts trained models (from nnTrainer.py) to Android-compatible format.

Usage:
    python convertToTFLiteWrapper.py --config nnConfig.json --quantize dynamic
    python convertToTFLiteWrapper.py --config nnConfig.json --model model.keras --quantize float16
"""

import argparse
import os
import sys
import json
import subprocess

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libosd.configUtils


def get_model_path_from_config(configObj, dataDir='.'):
    """
    Extract model filename from config and construct full path.
    
    Args:
        configObj: Configuration dictionary
        dataDir: Base data directory
    
    Returns:
        tuple: (model_path, framework) or (None, None) if not found
    """
    try:
        model_root = libosd.configUtils.getConfigParam("modelFname", configObj['modelConfig'])
        if not model_root:
            return None, None
        
        # Detect framework
        framework = configObj['modelConfig'].get('framework', 'tensorflow').lower()
        if framework in ['pytorch', 'torch']:
            ext = '.pt'
            framework = 'pytorch'
        else:
            ext = '.keras'
            framework = 'tensorflow'
        
        model_path = os.path.join(dataDir, f"{model_root}{ext}")
        return model_path, framework
    
    except (KeyError, TypeError):
        return None, None


def convert_model(model_path, output_path, framework='tensorflow', quantize=None, verbose=False):
    """
    Call convertToTFLite.py to perform the conversion.
    
    Args:
        model_path: Path to input model
        output_path: Path to save TFLite model
        framework: Model framework
        quantize: Quantization type
        verbose: Enable verbose output
    
    Returns:
        bool: True if successful, False otherwise
    """
    script_dir = os.path.dirname(__file__)
    convert_script = os.path.join(script_dir, 'convertToTFLite.py')
    
    if not os.path.exists(convert_script):
        print(f"ERROR: convertToTFLite.py not found at {convert_script}")
        return False
    
    cmd = [
        sys.executable,
        convert_script,
        '--model', model_path,
        '--output', output_path,
        '--framework', framework
    ]
    
    if quantize:
        cmd.extend(['--quantize', quantize])
    
    if verbose:
        cmd.append('--verbose')
    
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Conversion failed with return code {e.returncode}")
        return False
    except Exception as e:
        print(f"ERROR: Failed to run conversion: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert nnTrainer models to TensorFlow Lite for Android',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Convert model from config file with dynamic quantization:
    %(prog)s --config nnConfig.json --quantize dynamic
  
  Convert specific model with float16 quantization:
    %(prog)s --config nnConfig.json --model mymodel.keras --quantize float16
  
  Convert without quantization:
    %(prog)s --config nnConfig.json
        """
    )
    
    parser.add_argument('--config', default='nnConfig.json',
                        help='Path to nnTrainer configuration file (default: nnConfig.json)')
    parser.add_argument('--model', default=None,
                        help='Path to specific model file (optional; uses config if not provided)')
    parser.add_argument('--output', default=None,
                        help='Path to save TFLite model (default: model_name.tflite)')
    parser.add_argument('--quantize', choices=['dynamic', 'float16', 'integer'], default=None,
                        help='Quantization type for optimization')
    parser.add_argument('--datadir', default='.',
                        help='Base data directory (default: current directory)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"ERROR: Configuration file not found: {args.config}")
        sys.exit(1)
    
    try:
        configObj = libosd.configUtils.loadConfig(args.config)
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}")
        sys.exit(1)
    
    # Determine model path
    if args.model:
        model_path = args.model
        framework = 'pytorch' if model_path.endswith('.pt') else 'tensorflow'
    else:
        model_path, framework = get_model_path_from_config(configObj, args.datadir)
        if not model_path:
            print("ERROR: Could not determine model path from configuration")
            sys.exit(1)
    
    # Validate model exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(model_path)[0]
        output_path = f"{base_name}.tflite"
    
    print(f"Model conversion configuration:")
    print(f"  Input model: {model_path}")
    print(f"  Framework: {framework}")
    print(f"  Output file: {output_path}")
    if args.quantize:
        print(f"  Quantization: {args.quantize}")
    print()
    
    # Perform conversion
    success = convert_model(
        model_path,
        output_path,
        framework=framework,
        quantize=args.quantize,
        verbose=args.verbose
    )
    
    if success:
        print(f"\nModel is ready for Android deployment!")
        print(f"Transfer {output_path} to your Android project assets folder")
        sys.exit(0)
    else:
        print("\nERROR: Model conversion failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

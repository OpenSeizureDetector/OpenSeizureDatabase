#!/usr/bin/env python
"""
convertPt2Pte.py - Convert PyTorch models to ExecuTorch .pte format

This script converts PyTorch models (both .pt and .ptl formats) directly to ExecuTorch .pte format.
Supports:
  - .pt files: Regular PyTorch checkpoints or saved models
  - .ptl files: TorchScript modules (automatically reconstructed as regular PyTorch models)
  - XNNPACK delegation with CPU feature control (dotprod, fp16)

Usage:
    python convertPt2Pte.py input_model.pt -o output_model.pte
    python convertPt2Pte.py model.pt --xnnpack --no-dotprod
    
Prerequisites:
    pip install torch executorch
"""

import argparse
import sys
import os
import json
import importlib

try:
    import torch
    from executorch.exir import to_edge
    from torch.export import export

    # Optional XNNPACK support
    try:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        from executorch.backends.xnnpack.api import XnnpackBackendConfig
        XNNPACK_AVAILABLE = True
    except ImportError:
        XNNPACK_AVAILABLE = False

except ImportError as e:
    print(f"Error: Required libraries not found.", file=sys.stderr)
    print(f"Install with: pip install torch executorch", file=sys.stderr)
    print(f"Details: {e}", file=sys.stderr)
    sys.exit(1)

# Import the model architecture
try:
    from deepEpiCnnModel_torch import DeepEpiCnn
except ImportError:
    # Try alternative import path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from deepEpiCnnModel_torch import DeepEpiCnn
    except ImportError:
        print("Warning: Could not import DeepEpiCnn model architecture.", file=sys.stderr)
        print("Make sure deepEpiCnnModel_torch.py is in the same directory.", file=sys.stderr)
        DeepEpiCnn = None


def load_model_instance_from_checkpoint(checkpoint, config, input_length, num_classes, 
                                        conv_dropout, dense_dropout, input_shape, verbose=True):
    """
    Dynamically load and instantiate the model class from checkpoint configuration.
    
    Phase 2: Model class detection - tries to load the correct model class (DeepEpiCnn, CnnLstm, etc.)
    Falls back to DeepEpiCnn if model class not specified or loading fails.
    
    The checkpoint may contain a wrapper class (e.g., CnnLstmModelPyTorch) or the actual model class
    (e.g., DeepEpiCnn). This function detects which one and returns the underlying PyTorch model.
    
    Args:
        checkpoint: The full checkpoint dict
        config: Configuration dict from checkpoint
        input_length: Input sequence length
        num_classes: Number of output classes
        conv_dropout: Convolution layer dropout rate
        dense_dropout: Dense layer dropout rate
        input_shape: Input tensor shape (batch, channels, length)
        verbose: Print debug messages
    
    Returns:
        Instantiated PyTorch model object (with load_state_dict method)
    """
    model = None
    model_class_path = config.get('modelConfig', {}).get('modelClass', None)
    
    # Try to load the model class from checkpoint config (Phase 2 - Model class detection)
    if model_class_path:
        if verbose:
            print(f"Attempting to load model class: {model_class_path}")
        try:
            # Split module path and class name
            parts = model_class_path.rsplit('.', 1)
            if len(parts) == 2:
                module_name, class_name = parts
                if verbose:
                    print(f"  Module: {module_name}, Class: {class_name}")
                
                # Dynamically import the module
                try:
                    module = importlib.import_module(module_name)
                    TargetClass = getattr(module, class_name)
                    
                    # Instantiate the class
                    try:
                        instance = TargetClass(config['modelConfig'])
                        if verbose:
                            print(f"✓ Successfully instantiated class: {class_name}")
                    except TypeError:
                        # If config-based instantiation fails, try with individual parameters
                        if verbose:
                            print(f"  Instantiation with config failed, trying parameter-based init...")
                        instance = TargetClass(input_length=input_length, num_classes=num_classes,
                                             conv_dropout=conv_dropout, dense_dropout=dense_dropout)
                        if verbose:
                            print(f"✓ Successfully instantiated with parameters: {class_name}")
                    
                    # Check if this is a wrapper class (has .model attribute) or the actual model class
                    if hasattr(instance, 'model') and hasattr(instance, 'makeModel'):
                        # This is a wrapper class - call makeModel() to create the underlying model
                        if verbose:
                            print(f"  Detected wrapper class, calling makeModel()...")
                        model = instance.makeModel(input_shape=input_shape, num_classes=num_classes)
                        if verbose:
                            print(f"✓ Successfully created underlying model from wrapper")
                    elif hasattr(instance, 'load_state_dict'):
                        # This is already the actual PyTorch model
                        model = instance
                        if verbose:
                            print(f"✓ Using model class directly: {class_name}")
                    else:
                        if verbose:
                            print(f"  Instantiated class is neither a wrapper nor a PyTorch model")
                        model = None
                
                except AttributeError as ae:
                    if verbose:
                        print(f"  Could not find class {class_name} in module: {ae}")
                    model = None
                    
            else:
                if verbose:
                    print(f"  Invalid model class path format (expected 'module.ClassName')")
                model = None
                
        except Exception as e:
            if verbose:
                print(f"  Error loading model class {model_class_path}: {e}")
            model = None
    
    # Fallback to DeepEpiCnn if model class not specified or loading failed
    if model is None:
        if verbose:
            if model_class_path:
                print(f"✗ Using fallback DeepEpiCnn model (could not load {model_class_path})")
            else:
                print(f"Using default DeepEpiCnn model")
        model = DeepEpiCnn(input_length=input_length, num_classes=num_classes,
                         conv_dropout=conv_dropout, dense_dropout=dense_dropout)
    
    return model


def convert_pt_to_pte(input_path, output_path, input_shape=(1, 1, 750), num_classes=2,
                       use_xnnpack=False, use_dotprod=True, use_fp16=True, verbose=True):
    """
    Convert a PyTorch model to ExecuTorch .pte format.
    """
    try:
        if verbose:
            file_ext = os.path.splitext(input_path)[1].lower()
            print(f"Loading PyTorch model from {input_path} ({file_ext})...")
        
        # Load the model checkpoint
        checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
        
        # Reconstruct model logic (same as before)
        if isinstance(checkpoint, torch.jit.ScriptModule):
            state_dict = checkpoint.state_dict()
            input_length = input_shape[2] if len(input_shape) >= 3 else 750
            # Default dropout for TorchScript (no metadata available)
            model = DeepEpiCnn(input_length=input_length, num_classes=num_classes,
                             conv_dropout=0.0, dense_dropout=0.025)
            model.load_state_dict(state_dict)
        elif isinstance(checkpoint, dict):
            # Check if this is a checkpoint dict or just a state_dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                # Assume it's a state_dict directly
                state_dict = checkpoint
            
            # Extract configuration from checkpoint if available
            config = checkpoint.get('config', {}) if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else {}
            if config:
                if verbose:
                    print("Using configuration from checkpoint...")
                # Get input_length from config
                input_length = config.get('dataProcessing', {}).get('rawDataLength', 750)
                # Get num_classes from config
                num_classes = config.get('modelConfig', {}).get('numClasses', 2)
                # Update input_shape to match the extracted config
                input_shape = (1, 1, input_length)
                if verbose:
                    print(f"  input_length={input_length}, num_classes={num_classes}")
            else:
                # Fallback to provided parameters
                input_length = input_shape[2] if len(input_shape) >= 3 else 750
            
            # Extract dropout parameters from checkpoint
            # Priority: explicit checkpoint fields > config object > defaults
            conv_dropout = None
            dense_dropout = None
            
            # Try to get from explicit checkpoint fields (saved by nnTrainer.py)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                conv_dropout = checkpoint.get('conv_dropout')
                dense_dropout = checkpoint.get('dense_dropout')
            
            # If not available, try to read from config object
            if conv_dropout is None or dense_dropout is None:
                if config:
                    conv_dropout = config.get('convDropout', conv_dropout)
                    dense_dropout = config.get('denseDropout', dense_dropout)
            
            # Use defaults only if still not found
            if conv_dropout is None:
                conv_dropout = 0.0
            if dense_dropout is None:
                dense_dropout = 0.025
            
            if verbose and (conv_dropout != 0.0 or dense_dropout != 0.025):
                print(f"Using dropout parameters: conv_dropout={conv_dropout}, dense_dropout={dense_dropout}")
            
            # Phase 2: Use dynamic model loading instead of hardcoded DeepEpiCnn
            model = load_model_instance_from_checkpoint(
                checkpoint, config, input_length, num_classes,
                conv_dropout, dense_dropout, input_shape, verbose=verbose
            )
            model.load_state_dict(state_dict)
        else:
            model = checkpoint
        
        model.eval()
        example_inputs = (torch.randn(input_shape),)
        
        if verbose:
            print("Exporting model to ExecuTorch format...")
        
        # 1. Export to ATen dialect
        exported_program = export(model, example_inputs)

        # 2. Convert to Edge dialect
        edge_program = to_edge(exported_program)

        # 3. Optional XNNPACK Partitioning
        min_cpu_features = []
        if use_xnnpack:
            if not XNNPACK_AVAILABLE:
                print("Error: XNNPACK backend not available in this ExecuTorch installation.", file=sys.stderr)
                # Phase 1: Return just boolean
                return False
            
            if verbose:
                print(f"Partitioning for XNNPACK (dotprod={use_dotprod}, fp16={use_fp16})...")
            
            backend_config = XnnpackBackendConfig(
                use_dotprod=use_dotprod,
                use_fp16=use_fp16
            )
            
            if use_dotprod: min_cpu_features.append("dotprod")
            if use_fp16: min_cpu_features.append("fp16")
            
            partitioner = XnnpackPartitioner(backend_config)
            edge_program = edge_program.to_backend(partitioner)
        else:
            if verbose: print("Using portable kernels (no XNNPACK delegation).")

        # 4. Compile to ExecuTorch
        executorch_program = edge_program.to_executorch()
        
        # Save the program
        with open(output_path, 'wb') as f:
            executorch_program.write_to_file(f)
        
        if verbose:
            print(f"✓ Successfully converted to {output_path}")
            print(f"Suggested min_cpu_features for index.json: {json.dumps(min_cpu_features)}")
        
        # Phase 1: Return just boolean instead of tuple
        return True
        
    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        # Phase 1: Return just boolean instead of tuple
        return False


def parse_shape(shape_str):
    try:
        parts = [int(x.strip()) for x in shape_str.split(',')]
        return tuple(parts)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid shape format: {e}")


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch models to ExecuTorch .pte format')
    parser.add_argument('input', help='Input model file (.pt or .ptl)')
    parser.add_argument('-o', '--output', help='Output ExecuTorch file (.pte)')
    parser.add_argument('--input-shape', type=parse_shape, default=(1, 1, 750), help='Shape (default: 1,1,750)')
    parser.add_argument('--xnnpack', action='store_true', help='Use XNNPACK delegation for Android')
    parser.add_argument('--no-dotprod', action='store_false', dest='dotprod', help='Disable ARMv8.2 dotprod instructions')
    parser.add_argument('--no-fp16', action='store_false', dest='fp16', help='Disable FP16 instructions')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress progress messages')
    
    parser.set_defaults(dotprod=True, fp16=True)
    args = parser.parse_args()
    
    if args.output is None:
        args.output = os.path.splitext(args.input)[0] + '.pte'
    
    # Phase 1: Fixed - expects boolean return value
    success = convert_pt_to_pte(
        input_path=args.input,
        output_path=args.output,
        input_shape=args.input_shape,
        use_xnnpack=args.xnnpack,
        use_dotprod=args.dotprod,
        use_fp16=args.fp16,
        verbose=not args.quiet
    )
    
    if success:
        # Create a small json file with requirements next to the model
        meta_path = args.output + ".json"
        with open(meta_path, 'w') as f:
            json.dump({"min_cpu_features": []}, f)
        if not args.quiet:
            print(f"Metadata saved to {meta_path}")

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

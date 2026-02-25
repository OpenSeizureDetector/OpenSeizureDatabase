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
            model = DeepEpiCnn(input_length=input_length, num_classes=num_classes)
            model.load_state_dict(state_dict)
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            input_length = input_shape[2] if len(input_shape) >= 3 else 750
            model = DeepEpiCnn(input_length=input_length, num_classes=num_classes)
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
        
        return True, min_cpu_features
        
    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False, []


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
    
    success, features = convert_pt_to_pte(
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
            json.dump({"min_cpu_features": features}, f)
        if not args.quiet:
            print(f"Metadata saved to {meta_path}")

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

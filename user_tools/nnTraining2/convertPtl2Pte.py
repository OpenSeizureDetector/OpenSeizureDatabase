#!/usr/bin/env python
"""
convertPtl2Pte.py - Convert PyTorch Lite .ptl model to ExecuTorch .pte format

Simple CLI tool to convert PyTorch Lite models to ExecuTorch format for mobile deployment.
ExecuTorch is the successor to PyTorch Mobile and provides better performance and compatibility.

Usage:
    python convertPtl2Pte.py input_model.ptl -o output_model.pte
    python convertPtl2Pte.py model.ptl
    
Prerequisites:
    pip install executorch
"""

import argparse
import sys
import os

try:
    import torch
    from executorch.exir import to_edge
    from torch.export import export
except ImportError as e:
    print(f"Error: Required libraries not found.", file=sys.stderr)
    print(f"Install with: pip install torch executorch", file=sys.stderr)
    print(f"Details: {e}", file=sys.stderr)
    sys.exit(1)


def convert_ptl_to_pte(input_path, output_path, verbose=True):
    """
    Convert a PyTorch Lite .ptl model to ExecuTorch .pte format.
    
    Args:
        input_path: Path to input .ptl model file
        output_path: Path to output .pte file
        verbose: Print progress messages
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if verbose:
            print(f"Loading PyTorch Lite model from {input_path}...")
        
        # Load the mobile-optimized model
        model = torch.jit.load(input_path, map_location='cpu')
        model.eval()
        
        if verbose:
            print("Model loaded successfully")
            print("Analyzing model structure to determine input shape...")
        
        # Try to infer input shape from the model
        # Most seizure detection models use (batch, channels, length) format
        # Default to (1, 1, 750) which is the standard for this project
        input_shape = (1, 1, 750)
        
        # Try to get input shape from model if available
        try:
            # Attempt to extract from model's graph
            for node in model.graph.inputs():
                if node.debugName() != 'self':
                    # Found input node
                    node_type = node.type()
                    if hasattr(node_type, 'sizes'):
                        sizes = node_type.sizes()
                        if sizes and len(sizes) >= 3:
                            # Use the detected shape
                            input_shape = tuple(sizes)
                            if verbose:
                                print(f"  Detected input shape from model: {input_shape}")
                            break
        except Exception as e:
            if verbose:
                print(f"  Could not auto-detect shape, using default: {input_shape}")
        
        if verbose:
            print(f"Creating example input with shape {input_shape}...")
        
        # Create example input for export
        example_inputs = (torch.randn(input_shape),)
        
        if verbose:
            print("Exporting model to ExecuTorch format...")
        
        # Export the model using torch.export
        # Note: JIT models need to be converted back to eager mode first
        # For JIT traced models, we'll use the model directly with capture_pre_autograd_graph
        try:
            # Method 1: Try to export directly using torch.export
            # This works for models that can be captured by dynamo
            exported_program = export(model, example_inputs)
            
            if verbose:
                print("Converting to Edge dialect...")
            
            # Convert to Edge IR (ExecuTorch's intermediate representation)
            edge_program = to_edge(exported_program)
            
            if verbose:
                print("Compiling to ExecuTorch...")
            
            # Convert to ExecuTorch program
            executorch_program = edge_program.to_executorch()
            
        except Exception as e:
            if verbose:
                print(f"  Standard export failed: {e}")
                print("  Attempting alternative conversion method...")
            
            # Method 2: For TorchScript/JIT models, use a wrapper approach
            # Create a wrapper module that converts the JIT model to eager execution
            class JitToEagerWrapper(torch.nn.Module):
                def __init__(self, jit_model):
                    super().__init__()
                    self.jit_model = jit_model
                
                def forward(self, x):
                    return self.jit_model(x)
            
            wrapper = JitToEagerWrapper(model)
            wrapper.eval()
            
            # Try exporting the wrapper
            exported_program = export(wrapper, example_inputs)
            edge_program = to_edge(exported_program)
            executorch_program = edge_program.to_executorch()
        
        if verbose:
            print(f"Saving ExecuTorch program to {output_path}...")
        
        # Save the ExecuTorch program
        with open(output_path, 'wb') as f:
            executorch_program.write_to_file(f)
        
        if verbose:
            print(f"✓ Successfully converted to {output_path}")
            
            # Print file sizes for comparison
            input_size = os.path.getsize(input_path) / (1024 * 1024)
            output_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  Input size:  {input_size:.2f} MB")
            print(f"  Output size: {output_size:.2f} MB")
        
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found.", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        
        if "torch.export" in str(e) or "dynamo" in str(e).lower():
            print("\nNote: Some PyTorch models may not be compatible with ExecuTorch export.", file=sys.stderr)
            print("This is typically due to dynamic control flow or unsupported operations.", file=sys.stderr)
            print("Consider retraining the model with ExecuTorch compatibility in mind.", file=sys.stderr)
        
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert PyTorch Lite .ptl model to ExecuTorch .pte format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python convertPtl2Pte.py model.ptl
  
  # Specify output file
  python convertPtl2Pte.py model.ptl -o mobile_model.pte
  
  # Quiet mode
  python convertPtl2Pte.py model.ptl -q

Note:
  ExecuTorch is the successor to PyTorch Mobile and provides better
  performance and compatibility on mobile devices. The .pte format
  replaces the .ptl format for newer mobile applications.
  
  Install ExecuTorch with: pip install executorch
        """
    )
    
    parser.add_argument(
        'input',
        help='Input PyTorch Lite model file (.ptl)'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output ExecuTorch file (.pte). Default: input name with .pte extension',
        default=None
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if args.output is None:
        if args.input.endswith('.ptl'):
            args.output = args.input[:-4] + '.pte'
        elif args.input.endswith('.pt'):
            args.output = args.input[:-3] + '.pte'
        else:
            args.output = args.input + '.pte'
    
    # Perform conversion
    success = convert_ptl_to_pte(
        input_path=args.input,
        output_path=args.output,
        verbose=not args.quiet
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

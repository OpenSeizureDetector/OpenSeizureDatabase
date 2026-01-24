#!/usr/bin/env python
"""
convertPt2Pte.py - Convert PyTorch models to ExecuTorch .pte format

This script converts PyTorch models (both .pt and .ptl formats) directly to ExecuTorch .pte format.
Supports:
  - .pt files: Regular PyTorch checkpoints or saved models
  - .ptl files: TorchScript modules (automatically reconstructed as regular PyTorch models)

Usage:
    python convertPt2Pte.py input_model.pt -o output_model.pte
    python convertPt2Pte.py model.ptl -o output_model.pte
    python convertPt2Pte.py model.pt --input-shape 1,1,750
    
Prerequisites:
    pip install torch executorch
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


def convert_pt_to_pte(input_path, output_path, input_shape=(1, 1, 750), num_classes=2, verbose=True):
    """
    Convert a PyTorch model to ExecuTorch .pte format.
    
    Supports both .pt (PyTorch checkpoint) and .ptl (TorchScript) formats.
    TorchScript modules are automatically reconstructed as regular PyTorch models.
    
    Args:
        input_path: Path to input model file (.pt or .ptl)
        output_path: Path to output .pte file
        input_shape: Tuple of (batch, channels, length) for export
        num_classes: Number of output classes (default: 2)
        verbose: Print progress messages
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if verbose:
            file_ext = os.path.splitext(input_path)[1].lower()
            print(f"Loading PyTorch model from {input_path} ({file_ext})...")
        
        # Load the model checkpoint
        checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
        
        # Check if this is a TorchScript module (ScriptModule) - from .ptl or saved .pt
        if isinstance(checkpoint, torch.jit.ScriptModule):
            if verbose:
                print("Detected TorchScript module (.ptl or scripted .pt). Extracting weights to reconstruct as regular PyTorch model...")
            
            # Extract state dict from ScriptModule
            state_dict = checkpoint.state_dict()
            
            if DeepEpiCnn is None:
                print("Error: Cannot reconstruct model - DeepEpiCnn class not available", file=sys.stderr)
                return False
            
            # Create model instance
            input_length = input_shape[2] if len(input_shape) >= 3 else 750
            model = DeepEpiCnn(
                input_length=input_length,
                num_classes=num_classes,
                conv_dropout=0.0,
                dense_dropout=0.025
            )
            
            # Load the state dict
            try:
                model.load_state_dict(state_dict)
                if verbose:
                    print(f"Model reconstructed from ScriptModule with input_length={input_length}")
            except Exception as e:
                print(f"Warning: Could not load state dict from ScriptModule: {e}", file=sys.stderr)
                print("This may be a traced or optimized module with incompatible structure.", file=sys.stderr)
                raise
        
        # Handle different save formats
        elif isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                if verbose:
                    print("Detected training checkpoint format. Extracting model weights...")
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            if DeepEpiCnn is None:
                print("Error: Cannot reconstruct model - DeepEpiCnn class not available", file=sys.stderr)
                return False
            
            if verbose:
                print("Reconstructing model from state_dict...")
            
            # Extract input length and dropout values
            input_length = input_shape[2] if len(input_shape) >= 3 else 750
            conv_dropout = checkpoint.get('conv_dropout', 0.0) if 'model_state_dict' in checkpoint else 0.0
            dense_dropout = checkpoint.get('dense_dropout', 0.025) if 'model_state_dict' in checkpoint else 0.025
            
            # Create model instance
            model = DeepEpiCnn(
                input_length=input_length,
                num_classes=num_classes,
                conv_dropout=conv_dropout,
                dense_dropout=dense_dropout
            )
            
            # Load the state dict
            model.load_state_dict(state_dict)
            
            if verbose:
                print(f"Model reconstructed with input_length={input_length}, num_classes={num_classes}")
                print(f"  conv_dropout={conv_dropout}, dense_dropout={dense_dropout}")
        else:
            model = checkpoint
        
        # Set to evaluation mode
        model.eval()
        
        if verbose:
            print(f"Creating example input with shape {input_shape}...")
        
        # Create example input for export
        example_inputs = (torch.randn(input_shape),)
        
        if verbose:
            print("Exporting model to ExecuTorch format...")
        
        try:
            # Check if this is a ScriptModule (TorchScript)
            if isinstance(model, torch.jit.ScriptModule):
                if verbose:
                    print("Detected TorchScript module. Using TS2EPConverter...")
                
                try:
                    # Use TS2EPConverter for ScriptModule
                    ts2ep_converter = TS2EPConverter(model, example_inputs)
                    exported_program = ts2ep_converter.convert()
                except Exception as ts2ep_error:
                    print(f"TS2EPConverter failed: {ts2ep_error}", file=sys.stderr)
                    print("Attempting alternative approach with torch.export...", file=sys.stderr)
                    exported_program = export(model, example_inputs)
            else:
                # Export the model using torch.export for regular PyTorch modules
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
            print(f"Error during ExecuTorch export: {e}", file=sys.stderr)
            
            if "ScriptModule" in str(e) or "torch.export" in str(e) or "dynamo" in str(e).lower():
                print("\nNote: This model may not be compatible with ExecuTorch export.", file=sys.stderr)
                print("ScriptModule/TorchScript models may have limited support.", file=sys.stderr)
                print("Consider converting to .ptl first with convertPt2Ptl.py,", file=sys.stderr)
                print("then use convertPtl2Pte.py for the final conversion.", file=sys.stderr)
            
            raise
        
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
        return False


def parse_shape(shape_str):
    """Parse shape string like '1,1,750' into tuple."""
    try:
        parts = [int(x.strip()) for x in shape_str.split(',')]
        if len(parts) != 3:
            raise ValueError("Shape must have 3 dimensions (batch, channels, length)")
        return tuple(parts)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid shape format: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert PyTorch models (.pt or .ptl) to ExecuTorch .pte format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert .pt checkpoint
  python convertPt2Pte.py model.pt
  
  # Convert .ptl (TorchScript) file
  python convertPt2Pte.py model.ptl -o mobile_model.pte
  
  # Specify output file
  python convertPt2Pte.py model.pt -o mobile_model.pte
  
  # Custom input shape
  python convertPt2Pte.py model.pt --input-shape 1,1,750
  
  # Quiet mode
  python convertPt2Pte.py model.pt -q

Supported Input Formats:
  .pt files: PyTorch checkpoints or state dicts
  .ptl files: TorchScript modules (automatically reconstructed)

Note:
  If you encounter compatibility issues, try the two-step process:
    1. convertPt2Ptl.py model.pt       # Creates model.ptl
    2. convertPtl2Pte.py model.ptl     # Creates model.pte
  
  Install ExecuTorch with: pip install executorch
        """
    )
    
    parser.add_argument(
        'input',
        help='Input model file (.pt or .ptl)'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output ExecuTorch file (.pte). Default: input name with .pte extension',
        default=None
    )
    
    parser.add_argument(
        '--input-shape',
        type=parse_shape,
        default=(1, 1, 750),
        help='Input tensor shape as batch,channels,length (default: 1,1,750)'
    )
    
    parser.add_argument(
        '--num-classes',
        type=int,
        default=2,
        help='Number of output classes (default: 2)'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if args.output is None:
        if args.input.endswith('.pt'):
            args.output = args.input[:-3] + '.pte'
        else:
            args.output = args.input + '.pte'
    
    # Perform conversion
    success = convert_pt_to_pte(
        input_path=args.input,
        output_path=args.output,
        input_shape=args.input_shape,
        num_classes=args.num_classes,
        verbose=not args.quiet
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

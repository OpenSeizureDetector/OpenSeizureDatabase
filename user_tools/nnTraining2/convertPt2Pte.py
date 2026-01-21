#!/usr/bin/env python
"""
convertPt2Pte.py - Direct conversion from PyTorch .pt model to ExecuTorch .pte format

This script provides a direct conversion path from .pt to .pte, bypassing the .ptl step.
Use this when you want to create ExecuTorch models directly from PyTorch checkpoints.

Usage:
    python convertPt2Pte.py input_model.pt -o output_model.pte
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
    Convert a PyTorch .pt model directly to ExecuTorch .pte format.
    
    Args:
        input_path: Path to input .pt model file
        output_path: Path to output .pte file
        input_shape: Tuple of (batch, channels, length) for export
        num_classes: Number of output classes (default: 2)
        verbose: Print progress messages
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if verbose:
            print(f"Loading PyTorch model from {input_path}...")
        
        # Load the model checkpoint
        checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
        
        # Handle different save formats
        if isinstance(checkpoint, dict):
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
            # Export the model using torch.export
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
            
            if "torch.export" in str(e) or "dynamo" in str(e).lower():
                print("\nNote: This model may not be compatible with ExecuTorch export.", file=sys.stderr)
                print("Some operations or dynamic control flow may not be supported.", file=sys.stderr)
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
        description='Convert PyTorch .pt model directly to ExecuTorch .pte format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python convertPt2Pte.py model.pt
  
  # Specify output file
  python convertPt2Pte.py model.pt -o mobile_model.pte
  
  # Custom input shape
  python convertPt2Pte.py model.pt --input-shape 1,1,750
  
  # Quiet mode
  python convertPt2Pte.py model.pt -q

Note:
  This script converts .pt models directly to .pte format.
  If you encounter compatibility issues, try the two-step process:
    1. convertPt2Ptl.py model.pt       # Creates model.ptl
    2. convertPtl2Pte.py model.ptl     # Creates model.pte
  
  Install ExecuTorch with: pip install executorch
        """
    )
    
    parser.add_argument(
        'input',
        help='Input PyTorch model file (.pt)'
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

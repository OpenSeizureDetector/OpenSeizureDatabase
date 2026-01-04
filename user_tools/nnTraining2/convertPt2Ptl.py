#!/usr/bin/env python
"""
convertPt2Ptl.py - Convert PyTorch .pt model to .ptl (PyTorch Lite) format

Simple CLI tool to convert trained PyTorch models to mobile-optimized format.

Usage:
    python convertPt2Ptl.py input_model.pt -o output_model.ptl
    python convertPt2Ptl.py model.pt --input-shape 1,1,750
"""

import argparse
import sys
import os
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

# Import the model architecture
try:
    from deepEpiCnnModel_torch import DeepEpiCnn
except ImportError:
    # Try alternative import path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from deepEpiCnnModel_torch import DeepEpiCnn


def convert_pt_to_ptl(input_path, output_path, input_shape=(1, 1, 750), num_classes=2, verbose=True):
    """
    Convert a PyTorch .pt model to .ptl format for mobile deployment.
    
    Args:
        input_path: Path to input .pt model file
        output_path: Path to output .ptl file
        input_shape: Tuple of (batch, channels, length) for tracing
        num_classes: Number of output classes (default: 2)
        verbose: Print progress messages
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if verbose:
            print(f"Loading model from {input_path}...")
        
        # Load the model (weights_only=False for models saved with numpy objects)
        checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
        
        # Handle different save formats (model vs state_dict vs training checkpoint)
        if isinstance(checkpoint, dict):
            # Check if this is a training checkpoint with nested model_state_dict
            if 'model_state_dict' in checkpoint:
                if verbose:
                    print("Detected training checkpoint format. Extracting model weights...")
                state_dict = checkpoint['model_state_dict']
            else:
                # Direct state_dict
                state_dict = checkpoint
            
            if verbose:
                print("Reconstructing model from state_dict...")
            
            # Extract input length from shape parameter
            input_length = input_shape[2] if len(input_shape) >= 3 else 750
            
            # Create model instance
            model = DeepEpiCnn(
                input_length=input_length,
                num_classes=num_classes,
                dropout=0.025
            )
            
            # Load the state dict
            model.load_state_dict(state_dict)
            
            if verbose:
                print(f"Model reconstructed with input_length={input_length}, num_classes={num_classes}")
        else:
            model = checkpoint
        
        # Set to evaluation mode
        model.eval()
        
        if verbose:
            print(f"Creating example input with shape {input_shape}...")
        
        # Create example input for tracing
        example_input = torch.randn(input_shape)
        
        if verbose:
            print("Tracing model to TorchScript...")
        
        # Trace the model (converts to TorchScript)
        traced_model = torch.jit.trace(model, example_input)
        
        if verbose:
            print("Optimizing for mobile...")
        
        # Optimize for mobile deployment
        optimized_model = optimize_for_mobile(traced_model)
        
        if verbose:
            print(f"Saving to {output_path}...")
        
        # Save in lite interpreter format
        optimized_model._save_for_lite_interpreter(output_path)
        
        if verbose:
            print(f"✓ Successfully converted to {output_path}")
            
            # Print file sizes for comparison
            import os
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
        description='Convert PyTorch .pt model to .ptl (PyTorch Lite) format for mobile deployment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python convertPt2Ptl.py model.pt
  
  # Specify output file
  python convertPt2Ptl.py model.pt -o mobile_model.ptl
  
  # Custom input shape
  python convertPt2Ptl.py model.pt --input-shape 1,1,750
  
  # Quiet mode
  python convertPt2Ptl.py model.pt -q
        """
    )
    
    parser.add_argument(
        'input',
        help='Input PyTorch model file (.pt)'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output PyTorch Lite file (.ptl). Default: input name with .ptl extension',
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
            args.output = args.input[:-3] + '.ptl'
        else:
            args.output = args.input + '.ptl'
    
    # Perform conversion
    success = convert_pt_to_ptl(
        input_path=args.input,
        output_path=args.output,
        input_shape=args.input_shape,
        num_classes=args.num_classes,
        verbose=not args.quiet
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

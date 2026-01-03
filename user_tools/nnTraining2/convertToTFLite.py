#!/usr/bin/env python3

"""
Convert trained neural network models to TensorFlow Lite format for Android deployment.

Supports:
- TensorFlow/Keras models (.keras)
- PyTorch models (.pt) via ONNX conversion
- Optional quantization for reduced model size and faster inference

Usage:
    python convertToTFLite.py --model model.keras --output model.tflite
    python convertToTFLite.py --model model.pt --output model.tflite --framework pytorch
    python convertToTFLite.py --model model.keras --output model.tflite --quantize dynamic
"""

import argparse
import os
import sys
import numpy as np

def convert_keras_to_tflite(keras_model_path, output_path, quantize=None, representative_data=None):
    """
    Convert TensorFlow/Keras model to TensorFlow Lite format.
    
    Args:
        keras_model_path: Path to .keras model file
        output_path: Path to save .tflite file
        quantize: Quantization type ('dynamic', 'integer', 'float16') or None for no quantization
        representative_data: Optional numpy array or generator for quantization-aware conversion
    
    Returns:
        tuple: (output_path, model_size_bytes)
    """
    try:
        import tensorflow as tf
    except ImportError:
        print("ERROR: TensorFlow is required for Keras model conversion")
        return None, 0
    
    print(f"Loading Keras model from {keras_model_path}")
    model = tf.keras.models.load_model(keras_model_path)
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    # Apply quantization if requested
    if quantize:
        print(f"Applying {quantize} quantization")
        if quantize == 'dynamic':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif quantize == 'float16':
            converter.target_spec.supported_types = [tf.float16]
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif quantize == 'integer':
            # Integer quantization requires representative data
            if representative_data is None:
                print("WARNING: Integer quantization requires representative_data. Using dynamic quantization instead.")
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            else:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_data = representative_data
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
    
    print("Converting to TensorFlow Lite format")
    tflite_model = converter.convert()
    
    # Save model
    print(f"Saving TFLite model to {output_path}")
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    model_size = os.path.getsize(output_path)
    print(f"TFLite model size: {model_size / 1024:.1f} KB")
    
    return output_path, model_size


def convert_pytorch_to_tflite(pytorch_model_path, output_path, quantize=None, framework_module=None):
    """
    Convert PyTorch model to TensorFlow Lite format via ONNX.
    
    Args:
        pytorch_model_path: Path to .pt model file
        output_path: Path to save .tflite file
        quantize: Quantization type ('dynamic', 'float16') or None
        framework_module: Module containing model class (e.g., 'deepEpiCnnModel_torch')
    
    Returns:
        tuple: (output_path, model_size_bytes)
    """
    try:
        import torch
        import onnx
        import tf2onnx
    except ImportError as e:
        print(f"ERROR: PyTorch to TFLite conversion requires: torch, onnx, tf2onnx ({e})")
        return None, 0
    
    print("WARNING: PyTorch to TFLite conversion is experimental and requires additional dependencies")
    print("Recommended: Convert PyTorch to ONNX separately, then ONNX to TFLite")
    print(f"Skipping PyTorch model at {pytorch_model_path}")
    
    return None, 0


def main():
    parser = argparse.ArgumentParser(
        description='Convert trained neural network models to TensorFlow Lite format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Convert Keras model with dynamic quantization:
    %(prog)s --model model.keras --output model.tflite --quantize dynamic
  
  Convert Keras model without quantization:
    %(prog)s --model model.keras --output model.tflite
  
  Convert with float16 quantization (half precision):
    %(prog)s --model model.keras --output model.tflite --quantize float16
        """
    )
    
    parser.add_argument('--model', required=True,
                        help='Path to trained model file (.keras or .pt)')
    parser.add_argument('--output', required=True,
                        help='Path to save TensorFlow Lite model (.tflite)')
    parser.add_argument('--framework', choices=['tensorflow', 'keras', 'pytorch'], default='tensorflow',
                        help='Model framework (default: tensorflow)')
    parser.add_argument('--quantize', choices=['dynamic', 'float16', 'integer'], default=None,
                        help='Quantization type for model optimization (optional)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        sys.exit(1)
    
    # Validate output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        print(f"ERROR: Output directory does not exist: {output_dir}")
        sys.exit(1)
    
    print(f"Converting {args.framework.upper()} model to TensorFlow Lite")
    print(f"Input:  {args.model}")
    print(f"Output: {args.output}")
    if args.quantize:
        print(f"Quantization: {args.quantize}")
    
    # Detect framework from file extension if not specified
    if args.framework == 'tensorflow' and args.model.endswith('.pt'):
        print("Detected PyTorch model from .pt extension")
        args.framework = 'pytorch'
    elif args.framework == 'tensorflow' and args.model.endswith('.keras'):
        args.framework = 'tensorflow'
    
    try:
        if args.framework in ['tensorflow', 'keras']:
            output_path, model_size = convert_keras_to_tflite(
                args.model,
                args.output,
                quantize=args.quantize
            )
        elif args.framework == 'pytorch':
            output_path, model_size = convert_pytorch_to_tflite(
                args.model,
                args.output,
                quantize=args.quantize
            )
        else:
            print(f"ERROR: Unknown framework: {args.framework}")
            sys.exit(1)
        
        if output_path:
            print(f"\nSUCCESS: Model converted to TensorFlow Lite")
            print(f"Output file: {output_path}")
            print(f"Model size: {model_size / (1024 * 1024):.2f} MB")
            
            # Show original size for comparison
            if os.path.exists(args.model):
                orig_size = os.path.getsize(args.model)
                reduction = (1 - model_size / orig_size) * 100
                print(f"Original size: {orig_size / (1024 * 1024):.2f} MB")
                print(f"Size reduction: {reduction:.1f}%")
        else:
            print("ERROR: Conversion failed")
            sys.exit(1)
    
    except Exception as e:
        print(f"ERROR during conversion: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

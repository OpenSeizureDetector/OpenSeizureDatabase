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
        import tensorflow as tf
    except ImportError as e:
        print(f"ERROR: PyTorch to TFLite conversion requires: torch, onnx, tensorflow ({e})")
        print("Install with: pip install torch onnx tensorflow")
        return None, 0
    
    # Try to import onnx2tf (preferred) or ai_edge_torch
    converter_type = None
    try:
        import onnx2tf
        converter_type = 'onnx2tf'
    except ImportError:
        try:
            import ai_edge_torch
            converter_type = 'ai_edge_torch'
        except ImportError:
            print("ERROR: Need either onnx2tf or ai_edge_torch for conversion")
            print("Install with: pip install onnx2tf  (recommended)")
            print("         or: pip install ai-edge-torch")
            return None, 0
    
    import tempfile
    
    print(f"Loading PyTorch model from {pytorch_model_path}")
    
    # Load PyTorch model (checkpoint may be full model or state dict)
    model = None
    try:
        # Allow numpy scalar class during safe loading (trusted checkpoint)
        try:
            torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        except Exception:
            pass
        try:
            checkpoint = torch.load(pytorch_model_path, map_location='cpu', weights_only=False)
        except TypeError:
            # Older torch versions do not support weights_only
            checkpoint = torch.load(pytorch_model_path, map_location='cpu')
        
        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint
        else:
            # Assume state dict style checkpoint
            state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
            # Determine which module to use for model construction
            constructor = None
            if framework_module:
                import importlib
                module = importlib.import_module(framework_module)
                constructor = getattr(module, 'DeepEpiCnn', None) or getattr(module, 'build_model', None)
            else:
                # Default to bundled torch model
                try:
                    from user_tools.nnTraining2.deepEpiCnnModel_torch import DeepEpiCnn  # type: ignore
                    constructor = DeepEpiCnn
                except Exception:
                    try:
                        from deepEpiCnnModel_torch import DeepEpiCnn  # type: ignore
                        constructor = DeepEpiCnn
                    except Exception:
                        constructor = None
            
            if constructor is None:
                print("ERROR: Could not determine model constructor. Pass --framework-module module_path.ClassName")
                return None, 0
            
            model = constructor()
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"WARNING: Missing keys when loading state dict: {missing}")
            if unexpected:
                print(f"WARNING: Unexpected keys when loading state dict: {unexpected}")
    except Exception as e:
        print(f"ERROR loading PyTorch model: {e}")
        return None, 0
    
    model.eval()
    
    # Create dummy input - prefer model metadata, then layer introspection
    print("Creating dummy input for model tracing")
    try:
        input_length = getattr(model, 'input_length', None)
        channels = 1
        modules_iter = model.modules()
        next(modules_iter)  # skip the model itself
        first_layer = next(modules_iter, None)
        if hasattr(first_layer, 'in_channels') and getattr(first_layer, 'in_channels'):
            channels = first_layer.in_channels
        if hasattr(first_layer, 'in_features'):
            print("Determining input shape from first layer in_features")
            dummy_input = torch.randn(1, first_layer.in_features)
        else:
            if input_length is None:
                # Default length for DeepEpiCnn (30s @25Hz -> 750)
                input_length = 750
            print(f"Using channels={channels}, length={input_length}")
            dummy_input = torch.randn(1, channels, input_length)
    except Exception as e:
        print(f"WARNING: Could not infer input shape: {e}")
        print("Using default input shape (1, 1, 750)")
        dummy_input = torch.randn(1, 1, 750)
    
    print(f"Using input shape: {dummy_input.shape}")
    
    # Convert to ONNX (temporary file)
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp_onnx:
        onnx_path = tmp_onnx.name
    
    try:
        print(f"Converting PyTorch to ONNX (intermediate step)")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Load and prepare ONNX model
        print("Loading ONNX model")
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # Convert ONNX to TFLite based on available converter
        if converter_type == 'onnx2tf':
            print("Converting ONNX to TFLite using onnx2tf")
            # onnx2tf can convert directly to TFLite
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Convert ONNX to TensorFlow SavedModel, then to TFLite
                saved_model_path = os.path.join(tmp_dir, 'saved_model')
                
                # Run onnx2tf conversion
                onnx2tf.convert(
                    input_onnx_file_path=onnx_path,
                    output_folder_path=saved_model_path,
                    non_verbose=True
                )
                
                # Convert SavedModel to TFLite
                print("Converting TensorFlow SavedModel to TFLite")
                converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
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
                        print("WARNING: Integer quantization not fully supported for PyTorch conversion")
                        converter.optimizations = [tf.lite.Optimize.DEFAULT]
                
                tflite_model = converter.convert()
                
        elif converter_type == 'ai_edge_torch':
            print("Converting PyTorch to TFLite using ai_edge_torch")
            # ai_edge_torch can convert PyTorch models directly
            # Reload the PyTorch model for direct conversion
            model_reload = torch.load(pytorch_model_path, map_location='cpu')
            model_reload.eval()
            
            edge_model = ai_edge_torch.convert(model_reload.eval(), (dummy_input,))
            tflite_model = edge_model.tflite_model()
        
        # Save TFLite model
        print(f"Saving TFLite model to {output_path}")
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        model_size = os.path.getsize(output_path)
        print(f"TFLite model size: {model_size / 1024:.1f} KB")
        
        return output_path, model_size
            
    finally:
        # Clean up temporary ONNX file
        if os.path.exists(onnx_path):
            os.unlink(onnx_path)


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

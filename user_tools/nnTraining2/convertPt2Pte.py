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


def load_model_instance_from_checkpoint(checkpoint, config, num_classes,
                                         verbose=True, conv_dropout=0.0,
                                         dense_dropout=0.025):
    """
    Dynamically load and instantiate the model class from checkpoint configuration.

    Tries to load the correct model class (DeepEpiCnnModelPyTorch, CnnLstmModelPyTorch,
    etc.) via the checkpoint's modelClass entry. Falls back to DeepEpiCnn if model
    class not specified or loading fails.

    The checkpoint may contain a wrapper class (e.g., CnnLstmModelPyTorch) or the actual model class
    (e.g., DeepEpiCnn). This function detects which one and returns the underlying PyTorch model.

    Model geometry (input length, channels, ...) always comes from the wrapper's
    own configuration - never from hardcoded constants - so exports match the
    trained model for any architecture. Callers must therefore obtain example
    inputs from the wrapper (see export_example_inputs) rather than assuming a
    shape.

    Args:
        checkpoint: The full checkpoint dict
        config: Configuration dict from checkpoint
        num_classes: Number of output classes
        verbose: Print debug messages

    Returns:
        (model, wrapper) tuple: instantiated PyTorch model object (with
        load_state_dict method) and the wrapper instance, or None if the model
        is not wrapper-based (raw nn.Module / ScriptModule paths).
    """
    model = None
    wrapper = None
    model_class_path = config.get('modelConfig', {}).get('modelClass', None)

    # Try to load the model class from checkpoint config
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

                    # Instantiate the class with its own training configuration
                    # so geometry (buffer/window lengths, channels, dropout)
                    # matches the trained weights.
                    try:
                        instance = TargetClass(config['modelConfig'])
                        if verbose:
                            print(f"✓ Successfully instantiated class: {class_name}")
                    except TypeError:
                        # If config-based instantiation fails, try with individual parameters
                        if verbose:
                            print(f"  Instantiation with config failed, trying parameter-based init...")
                        instance = TargetClass(num_classes=num_classes)
                        if verbose:
                            print(f"✓ Successfully instantiated with parameters: {class_name}")
                    # Check if this is a wrapper class (has .model attribute) or the actual model class
                    if hasattr(instance, 'model') and hasattr(instance, 'makeModel'):
                        # This is a wrapper class - call makeModel() to create the underlying model.
                        # input_shape=None lets the wrapper derive geometry from
                        # its own config (passing a hardcoded shape here is what
                        # used to break exports for non-default window lengths).
                        if verbose:
                            print(f"  Detected wrapper class, calling makeModel()...")
                        model = instance.makeModel(input_shape=None, num_classes=num_classes)
                        wrapper = instance
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

    # Fallback to DeepEpiCnn if model class not specified or loading failed.
    # NOTE: the fallback can only serve the default 750-sample geometry; for
    # any other architecture the checkpoint must carry a loadable modelClass.
    if model is None:
        if verbose:
            if model_class_path:
                print(f"✗ Using fallback DeepEpiCnn model (could not load {model_class_path})")
            else:
                print(f"Using default DeepEpiCnn model")
        model = DeepEpiCnn(input_length=750, num_classes=num_classes,
                         conv_dropout=conv_dropout, dense_dropout=dense_dropout)

    return model, wrapper


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
        wrapper = None
        if isinstance(checkpoint, torch.jit.ScriptModule):
            state_dict = checkpoint.state_dict()
            input_length = input_shape[2] if len(input_shape) >= 3 else 750
            # Default dropout for TorchScript (no metadata available).
            # NOTE: a TorchScript file carries no architecture metadata, so
            # this path can only serve DeepEpiCnn-geometry models.
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

            # Extract configuration from checkpoint if available. The config is
            # the training config: model geometry comes from the wrapper's own
            # config handling, NOT from hardcoded keys here. (A previous
            # version overwrote the caller's input_shape from a stale
            # dataProcessing.rawDataLength key, defaulting to 750 - which broke
            # every model not trained on exactly 750 samples, e.g. the 45 s /
            # 1125-sample CNN-LSTM.)
            config = checkpoint.get('config', {}) if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else {}
            if config and verbose:
                print("Using configuration from checkpoint...")
                num_classes = config.get('modelConfig', {}).get('numClasses', num_classes)
                print(f"  num_classes={num_classes}")

            # Extract dropout parameters from checkpoint (used only for the
            # DeepEpiCnn fallback / direct-class construction; wrapper classes
            # take dropout from their own config).
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

            # Reconstruct via the recorded model class; geometry comes from
            # that class's own config handling.
            model, wrapper = load_model_instance_from_checkpoint(
                checkpoint, config, num_classes,
                verbose=verbose, conv_dropout=conv_dropout,
                dense_dropout=dense_dropout
            )
            model.load_state_dict(state_dict)
        else:
            model = checkpoint

        model.eval()

        # Fix device mismatch: move model to CPU and create example_inputs on same device
        # ExecuTorch requires CPU-compatible models, and all tensors must be on the same device
        if verbose:
            print("Preparing model for export (moving to CPU for device consistency)...")
        model = model.cpu()

        # If model has internal device tracking (e.g., DeepEpiCnnModelPyTorch.device),
        # update it to CPU as well to ensure forward pass operations are on CPU
        if hasattr(model, 'device'):
            model.device = torch.device('cpu')
            if verbose:
                print("  Updated model internal device to CPU")

        # Example inputs MUST come from the model wrapper, which is the only
        # place that knows the true input geometry (sequence length, channel
        # order). They are traced in the exact (batch, channels, length)
        # layout the .pte runtime is fed at inference time.
        example_inputs = None
        if wrapper is not None:
            get_examples = getattr(wrapper, 'export_example_inputs', None)
            if callable(get_examples):
                try:
                    example_inputs = get_examples(batch_size=1)
                    if verbose:
                        print(f"  Example inputs from {type(wrapper).__name__}: "
                              f"{[tuple(t.shape) for t in example_inputs]}")
                except Exception as e:
                    if verbose:
                        print(f"  Wrapper example inputs failed ({e}); using fallback shape")
                    example_inputs = None
        if example_inputs is None:
            # Legacy fallback: caller's input_shape (nnTrainer derives this
            # from the training data) or the (1, 1, 750) default.
            if verbose:
                print(f"  No wrapper example inputs; falling back to input_shape={tuple(input_shape)}")
            example_inputs = (torch.randn(tuple(input_shape), device='cpu'),)
        else:
            example_inputs = tuple(
                t.to(device='cpu', dtype=torch.float32) if isinstance(t, torch.Tensor)
                else torch.as_tensor(t, dtype=torch.float32)
                for t in example_inputs
            )

        # Smoke-test the example inputs through the model BEFORE tracing, so a
        # geometry mismatch raises the model's own clear error instead of a
        # dynamo stack trace.
        if verbose:
            print("Smoke-testing example inputs through the model...")
        with torch.no_grad():
            smoke_out = model(*example_inputs)
        if verbose:
            try:
                print(f"  Smoke test output shape: {tuple(smoke_out.shape)}")
            except Exception:
                print("  Smoke test ran (non-tensor output)")

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
    parser.add_argument('--input-shape', type=parse_shape, default=(1, 1, 750), help='Fallback input shape (batch,channels,length) - only used when the checkpoint model class cannot provide example inputs itself')
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

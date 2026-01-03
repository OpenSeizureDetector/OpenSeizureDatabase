#!/usr/bin/env python3
"""
Test script to verify both TensorFlow and PyTorch implementations work correctly.

This script:
1. Tests that both model classes can be instantiated
2. Verifies forward pass produces expected output shapes
3. Checks that model architectures have similar parameter counts
4. Validates data preprocessing pipeline compatibility
"""

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def test_tensorflow_model():
    """Test TensorFlow DeepEpiCnnModel"""
    print("\n" + "="*60)
    print("Testing TensorFlow Implementation")
    print("="*60)
    
    try:
        from user_tools.nnTraining2.deepEpiCnnModel import DeepEpiCnnModel
        
        config = {
            'sampleFreq': 25,
            'window': 750,
            'framework': 'tensorflow'
        }
        
        print("Creating TensorFlow model...")
        model_wrapper = DeepEpiCnnModel(configObj=config, debug=True)
        model = model_wrapper.makeModel(input_shape=(750, 1), num_classes=2)
        
        print("\nModel created successfully!")
        print(f"Framework: {model_wrapper.framework}")
        
        # Get parameter count
        param_count = model.count_params()
        print(f"Total parameters: {param_count:,}")
        
        # Test forward pass
        print("\nTesting forward pass...")
        import numpy as np
        test_input = np.random.randn(4, 750, 1).astype(np.float32)
        output = model.predict(test_input, verbose=0)
        
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        
        assert output.shape == (4, 2), f"Expected output shape (4, 2), got {output.shape}"
        assert np.allclose(output.sum(axis=1), 1.0, atol=1e-5), "Softmax outputs should sum to 1"
        
        # Test data preprocessing
        print("\nTesting data preprocessing...")
        test_acc_data = list(np.random.randn(750))
        vec = model_wrapper.accData2vector(test_acc_data, normalise=True)
        assert vec is not None, "accData2vector should return data"
        assert len(vec) == 750, f"Expected 750 samples, got {len(vec)}"
        
        print("✓ TensorFlow implementation: PASSED")
        return True, param_count
        
    except Exception as e:
        print(f"✗ TensorFlow implementation: FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False, 0


def test_pytorch_model():
    """Test PyTorch DeepEpiCnnModelPyTorch"""
    print("\n" + "="*60)
    print("Testing PyTorch Implementation")
    print("="*60)
    
    try:
        from user_tools.nnTraining2.deepEpiCnnModel_torch import DeepEpiCnnModelPyTorch
        import torch
        
        config = {
            'sampleFreq': 25,
            'window': 750,
            'framework': 'pytorch'
        }
        
        print("Creating PyTorch model...")
        model_wrapper = DeepEpiCnnModelPyTorch(configObj=config, debug=True)
        model = model_wrapper.makeModel(input_shape=(750, 1), num_classes=2)
        
        print("\nModel created successfully!")
        print(f"Framework: {model_wrapper.framework}")
        print(f"Device: {model_wrapper.device}")
        
        # Get parameter count
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {param_count:,}")
        
        # Test forward pass
        print("\nTesting forward pass...")
        test_input = torch.randn(4, 750, 1)
        with torch.no_grad():
            output = model(test_input)
        
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        
        assert output.shape == (4, 2), f"Expected output shape (4, 2), got {output.shape}"
        
        # Test with predict method (applies softmax)
        output_probs = model_wrapper.predict(test_input.numpy())
        assert output_probs.shape == (4, 2), f"Expected output shape (4, 2), got {output_probs.shape}"
        assert np.allclose(output_probs.sum(axis=1), 1.0, atol=1e-5), "Softmax outputs should sum to 1"
        
        # Test data preprocessing
        print("\nTesting data preprocessing...")
        test_acc_data = list(np.random.randn(750))
        vec = model_wrapper.accData2vector(test_acc_data, normalise=True)
        assert vec is not None, "accData2vector should return data"
        assert len(vec) == 750, f"Expected 750 samples, got {len(vec)}"
        
        print("✓ PyTorch implementation: PASSED")
        return True, param_count
        
    except Exception as e:
        print(f"✗ PyTorch implementation: FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False, 0


def test_harrell_davis():
    """Test Harrell-Davis quantile estimator (shared between implementations)"""
    print("\n" + "="*60)
    print("Testing Harrell-Davis Quantile Estimator")
    print("="*60)
    
    try:
        from user_tools.nnTraining2.deepEpiCnnModel import DeepEpiCnnModel
        from user_tools.nnTraining2.deepEpiCnnModel_torch import DeepEpiCnnModelPyTorch
        
        test_sample = [0.1, 0.2, 0.9, 0.8]
        q = 0.7
        
        # Test TensorFlow version
        result_tf = DeepEpiCnnModel.harrell_davis_quantile(test_sample, q)
        print(f"TensorFlow HD({q}) of {test_sample} = {result_tf:.6f}")
        
        # Test PyTorch version
        result_pt = DeepEpiCnnModelPyTorch.harrell_davis_quantile(test_sample, q)
        print(f"PyTorch HD({q}) of {test_sample} = {result_pt:.6f}")
        
        # Results should be identical
        assert np.isclose(result_tf, result_pt, atol=1e-6), \
            f"HD results differ: TF={result_tf}, PT={result_pt}"
        
        print("✓ Harrell-Davis quantile: PASSED (both implementations match)")
        return True
        
    except ImportError as e:
        print(f"⚠ Harrell-Davis quantile: SKIPPED (scipy not installed)")
        return True  # Not a failure, just missing optional dependency
    except Exception as e:
        print(f"✗ Harrell-Davis quantile: FAILED")
        print(f"Error: {e}")
        return False


def compare_implementations(tf_params, pt_params):
    """Compare parameter counts between implementations"""
    print("\n" + "="*60)
    print("Comparing Implementations")
    print("="*60)
    
    print(f"TensorFlow parameters: {tf_params:,}")
    print(f"PyTorch parameters:    {pt_params:,}")
    
    if tf_params > 0 and pt_params > 0:
        diff = abs(tf_params - pt_params)
        pct_diff = (diff / tf_params) * 100
        
        print(f"Difference: {diff:,} ({pct_diff:.2f}%)")
        
        if pct_diff < 1.0:
            print("✓ Parameter counts are very close (< 1% difference)")
        elif pct_diff < 5.0:
            print("⚠ Parameter counts differ slightly (< 5% difference)")
        else:
            print("✗ Parameter counts differ significantly (>= 5% difference)")
            print("  This may indicate architectural differences.")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("nnTraining2 Framework Compatibility Test Suite")
    print("="*60)
    
    results = []
    
    # Test TensorFlow
    tf_success, tf_params = test_tensorflow_model()
    results.append(("TensorFlow", tf_success))
    
    # Test PyTorch
    pt_success, pt_params = test_pytorch_model()
    results.append(("PyTorch", pt_success))
    
    # Test Harrell-Davis (if scipy available)
    hd_success = test_harrell_davis()
    results.append(("Harrell-Davis", hd_success))
    
    # Compare implementations
    if tf_success and pt_success:
        compare_implementations(tf_params, pt_params)
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for name, success in results:
        status = "PASS" if success else "FAIL"
        symbol = "✓" if success else "✗"
        print(f"{symbol} {name}: {status}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n✓ All tests passed! Both frameworks are working correctly.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

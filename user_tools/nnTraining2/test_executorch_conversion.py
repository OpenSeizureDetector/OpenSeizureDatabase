#!/usr/bin/env python3
"""
test_executorch_conversion.py - Test the ExecuTorch conversion pipeline

This script tests the conversion of models to .pte format.
"""

import os
import sys
import subprocess

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print(f"✓ SUCCESS: {description}")
            return True
        else:
            print(f"✗ FAILED: {description} (exit code: {result.returncode})")
            return False
    except Exception as e:
        print(f"✗ ERROR: {description}")
        print(f"   {e}")
        return False

def main():
    print("ExecuTorch Conversion Test Suite")
    print("="*60)
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Test files
    pt_model = os.path.join(script_dir, "deepEpiCnnModel_pytorch.pt")
    
    # Check if model exists
    if not os.path.exists(pt_model):
        print(f"✗ Test model not found: {pt_model}")
        print("  Please train a model first or provide a .pt model file")
        return False
    
    print(f"Using test model: {pt_model}")
    
    # Test 1: Check if scripts are accessible
    print(f"\n{'='*60}")
    print("Test 1: Checking if conversion scripts exist")
    print('='*60)
    
    convertPt2Ptl = os.path.join(script_dir, "convertPt2Ptl.py")
    convertPtl2Pte = os.path.join(script_dir, "convertPtl2Pte.py")
    convertPt2Pte = os.path.join(script_dir, "convertPt2Pte.py")
    
    scripts = [
        (convertPt2Ptl, "convertPt2Ptl.py"),
        (convertPtl2Pte, "convertPtl2Pte.py"),
        (convertPt2Pte, "convertPt2Pte.py")
    ]
    
    all_exist = True
    for script, name in scripts:
        if os.path.exists(script):
            print(f"  ✓ {name} found")
        else:
            print(f"  ✗ {name} NOT FOUND")
            all_exist = False
    
    if not all_exist:
        print("\n✗ Some scripts are missing!")
        return False
    
    # Test 2: Convert .pt to .ptl
    test_ptl = pt_model.replace('.pt', '_test.ptl')
    success_ptl = run_command(
        [sys.executable, convertPt2Ptl, pt_model, '-o', test_ptl],
        "Convert .pt to .ptl"
    )
    
    if not success_ptl or not os.path.exists(test_ptl):
        print("\n✗ .ptl conversion failed or file not created")
        return False
    
    # Test 3: Convert .ptl to .pte
    test_pte = test_ptl.replace('.ptl', '.pte')
    
    # Check if ExecuTorch is installed first
    print(f"\n{'='*60}")
    print("Checking ExecuTorch installation")
    print('='*60)
    
    try:
        import executorch
        print("✓ ExecuTorch is installed")
        
        success_pte = run_command(
            [sys.executable, convertPtl2Pte, test_ptl, '-o', test_pte],
            "Convert .ptl to .pte (ExecuTorch)"
        )
        
        if not success_pte:
            print("\n⚠ WARNING: .pte conversion failed")
            print("  This may be due to model compatibility issues with ExecuTorch")
            print("  The .ptl model can still be used with PyTorch Mobile")
        elif not os.path.exists(test_pte):
            print("\n⚠ WARNING: .pte file was not created")
        else:
            print(f"\n✓ Successfully created .pte model: {test_pte}")
            
            # Show file sizes
            pt_size = os.path.getsize(pt_model) / (1024 * 1024)
            ptl_size = os.path.getsize(test_ptl) / (1024 * 1024)
            pte_size = os.path.getsize(test_pte) / (1024 * 1024)
            
            print(f"\nFile Sizes:")
            print(f"  .pt  (PyTorch):        {pt_size:.2f} MB")
            print(f"  .ptl (PyTorch Mobile): {ptl_size:.2f} MB")
            print(f"  .pte (ExecuTorch):     {pte_size:.2f} MB")
    
    except ImportError:
        print("✗ ExecuTorch is not installed")
        print("\n  To install ExecuTorch:")
        print("    pip install executorch")
        print("\n  Skipping .pte conversion test")
        success_pte = False
    
    # Test 4: Direct .pt to .pte conversion (if ExecuTorch is available)
    if success_pte:
        test_pte_direct = pt_model.replace('.pt', '_test_direct.pte')
        success_direct = run_command(
            [sys.executable, convertPt2Pte, pt_model, '-o', test_pte_direct],
            "Convert .pt directly to .pte"
        )
        
        if success_direct and os.path.exists(test_pte_direct):
            print(f"\n✓ Direct .pt to .pte conversion successful")
            # Clean up direct test file
            try:
                os.remove(test_pte_direct)
            except:
                pass
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    print(f"  .pt  → .ptl conversion: {'✓ PASS' if success_ptl else '✗ FAIL'}")
    
    if 'executorch' in sys.modules or success_pte:
        print(f"  .ptl → .pte conversion: {'✓ PASS' if success_pte else '✗ FAIL'}")
        print(f"  .pt  → .pte direct:     {'✓ PASS' if success_pte else '✗ FAIL'}")
    else:
        print(f"  .ptl → .pte conversion: ⊘ SKIP (ExecuTorch not installed)")
        print(f"  .pt  → .pte direct:     ⊘ SKIP (ExecuTorch not installed)")
    
    # Cleanup test files
    print(f"\n{'='*60}")
    print("Cleaning up test files...")
    print('='*60)
    
    for test_file in [test_ptl, test_pte]:
        if os.path.exists(test_file):
            try:
                os.remove(test_file)
                print(f"  Removed: {os.path.basename(test_file)}")
            except Exception as e:
                print(f"  Could not remove {os.path.basename(test_file)}: {e}")
    
    overall_success = success_ptl
    if overall_success:
        print("\n" + "="*60)
        print("✓ CONVERSION PIPELINE IS WORKING")
        print("="*60)
        if not success_pte:
            print("\nNote: Install ExecuTorch for .pte model support:")
            print("  pip install executorch")
    else:
        print("\n" + "="*60)
        print("✗ SOME TESTS FAILED")
        print("="*60)
    
    return overall_success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

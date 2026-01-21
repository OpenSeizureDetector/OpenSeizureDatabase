# ExecuTorch Model Export Guide

## Overview

The training pipeline now supports **ExecuTorch (.pte)** format in addition to PyTorch (.pt) and PyTorch Mobile (.ptl) formats. ExecuTorch is the successor to PyTorch Mobile and provides better performance and compatibility for mobile deployment.

## What Changed

### Model Formats Produced

After training, three model formats are now generated:

1. **`.pt`** - Standard PyTorch checkpoint (used for training and testing)
2. **`.ptl`** - PyTorch Mobile format (legacy mobile format)
3. **`.pte`** - ExecuTorch format (modern mobile format - **recommended**)

### Files Added

- **`convertPtl2Pte.py`** - Convert existing .ptl models to .pte format
- **`convertPt2Pte.py`** - Convert .pt models directly to .pte format
- **`EXECUTORCH_EXPORT.md`** - This documentation file

### Files Modified

- **`nnTrainer.py`** - Now exports .pte models during training
- **`runSequence.py`** - Now copies .pte models along with other formats
- **`requirements_pytorch.txt`** - Added ExecuTorch dependency

## Installation

### Installing ExecuTorch

```bash
# Install ExecuTorch (required for .pte export)
pip install executorch

# Or install all PyTorch requirements including ExecuTorch
pip install -r requirements_pytorch.txt
```

**Note:** ExecuTorch is optional but recommended. The training will work without it, but .pte models will not be generated.

## Usage

### During Training

The training pipeline automatically generates .pte models:

```bash
# Run the training sequence as usual
python runSequence.py --config nnConfig.json

# After training, you'll have:
# - model_pytorch.pt
# - model_pytorch.ptl
# - model_pytorch.pte  (NEW!)
```

### Converting Existing Models

#### Option 1: Convert .ptl to .pte (Recommended)

```bash
# Convert an existing .ptl model to .pte
python convertPtl2Pte.py model.ptl

# Specify output filename
python convertPtl2Pte.py model.ptl -o mymodel.pte

# Quiet mode (suppress progress messages)
python convertPtl2Pte.py model.ptl -q
```

#### Option 2: Convert .pt directly to .pte

```bash
# Convert a .pt checkpoint directly to .pte
python convertPt2Pte.py model.pt

# With custom input shape
python convertPt2Pte.py model.pt --input-shape 1,1,750 --num-classes 2

# Specify output file
python convertPt2Pte.py model.pt -o mymodel.pte
```

#### Option 3: Two-step conversion (if direct conversion fails)

```bash
# Step 1: Convert .pt to .ptl
python convertPt2Ptl.py model.pt

# Step 2: Convert .ptl to .pte
python convertPtl2Pte.py model.ptl
```

### Batch Conversion Script

Convert multiple models at once:

```bash
# Convert all .ptl models in current directory to .pte
for file in *.ptl; do
    python convertPtl2Pte.py "$file"
done

# Or convert all .pt models directly
for file in *.pt; do
    python convertPt2Pte.py "$file"
done
```

## Mobile Application Integration

### For Android (Java/Kotlin)

```java
// Old PyTorch Mobile approach (.ptl)
// Module module = Module.load(modelPath);

// New ExecuTorch approach (.pte)
import org.pytorch.executorch.Module;

Module module = Module.load(modelPath);
float[] input = new float[750];  // Your input data
Tensor inputTensor = Tensor.fromBlob(input, new long[]{1, 1, 750});
Tensor output = module.forward(inputTensor);
```

### For iOS (Swift)

```swift
// Old PyTorch Mobile approach (.ptl)
// let module = try! TorchModule(fileAtPath: modelPath)

// New ExecuTorch approach (.pte)
import ExecuTorch

let module = try! Module.load(path: modelPath)
let input = [Float](repeating: 0.0, count: 750)
let output = try! module.forward(input)
```

## Troubleshooting

### ExecuTorch Not Installed

**Symptom:** Warning message during training: "Could not convert model to .pte format"

**Solution:**
```bash
pip install executorch
```

### Conversion Fails with Dynamo Errors

**Symptom:** Error messages mentioning "torch.export" or "dynamo"

**Cause:** Some PyTorch operations may not be compatible with ExecuTorch export.

**Solution:** Use the two-step conversion process:
```bash
python convertPt2Ptl.py model.pt
python convertPtl2Pte.py model.ptl
```

### Input Shape Mismatch

**Symptom:** Model export fails with shape-related errors

**Solution:** Explicitly specify the input shape:
```bash
# Default is (1, 1, 750) for this project
python convertPt2Pte.py model.pt --input-shape 1,1,750
```

### Model Architecture Not Found

**Symptom:** "Could not import DeepEpiCnn model architecture"

**Solution:** Ensure you're running the script from the correct directory:
```bash
cd user_tools/nnTraining2
python convertPt2Pte.py path/to/model.pt
```

## Performance Comparison

| Format | Size | Mobile Performance | Compatibility |
|--------|------|-------------------|---------------|
| .pt    | ~3 MB | ❌ Desktop only | PyTorch |
| .ptl   | ~3 MB | ⚠️ Legacy mobile | PyTorch Mobile (deprecated) |
| .pte   | ~3 MB | ✅ Optimized | ExecuTorch (modern) |

## Migration Guide

### For Existing Mobile Apps

If your app currently uses .ptl models:

1. **Generate .pte models** for all your trained models
   ```bash
   python convertPtl2Pte.py old_model.ptl -o new_model.pte
   ```

2. **Update your app** to use ExecuTorch instead of PyTorch Mobile
   - Android: Update gradle dependencies
   - iOS: Update pod dependencies

3. **Test thoroughly** - .pte models may have slightly different behavior

4. **Keep .ptl as fallback** during transition period

### For New Projects

- Use .pte format exclusively
- No need to generate .ptl models (but they're created automatically for backward compatibility)

## Technical Details

### ExecuTorch Export Process

The conversion follows this pipeline:

```
.pt (PyTorch checkpoint)
  ↓ (load and reconstruct model)
.ptl (TorchScript traced + mobile optimized)
  ↓ (torch.export → to_edge → to_executorch)
.pte (ExecuTorch program)
```

### Model Architecture Requirements

For successful ExecuTorch export, models should:
- ✅ Use static shapes (no dynamic dimensions)
- ✅ Avoid dynamic control flow (if/else based on input values)
- ✅ Use standard PyTorch operations
- ❌ Avoid custom CUDA kernels
- ❌ Avoid Python-side control flow

The DeepEpiCnn model used in this project meets all these requirements.

## Additional Resources

- [ExecuTorch Official Documentation](https://pytorch.org/executorch/)
- [ExecuTorch GitHub](https://github.com/pytorch/executorch)
- [Migration Guide from PyTorch Mobile](https://pytorch.org/executorch/stable/migration-guide.html)

## Support

For issues related to:
- **Model training**: See main project README
- **ExecuTorch conversion**: Check this guide's troubleshooting section
- **Mobile integration**: Refer to ExecuTorch documentation

## Version History

- **v1.0** (January 2026) - Initial ExecuTorch support added
  - Added convertPtl2Pte.py
  - Added convertPt2Pte.py
  - Updated nnTrainer.py to auto-generate .pte models
  - Updated runSequence.py to handle .pte files

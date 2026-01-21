# ExecuTorch Quick Reference Card

## Installation
```bash
pip install executorch
# or
pip install -r requirements_pytorch.txt
```

## Conversion Commands

### Convert .ptl to .pte (Recommended)
```bash
python convertPtl2Pte.py model.ptl
python convertPtl2Pte.py model.ptl -o output.pte
```

### Convert .pt directly to .pte
```bash
python convertPt2Pte.py model.pt
python convertPt2Pte.py model.pt -o output.pte
```

### Batch Conversion
```bash
./batch_convert_to_pte.sh                    # All .ptl in current dir
./batch_convert_to_pte.sh output/fold_1      # All .ptl in specified dir
./batch_convert_to_pte.sh --from-pt          # All .pt in current dir
```

## Training (Auto-generates .pte)
```bash
python runSequence.py --config nnConfig.json
# Produces: model.pt, model.ptl, model.pte
```

## Testing
```bash
python test_executorch_conversion.py
```

## Model Formats

| Format | Purpose | Size | Mobile |
|--------|---------|------|--------|
| .pt    | Training/Testing | ~3 MB | ❌ |
| .ptl   | Legacy Mobile | ~3 MB | ⚠️ |
| .pte   | Modern Mobile (ExecuTorch) | ~3 MB | ✅ |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| ExecuTorch not installed | `pip install executorch` |
| Conversion fails | Try two-step: .pt → .ptl → .pte |
| Model architecture error | Run from nnTraining2 directory |
| Input shape mismatch | Use `--input-shape 1,1,750` |

## Documentation Files
- `EXECUTORCH_EXPORT.md` - Complete user guide
- `IMPLEMENTATION_SUMMARY_EXECUTORCH.md` - Developer notes
- `QUICKREF_EXECUTORCH.md` - This file

## Support
- ExecuTorch docs: https://pytorch.org/executorch/
- Project docs: See README.md

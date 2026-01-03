# TensorFlow Lite Model Conversion

Converts trained neural network models from the nnTrainer toolchain into TensorFlow Lite format for deployment on Android devices.

## Overview

This toolchain provides two scripts for model conversion:

- **`convertToTFLite.py`**: Low-level conversion utility that handles the actual model conversion
- **`convertToTFLiteWrapper.py`**: Integration wrapper that reads configuration and orchestrates conversion

## Supported Models

- **TensorFlow/Keras** (`.keras` files) — Fully supported with quantization options
- **PyTorch** (`.pt` files) — Basic support (recommend converting to ONNX first)

## Quantization Options

Quantization reduces model size and improves inference speed on mobile devices:

### 1. Dynamic Quantization (Recommended for most cases)
- Reduces model size by ~75%
- Maintains accuracy well
- Fast inference on CPU

```bash
python convertToTFLiteWrapper.py --config nnConfig.json --quantize dynamic
```

### 2. Float16 Quantization
- Good balance between size reduction and accuracy
- Suitable when FP16 hardware support is available
- Model size reduction ~50%

```bash
python convertToTFLiteWrapper.py --config nnConfig.json --quantize float16
```

### 3. Integer Quantization (Experimental)
- Maximum size reduction and speed
- Requires representative calibration data
- Most aggressive optimization

```bash
python convertToTFLiteWrapper.py --config nnConfig.json --quantize integer
```

### No Quantization
- Largest model size
- Highest accuracy (floating-point precision)
- Use for development/testing

```bash
python convertToTFLiteWrapper.py --config nnConfig.json
```

## Usage

### Basic Usage

Convert model using configuration file:
```bash
python convertToTFLiteWrapper.py --config nnConfig.json --quantize dynamic
```

### Specify Custom Model

Convert a specific model file:
```bash
python convertToTFLiteWrapper.py --config nnConfig.json --model mymodel.keras --quantize dynamic
```

### Custom Output Path

Save converted model to specific location:
```bash
python convertToTFLiteWrapper.py --config nnConfig.json --output ~/Android/assets/model.tflite --quantize dynamic
```

### Verbose Output

Enable detailed conversion information:
```bash
python convertToTFLiteWrapper.py --config nnConfig.json --quantize dynamic --verbose
```

## Integration into nnTrainer Workflow

Add conversion step after training completes:

```bash
# Train model
python nnTrainer.py --config nnConfig.json

# Convert to TFLite
python convertToTFLiteWrapper.py --config nnConfig.json --quantize dynamic

# Model is now ready at: model_name.tflite
```

## Android Integration

### 1. Add TensorFlow Lite Dependency

In `build.gradle`:
```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.13.0'
}
```

### 2. Copy Model to Assets

Place the `.tflite` file in `src/main/assets/`:
```
MyApp/
  src/main/
    assets/
      model.tflite
    java/
      ...
```

### 3. Load Model in Code

```kotlin
// Load model from assets
val interpreter = Interpreter(loadModelFile("model.tflite"))

// Run inference
val input = Array(1) { FloatArray(inputSize) { ... } }
val output = Array(1) { FloatArray(outputSize) }
interpreter.run(input, output)

private fun loadModelFile(modelName: String): MappedByteBuffer {
    val fileDescriptor = assets.openFd(modelName)
    val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
    val fileChannel = inputStream.channel
    val startOffset = fileDescriptor.startOffset
    val declaredLength = fileDescriptor.declaredLength
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
}
```

## Performance Considerations

### Model Size Comparison

Example reduction from dynamic quantization on a typical seizure detection model:

| Format | Size | Speed (CPU) |
|--------|------|-------------|
| Keras (FP32) | 8.5 MB | Baseline |
| TFLite (FP32) | 8.2 MB | 1.1x faster |
| TFLite (Dynamic INT8) | 2.1 MB | 1.8x faster |
| TFLite (Float16) | 4.3 MB | 1.5x faster |

### Inference Time on Mobile

Approximate inference time for 5-second (125 sample) seizure detection window:

- **Pixel 4 (CPU)**: ~20-50 ms (FP32) → ~10-25 ms (INT8)
- **Pixel 6 (CPU)**: ~15-30 ms (FP32) → ~8-15 ms (INT8)
- **With GPU Acceleration**: 5-10 ms (INT8)

## Troubleshooting

### "TensorFlow is required for Keras model conversion"

Install TensorFlow:
```bash
pip install tensorflow
```

### "Model file not found"

Ensure the model path is correct. Check in configuration file:
```bash
python convertToTFLiteWrapper.py --config nnConfig.json --verbose
```

### Large Model Size After Conversion

- Try different quantization: `--quantize dynamic`
- Check original model architecture (may have unnecessary layers)
- Reduce number of model parameters if possible

### Low Accuracy After Quantization

- Use `--quantize float16` instead of `dynamic`
- Retrain with quantization-aware training (advanced)
- Validate model on real Android device data

## References

- [TensorFlow Lite Guide](https://www.tensorflow.org/lite)
- [TensorFlow Lite Quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)
- [Android ML Kit (alternative)](https://developers.google.com/ml-kit)

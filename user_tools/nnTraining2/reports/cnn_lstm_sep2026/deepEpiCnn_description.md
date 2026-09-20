# Description of the `deepEpiCnn_torch` Model

The `deepEpiCnn_torch` model is a PyTorch re-implementation of the 14-layer one-dimensional convolutional neural network (1D-CNN) originally described by Spahr et al. (2025) for the detection of generalized convulsive seizures (CSs) using wrist-worn accelerometer data. The implementation is provided as part of the OpenSeizureDatabase (OSD) neural network training pipeline (`deepEpiCnnModel_torch.py`) and is designed to be functionally equivalent to the TensorFlow/Keras counterpart (`deepEpiCnnModel.py`) while enabling deployment to mobile and embedded platforms via PyTorch Lite (`.ptl`) and ExecuTorch (`.pte`) export formats.

## Architecture

The layer structure of the model is identical to that described in Spahr et al. (2025), though the regularisation configuration differs (see below). The model accepts a single-channel one-dimensional input representing the amplitude (vector magnitude) of a tri-axial wrist-worn accelerometer sampled at 25 Hz over a 30-second window. This produces an input vector of 750 samples (25 samples/second × 30 seconds), shaped as (batch, 1, 750) for processing.

### Convolutional Feature Extraction

The core of the model is a stack of 14 one-dimensional convolutional layers. A convolutional layer is a mathematical operation that slides a small window — called a *kernel* or *filter* — across the input signal, computing a dot product at each position to detect local patterns. In this model, each filter has a width of 5 samples, meaning it examines 5 consecutive accelerometer readings at a time. The number of filters (sometimes called *channels*) determines how many different patterns the layer can detect simultaneously.

The 14 convolutional layers are arranged in sequence, with the number of filters progressing as follows: the first layer has 16 filters, the next eleven layers each have 32 filters, and the final two layers each have 64 filters. This progression means the network initially detects a small number of simple patterns (e.g., sudden changes in acceleration) and gradually learns to combine these into a larger number of more complex representations.

Each convolutional layer uses a stride of 1 (meaning the filter moves one sample at a time) except for every fifth layer, which uses a stride of 2 (moving two samples at a time). Stride-2 layers effectively halve the temporal resolution of the signal, allowing the network to process progressively larger regions of the input with fewer computations. The network uses *valid* padding (no zero-padding is added at the edges), so the output of each layer is slightly shorter than its input.

After each convolution, the output is normalized using *batch normalization*, a technique that rescales the activations to have zero mean and unit variance within each mini-batch. This stabilises training and allows higher learning rates to be used. The normalised output is then passed through a *rectified linear unit* (ReLU) activation function, which introduces non-linearity into the model by replacing all negative values with zero. Without non-linear activation functions, a stack of convolutional layers would be mathematically equivalent to a single linear transformation and would be unable to learn complex patterns. A *dropout* layer (probability *p* = 0.08) follows each convolutional block, randomly setting 8% of activations to zero during training. This is an additional regularisation measure not present in Spahr et al. (2025), where dropout was applied only to the dense layers.

### Global Average Pooling

After the 14 convolutional layers, the output is a three-dimensional tensor with shape (batch, 64, *T*), where 64 is the number of filters in the final layer and *T* is the remaining temporal length. *Global average pooling* reduces this to a fixed-length vector by computing the arithmetic mean of each filter's output across the entire time dimension. The result is a single 64-dimensional vector (one value per filter) that summarises the presence and strength of each detected pattern across the full 30-second window, regardless of when it occurred. This operation also makes the model robust to small shifts in the timing of seizure-related movements.

### Classification Layers

The 64-dimensional vector from the pooling layer is passed to a sequence of four *fully connected* (also called *dense*) layers. In a fully connected layer, every input element is connected to every output element through a learned weight, allowing the layer to combine information from all detected patterns simultaneously.

The first three dense layers transform the feature vector through a series of non-linear transformations with the following dimensions: 64 → 64 → 32 → 16. Each of these layers is followed by batch normalization (as described above), ReLU activation, and *dropout*. Dropout is a regularisation technique that randomly sets a fraction of the layer's outputs to zero during training (here, *p* = 0.15, meaning 15% of activations are dropped). This prevents the network from becoming overly reliant on any single feature and improves generalisation to unseen data. Dropout is disabled during inference (i.e., when making predictions on new data). The dropout rate of 0.15 is substantially higher than the *p* = 0.025 used by Spahr et al. (2025), reflecting a stronger regularisation regime.

The fourth and final dense layer maps from 16 dimensions to the number of output classes (2 for binary seizure/non-seizure classification). This layer produces *un-normalized logits* — raw numerical scores that are passed to a cross-entropy loss function during training, or to a softmax function during inference to obtain class probabilities.

### Summary of Layer Dimensions

| Stage | Input Shape | Operation | Output Shape |
|---|---|---|---|
| Input | — | 30 s @ 25 Hz accelerometer magnitude | (batch, 1, 750) |
| Conv layers 1–14 | (batch, *C*<sub>in</sub>, *T*) | Conv1D(k=5) → BatchNorm → ReLU → Dropout | (batch, 64, *T*') |
| Global average pooling | (batch, 64, *T*') | Mean across time | (batch, 64) |
| Dense layer 1 | (batch, 64) | Linear → BatchNorm → ReLU → Dropout | (batch, 64) |
| Dense layer 2 | (batch, 64) | Linear → BatchNorm → ReLU → Dropout | (batch, 32) |
| Dense layer 3 | (batch, 32) | Linear → BatchNorm → ReLU → Dropout | (batch, 16) |
| Output layer | (batch, 16) | Linear | (batch, 2) |

### Total Parameters

The complete model comprises approximately 97,362 trainable parameters:

| Component | Parameters | Proportion |
|---|---|---|
| Convolutional layers | 86,048 | 88% |
| Batch normalization | 352 | <1% |
| Dense/linear layers | 10,962 | 11% |
| **Total** | **97,362** | **100%** |

For comparison, the CNN-LSTM model has approximately 281,000 parameters — roughly three times as many. The majority of the CNN-LSTM's parameters reside in its LSTM layers (85%), whereas the `deepEpiCnn` model concentrates 88% of its parameters in the convolutional layers, reflecting the absence of recurrent connections.

## Training Protocol

The training protocol closely follows the methodology described by Spahr et al. (2025) with several notable adaptations:

1. **Optimizer.** The model is trained using the AdamW optimizer (β₁ = 0.9, β₂ = 0.999, weight decay = 0.002). Spahr et al. (2025) used an equivalent optimizer but with weight decay set to 0.0; the nonzero weight decay here provides additional regularisation to reduce overfitting.

2. **Learning rate schedule.** A three-phase step-based learning rate schedule is employed: (i) a linear warmup phase from 1×10⁻⁶ to 1×10⁻⁴ over 2,500 steps; (ii) a cosine-annealing main phase from 1×10⁻⁴ to 3×10⁻⁵ over 45,000 steps; and (iii) a linear cooldown phase to near-zero over 2,500 steps, totalling 50,000 training steps. This schedule is functionally equivalent to the warmup–cosine–cooldown schedule described by Spahr et al.

3. **Batch size and sampling.** A batch size of 512 is used. Balanced batch sampling via a weighted random sampler ensures that seizure and non-seizure classes are represented equally within each mini-batch, addressing the inherent class imbalance in seizure detection datasets. This corresponds to the balanced batch strategy described in the reference paper.

4. **Data augmentation.** Noise augmentation is applied during training (additive Gaussian noise with a configurable factor of 5× at a standard deviation of 20 mG), in addition to the existing data preparation pipeline of the OSD framework. This extends the augmentation approach described by Spahr et al., which relied primarily on balanced sampling.

5. **Model selection.** A multi-criteria checkpoint strategy is employed: the best model is saved when both sensitivity and false alarm rate (FAR) improve simultaneously, or when FAR decreases by more than 5% while sensitivity remains within a tolerance of 0.2 of the best observed value, and sensitivity does not fall below 0.60. This criteria is adapted from the dual-improvement and FAR-reduction heuristics described in the reference paper.

6. **Evaluation frequency.** Model performance is evaluated every 5,000 steps on a held-out validation set, with sensitivity (TP/(TP+FN)) and FAR (FP/(FP+TN)) reported alongside standard cross-entropy loss and classification accuracy.

## Key Differences from Spahr et al. (2025)

While the `deepEpiCnn_torch` model replicates the core architecture and training philosophy of Spahr et al. (2025), several implementation-level differences exist:

| Aspect | Spahr et al. (2025) | `deepEpiCnn_torch` |
|---|---|---|
| **Framework** | TensorFlow/Keras (TFLite for deployment) | PyTorch (with `.ptl` and `.pte` export) |
| **Ensemble** | 30 models trained via cross-validation; 10 best selected for ensemble; predictions aggregated via Harrell-Davis quantile | Single model training; ensemble support available via `ensemblePredictor.py` with identical Harrell-Davis aggregation |
| **Sensitivity tuning** | Post-hoc quantile parameter (*q*) adjusts sensitivity/FAR trade-off without retraining | Same Harrell-Davis quantile mechanism provided for ensemble prediction; single-model operating point fixed at training |
| **Dataset** | 37 patients (54 CSs) for training; 347 patients (49 CSs) for independent testing across 8 European EMUs | Trained on OSD multi-center dataset; patient counts and seizure types configurable via OSDB files |
| **Weight decay** | 0.0 (no explicit L2 regularization) | 0.002 |
| **Conv dropout** | Not applied (dropout only in dense layers) | *p* = 0.08 (added; not in Spahr et al.) |
| **Dense dropout** | *p* = 0.025 | *p* = 0.15 (6× higher than Spahr et al.) |
| **Output activation** | Softmax (2-class) | Raw logits (softmax applied externally via `CrossEntropyLoss` or `torch.softmax` at inference) |
| **Deployment target** | TicWatch Pro 3 (Wear OS, TFLite) | Cross-platform: PyTorch, ONNX, TFLite, ExecuTorch (`.pte`) for Android |

## Key Similarities

The `deepEpiCnn_torch` model retains the following core properties from Spahr et al. (2025):

- Identical 14-layer 1D-CNN architecture with the same filter progression, kernel sizes, stride pattern, batch normalization, and global average pooling.
- Same input representation: accelerometer amplitude (vector magnitude) derived from a tri-axial wrist-worn sensor, sampled at 25 Hz over 30-second non-overlapping windows.
- Same training optimizer (AdamW) and three-phase learning rate schedule (warmup → cosine annealing → cooldown) with equivalent hyperparameters.
- Same balanced batch sampling strategy to address class imbalance.
- Same Harrell-Davis quantile estimator for ensemble aggregation, enabling tunable sensitivity without model retraining.
- Same evaluation metrics: sensitivity, false alarm rate (FAR), and event-level detection latency.

## Summary

The `deepEpiCnn_torch` model constitutes a PyTorch re-implementation of the Spahr et al. (2025) seizure detection architecture, sharing the same 14-layer 1D-CNN layer structure, training protocol, and ensemble-based tunable sensitivity framework. The regularisation regime has been strengthened relative to the reference paper: convolutional dropout (*p* = 0.08) has been added after each convolutional block (absent from Spahr et al.), dense dropout has been increased from *p* = 0.025 to *p* = 0.15, and weight decay of 0.002 has been applied (Spahr et al. used 0.0). These modifications are intended to improve generalisation on the OSD dataset. Additional extensions include framework portability (PyTorch ecosystem) and multi-platform deployment capability via ExecuTorch and PyTorch Lite export, facilitating integration with consumer-grade wearable devices for real-time convulsive seizure detection.

**Reference:** Spahr A, Bernini A, Ducouret P, et al. Deep learning-based detection of generalized convulsive seizures using a wrist-worn accelerometer. *Epilepsia*. 2025;66(S3):53–63. doi:10.1111/epi.18406

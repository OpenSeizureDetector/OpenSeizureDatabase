# Description of the CNN-LSTM Model

The CNN-LSTM model (`cnnLstmModel_torch.py`) is a PyTorch implementation of a hybrid convolutional-recurrent neural network for the detection of generalized convulsive seizures using wrist-worn accelerometer data. The architecture is designed to address the high false alarm rate of pure CNN approaches (such as the `deepEpiCnn` model) by augmenting short-window convolutional feature extraction with long-range temporal modelling via a stacked LSTM. The model is trained and evaluated within the OpenSeizureDatabase (OSD) neural network training pipeline (`runSequence.py`) using the configuration defined in `nnConfig_lstm_1D.json`.

## Architecture

The model consists of three functional stages: (1) a convolutional feature extractor that converts raw accelerometer data into compact feature representations, (2) a recurrent temporal processor that analyses how these features evolve over time, and (3) a classification decision layer that determines whether the observed pattern corresponds to a seizure.

### 1. CNN Feature Extractor

The first stage uses a four-layer one-dimensional convolutional neural network (CNN) to extract features from short segments of the accelerometer signal. Unlike the `deepEpiCnn` model, which processes the entire 30-second window at once, the CNN-LSTM divides the input into 30 non-overlapping 1-second windows (each containing 25 samples at 25 Hz) and processes each window independently through the same CNN.

A convolutional layer slides a small window — called a *kernel* or *filter* — across the input signal, computing a dot product at each position to detect local patterns. In this model, each filter has a width of 5 samples, meaning it examines 5 consecutive accelerometer readings at a time. The number of filters (sometimes called *channels*) determines how many different patterns the layer can detect simultaneously.

The four convolutional layers have the following configuration:

| Layer | Input Channels | Output Channels | Kernel Size | Output Length | Purpose |
|---|---|---|---|---|---|
| 1 | 1 (magnitude) | 16 | 5 | 22 | Detects basic signal shapes |
| 2 | 16 | 32 | 5 | 19 | Combines basic shapes |
| 3 | 32 | 32 | 5 | 16 | Refines representations |
| 4 | 32 | 64 | 5 | 12 | Produces high-level features |

Each convolutional layer is followed by batch normalization (which rescales activations to have zero mean and unit variance, stabilising training), ReLU activation (which replaces negative values with zero, introducing non-linearity so the network can learn complex patterns), and dropout (which randomly sets a fraction of activations to zero during training to prevent overfitting; default *p* = 0.08, meaning 8% of activations are dropped).

After the four convolutional layers, *global average pooling* computes the arithmetic mean of each filter's output across the time dimension, reducing the (batch, 64, 12) tensor to a (batch, 64) vector. This vector is then projected to a 64-dimensional feature vector through a linear (fully connected) layer.

This extractor is applied independently to each of the 30 one-second sub-windows, producing a sequence of 30 feature vectors of dimension 64. This sequence represents the temporal evolution of accelerometer patterns over the full 30-second observation window.

### 2. LSTM Temporal Processor

The second stage uses a *Long Short-Term Memory* (LSTM) network to process the sequence of 30 feature vectors produced by the CNN. An LSTM is a type of recurrent neural network (RNN) that maintains an internal *hidden state* — a running summary of the information it has seen so far — which allows it to capture temporal dependencies across the sequence. Unlike a simple RNN, an LSTM uses *gating mechanisms* (input, forget, and output gates) that control how much new information to incorporate, how much old information to retain, and how much of the hidden state to expose at each time step. This design enables the network to learn long-range dependencies without suffering from the *vanishing gradient* problem that limits simple RNNs.

The model uses a two-layer stacked LSTM:

- **First LSTM layer:** Maps the 64-dimensional CNN feature vectors to a 128-dimensional hidden state. At each of the 30 time steps, this layer reads the corresponding CNN feature vector and updates its hidden state to incorporate the new information.
- **Second LSTM layer:** Takes the 128-dimensional hidden state output from the first layer as its input and produces a new 128-dimensional hidden state. This allows the network to learn more abstract temporal patterns by building on the representations learned by the first layer.

Dropout (*p* = 0.25) is applied between the two LSTM layers to regularize the recurrent connections, reducing the risk that the model becomes overly dependent on specific temporal patterns seen during training.

Only the final hidden state of the sequence (corresponding to the most recent time step, *t* = 30) is retained for classification. This provides a fixed-length 128-dimensional summary that encapsulates the temporal dynamics observed across the entire 30-second input window.

### 3. Classification Layers

The 128-dimensional vector from the LSTM is passed to a sequence of three *fully connected* (also called *dense*) layers. In a fully connected layer, every input element is connected to every output element through a learned weight, allowing the layer to combine information from all temporal features simultaneously.

The three dense layers have the following dimensions: 128 → 64 → 32 → 2. Each of the first three layers is followed by batch normalization (rescaling activations to stabilise training), ReLU activation (replacing negative values with zero to introduce non-linearity), and dropout (*p* = 0.15, meaning 15% of activations are randomly set to zero during training to prevent overfitting). The fourth and final layer maps from 32 dimensions to the two output classes (seizure vs. non-seizure), producing *un-normalized logits* — raw numerical scores that are passed to a cross-entropy loss function during training, or to a softmax function during inference to obtain class probabilities.

### Total Parameters

The complete model comprises approximately 281,000 trainable parameters:

| Component | Parameters | Proportion |
|---|---|---|
| CNN feature extractor | ~20,000 | 7% |
| LSTM layers | ~240,000 | 85% |
| Classification layers | ~21,000 | 8% |
| **Total** | **~281,000** | **100%** |

## Training Protocol

The training protocol shares the Spahr et al. (2025) three-phase learning rate schedule and AdamW optimizer with the `deepEpiCnn` model, but introduces several modifications to accommodate the recurrent architecture and improve temporal discrimination.

1. **Optimizer.** AdamW with β₁ = 0.9, β₂ = 0.999, and weight decay = 0.002 (compared to 0.0001 for the pure CNN).

2. **Learning rate schedule.** A three-phase step-based schedule: linear warmup from 1×10⁻⁶ to 1×10⁻⁴ over 3,750 steps; cosine annealing from 1×10⁻⁴ to 3×10⁻⁵ over 107,500 steps; linear cooldown to near-zero over 3,750 steps, totalling 115,000 training steps. The extended training duration (compared to 50,000 steps for the pure CNN) reflects the slower convergence characteristic of recurrent architectures.

3. **Batch size and sampling.** A batch size of 256 (reduced from 512 to accommodate the greater memory footprint of the LSTM). Balanced batch sampling via a weighted random sampler ensures equal representation of seizure and non-seizure classes. Subtype-aware weighting is additionally applied, assigning a weight of 2.0 to tonic-clonic seizures, 0.4 to aura events, and 1.0 to all other subtypes, reflecting their relative clinical importance.

4. **Class-weighted loss.** A positive class weight multiplier of 2.0 is applied to the cross-entropy loss, penalizing false negatives (missed seizures) more heavily than false positives. This is a departure from the pure CNN training, which does not employ asymmetric class weighting.

5. **Data augmentation.** Noise augmentation is applied selectively to non-seizure events of specific subtypes (e.g., cooking, walking, motor vehicle, typing) at configurable noise levels, rather than uniformly to all non-seizure data. User-based augmentation is also enabled, balancing seizure event counts across participants by duplicating events from under-represented users.

6. **Model selection.** The Youden index (sensitivity − specificity, equivalently TPR − FPR) is used as the primary model selection metric, replacing the dual-improvement or FAR-reduction criteria of the pure CNN. A minimum sensitivity threshold of 0.15 and maximum FPR of 0.15 are enforced.

7. **Evaluation frequency.** Model performance is evaluated every 5,000 steps on a held-out validation set.

## Key Differences from the AMBER Model

The AMBER model (Pordoy, 2024) is a multi-input CNN–Bidirectional LSTM–Multi-Head Attention architecture developed for the same clinical task. While both models share the conceptual framework of combining convolutional feature extraction with recurrent temporal modelling, they differ substantially in architectural design, input representation, and training methodology.

| Aspect | CNN-LSTM (`cnnLstmModel_torch`) | AMBER (`amber/model.py`) |
|---|---|---|
| **Framework** | PyTorch | TensorFlow/Keras |
| **Input representation** | Accelerometer magnitude only (1 channel) | Multi-feature: magnitude, heart rate, and FFT (3 channels) processed in parallel pipelines |
| **Temporal window** | 30 seconds (30 × 1-second sub-windows) | 5 seconds (125 samples at 25 Hz) |
| **CNN depth** | 4 Conv1D layers (16→32→32→64), kernel=5, valid padding | 3 Conv1D blocks (64→128→256), kernel=3, same padding, each with MaxPooling1D(2) |
| **Recurrent architecture** | Unidirectional stacked LSTM (2 layers, 128 hidden units) | Bidirectional LSTM (128 hidden) followed by a unidirectional LSTM (128 hidden) |
| **Sequence length** | 30 timesteps (1-second feature vectors) | Variable (dependent on 5-second input after 3 MaxPooling reductions) |
| **Attention mechanism** | None | Per-feature self-attention + Multi-Head Attention fusion layer (4 heads, key_dim=64) with residual connection |
| **Multi-feature fusion** | N/A (single input) | `EnhancedFusionLayer`: concatenation of attention outputs from all input features, followed by multi-head attention and residual addition |
| **Classification head** | 3-layer MLP (128→64→32→2) with BatchNorm + ReLU + Dropout | Single Dense(128) + BatchNorm + Dropout(0.1) + GlobalMaxPooling + Dense(2, softmax) |
| **Output activation** | Raw logits (softmax applied externally) | Softmax (applied in-model) |
| **Loss function** | Cross-entropy with positive class weighting (×2.0) | Mean squared error |
| **Optimizer** | AdamW (β₁=0.9, β₂=0.999, weight decay=0.002) | RMSprop (lr=1×10⁻⁵) |
| **Learning rate schedule** | Three-phase: warmup → cosine annealing → cooldown (115,000 steps) | ReduceLROnPlateau (factor=0.2, patience=3) |
| **Training duration** | 115,000 steps | 5 epochs |
| **Regularization** | Dropout (CNN: 0.08, LSTM: 0.25, Dense: 0.15) + weight decay | Dropout (0.1) + L2 regularization (0.0001) + EarlyStopping (patience=10) |
| **Model selection** | Youden index (TPR − FPR) | Validation loss (EarlyStopping with restore_best_weights) |
| **Input features** | Single channel (accelerometer magnitude) | Up to 3 features (accelerometer magnitude, heart rate, FFT) |
| **Deployment** | PyTorch (`.pt`), exportable to `.ptl`/`.pte` | TensorFlow/Keras (`.keras`) |

## Key Similarities with the AMBER Model

Despite the architectural differences, the two models share several fundamental design principles:

- **Hybrid CNN-RNN paradigm.** Both models employ a convolutional front-end for local feature extraction followed by a recurrent back-end for temporal modelling, reflecting the consensus that seizure detection from accelerometry benefits from both spatial and temporal processing.
- **CNN feature extraction.** Both use a stack of Conv1D layers with batch normalization and ReLU activation to transform raw accelerometer signals into higher-level feature representations before temporal modelling.
- **LSTM temporal processing.** Both employ LSTM layers to capture long-range temporal dependencies in the feature sequence, though the CNN-LSTM uses a unidirectional architecture while AMBER uses a bidirectional design.
- **Binary classification.** Both models produce a two-class output (seizure vs. non-seizure) with softmax or equivalent probability output.
- **Input modality.** Both use wrist-worn accelerometer data as the primary input signal, though AMBER additionally incorporates heart rate and FFT features.
- **K-fold cross-validation.** Both support k-fold cross-validation for model evaluation (k=5 in AMBER's default configuration).

## Key Similarities with the `deepEpiCnn` Model

The CNN-LSTM model also shares several training protocol elements with the pure CNN (`deepEpiCnn`) model, reflecting a common OSD training philosophy:

- Identical three-phase learning rate schedule (warmup → cosine annealing → cooldown) with the same peak and end learning rates.
- Same AdamW optimizer with identical β₁ and β₂ parameters.
- Same balanced batch sampling strategy.
- Same Harrell-Davis quantile estimator available for ensemble prediction.
- Same evaluation metrics: sensitivity, false alarm rate (FAR), and event-level detection latency.

## Summary

The CNN-LSTM model represents an evolution of the pure CNN seizure detection approach, introducing recurrent temporal modelling to capture the sequential dynamics of convulsive seizures that are poorly represented by single-window convolutional classifiers. While the AMBER model shares this hybrid philosophy, it additionally incorporates multi-feature input, bidirectional recurrence, and multi-head attention fusion, reflecting a more complex architectural design. The CNN-LSTM model, by contrast, prioritizes architectural simplicity and longer temporal context (30 seconds vs. 5 seconds), with the goal of reducing false alarms through temporal pattern recognition rather than attention-based feature fusion. The choice between these approaches depends on the specific clinical deployment requirements, including the availability of heart rate data, computational constraints on the target platform, and the desired balance between sensitivity and false alarm rate.

**References:**

1. Spahr A, Bernini A, Ducouret P, et al. Deep learning-based detection of generalized convulsive seizures using a wrist-worn accelerometer. *Epilepsia*. 2025;66(S3):53–63. doi:10.1111/epi.18406
2. Pordoy J. AMBER: A Multi-Head Attention-Based LSTM Model for Epileptic Seizure Detection. University of West London. 2024. Available at: https://github.com/jpordoy/AMBER/tree/Amber_beta_1.0.1

# Experiments & Workflow: Gravitational Lens Classification

Here, I detail an in-depth analysis of the experimental methodology, technical design decisions, and hyperparameter optimization strategies utilized to develop a high-performance classifier for gravitational lensing substructures.

## 1. Project's Objective

**Primary Goal**: To develop a robust deep learning model capable of distinguishing between three distinct classes of strong gravitational lensing: `no substructure`, `subhalo substructure`, and `vortex substructure`.

**Research Questions & Hypotheses**:
- **H1: Transfer Learning Benefits**: Does leveraging pre-trained ImageNet weights provide a superior starting point for astrophysics-domain images compared to random initialization?
- **H2: Resolution Sensitivity**: Does increasing input resolution from 150x150 to 224x224 allow the model to capture finer "vortex" perturbations, or does it lead to overfitting?
- **H3: Class Imbalance Sensitivity**: How resilient is the AUC metric to potential imbalances in substructure samples, and are class weights necessary for minority class stability?

---

## 2. Baseline Configuration

The "Baseline" serves as our control group. Any deviation is measured against these metrics.

- **Architecture**: **ResNet18** (Modified)
  - Chosen for its depth-to-complexity ratio, making it ideal for the ~37k image dataset size.
- **Input Dimensions**: 150x150 pixels, Grayscale (1 Channel).
- **Optimization Strategy**:
  - **Optimizer**: Adam (Adaptive Moment Estimation) for rapid initial convergence.
  - **Initial LR**: 1e-3 (10^-3).
  - **Batch Size**: 32 (Balanced for gradient stability vs. compute efficiency).
- **Evaluation Protocol**:
  - **Loss Function**: CrossEntropyLoss (with log-softmax).
  - **Data Split**: 80% Training, 20% Validation.

---

## 3. Comprehensive Experiment Summary

| Exp ID | Variation Introduced | Validation AUC | Training Dynamics | Key Insight |
| :--- | :--- | :--- | :--- | :--- |
| **E1** | Baseline (Standard) | 0.962 | Smooth convergence | Effective starting point. |
| **E2** | Learning Rate Decay (1e-4) | **0.975** | Slower, more stable | 1e-4 is the "sweet spot" for generalization. |
| **E3** | Random Initialization | 0.891 | High volatility | Pre-trained feature extractors are critical. |
| **E4** | Upsampling to 224x224 | 0.968 | Slower training (3x) | Negligible gain for the high compute cost. |
| **E5** | Weighted Loss Function | 0.972 | Targeted optimization | Improves "vortex" recall significantly. |

---

## 4. Hyperparameter Exploration & Analysis

### Learning Rate (LR) Optimization
- **Testing Range**: `[1e-3, 1e-4, 5e-5]`
- **Observation**: At `1e-3`, the validation loss exhibited minor "jittering" in later epochs.
- **Conclusion**: Dropping to `1e-4` after an initial warmup allowed the model to find a deeper local minimum without bouncing out of the gradient valley.

### Batch Size Dynamics
- **Testing Range**: `[16, 32, 64, 128]`
- **Result**: `32` and `64` performed similarly. `128` caused a slight drop in validation performance (generalization gap).
- **Decision**: Stuck with `64` for final production to maximize GPU utilization (MPS) without losing accuracy.

---

## 5. Intentional Model Design Choices

### 1-Channel conv1 Adaptation
Standard ResNet18 expects 3-channel RGB. Instead of simply duplicating our grayscale image three times (which increases memory overhead), I modified the `conv1` weight shape. I initialized the new 1-channel kernel using the mean of the original 3-channel weights, preserving the spatial feature extraction knowledge captured from ImageNet.

### Dropout & Regularization
Early experiments showed a trend towards overfitting (Train Acc: 99.5%, Val Acc: 92%). I introduced a small Dropout (p=0.2) before the final `fc` layer to force feature redundancy.

---

## 6. Training Behavior & Diagnostics

### Convergence Profile
- **Epochs 1-5**: Rapid loss reduction (Feature learning).
- **Epochs 8-12**: Accuracy plateaued; validation loss stabilized (Convergence).
- **Post-Epoch 15**: Divergence began (Overfitting); Early Stopping was triggered to keep the best version.

### Diagnostic Metrics
- **Final Train Accuracy**: ~98.2%
- **Final Validation Accuracy**: ~94.1%
- **ROC Curve Analysis**: The curves are exceptionally convex, especially for the "No Substructure" class, indicating excellent zero-false-positive thresholds.

---

## 7. Failure Analysis (The "Why")

Despite high AUC, the model occasionally confuses **Subhalo** and **Vortex** patterns.

**Root Causes**:
1. **Geometric Similarity**: At low signal-to-noise ratios, the "kink" in a lensing arc caused by a vortex can look remarkably similar to a spherical mass perturbation.
2. **Normalisation Artifacts**: Extreme values in min-max normalization on certain edge-case images may wash out subtle substructure contrast.

**Proposed Mitigation**:
- Implement **data augmentation** specifically for rotation and slight perspective shifts to make the model invariant to the orientation of the substructure.

---

## 8. Final Insights & Learnings

1. **Domain Transfer**: Even though ImageNet contains "real-world" objects and this is astrophysics, the low-level edge detectors (Gabor filters) learned by a pre-trained model are highly transferrable.
2. **Compute Efficiency**: Increasing resolution doesn't always help. For this specific morphology, 150x150 captures 99% of the relevant spatial information.
3. **Class Weighting**: Essential for high-stakes scientific classification to ensure that rare substructures (the "signal") aren't ignored by the majority class ("noise").

# 4. Methodology

## 4.1 PatchCore Architecture

We adopt **PatchCore** [1] as our anomaly detection model due to its state-of-the-art performance and suitability for federated adaptation. The architecture consists of three main components:

### Feature Extraction

A pre-trained **WideResNet-50-2** backbone extracts multi-scale features:
- Layer 2: [512, H/4, W/4] - local features
- Layer 3: [1024, H/8, W/8] - mid-level features

Features are concatenated after upsampling Layer 2 to match Layer 3 dimensions, yielding patch embeddings of dimension 1536.

### Memory Bank

The memory bank stores representative normal patch features using **greedy coreset selection**:

1. Initialize with a random patch
2. Iteratively add the patch maximizing minimum distance to selected set
3. Continue until target size (10% of total patches)

This selection ensures diverse coverage of the normal feature space while maintaining computational efficiency.

### Anomaly Scoring

During inference, anomaly scores are computed as:
$$s(x, p) = \min_{m \in M} \|f(x, p) - m\|_2$$

where $f(x, p)$ is the feature at patch position $p$ in image $x$, and $M$ is the memory bank. Scores are upsampled to pixel resolution for localization.

**Table 2: Model Hyperparameters**

| Parameter | Value |
|-----------|-------|
| Backbone | WideResNet-50-2 |
| Feature layers | Layer 2 + Layer 3 |
| Feature dimension | 1536 |
| Coreset percentage | 10% |
| Neighborhood size | 3 |

## 4.2 Federated Learning Adaptation

### Federated Memory Bank Aggregation

Unlike traditional FL that aggregates gradient updates, our approach aggregates **memory banks** directly:

```
Algorithm: Federated PatchCore
1. Server initializes empty global memory bank
2. For each client k in parallel:
   a. Extract features from local training data
   b. Build local coreset (10% of local patches)
   c. Send local coreset to server
3. Server aggregates:
   a. Concatenate all local coresets
   b. Apply global coreset selection
   c. Build global memory bank
4. Broadcast global memory bank to clients
```

### Aggregation Strategy

We use **weighted federated coreset** aggregation:
1. Weight client contributions by local dataset size (fairness)
2. Oversample from each client (2× target allocation)
3. Apply global greedy coreset selection for diversity

This ensures balanced representation across clients while maintaining efficient memory bank size.

## 4.3 Evaluation Metrics

### AUC-sPRO (Localization)

The **saturated Per-Region Overlap (sPRO)** metric measures pixel-level localization accuracy with saturation to prevent over-crediting large detections:

$$\text{sPRO}(d) = \min\left(\frac{\text{TP}_d}{\text{Sat}_d}, 1.0\right)$$

We compute **AUC-sPRO** at multiple FPR limits: 0.01, 0.05, 0.1, 0.3, 1.0.

### AUC-ROC (Classification)

Image-level anomaly classification using maximum anomaly score:
$$\text{image\_score}(x) = \max_{p} s(x, p)$$

ROC curve computed over good vs anomalous test images.

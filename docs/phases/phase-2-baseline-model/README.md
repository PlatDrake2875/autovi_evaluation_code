# Phase 2: Baseline Model (PatchCore)

> **Objective**: Implement and train a centralized PatchCore model as the baseline for comparison with federated approaches.

---

## Overview

PatchCore is a state-of-the-art anomaly detection method that uses:
1. **Pre-trained CNN backbone** (WideResNet-50-2) for feature extraction
2. **Memory bank** of representative normal patch features
3. **Nearest neighbor** scoring for anomaly detection

---

## PatchCore Architecture

```mermaid
flowchart TB
    subgraph Input["Input"]
        I1["Image\n[3, H, W]"]
    end

    subgraph Backbone["Feature Extractor (Frozen)"]
        B1["WideResNet-50-2\n(ImageNet pretrained)"]
        B1 --> B2["Layer 2 Features\n[512, H/4, W/4]"]
        B1 --> B3["Layer 3 Features\n[1024, H/8, W/8]"]
    end

    subgraph LocalAware["Locally-Aware Features"]
        B2 --> L1["Upsample to\nH/8 x W/8"]
        L1 --> L2["Concatenate"]
        B3 --> L2
        L2 --> L3["Local Neighborhood\nAggregation"]
        L3 --> L4["Patch Features\n[1536, H/8, W/8]"]
    end

    subgraph MemoryBank["Memory Bank"]
        M1["All Training\nPatch Features"]
        M1 --> M2["Coreset\nSubsampling"]
        M2 --> M3["Memory Bank\n(10% of patches)"]
    end

    subgraph Inference["Anomaly Scoring"]
        L4 --> S1["For Each\nTest Patch"]
        M3 --> S1
        S1 --> S2["Compute Distance\nto Nearest Neighbor"]
        S2 --> S3["Anomaly Map\n[H/8, W/8]"]
        S3 --> S4["Upsample to\nOriginal Size"]
        S4 --> S5["Pixel-wise\nAnomaly Scores"]
    end

    Input --> Backbone
    LocalAware --> MemoryBank
    LocalAware --> Inference

    style Input fill:#e1f5fe
    style Backbone fill:#e8f5e9
    style LocalAware fill:#fff3e0
    style MemoryBank fill:#fce4ec
    style Inference fill:#f3e5f5
```

---

## Training Workflow

```mermaid
flowchart TB
    subgraph DataLoad["1. Data Loading"]
        D1["Load Training Images\n(good only)"]
        D1 --> D2["Apply Transforms:\nResize, Normalize"]
    end

    subgraph FeatureExtract["2. Feature Extraction"]
        D2 --> F1["Load WideResNet-50-2\n(frozen weights)"]
        F1 --> F2["For Each Training Image"]
        F2 --> F3["Extract Layer 2+3 Features"]
        F3 --> F4["Reshape to Patch Grid"]
        F4 --> F5{"More\nImages?"}
        F5 -->|Yes| F2
        F5 -->|No| F6["Collect All Patch Features"]
    end

    subgraph Coreset["3. Coreset Selection"]
        F6 --> C1["Initialize: Select Random Patch"]
        C1 --> C2["Compute Distance to\nAll Remaining Patches"]
        C2 --> C3["Select Furthest Patch\n(maximize coverage)"]
        C3 --> C4{"Target Size\nReached?"}
        C4 -->|No| C2
        C4 -->|Yes| C5["Memory Bank Complete"]
    end

    subgraph Save["4. Save Model"]
        C5 --> S1["Save Memory Bank\n(numpy array)"]
        S1 --> S2["Save Metadata\n(config, stats)"]
    end

    style DataLoad fill:#e1f5fe
    style FeatureExtract fill:#e8f5e9
    style Coreset fill:#fff3e0
    style Save fill:#fce4ec
```

---

## Implementation Details

### Backbone: WideResNet-50-2

```python
import torchvision.models as models

backbone = models.wide_resnet50_2(pretrained=True)
backbone.eval()  # Freeze weights

# Hook layers for feature extraction
features = {}
def hook_layer2(module, input, output):
    features['layer2'] = output
def hook_layer3(module, input, output):
    features['layer3'] = output

backbone.layer2.register_forward_hook(hook_layer2)
backbone.layer3.register_forward_hook(hook_layer3)
```

### Feature Dimensions

| Layer | Output Shape | Receptive Field |
|-------|--------------|-----------------|
| Layer 2 | [B, 512, H/4, W/4] | Local |
| Layer 3 | [B, 1024, H/8, W/8] | Mid-level |
| Concatenated | [B, 1536, H/8, W/8] | Combined |

### Coreset Subsampling

Greedy k-center algorithm:
1. Start with random patch
2. Iteratively add patch furthest from current set
3. Continue until target size (10% of total patches)

```python
def greedy_coreset(features, target_percentage=0.1):
    n_samples = features.shape[0]
    target_size = int(n_samples * target_percentage)

    # Initialize with random sample
    selected = [np.random.randint(n_samples)]
    min_distances = np.full(n_samples, np.inf)

    for _ in range(target_size - 1):
        # Update distances
        last_selected = features[selected[-1]]
        distances = np.linalg.norm(features - last_selected, axis=1)
        min_distances = np.minimum(min_distances, distances)
        min_distances[selected] = -1  # Exclude already selected

        # Select furthest
        selected.append(np.argmax(min_distances))

    return features[selected]
```

---

## Hyperparameters

```yaml
# experiments/configs/baseline/patchcore_config.yaml
model:
  backbone: "wide_resnet50_2"
  layers: ["layer2", "layer3"]
  coreset_percentage: 0.1
  neighborhood_size: 3

preprocessing:
  resize_small: [400, 400]   # engine_wiring, pipe_clip, pipe_staple
  resize_large: [1000, 750]  # tank_screw, underbody_*
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std: [0.229, 0.224, 0.225]

inference:
  anomaly_threshold: null    # Computed from validation
  num_neighbors: 9           # For local aggregation
```

---

## Training Script

```bash
python experiments/scripts/train_centralized.py \
    --config experiments/configs/baseline/patchcore_config.yaml \
    --data_dir /path/to/autovi \
    --output_dir outputs/baseline/ \
    --objects all
```

---

## Expected Outputs

```
outputs/baseline/
├── models/
│   ├── patchcore_engine_wiring.pt      # Per-object memory banks
│   ├── patchcore_pipe_clip.pt
│   ├── patchcore_pipe_staple.pt
│   ├── patchcore_tank_screw.pt
│   ├── patchcore_underbody_pipes.pt
│   └── patchcore_underbody_screw.pt
├── features/
│   └── feature_cache.npz               # Cached features (optional)
└── logs/
    └── training.log
```

---

## Performance Expectations

Based on PatchCore literature, expected centralized performance:

| Object | AUC-sPRO@0.05 | AUC-ROC |
|--------|---------------|---------|
| engine_wiring | ~0.85 | ~0.95 |
| pipe_clip | ~0.82 | ~0.92 |
| pipe_staple | ~0.80 | ~0.90 |
| tank_screw | ~0.78 | ~0.88 |
| underbody_pipes | ~0.75 | ~0.85 |
| underbody_screw | ~0.70 | ~0.80 |
| **Mean** | **~0.78** | **~0.88** |

---

## Implementation Checklist

- [ ] Implement `PatchCore` class in `src/models/patchcore.py`
- [ ] Implement feature extractor in `src/models/backbone.py`
- [ ] Implement memory bank in `src/models/memory_bank.py`
- [ ] Implement coreset selection algorithm
- [ ] Create centralized training script
- [ ] Create notebook `notebooks/02_baseline_training.ipynb`
- [ ] Train on all 6 object categories
- [ ] Validate against expected performance

---

## FAISS Optimization

For efficient nearest neighbor search:

```python
import faiss

# Build index
index = faiss.IndexFlatL2(feature_dim)
index.add(memory_bank)

# Query (during inference)
distances, indices = index.search(query_features, k=1)
anomaly_scores = distances.squeeze()
```

---

## Related Documentation

- [PatchCore Training Workflow](workflows/training-workflow.md) - Detailed BPMN
- [Phase 1: Data Preparation](../phase-1-data-preparation/README.md) - Previous phase
- [Phase 3: Federated Setup](../phase-3-federated-setup/README.md) - Next phase

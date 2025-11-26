# PatchCore Training Workflow

> Detailed BPMN diagram for centralized PatchCore model training.

---

## Complete Training Pipeline

```mermaid
flowchart TB
    START([Start Training]) --> CONFIG

    subgraph CONFIG["Configuration"]
        C1["Load Config YAML"] --> C2["Set Random Seeds"]
        C2 --> C3["Initialize Logging"]
    end

    subgraph DATALOAD["Data Loading"]
        C3 --> D1["Create AutoVIDataset\nfor Object Category"]
        D1 --> D2["Filter: train/good only"]
        D2 --> D3["Apply Transforms:\nResize + Normalize"]
        D3 --> D4["Create DataLoader\n(batch_size=32)"]
    end

    subgraph BACKBONE["Backbone Setup"]
        D4 --> B1["Load WideResNet-50-2\n(pretrained=True)"]
        B1 --> B2["Set eval() mode\n(freeze weights)"]
        B2 --> B3["Register Forward Hooks\non Layer2 + Layer3"]
        B3 --> B4["Move to GPU\n(if available)"]
    end

    subgraph EXTRACT["Feature Extraction Loop"]
        B4 --> E1["Initialize Feature List"]
        E1 --> E2["For Each Batch\nin DataLoader"]
        E2 --> E3["Forward Pass\nthrough Backbone"]
        E3 --> E4["Extract Hooked\nFeatures"]
        E4 --> E5["Upsample Layer2\nto Layer3 Size"]
        E5 --> E6["Concatenate\nLayer2 + Layer3"]
        E6 --> E7["Apply Local\nNeighborhood Averaging"]
        E7 --> E8["Reshape:\n[B,C,H,W] -> [B*H*W, C]"]
        E8 --> E9["Append to\nFeature List"]
        E9 --> E10{"More\nBatches?"}
        E10 -->|Yes| E2
        E10 -->|No| E11["Concatenate All\nPatch Features"]
    end

    subgraph CORESET["Coreset Selection"]
        E11 --> CS1["Log: Total patches = N"]
        CS1 --> CS2["Target size =\nN * coreset_percentage"]
        CS2 --> CS3["Initialize:\nSelect random patch"]
        CS3 --> CS4["Compute distances\nto all patches"]
        CS4 --> CS5["Update min_distances"]
        CS5 --> CS6["Select patch with\nmax min_distance"]
        CS6 --> CS7{"Target\nsize?"}
        CS7 -->|No| CS4
        CS7 -->|Yes| CS8["Memory Bank\nComplete"]
    end

    subgraph SAVE["Save Artifacts"]
        CS8 --> S1["Save Memory Bank\nas .pt file"]
        S1 --> S2["Save Config\nMetadata"]
        S2 --> S3["Log Statistics:\n- n_patches\n- memory_size\n- feature_dim"]
    end

    S3 --> END([Training Complete])

    style CONFIG fill:#e1f5fe
    style DATALOAD fill:#e8f5e9
    style BACKBONE fill:#fff3e0
    style EXTRACT fill:#fce4ec
    style CORESET fill:#f3e5f5
    style SAVE fill:#e0f7fa
```

---

## Multi-Object Training Loop

```mermaid
flowchart TB
    START([Start]) --> L1["Load Object List:\nengine_wiring, pipe_clip,\npipe_staple, tank_screw,\nunderbody_pipes, underbody_screw"]

    L1 --> L2["For Each Object\nin List"]
    L2 --> L3["Run Training Pipeline\n(see above)"]
    L3 --> L4["Save Model:\npatchcore_{object}.pt"]
    L4 --> L5{"More\nObjects?"}
    L5 -->|Yes| L2
    L5 -->|No| L6["Generate Summary\nReport"]
    L6 --> END([End])
```

---

## Coreset Algorithm Detail

```mermaid
flowchart LR
    subgraph Init["Initialization"]
        I1["All patches P"] --> I2["Select random p0"]
        I2 --> I3["S = {p0}"]
    end

    subgraph Iterate["Iteration k"]
        IT1["For each p in P\\S"] --> IT2["d(p) = min dist to S"]
        IT2 --> IT3["Select p* = argmax d(p)"]
        IT3 --> IT4["S = S ∪ {p*}"]
    end

    subgraph Stop["Stopping"]
        ST1{"|S| = target?"} -->|Yes| ST2["Return S"]
        ST1 -->|No| IT1
    end

    Init --> Iterate
    Iterate --> Stop

    style Init fill:#e1f5fe
    style Iterate fill:#e8f5e9
    style Stop fill:#fce4ec
```

---

## Feature Extraction Detail

```mermaid
sequenceDiagram
    participant DL as DataLoader
    participant BB as Backbone
    participant H as Hooks
    participant FE as FeatureExtractor

    loop For each batch
        DL->>BB: Input images [B, 3, H, W]
        BB->>H: Layer2 output [B, 512, H/4, W/4]
        BB->>H: Layer3 output [B, 1024, H/8, W/8]
        H->>FE: Collected features
        FE->>FE: Upsample Layer2 to H/8
        FE->>FE: Concat [B, 1536, H/8, W/8]
        FE->>FE: Local averaging
        FE->>FE: Reshape to patches
    end

    FE->>FE: Stack all patches [N, 1536]
```

---

## Memory Requirements

| Component | Size (approx) |
|-----------|---------------|
| WideResNet-50-2 | ~270 MB |
| All patch features (1 object) | ~500 MB - 2 GB |
| Memory bank (10% coreset) | ~50 MB - 200 MB |

---

## Performance Optimization Tips

1. **Batch processing**: Use largest batch size that fits in GPU memory
2. **Feature caching**: Save extracted features to disk for reuse
3. **FAISS indexing**: Use GPU-accelerated FAISS for large memory banks
4. **Mixed precision**: Use fp16 for feature extraction (not storage)

```python
# Example: Feature caching
if os.path.exists(cache_path):
    features = np.load(cache_path)
else:
    features = extract_features(dataloader, backbone)
    np.save(cache_path, features)
```

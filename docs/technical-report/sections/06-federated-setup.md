# 6. Federated Setup - Stage 1

## 6.1 Client Configuration

For Stage 1, we configure **6 federated clients** representing different production lines/inspection stations in an automotive manufacturing facility. Each client is assigned one product category and trains independently **without aggregation**.

### Stage 1: Category-Based Independent Training

| Client | Product Category | Train Images | Test Images | Role |
|--------|-----------------|--------------|-------------|------|
| Client 1 | engine_wiring | 285 | 607 | Engine Assembly |
| Client 2 | pipe_clip | 195 | 337 | Clip Inspection |
| Client 3 | pipe_staple | 191 | 305 | Fastener Station |
| Client 4 | tank_screw | 318 | 413 | Fuel Tank Assembly |
| Client 5 | underbody_pipes | 161 | 345 | Underbody Line |
| Client 6 | underbody_screw | 373 | 392 | Underbody Fastening |
| **Total** | **6 categories** | **1,523** | **2,399** | - |

Each client operates independently in Stage 1, establishing category-specific baseline performance. This reflects real industrial scenarios where different facilities specialize in different components and cannot share proprietary data.

## 6.2 Stage 1 Local Training Protocol

```
Stage 1: Independent Local Training (No Aggregation)
├── Client 1-6: Load assigned category data (good images only)
├── Client 1-6: Extract features from local training data
│   └── Use shared pre-trained WideResNet-50-2 backbone
├── Client 1-6: Build local memory bank
│   ├── Extract patches from all training images
│   ├── Apply greedy coreset selection (10% of patches)
│   └── Store local memory bank (category-specific)
├── Client 1-6: Evaluate on local test set
│   ├── Compute AUC-sPRO @ multiple FPR thresholds
│   └── Compute AUC-ROC for image-level classification
└── Collect baseline performance metrics per client
```

**Stage 1 Scope**: Each client trains independently on their assigned category without communication or aggregation. This establishes baseline accuracy for each product type and identifies category-specific challenges before introducing federated mechanisms in Stage 2.

## 6.3 Stage 2 Preview: Federated Aggregation

Future work (Stage 2) will introduce memory bank aggregation:

```
Stage 2: Federated Memory Bank Aggregation
├── Round 1: Clients send local coresets to server
├── Round 2: Server aggregates
│   ├── Concatenate all local coresets
│   ├── Apply global greedy coreset selection
│   └── Build global memory bank
└── Round 3: Server broadcasts global memory bank
    └── Clients can evaluate using aggregated model
```

This approach requires only **one communication round** (highly efficient vs. gradient-based FL).

## 6.4 Privacy Considerations

While Stage 1 does not include formal privacy guarantees, the architecture enables future privacy enhancement:

1. **Architecture enabler**: Foundation for feature-level privacy mechanisms
2. **Memory bank abstraction**: Future DP-SGD can add noise at aggregation layer
3. **Preparation for Stage 2**: Privacy constraints will be introduced systematically

Stage 2 will introduce **Differential Privacy (DP-SGD)** with formal privacy guarantees.

## 6.5 Implementation Status

**Completed**:
- Data loader for all 6 categories ✓
- Client configuration with category assignment ✓

**In Progress**:
- PatchCore baseline model implementation

**Planned (Stage 2)**:
- Aggregation mechanism for federated setup
- Privacy-preserving feature sharing (DP-SGD)
- Fairness mechanisms for imbalanced clients

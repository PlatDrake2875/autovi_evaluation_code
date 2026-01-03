# 6. Federated Setup

## 6.1 Client Configuration

We simulate **5 federated clients** representing different production lines in an automotive manufacturing facility. Each client holds a local data partition and trains independently before sharing memory bank updates with the central server.

### IID Configuration

For the IID baseline, training data is randomly distributed:

| Client | Categories | Images | Proportion |
|--------|------------|--------|------------|
| Client 1-5 | All (uniform) | ~305 each | 20% each |

### Category-Based Configuration (Non-IID)

For realistic simulation, categories are assigned to specific clients:

| Client | Role | Categories | Images |
|--------|------|------------|--------|
| Client 1 | Engine Assembly | engine_wiring | 285 |
| Client 2 | Underbody Line | underbody_pipes, underbody_screw | 534 |
| Client 3 | Fastener Station | tank_screw, pipe_staple | 509 |
| Client 4 | Clip Inspection | pipe_clip | 195 |
| Client 5 | Quality Control | All (10% each) | ~150 |

This distribution reflects real industrial scenarios where different facilities specialize in different components.

## 6.2 Federated Training Protocol

```
Round 1: Local Feature Extraction
├── Server broadcasts backbone weights (shared)
├── Each client extracts features from local data
├── Each client builds local coreset (10%)
└── Clients send local coresets to server

Round 2: Server Aggregation
├── Server concatenates all local coresets
├── Server applies global coreset selection
└── Server builds global memory bank

Round 3: Distribution
└── Server broadcasts global memory bank to all clients
```

**Key Observation**: PatchCore-based FL requires only **one round** of communication (memory bank exchange), unlike gradient-based methods that require multiple rounds.

## 6.3 Communication Analysis

| Metric | IID | Category-based |
|--------|-----|----------------|
| Local coreset size (avg) | ~50 MB | ~50 MB |
| Total upload | ~250 MB | ~250 MB |
| Global memory download | ~200 MB | ~200 MB |
| Total communication | ~450 MB | ~450 MB |
| Rounds required | 1 | 1 |

The memory bank aggregation approach is highly **communication-efficient** compared to gradient-based FL, which may require hundreds of rounds.

## 6.4 Privacy Considerations

While our Stage 1 implementation does not include formal privacy guarantees, the architecture provides inherent privacy benefits:

1. **Raw data never leaves clients**: Only aggregated feature statistics are shared
2. **Memory bank abstraction**: Individual images cannot be reconstructed from memory bank features
3. **Feature anonymization**: Pre-trained backbone features are less identifiable than raw pixels

Stage 2 will enhance privacy through Differential Privacy (DP-SGD) integration.

## 6.5 Implementation Details

We implement federated training using custom simulation (alternatively compatible with Flower framework):

```python
class FederatedPatchCore:
    def __init__(self, num_clients=5):
        self.clients = []
        self.global_memory = None

    def train_round(self):
        # Collect local coresets
        local_coresets = [c.build_coreset() for c in self.clients]

        # Aggregate with weighted coreset selection
        self.global_memory = aggregate_coresets(
            local_coresets,
            weights=[c.data_size for c in self.clients]
        )

        return self.global_memory
```

The framework supports both IID and category-based partitioning through configuration files.

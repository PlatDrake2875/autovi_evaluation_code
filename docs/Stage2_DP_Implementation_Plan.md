# Stage 2 Implementation Plan: Differential Privacy for Federated PatchCore

## Overview
Add Differential Privacy (DP) to the federated PatchCore anomaly detection system using Gaussian noise mechanism on embeddings.

## Mathematical Foundation

**Gaussian Mechanism**: For embeddings with L2 sensitivity Δ₂, add noise:
```
σ = Δ₂ × √(2 × ln(1.25/δ)) / ε
```

**Process**:
1. **Clip** embeddings to L2 norm ≤ C (bounds sensitivity to C)
2. **Add** Gaussian noise N(0, σ²I) to each embedding
3. **Track** privacy budget via composition theorem

**Privacy Budgets**:
| ε | δ | σ (C=1.0) | Expected Utility Loss |
|---|---|-----------|----------------------|
| 1.0 | 1e-5 | 4.83 | High (15-25% AUROC drop) |
| 5.0 | 1e-5 | 0.97 | Medium (5-10% drop) |
| 10.0 | 1e-5 | 0.48 | Low (2-5% drop) |

---

## Implementation Steps

### Step 1: Create Privacy Module

**Create directory**: `src/privacy/`

**Files to create**:

#### `src/privacy/__init__.py`
```python
from .gaussian_mechanism import GaussianMechanism
from .embedding_sanitizer import EmbeddingSanitizer, DPConfig
from .privacy_accountant import PrivacyAccountant
```

#### `src/privacy/gaussian_mechanism.py`
- Class `GaussianMechanism`
- Computes σ from (ε, δ, sensitivity)
- `add_noise(data, seed)` method

#### `src/privacy/embedding_sanitizer.py`
- Dataclass `DPConfig(enabled, epsilon, delta, clipping_norm)`
- Class `EmbeddingSanitizer`
- `clip_embeddings(embeddings)` - L2 norm clipping
- `sanitize(embeddings, seed)` - clip + noise
- `get_privacy_spent()` - returns (ε, δ)

#### `src/privacy/privacy_accountant.py`
- Class `PrivacyAccountant`
- `record_expenditure(epsilon, delta, round_num)`
- `get_total_privacy()` - uses advanced composition
- `get_report()` - comprehensive stats

---

### Step 2: Modify PatchCoreClient

**File**: `src/federated/client.py`

**Changes**:

1. **Add import** (line 7):
```python
from src.privacy import EmbeddingSanitizer, DPConfig
```

2. **Modify `__init__`** (add parameter and initialization):
```python
def __init__(
    self,
    client_id: int,
    ...
    dp_config: Optional[DPConfig] = None,  # NEW
):
    ...
    # NEW: Initialize DP sanitizer
    self.dp_config = dp_config or DPConfig(enabled=False)
    self.sanitizer = EmbeddingSanitizer(self.dp_config) if self.dp_config.enabled else None
    self.stats["dp_enabled"] = self.dp_config.enabled
```

3. **Modify `build_local_coreset`** (after line 159, before updating stats):
```python
# NEW: Apply DP if enabled
if self.sanitizer is not None:
    sanitizer_seed = seed + self.client_id + 1000
    self.local_coreset = self.sanitizer.sanitize(self.local_coreset, seed=sanitizer_seed)
    self.stats["dp_stats"] = self.sanitizer.get_stats()
```

4. **Add new method** (after line 201):
```python
def get_privacy_spent(self) -> Tuple[float, float]:
    if self.sanitizer is None:
        return (0.0, 0.0)
    return self.sanitizer.get_privacy_spent()
```

---

### Step 3: Modify FederatedServer

**File**: `src/federated/server.py`

**Changes**:

1. **Add import**:
```python
from src.privacy import PrivacyAccountant
```

2. **Modify `__init__`** (add parameters):
```python
def __init__(
    self,
    ...
    track_privacy: bool = False,  # NEW
    target_epsilon: Optional[float] = None,  # NEW
):
    ...
    self.privacy_accountant = PrivacyAccountant(target_epsilon) if track_privacy else None
```

3. **Modify `receive_client_coresets`** (add privacy tracking):
```python
def receive_client_coresets(self, ..., round_num: int = 1):
    ...
    # NEW: Record privacy if tracking
    if self.privacy_accountant and client_stats:
        for stats in client_stats:
            if stats.get("dp_enabled"):
                self.privacy_accountant.record_expenditure(
                    stats.get("dp_epsilon", 0), stats.get("dp_delta", 0), round_num
                )
```

4. **Add new method**:
```python
def get_privacy_report(self) -> Optional[dict]:
    return self.privacy_accountant.get_report() if self.privacy_accountant else None
```

---

### Step 4: Modify FederatedPatchCore Orchestrator

**File**: `src/federated/federated_patchcore.py`

**Changes**:

1. **Add import**:
```python
from src.privacy import DPConfig
```

2. **Modify `__init__`** (add DP parameters):
```python
def __init__(
    self,
    ...
    dp_enabled: bool = False,  # NEW
    dp_epsilon: float = 1.0,   # NEW
    dp_delta: float = 1e-5,    # NEW
    dp_clipping_norm: float = 1.0,  # NEW
):
    self.dp_config = DPConfig(
        enabled=dp_enabled, epsilon=dp_epsilon,
        delta=dp_delta, clipping_norm=dp_clipping_norm
    )

    # Pass dp_config to each client
    for i in range(num_clients):
        client = PatchCoreClient(..., dp_config=self.dp_config)

    # Enable privacy tracking on server
    self.server = FederatedServer(..., track_privacy=dp_enabled)
```

3. **Modify `save`** method to include privacy report

---

### Step 5: Update Configuration

**File**: `experiments/configs/federated/fedavg_iid_dp_config.yaml` (NEW)

```yaml
federated:
  num_clients: 5
  partitioning: "iid"
  num_rounds: 1

privacy:
  enabled: true
  epsilon: 5.0        # Medium privacy
  delta: 1e-5
  clipping_norm: 1.0

model:
  backbone: "wide_resnet50_2"
  layers: ["layer2", "layer3"]
  coreset_ratio: 0.1
  ...

output:
  dir: "outputs/federated/iid_dp_eps5"
```

---

### Step 6: Update Training Script

**File**: `scripts/train_federated.py`

**Changes**:
- Load privacy config section
- Pass DP parameters to FederatedPatchCore
- Save privacy report after training

---

## Files Summary

| File | Action | Key Changes |
|------|--------|-------------|
| `src/privacy/__init__.py` | Create | Module exports |
| `src/privacy/gaussian_mechanism.py` | Create | Noise calibration |
| `src/privacy/embedding_sanitizer.py` | Create | Clip + noise |
| `src/privacy/privacy_accountant.py` | Create | Budget tracking |
| `src/federated/client.py` | Modify | Add `dp_config`, sanitize in `build_local_coreset` |
| `src/federated/server.py` | Modify | Add `PrivacyAccountant` |
| `src/federated/federated_patchcore.py` | Modify | Add DP params, pass to clients/server |
| `experiments/configs/federated/fedavg_iid_dp_config.yaml` | Create | DP-enabled config |
| `scripts/train_federated.py` | Modify | Load DP config |

---

## Testing Plan

1. **Unit tests** for each privacy module class
2. **Integration test**: Client with DP produces different embeddings
3. **Statistical test**: Verify noise follows Gaussian distribution
4. **End-to-end**: Train with ε=5, compare AUROC to non-DP baseline

---

## Expected Outcomes

After implementation:
- Run `python scripts/train_federated.py --config experiments/configs/federated/fedavg_iid_dp_config.yaml`
- Generates `privacy_report.json` with budget accounting
- Embeddings sanitized before aggregation
- Quantifiable privacy-utility trade-off

---

## Next Stage 2 Features (After DP)

1. **Fairness Enhancement**: Cross-category equity and client contribution balancing
2. **Centralized Baseline**: Comparison experiments
3. **Analysis & Reporting**: Trade-off visualizations, statistical significance

# Branch: AIT-9/robustness-core

## Overview
This branch implements the core robustness module for federated learning, including robust aggregation methods, client anomaly detection, and attack simulations for testing.

## Tickets to Complete

### AIT-9: Create robustness module structure and RobustnessConfig
**Status:** To Do
**Priority:** Start here - foundation for other tickets

Create the `src/robustness/` module with:
- `__init__.py`
- `config.py` - RobustnessConfig dataclass with: enabled, aggregation_method, num_byzantine, trim_fraction, client_scoring_method, zscore_threshold
- `aggregators.py` (empty placeholder)
- `client_scoring.py` (empty placeholder)
- `attacks.py` (empty placeholder)

**Reference:** `src/privacy/embedding_sanitizer.py` for config pattern

---

### AIT-10: Implement Coordinate Median robust aggregator
**Status:** To Do
**Depends on:** AIT-9

Implement `CoordinateMedianAggregator` in `src/robustness/aggregators.py`:
- Sample fixed number of vectors from each client coreset
- Compute median across clients for each dimension
- Robust to up to 50% malicious clients
- Return aggregated result + stats dict

**Tests:** Add basic unit test in `tests/robustness/test_aggregators.py`

---

### AIT-11: Implement Z-score client anomaly detector
**Status:** To Do
**Depends on:** AIT-9

Implement `ZScoreDetector` in `src/robustness/client_scoring.py`:
- Compute statistics for each client (mean_norm, std, max_norm)
- Calculate z-scores across clients
- Flag clients exceeding threshold (default: 3.0)
- Return ClientScore with score, is_outlier, details

**Tests:** Add unit test in `tests/robustness/test_client_scoring.py`

---

### AIT-12: Integrate robustness with FederatedServer
**Status:** To Do
**Depends on:** AIT-9, AIT-10, AIT-11

Modify `src/federated/server.py` to support robust aggregation:
- Add optional `robustness_config` parameter to `__init__`
- Initialize robust_aggregator and client_scorer if enabled
- Add `_robust_aggregate()` method
- Update `aggregate()` to use robust path when configured
- Track client trust scores

Also update `src/federated/federated_patchcore.py` to pass robustness config

---

### AIT-13: Implement attack simulations for robustness testing
**Status:** To Do
**Depends on:** AIT-9

Implement attack simulations in `src/robustness/attacks.py`:
- `ModelPoisoningAttack`: scaling attack (multiply by large factor)
- `ModelPoisoningAttack`: noise attack (add random noise)
- `apply(client_updates, malicious_indices)` method

**Note:** These are for evaluation only - simulate what happens when clients are compromised

---

## Suggested Order
1. AIT-9 (module structure)
2. AIT-10 (aggregator) + AIT-11 (detector) - can be parallel
3. AIT-13 (attacks) - can be done anytime after AIT-9
4. AIT-12 (integration) - do last, requires AIT-10 and AIT-11

## Files to Create/Modify
- `src/robustness/__init__.py` (new)
- `src/robustness/config.py` (new)
- `src/robustness/aggregators.py` (new)
- `src/robustness/client_scoring.py` (new)
- `src/robustness/attacks.py` (new)
- `src/federated/server.py` (modify)
- `src/federated/federated_patchcore.py` (modify)
- `tests/robustness/test_aggregators.py` (new)
- `tests/robustness/test_client_scoring.py` (new)

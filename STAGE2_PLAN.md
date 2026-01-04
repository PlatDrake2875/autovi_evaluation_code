# Stage 2: Trustworthiness Enhancements Plan

## Overview

**Project**: AI for Trustworthy Decision Making - Industrial Anomaly Detection
**Domain**: Federated PatchCore for Visual Inspection (Renault/eVantage dataset)
**Team Size**: 2 members
**Deadline**: 1-2 weeks (TIGHT)

### Current Status
- **Differential Privacy**: COMPLETE
- **Fairness**: NOT STARTED (Priority 1) - NEW
- **Robustness**: NOT STARTED (Priority 2)
- **Interpretability/XAI**: NOT STARTED (Priority 3)

### Target: Implement 3 Additional Trust Dimensions
1. **Fairness** - Client/category performance parity evaluation (SIMPLEST - metrics only)
2. **Robustness** - Byzantine-robust aggregation + client anomaly detection
3. **Interpretability** - Distance-based attribution for PatchCore

### Scope Reduction for Tight Deadline
- Focus on **core implementations** only
- Skip advanced features (Isolation Forest, complex SHAP)
- Minimal but sufficient tests
- Prioritize working demo over comprehensive coverage

---

## Part 1: Fairness Implementation (NEW)

### 1.1 Module Structure

```
src/fairness/
├── __init__.py
├── config.py              # FairnessConfig dataclass
├── metrics.py             # Fairness metric calculations
├── evaluator.py           # Per-client/category evaluation runner
└── visualization.py       # Fairness plots (bar charts, heatmaps)

tests/fairness/
├── test_metrics.py
└── test_evaluator.py
```

**MVP Scope**: Metrics + per-client evaluation (no model changes needed!)

### 1.2 Key Fairness Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Jain's Fairness Index** | (Σx)² / (n × Σx²) | 1.0 = perfect fairness, 1/n = worst |
| **Performance Variance** | var(client_aucs) | Lower = more fair |
| **Performance Gap** | max - min | Lower = more fair |
| **Worst-Case AUC** | min(client_aucs) | Rawlsian fairness |
| **Coefficient of Variation** | std / mean | Normalized disparity |

**References:**
- Li et al. (2021). "New Metrics to Evaluate the Performance and Fairness of Personalized Federated Learning" - https://arxiv.org/abs/2107.13173
- Nguyen et al. (2025). "Fairness in Federated Learning: Fairness for Whom?" - https://arxiv.org/abs/2505.21584
- Salazar et al. (2024). "Federated Fairness Analytics: Quantifying Fairness in Federated Learning" - https://arxiv.org/abs/2408.08214
- Huang et al. (2024). "A New Perspective to Boost Performance Fairness For Medical Federated Learning" - MICCAI 2024

### 1.3 Fairness Dimensions

| Dimension | Groups | Description |
|-----------|--------|-------------|
| **Client Parity** | 5 clients | Equal AUC across factory sites |
| **Category Parity** | 6 categories | Equal AUC across product types |
| **Defect-Type Parity** | ~10 defect types | Equal detection across defect types |

### 1.4 Integration Points

**Files to modify:**
- `src/evaluation/metrics_wrapper.py` - Add per-client evaluation mode
- `evaluate_experiment.py` - Add fairness evaluation option

**No model changes required** - fairness is purely evaluation-side!

### 1.5 Evaluation

Compare fairness metrics across:
1. Centralized baseline
2. Federated IID
3. Federated Category-based (Non-IID)
4. Federated + DP

**Key questions to answer:**
- Does federated learning hurt fairness vs centralized?
- Does category-based partitioning create unfair models?
- Does DP disproportionately hurt certain clients?

### 1.6 Jira Tickets (Fairness)

| Key | Summary | Assignee |
|-----|---------|----------|
| AIT-32 | Create fairness module structure and FairnessConfig | Adrian |
| AIT-33 | Implement core fairness metrics (Jain's Index, variance, gap) | Adrian |
| AIT-34 | Implement per-client fairness evaluator | Adrian |
| AIT-35 | Implement fairness visualization (bar charts, heatmaps) | Adrian |
| AIT-36 | Run fairness evaluation experiments | Both |
| AIT-37 | Write unit tests for fairness module | Adrian |

---

## Part 2: Robustness Implementation

### 2.1 Module Structure (Simplified for Deadline)

```
src/robustness/
├── __init__.py
├── config.py                    # RobustnessConfig dataclass
├── aggregators.py               # All aggregators in one file (Median, Krum, TrimmedMean)
├── client_scoring.py            # Z-score detector (single file)
└── attacks.py                   # Simple attack simulations

tests/robustness/
├── test_aggregators.py
├── test_client_scoring.py
└── test_integration.py
```

**MVP Scope**: Coordinate Median + Z-score detector + scaling attack

### 2.2 Key Components

| Component | Purpose |
|-----------|---------|
| **Krum/Multi-Krum** | Select clients closest to neighbors, robust to f Byzantine clients |
| **Coordinate Median** | Median per dimension, robust to 50% malicious |
| **Trimmed Mean** | Remove extremes before averaging |
| **Z-Score Detector** | Flag clients with unusual update statistics |
| **IQR Detector** | Interquartile range outlier detection |
| **Isolation Forest** | ML-based anomaly detection on client updates |

### 2.3 Integration Points

**Files to modify:**
- `src/federated/server.py` - Add robust aggregation support
- `src/federated/strategies/federated_memory.py` - Register robust strategies
- `src/federated/federated_patchcore.py` - Add robustness config

### 2.4 Evaluation

- **Attack success rate**: AUC degradation under attack
- **Detection rate**: True positive rate for malicious clients
- **False positive rate**: Honest clients incorrectly flagged
- Attacks to test: scaling, noise injection, random, label flipping
- Malicious fractions: 10%, 20%, 30%, 40%

---

## Part 3: Interpretability/XAI Implementation

### 3.1 Module Structure (Simplified for Deadline)

```
src/interpretability/
├── __init__.py
├── config.py                    # XAIConfig dataclass
├── attribution.py               # Distance-based patch attribution
├── visualization.py             # Heatmap overlays
└── evaluation.py                # IoU localization metric

tests/interpretability/
├── test_attribution.py
└── test_evaluation.py
```

**MVP Scope**: Distance-based attribution + heatmap visualization + IoU metric
**Deferred**: SHAP (too complex), faithfulness metrics, federated XAI

### 3.2 Key Components

| Component | Purpose |
|-----------|---------|
| **PatchAttributor** | Compute which patches contribute to anomaly score |
| **SHAP Explainer** | Feature attribution via SHAP (adapted for distance-based model) |
| **Spatial Attribution** | Upscale patch attributions to pixel-level heatmaps |
| **Memory Bank Analyzer** | Which stored patches are matched, client provenance |
| **Federated XAI** | Compare local vs global explanations, track client contributions |

### 3.3 Key Challenge

PatchCore is **distance-based** (not gradient-based), so traditional Grad-CAM doesn't apply directly. Instead:
- Attribution = distance contribution per patch
- SHAP treats distance computation as black-box function
- Occlusion-based attribution as alternative

### 3.4 Integration Points

**Files to modify:**
- `src/models/patchcore.py` - Add `predict_with_explanations()`
- `src/evaluation/anomaly_scorer.py` - Generate explanations alongside maps
- `src/federated/federated_patchcore.py` - Add XAI analyzer

### 3.5 Evaluation

- **Faithfulness**: Deletion AUC, Insertion AUC, Infidelity
- **Localization**: IoU with ground truth masks, Pixel-AUC, Pointing game

---

## Part 4: Team Role Allocation (2 Members)

| Member | Focus | Deliverables |
|--------|-------|--------------|
| **Adrian** | Fairness + Robustness | Fairness metrics/evaluation, robust aggregation, client scoring |
| **Miriam** | Interpretability | Attribution, visualization, localization metrics |
| **Both** | Integration & Report | Trade-off analysis, final report, presentation |

---

## Part 5: Implementation Schedule (1-2 Weeks)

### Week 1: Core Implementations

**Days 1-3: Robustness MVP**
1. Create `src/robustness/` module structure
2. Implement `RobustnessConfig` dataclass
3. Implement Coordinate Median aggregator (simplest, most robust)
4. Implement Z-score client detector
5. Integrate with `FederatedServer`
6. Basic test coverage

**Days 4-5: Robustness Evaluation**
1. Implement simple attack (scaling + noise)
2. Run experiments: robust vs non-robust under attack
3. Generate comparison table

**Days 3-5: Interpretability MVP (parallel)**
1. Create `src/interpretability/` module structure
2. Implement distance-based `PatchAttributor`
3. Implement basic heatmap visualization
4. Skip SHAP (too complex for deadline)

### Week 2: Polish & Report

**Days 6-7: Interpretability Evaluation**
1. Implement localization metrics (IoU with GT masks)
2. Run explanation experiments
3. Generate visualization outputs

**Days 8-10: Integration & Documentation**
1. Trade-off analysis (accuracy vs DP vs robustness)
2. Write final report sections
3. Prepare presentation
4. Code cleanup and README

---

## Part 6: Critical Files

### Robustness
- `src/federated/server.py` - Main integration point
- `src/federated/strategies/federated_memory.py` - Strategy patterns
- `src/privacy/embedding_sanitizer.py` - Config pattern reference

### Interpretability
- `src/models/patchcore.py` - Core model to extend
- `src/models/memory_bank.py` - Query logic for NN analysis
- `src/evaluation/anomaly_scorer.py` - Scoring logic to extend
- `src/evaluation/visualization.py` - Plot patterns

---

## Part 7: Dependencies to Add

```toml
# pyproject.toml
shap = ">=0.42.0"
scikit-learn = ">=1.3.0"  # For IsolationForest
scipy = ">=1.11.0"  # For trim_mean
```

---

## Part 8: Deliverables Checklist (MVP for Deadline)

### Code (Must Have)
- [ ] `src/fairness/` module: Metrics + per-client evaluation
- [ ] `src/robustness/` module: Coordinate Median + Z-score detector
- [ ] `src/interpretability/` module: Distance attribution + heatmaps
- [ ] Integration with `FederatedServer`
- [ ] Basic tests (at least 1 per module)
- [ ] Example config YAML

### Evaluation (Must Have)
- [ ] Fairness: Per-client/category performance comparison
- [ ] Robustness: Attack success rate comparison (robust vs baseline)
- [ ] XAI: Sample explanation visualizations
- [ ] Trade-off table: Accuracy vs DP epsilon vs robustness vs fairness

### Report (Must Have)
- [ ] Stage 2 section in final report (18-20 pages total)
- [ ] Quantitative results tables
- [ ] Ethical discussion section

### Nice to Have (If Time Permits)
- [ ] Additional aggregators (Krum, Trimmed Mean)
- [ ] IoU localization metrics
- [ ] Client contribution analysis for XAI

---

## Part 9: Jira Tickets

### Epic
- **AIT-8**: Stage 2: Trustworthiness Enhancements (Fairness + Robustness + Interpretability)

### Adrian's Tasks (Fairness)
| Key | Summary |
|-----|---------|
| AIT-32 | Create fairness module structure and FairnessConfig |
| AIT-33 | Implement core fairness metrics (Jain's Index, variance, gap) |
| AIT-34 | Implement per-client fairness evaluator |
| AIT-35 | Implement fairness visualization (bar charts, heatmaps) |
| AIT-36 | Run fairness evaluation experiments |
| AIT-37 | Write unit tests for fairness module |

### Adrian's Tasks (Robustness)
| Key | Summary |
|-----|---------|
| AIT-9 | Create robustness module structure and RobustnessConfig |
| AIT-10 | Implement Coordinate Median robust aggregator |
| AIT-11 | Implement Z-score client anomaly detector |
| AIT-12 | Integrate robustness with FederatedServer |
| AIT-13 | Implement attack simulations for robustness testing |
| AIT-14 | Run robustness evaluation experiments |
| AIT-20 | Trade-off analysis: Accuracy vs DP vs Robustness |
| AIT-22 | Prepare final presentation |

### Miriam's Tasks (Interpretability)
| Key | Summary |
|-----|---------|
| AIT-15 | Create interpretability module structure and XAIConfig |
| AIT-16 | Implement distance-based PatchAttributor |
| AIT-17 | Implement XAI heatmap visualization |
| AIT-18 | Implement IoU localization evaluation |
| AIT-19 | Run XAI evaluation experiments |
| AIT-21 | Write Stage 2 sections in final report |

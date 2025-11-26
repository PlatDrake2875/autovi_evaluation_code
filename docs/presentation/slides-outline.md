# Slides Outline

## Slide 1: Title

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│     FEDERATED LEARNING FOR INDUSTRIAL ANOMALY DETECTION     │
│                                                             │
│        Stage 1: Baseline Development & Federated Setup      │
│                                                             │
│     ─────────────────────────────────────────────────────   │
│                                                             │
│                     Team Members:                           │
│              Member 1 - Data & Experiment Lead              │
│              Member 2 - Modeling & Privacy Lead             │
│              Member 3 - Evaluation & Fairness Lead          │
│                                                             │
│            AI for Trustworthy Decision Making               │
│                      [Date]                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Slide 2: Problem & Motivation

```
┌─────────────────────────────────────────────────────────────┐
│  THE CHALLENGE: Privacy in Industrial Quality Control       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [DIAGRAM: Multiple factories with isolated data silos]     │
│                                                             │
│  Factory A    Factory B    Factory C                        │
│  ┌─────┐      ┌─────┐      ┌─────┐                         │
│  │Data │      │Data │      │Data │   ← Cannot share        │
│  └─────┘      └─────┘      └─────┘     (privacy, legal)    │
│                                                             │
│  CHALLENGES:                                                │
│  • Manufacturing data is proprietary                        │
│  • Regulatory restrictions on data sharing                  │
│  • Different facilities = different components              │
│                                                             │
│  SOLUTION: Federated Learning                               │
│  • Train models locally, share only updates                 │
│  • No raw data leaves the factory                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Slide 3: AutoVI Dataset

```
┌─────────────────────────────────────────────────────────────┐
│  DATASET: Automotive Visual Inspection (AutoVI)             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [GRID: 6 sample images, one per category]                  │
│                                                             │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐           │
│  │eng_ │ │pipe_│ │pipe_│ │tank_│ │under│ │under│           │
│  │wire │ │clip │ │stap │ │screw│ │_pipe│ │_scrw│           │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘           │
│                                                             │
│  STATISTICS:                                                │
│  ┌────────────────┬───────┬───────┬─────────────┐          │
│  │ Category       │ Train │ Test  │ Defect Types│          │
│  ├────────────────┼───────┼───────┼─────────────┤          │
│  │ Total          │ 1,523 │ 2,399 │ 10          │          │
│  └────────────────┴───────┴───────┴─────────────┘          │
│                                                             │
│  • Real production line data (Renault Group)                │
│  • Unsupervised: train on "good" images only               │
│  • Pixel-level ground truth annotations                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Slide 4: PatchCore Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  MODEL: PatchCore Architecture                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [DIAGRAM: PatchCore pipeline]                              │
│                                                             │
│  Input     →  Backbone    →  Memory Bank  →  Anomaly Map    │
│  Image        (WideResNet)    (Coreset)       (NN Search)   │
│                                                             │
│  ┌────┐      ┌─────────┐     ┌─────────┐     ┌─────────┐   │
│  │    │  →   │ Layer2  │  →  │ Normal  │  →  │ Distance│   │
│  │    │      │ Layer3  │     │ Patches │     │  Map    │   │
│  └────┘      └─────────┘     └─────────┘     └─────────┘   │
│                                                             │
│  KEY COMPONENTS:                                            │
│  1. Pre-trained CNN extracts patch features                 │
│  2. Memory bank stores representative normal patches        │
│  3. Anomaly = distance to nearest normal patch              │
│                                                             │
│  WHY PATCHCORE?                                             │
│  ✓ State-of-the-art performance                            │
│  ✓ Memory bank can be aggregated (FL-friendly)             │
│  ✓ Interpretable anomaly maps                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Slide 5: Federated Setup

```
┌─────────────────────────────────────────────────────────────┐
│  FEDERATED ARCHITECTURE: 5 Clients                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [DIAGRAM: Server-client architecture]                      │
│                                                             │
│                    ┌─────────┐                              │
│                    │ SERVER  │                              │
│                    │ Global  │                              │
│                    │ Memory  │                              │
│                    └────┬────┘                              │
│           ┌─────┬─────┼─────┬─────┐                        │
│           ▼     ▼     ▼     ▼     ▼                        │
│        ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐                       │
│        │C1 │ │C2 │ │C3 │ │C4 │ │C5 │                       │
│        │eng│ │und│ │fas│ │cli│ │mix│                       │
│        └───┘ └───┘ └───┘ └───┘ └───┘                       │
│                                                             │
│  DATA PARTITIONING:                                         │
│  ┌────────────────────────────────────────────────────┐    │
│  │ IID:      Uniform random (20% each)                │    │
│  │ Non-IID:  Category-based (realistic simulation)    │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  AGGREGATION: Federated Coreset Selection                   │
│  • Only 1 communication round needed (efficient!)          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Slide 6: Baseline Results

```
┌─────────────────────────────────────────────────────────────┐
│  RESULTS: Centralized Baseline Performance                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [BAR CHART: AUC-sPRO per category]                        │
│                                                             │
│  AUC-sPRO@0.05                                              │
│  1.0 ┤                                                      │
│      │  ████                                                │
│  0.8 ┤  ████ ████ ████ ████                                │
│      │  ████ ████ ████ ████ ████ ████                      │
│  0.6 ┤  ████ ████ ████ ████ ████ ████                      │
│      │  ████ ████ ████ ████ ████ ████                      │
│  0.4 ┤  ████ ████ ████ ████ ████ ████                      │
│      │  ████ ████ ████ ████ ████ ████                      │
│  0.2 ┤  ████ ████ ████ ████ ████ ████                      │
│      │  ████ ████ ████ ████ ████ ████                      │
│  0.0 └──eng──clip─stap─tank─u_pp─u_sc───                   │
│                                                             │
│  MEAN AUC-sPRO@0.05: 0.78                                   │
│  MEAN AUC-ROC: 0.88                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Slide 7: Federated Comparison

```
┌─────────────────────────────────────────────────────────────┐
│  RESULTS: Centralized vs Federated                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [GROUPED BAR CHART: 3 methods comparison]                  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Method          │ AUC-sPRO@0.05 │ Gap vs Central   │   │
│  ├─────────────────┼───────────────┼──────────────────┤   │
│  │ Centralized     │ 0.78          │ -                │   │
│  │ Federated (IID) │ 0.76          │ -3.2%            │   │
│  │ Fed (Category)  │ 0.71          │ -9.4%            │   │
│  └─────────────────┴───────────────┴──────────────────┘   │
│                                                             │
│  KEY OBSERVATIONS:                                          │
│  • IID federated achieves 97% of centralized performance   │
│  • Non-IID shows significant degradation (-9.4%)           │
│  • Trade-off: Privacy vs Accuracy                          │
│                                                             │
│  [HIGHLIGHT BOX]                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ "Memory bank aggregation is communication-efficient  │   │
│  │  but non-IID data remains a key challenge"          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Slide 8: Key Findings

```
┌─────────────────────────────────────────────────────────────┐
│  ANALYSIS: Key Findings & Limitations                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ✓ SUCCESSES:                                               │
│    • Federated PatchCore is feasible and efficient         │
│    • Single communication round (vs 100s for gradient FL)  │
│    • IID performance close to centralized                  │
│                                                             │
│  ✗ CHALLENGES:                                              │
│    • Non-IID data causes significant performance drop      │
│    • Smaller clients (fewer samples) suffer more           │
│    • No formal privacy guarantees yet                      │
│                                                             │
│  [FPR-sPRO CURVE COMPARISON]                               │
│                                                             │
│  sPRO │    ___--- Centralized                              │
│   1.0 │  _/  _--- IID Federated                            │
│       │ / _/  _-- Category Federated                       │
│   0.5 │/  /  /                                              │
│       │  /  /                                               │
│   0.0 └────────────────────────                            │
│        0    0.1   0.2   0.3  FPR                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Slide 9: Stage 2 Roadmap

```
┌─────────────────────────────────────────────────────────────┐
│  NEXT STEPS: Stage 2 Trustworthiness Enhancements           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  TRUST DIMENSIONS:                                          │
│                                                             │
│  ┌─────────────────────┐  ┌─────────────────────┐          │
│  │     PRIVACY         │  │     FAIRNESS        │          │
│  │  (DP-SGD)          │  │ (Cross-Category)    │          │
│  │                     │  │                     │          │
│  │  • Add noise to     │  │  • Balance client   │          │
│  │    feature sharing  │  │    contributions    │          │
│  │  • Track ε budget   │  │  • Reduce variance  │          │
│  │  • Privacy-utility  │  │    across objects   │          │
│  │    trade-off        │  │                     │          │
│  └─────────────────────┘  └─────────────────────┘          │
│                                                             │
│  EXPECTED OUTCOMES:                                         │
│  • Quantified privacy guarantees (ε = 1, 5, 10)            │
│  • Reduced performance variance across categories          │
│  • Trade-off analysis and recommendations                  │
│                                                             │
│  DELIVERABLES:                                              │
│  • Final Report (18-20 pages)                              │
│  • Complete Code Repository                                │
│  • Group Presentation                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Slide 10: Q&A

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│                      QUESTIONS?                             │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│                      THANK YOU                              │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  Repository: [GitHub URL]                                   │
│  Dataset: doi.org/10.5281/zenodo.10459003                  │
│                                                             │
│  Team Contacts:                                             │
│  • Member 1: [email]                                        │
│  • Member 2: [email]                                        │
│  • Member 3: [email]                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Backup Slides

### Backup 1: Detailed Dataset Statistics

Full table of per-category train/test/defect breakdown.

### Backup 2: PatchCore Algorithm Details

Coreset selection algorithm, feature dimensions, FAISS usage.

### Backup 3: Aggregation Strategy Comparison

Comparison of different memory bank aggregation approaches considered.

### Backup 4: Statistical Analysis

Paired t-test results, confidence intervals, effect sizes.

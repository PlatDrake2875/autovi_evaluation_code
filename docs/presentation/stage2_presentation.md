# Stage 2: Trustworthiness Enhancements - Presentation Outline

**Project**: AI for Trustworthy Decision Making - Industrial Anomaly Detection
**Team**: Adrian & Miriam
**Date**: [Presentation Date]

---

## Slide 1: Title Slide

**Title**: Trustworthy Federated Learning for Industrial Anomaly Detection

**Subtitle**: Stage 2 - Privacy, Robustness, and Fairness

**Team Members**: Adrian [Last Name], Miriam [Last Name]

---

## Slide 2: Recap - Stage 1 Results

**Key Points**:
- Federated PatchCore for visual inspection on Renault/eVantage dataset
- 5 clients, 6 product categories
- IID and category-based partitioning
- **Stage 1 Result**: Mean AUC-sPRO = 0.167 (FPR=0.05)

**Visual**: Comparison table showing Stage 1 results by object

| Object | AUC-sPRO |
|--------|----------|
| engine_wiring | 0.204 |
| pipe_clip | 0.205 |
| pipe_staple | 0.255 |
| tank_screw | 0.118 |
| underbody_pipes | 0.094 |
| underbody_screw | 0.131 |
| **Mean** | **0.167** |

---

## Slide 3: Stage 2 Overview

**Three Trustworthiness Dimensions**:

1. **Differential Privacy (DP)** - Protect sensitive manufacturing data
2. **Robustness** - Resist Byzantine/malicious client attacks
3. **Fairness** - Ensure equal performance across clients and categories

**Visual**: Three-pillar diagram showing the trust dimensions

---

## Slide 4: Differential Privacy - Method

**Presenter**: [Name]

**Key Points**:
- Gaussian mechanism for embedding sanitization
- Privacy budget (epsilon, delta) accounting
- L2 norm clipping before adding noise
- Configurable privacy levels: epsilon = {1.0, 5.0, 10.0}

**Visual**: DP mechanism diagram

```
Client Embeddings → L2 Clipping → Add Gaussian Noise → Private Embeddings
                         ↓
                  Privacy Accountant tracks (ε, δ)
```

---

## Slide 5: Differential Privacy - Results

**Privacy-Accuracy Trade-off**:

| Privacy (epsilon) | AUC-sPRO | Change vs Baseline |
|-------------------|----------|-------------------|
| No DP (baseline) | 0.167 | - |
| epsilon = 10.0 | TBD | -X% |
| epsilon = 5.0 | TBD | -X% |
| epsilon = 1.0 | TBD | -X% |

**Key Finding**: Lower epsilon provides stronger privacy but reduces accuracy

**Visual**: Line plot showing epsilon vs accuracy trade-off

---

## Slide 6: Robustness - Method

**Presenter**: Adrian

**Key Points**:
- Byzantine-robust aggregation: Coordinate Median
  - Computes per-dimension median across clients
  - Robust to 50% malicious clients (theoretically)
- Client anomaly detection: Z-Score Detector
  - Flags clients with unusual update statistics
  - Configurable threshold (default: 2.5)

**Attack Types Tested**:
- Scaling attack: Multiply updates by large factor (100x)
- Noise injection: Add high-variance noise
- Sign flip: Negate update values

---

## Slide 7: Robustness - Results

**Attack Resistance Comparison**:

| Attack | Malicious % | Standard | Robust | Improvement |
|--------|-------------|----------|--------|-------------|
| Scaling | 20% | 0.34 dev | 0.01 dev | 97% |
| Scaling | 40% | 0.48 dev | 0.01 dev | 98% |
| Noise | 20% | 0.04 dev | 0.01 dev | 75% |
| Noise | 40% | 0.05 dev | 0.01 dev | 80% |

**Key Finding**: Coordinate median maintains near-baseline performance even under 40% malicious clients

**Visual**: Bar chart comparing robust vs standard under attacks

---

## Slide 8: Fairness - Method

**Key Points**:
- Evaluate performance parity across clients and categories
- Metrics computed:
  - **Jain's Fairness Index**: (sum(x))^2 / (n * sum(x^2)), 1.0 = perfect
  - **Performance Gap**: max - min AUC
  - **Worst-Case AUC**: Rawlsian fairness (max-min principle)
  - **Coefficient of Variation**: std / mean

**Visual**: Fairness metrics formulas

---

## Slide 9: Fairness - Results

**Current Model Fairness**:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Jain's Index | 0.896 | Good (1.0 = perfect) |
| Performance Gap | 0.161 | Moderate disparity |
| Worst-Case AUC | 0.094 | underbody_pipes struggles |
| Coefficient of Variation | 0.340 | 34% relative disparity |

**Per-Category Performance**:
- Best: pipe_staple (0.255)
- Worst: underbody_pipes (0.094)

**Key Finding**: Some categories benefit more from federated learning than others

---

## Slide 10: Trade-off Analysis

**Three-Way Trade-off: Accuracy vs Privacy vs Robustness**

| Configuration | AUC-sPRO | Privacy | Robustness |
|---------------|----------|---------|------------|
| Baseline (Federated) | 0.167 | None | None |
| + DP (eps=5.0) | TBD | Strong | None |
| + Robust | TBD | None | Strong |
| + DP + Robust | TBD | Strong | Strong |

**Visual**: 3D scatter plot or radar chart showing trade-offs

---

## Slide 11: Key Findings

1. **Differential Privacy**: Achievable with modest accuracy trade-off
   - [Specific numbers when experiments complete]

2. **Robustness**: Coordinate median effectively resists attacks
   - Up to 40% malicious clients tolerated
   - Mean deviation reduced by 97%+ under scaling attacks

3. **Fairness**: Room for improvement
   - Jain's Index = 0.896 (good but not perfect)
   - 2.7x performance gap between best and worst categories

4. **Combined Trust**: [Results when available]
   - DP + Robustness achieves [X] accuracy with strong protections

---

## Slide 12: Limitations & Future Work

**Limitations**:
- Only tested synthetic attacks (robustness)
- DP experiments pending (need more compute time)
- No interpretability/XAI implemented yet

**Future Work**:
- Additional robust aggregators (Krum, Trimmed Mean)
- Fairness-aware federated optimization
- Interpretability for federated anomaly detection
- Real-world deployment considerations

---

## Slide 13: Ethical Considerations

**Privacy**:
- DP protects sensitive manufacturing data
- Trade-off: privacy vs utility must be balanced

**Fairness**:
- Ensure all clients/categories receive fair model performance
- Consider downstream impact of biased models

**Safety**:
- Robustness prevents adversarial manipulation
- Critical for safety-related anomaly detection (automotive)

---

## Slide 14: Conclusion

**Summary**:
- Implemented three trustworthiness dimensions
- Privacy: Gaussian mechanism DP with configurable epsilon
- Robustness: Coordinate median resists 40%+ Byzantine clients
- Fairness: Metrics show 0.896 Jain's Index (room for improvement)

**Contribution**:
- First trustworthy federated learning framework for industrial anomaly detection
- Trade-off analysis quantifies privacy-accuracy-robustness relationships

---

## Slide 15: Q&A

**Questions?**

**Resources**:
- Code: [Repository URL]
- Results: `results/` directory
- Documentation: See README.md Stage 2 section

---

## Speaker Notes

### Slide 6 (Robustness - Adrian)
- Explain why median is robust: 50% breakdown point
- Mention that coordinate-wise median preserves efficiency
- Z-score detector: simple but effective for detecting outliers

### Slide 7 (Robustness Results - Adrian)
- "Mean deviation" measures how far aggregated result is from honest mean
- Lower is better
- Highlight: Standard aggregation degrades severely under scaling (0.48)
- Robust aggregation maintains 0.01 even at 40% malicious

### Time Allocation
- Intro/Recap: 2-3 minutes
- DP: 3-4 minutes
- Robustness: 3-4 minutes (Adrian)
- Fairness: 2-3 minutes
- Trade-offs: 2-3 minutes
- Conclusions: 1-2 minutes
- Q&A: 5+ minutes
- **Total**: ~20 minutes

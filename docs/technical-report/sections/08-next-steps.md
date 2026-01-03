# 8. Next Steps (Stage 2 Roadmap)

## 8.1 Trust Dimensions to Address

Based on our Stage 1 findings, we will focus on **two trust dimensions** in Stage 2:

### Privacy Enhancement: Differential Privacy (DP-SGD)

**Motivation**: While our current implementation keeps raw data local, memory bank features could theoretically leak information about training samples. Formal privacy guarantees are essential for deployment in sensitive industrial settings.

**Approach**:
- Integrate DP-SGD for feature extraction phase
- Add calibrated noise to local coresets before sharing
- Track privacy budget (ε, δ) across communication
- Analyze privacy-utility trade-off

**Expected Outcome**: Quantified privacy guarantees with measured impact on anomaly detection performance.

### Fairness: Cross-Category Performance Equity

**Motivation**: Our results show that clients with smaller datasets or unique categories suffer disproportionate performance degradation in federated settings. This raises fairness concerns for industrial deployment.

**Approach**:
- Implement fairness-aware aggregation weighting
- Develop per-client performance monitoring
- Create balanced sampling strategies for global coreset
- Evaluate performance variance across categories as fairness metric

**Expected Outcome**: Reduced performance variance across object categories while maintaining overall accuracy.

## 8.2 Technical Enhancements

| Enhancement | Priority | Estimated Effort |
|-------------|----------|------------------|
| DP-SGD integration | High | Medium |
| Fairness metrics | High | Low |
| FedProx regularization | Medium | Low |
| Iterative memory refinement | Medium | Medium |
| Grad-CAM interpretability | Low | Low |

## 8.3 Evaluation Plan

Stage 2 evaluation will include:

1. **Privacy Analysis**
   - Privacy budget tracking (ε values: 1, 5, 10)
   - Membership inference attack resistance
   - Accuracy vs privacy curves

2. **Fairness Analysis**
   - Per-category performance variance
   - Gini coefficient of client contributions
   - Pareto frontier: accuracy vs fairness

3. **Trade-off Analysis**
   - Multi-objective optimization results
   - Recommended configurations for different use cases

## 8.4 Deliverables

Stage 2 will produce:

1. **Final Report** (18-20 pages): Comprehensive documentation including Stage 1 baseline and Stage 2 enhancements
2. **Complete Code Repository**: Modular, reproducible implementation with documentation
3. **Group Presentation**: Each team member presents their contribution

## 8.5 Team Coordination

| Member | Stage 2 Focus |
|--------|---------------|
| Member 1 | Fairness-aware aggregation, data analysis |
| Member 2 | DP-SGD integration, privacy analysis |
| Member 3 | Evaluation framework, trade-off analysis |

Cross-evaluation will continue, with each member testing enhanced models on their assigned data splits.

---

## References

[1] Roth, K., et al. "Towards Total Recall in Industrial Anomaly Detection." CVPR 2022.

[2] McMahan, B., et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS 2017.

[3] Zhao, Y., et al. "Federated Learning with Non-IID Data." arXiv 2018.

[4] Carvalho, P., et al. "The Automotive Visual Inspection Dataset (AutoVI)." Zenodo 2024.

[5] Li, T., et al. "Federated Optimization in Heterogeneous Networks (FedProx)." MLSys 2020.

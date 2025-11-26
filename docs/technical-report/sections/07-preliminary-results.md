# 7. Preliminary Results - Stage 1

## 7.1 Status: In Progress

Stage 1 implementation is currently in progress with the following completion status:

| Component | Status | Notes |
|-----------|--------|-------|
| Data Loader | ✓ Complete | All 6 categories loaded and partitioned |
| PatchCore Model | In Progress | Feature extraction and memory bank in development |
| Client Training | Pending | Awaiting model implementation |
| Baseline Evaluation | Pending | Will benchmark centralized aggregated model |
| Stage 1 Results | Pending | Will show independent baseline per client |

## 7.2 Expected Results Structure

Once PatchCore implementation is complete, this section will contain:

**Table 4: Per-Client Baseline Performance (AUC-sPRO@0.05)**

| Client | Category | Train Images | @FPR=0.01 | @FPR=0.05 | @FPR=0.1 | AUC-ROC |
|--------|----------|--------------|-----------|-----------|----------|---------|
| 1 | engine_wiring | 285 | % TODO | % TODO | % TODO | % TODO |
| 2 | pipe_clip | 195 | % TODO | % TODO | % TODO | % TODO |
| 3 | pipe_staple | 191 | % TODO | % TODO | % TODO | % TODO |
| 4 | tank_screw | 318 | % TODO | % TODO | % TODO | % TODO |
| 5 | underbody_pipes | 161 | % TODO | % TODO | % TODO | % TODO |
| 6 | underbody_screw | 373 | % TODO | % TODO | % TODO | % TODO |
| **Mean** | **-** | **1,523** | **% TODO** | **% TODO** | **% TODO** | **% TODO** |

## 7.3 Performance Analysis (Placeholder)

% TODO: Add per-category performance analysis
% Expected observations:
% - Categories with more training data (e.g., underbody_screw with 373 samples) may achieve better performance
% - Smaller categories (e.g., pipe_clip with 195 samples) may show higher variance
% - Defect type complexity affects performance (structural vs logical anomalies)

## 7.4 Category-wise Comparisons

% TODO: Figure 1 - Bar chart comparing AUC-sPRO across 6 clients
% Format: Grouped bars per category at different FPR thresholds
% Should show performance variance across categories

## 7.5 Statistical Analysis

% TODO: Add statistical summaries after experiments
% Include: mean, std, confidence intervals per category
% Add: paired comparisons if performing federated aggregation trials

## 7.6 Key Observations (To Be Updated)

This section will document:

1. **Training dynamics**: How quickly each client's memory bank converges
2. **Category-specific challenges**: Which object types are harder to detect
3. **Data imbalance effects**: Performance correlation with dataset size
4. **Anomaly type patterns**: Difficulty of structural vs logical defects per category

---

**Next Steps for Stage 1**:
1. Complete PatchCore model implementation
2. Train independent models for all 6 clients
3. Collect and document baseline performance
4. Generate anomaly visualizations per category
5. Update this section with actual results and analysis

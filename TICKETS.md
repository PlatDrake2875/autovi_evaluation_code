# Branch: AIT-20/analysis-docs

## Overview
This branch focuses on analysis, documentation, and presentation tasks. It includes trade-off analysis, quantitative comparisons, README updates, and final presentation preparation.

**Prerequisites:** Implementation and testing branches should be completed first to have results to analyze.

**Note:** Robustness (AIT-14) and Fairness (AIT-32) modules have been merged from main.

## Tickets to Complete

### AIT-20: Trade-off analysis: Accuracy vs DP vs Robustness
**Status:** To Do
**Priority:** Start here (needs experiment results)

Analyze trade-offs between trustworthiness mechanisms:
- Baseline accuracy (no trust mechanisms)
- Accuracy with DP at different epsilon values (1.0, 5.0, 10.0)
- Accuracy with robust aggregation vs baseline under attack
- Combined: DP + Robustness

**Output:** `trade_off_table.csv` and visualization showing accuracy-privacy-robustness trade-offs

---

### AIT-27: Quantitative comparison: Centralized vs Federated vs Trust-Enhanced
**Status:** To Do
**Depends on:** AIT-20 (uses same data)

Per guidelines, must provide quantitative comparison:
- Baseline centralized model accuracy
- Federated model accuracy (Stage 1)
- Federated + DP accuracy at various epsilon
- Federated + Robustness accuracy (clean and under attack)
- Combined: Federated + DP + Robustness

**Output:** Create comparison table and visualization for report

---

### AIT-30: Update README with Stage 2 documentation
**Status:** To Do
**Depends on:** AIT-20, AIT-27 (need to know final features)

Update project README.md with:
- New module descriptions (`src/robustness/`, `src/fairness/`, `src/privacy/`)
- Usage examples for robustness, fairness, and DP features
- Configuration options explained
- Link scripts to corresponding report sections

---

### AIT-22: Prepare final presentation
**Status:** To Do
**Priority:** Do last (needs all results)

Prepare the final presentation for Stage 2:
- Introduction and recap of Stage 1
- Differential Privacy: methods and privacy-accuracy trade-off
- Robustness: methods and key results (Adrian presents)
- Fairness: metrics and performance parity analysis
- Trade-off analysis and conclusions

**Note:** Per guidelines - each member must present part of the work

---

## Suggested Order
1. AIT-20 (trade-off analysis) - requires experiment results
2. AIT-27 (quantitative comparison) - builds on AIT-20
3. AIT-30 (README) - document final features
4. AIT-22 (presentation) - do last once all results ready

## Files to Create/Modify
- `experiments/scripts/trade_off_analysis.py` (new)
- `results/trade_off_table.csv` (output)
- `results/comparison_table.csv` (output)
- `README.md` (modify)
- `docs/presentation/` (new directory)

## Notes
- This branch should be worked on after implementation and testing are complete
- Robustness and fairness modules are now available from main
- Presentation requires both team members' input

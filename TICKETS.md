# Branch: AIT-14/robustness-testing

## Overview
This branch focuses on testing and evaluation of the robustness implementation, including unit tests, integration tests, cross-evaluation, and running experiments to measure robustness against attacks.

**Prerequisites:** The `AIT-9/robustness-core` branch should be completed and merged first.

## Tickets to Complete

### AIT-23: Write unit tests for robustness module
**Status:** To Do
**Priority:** Start here

Create `tests/robustness/` directory with:
- `test_aggregators.py` - test Coordinate Median output shape, robustness to outliers
- `test_client_scoring.py` - test Z-score detector flags outliers correctly
- `test_integration.py` - test FederatedServer with robustness config

---

### AIT-14: Run robustness evaluation experiments
**Status:** To Do
**Depends on:** AIT-23 (tests should pass first)

Run experiments comparing robust vs baseline aggregation under attack:
- Test with 10%, 20%, 30% malicious clients
- Measure attack success rate (AUC degradation)
- Measure detection rate if client scoring enabled
- Generate comparison table for report

**Output:** `robustness_results.csv` and comparison plots

---

### AIT-26: Cross-evaluation: Test each other's implementations
**Status:** To Do
**Depends on:** AIT-14 (need working implementation to test)

Per project guidelines, team must do cross-evaluation:
- Adrian tests Miriam's XAI implementation
- Miriam tests Adrian's robustness implementation
- Document any issues found and how they were resolved
- Include cross-evaluation results in report

---

## Suggested Order
1. AIT-23 (unit tests) - write tests for robustness module
2. AIT-14 (experiments) - run evaluation experiments
3. AIT-26 (cross-eval) - coordinate with team member

## Files to Create/Modify
- `tests/robustness/__init__.py` (new)
- `tests/robustness/test_aggregators.py` (new)
- `tests/robustness/test_client_scoring.py` (new)
- `tests/robustness/test_integration.py` (new)
- `experiments/robustness_evaluation.py` (new or modify)
- `results/robustness_results.csv` (output)

## Notes
- Ensure robustness-core branch is merged before starting experiments
- Cross-evaluation requires coordination with team member Miriam

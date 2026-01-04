# Missing Experiments for Trade-off Analysis

These experiments need to be run to complete the trade-off analysis (AIT-20).

## Summary

| # | Experiment | Status | Est. Time |
|---|------------|--------|-----------|
| 1 | Centralized baseline | Missing | ~15 min |
| 2 | Federated + DP (ε=1.0) | Missing | ~15 min |
| 3 | Federated + DP (ε=5.0) | Missing | ~15 min |
| 4 | Federated + DP (ε=10.0) | Missing | ~15 min |
| 5 | Federated + Robust (clean) | Missing | ~15 min |
| 6 | Federated + Robust (attack) | Missing | ~15 min |

**Total estimated time**: ~90 minutes

---

## 1. Centralized Baseline

**Purpose**: Baseline accuracy without federation

**Known Issue**: Previously failed with CUDA OOM on 3/6 objects. May need to reduce batch size.

**Command** (example):
```bash
# Train centralized model
python scripts/train_centralized.py \
    --config experiments/configs/baseline/patchcore_config.yaml \
    --output_dir outputs/centralized

# Generate anomaly maps
python experiments/scripts/evaluate_all.py \
    --dataset_dir /path/to/autovi \
    --models_dir outputs/centralized \
    --output_dir outputs/evaluation \
    --methods centralized
```

**Output**: `outputs/evaluation/metrics/centralized/`

---

## 2. Federated + DP (ε=1.0)

**Purpose**: Strong privacy (low epsilon = more noise)

**Config modification**: Set `epsilon: 1.0` in DP config

**Command**:
```bash
# Modify config or pass as argument
python scripts/train_federated.py \
    --config experiments/configs/federated/fedavg_dp_config.yaml \
    --dp_epsilon 1.0 \
    --output_dir outputs/federated/dp_eps1

# Evaluate
python experiments/scripts/evaluate_all.py \
    --dataset_dir /path/to/autovi \
    --models_dir outputs/federated/dp_eps1 \
    --output_dir outputs/evaluation \
    --methods federated_dp_eps1
```

**Output**: `outputs/evaluation/metrics/federated_dp_eps1/`

---

## 3. Federated + DP (ε=5.0)

**Purpose**: Moderate privacy

**Config modification**: Set `epsilon: 5.0` in DP config

**Command**:
```bash
python scripts/train_federated.py \
    --config experiments/configs/federated/fedavg_dp_config.yaml \
    --dp_epsilon 5.0 \
    --output_dir outputs/federated/dp_eps5

python experiments/scripts/evaluate_all.py \
    --dataset_dir /path/to/autovi \
    --models_dir outputs/federated/dp_eps5 \
    --output_dir outputs/evaluation \
    --methods federated_dp_eps5
```

**Output**: `outputs/evaluation/metrics/federated_dp_eps5/`

---

## 4. Federated + DP (ε=10.0)

**Purpose**: Weak privacy (high epsilon = less noise)

**Config modification**: Set `epsilon: 10.0` in DP config

**Command**:
```bash
python scripts/train_federated.py \
    --config experiments/configs/federated/fedavg_dp_config.yaml \
    --dp_epsilon 10.0 \
    --output_dir outputs/federated/dp_eps10

python experiments/scripts/evaluate_all.py \
    --dataset_dir /path/to/autovi \
    --models_dir outputs/federated/dp_eps10 \
    --output_dir outputs/evaluation \
    --methods federated_dp_eps10
```

**Output**: `outputs/evaluation/metrics/federated_dp_eps10/`

---

## 5. Federated + Robust (clean)

**Purpose**: Robust aggregation without attack (baseline for robustness)

**Config**: Enable coordinate median aggregation, no attack

**Command**:
```bash
python scripts/train_federated.py \
    --config experiments/configs/federated/fedavg_iid_config.yaml \
    --robust_aggregation coordinate_median \
    --output_dir outputs/federated/robust_clean

python experiments/scripts/evaluate_all.py \
    --dataset_dir /path/to/autovi \
    --models_dir outputs/federated/robust_clean \
    --output_dir outputs/evaluation \
    --methods federated_robust_clean
```

**Output**: `outputs/evaluation/metrics/federated_robust_clean/`

---

## 6. Federated + Robust (under attack)

**Purpose**: Test robustness under Byzantine attack

**Config**: Enable coordinate median + simulate 20% malicious clients with scaling attack

**Command**:
```bash
python scripts/train_federated.py \
    --config experiments/configs/federated/fedavg_iid_config.yaml \
    --robust_aggregation coordinate_median \
    --simulate_attack scaling \
    --malicious_fraction 0.2 \
    --output_dir outputs/federated/robust_attack

python experiments/scripts/evaluate_all.py \
    --dataset_dir /path/to/autovi \
    --models_dir outputs/federated/robust_attack \
    --output_dir outputs/evaluation \
    --methods federated_robust_attack
```

**Output**: `outputs/evaluation/metrics/federated_robust_attack/`

---

## After Running All Experiments

Re-run the trade-off analysis to generate updated results:

```bash
python experiments/scripts/trade_off_analysis.py \
    --metrics_dir outputs/evaluation/metrics \
    --robustness_dir results/robustness \
    --output_dir results
```

This will update:
- `results/trade_off_table.csv`
- `results/trade_off_plot.png`
- `results/trade_off_summary.md`
- `results/comparison_analysis.json`

---

## Notes

1. **CUDA OOM**: If centralized training fails, try:
   - Reduce batch size: `--batch_size 16` or `--batch_size 8`
   - Process objects sequentially instead of in parallel
   - Use gradient checkpointing if available

2. **Dataset path**: Replace `/path/to/autovi` with actual dataset location

3. **Verify configs**: Check that training scripts accept the command-line arguments shown above, or modify the YAML configs directly

4. **Robustness evaluation**: The synthetic robustness experiments (`results/robustness/`) are already complete (39 experiments). These new experiments test robustness in the actual federated training pipeline.

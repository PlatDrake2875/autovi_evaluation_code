#!/bin/bash
# Run all missing experiments sequentially to avoid GPU OOM

set -e
cd /home/adrian/autovi_evaluation_code

CATEGORIES="engine_wiring pipe_clip pipe_staple tank_screw underbody_pipes underbody_screw"

echo "=============================================="
echo "Running all experiments sequentially"
echo "=============================================="

# 1. Centralized baseline
echo ""
echo "=== EXPERIMENT 1: Centralized Baseline ==="
~/.local/bin/uv run python scripts/train_centralized.py \
    --config experiments/configs/baseline/patchcore_config.yaml \
    --data_dir dataset \
    --output_dir outputs/centralized

# 2. Federated + DP (epsilon=1.0)
echo ""
echo "=== EXPERIMENT 2: Federated + DP (eps=1.0) ==="
for cat in $CATEGORIES; do
    echo "  Training $cat..."
    ~/.local/bin/uv run python scripts/train_federated.py \
        --config experiments/configs/federated/fedavg_dp_config.yaml \
        --data_root dataset \
        --dp_epsilon 1.0 \
        --categories $cat \
        --output_dir outputs/federated/dp_eps1/$cat
done

# 3. Federated + DP (epsilon=5.0)
echo ""
echo "=== EXPERIMENT 3: Federated + DP (eps=5.0) ==="
for cat in $CATEGORIES; do
    echo "  Training $cat..."
    ~/.local/bin/uv run python scripts/train_federated.py \
        --config experiments/configs/federated/fedavg_dp_config.yaml \
        --data_root dataset \
        --dp_epsilon 5.0 \
        --categories $cat \
        --output_dir outputs/federated/dp_eps5/$cat
done

# 4. Federated + DP (epsilon=10.0)
echo ""
echo "=== EXPERIMENT 4: Federated + DP (eps=10.0) ==="
for cat in $CATEGORIES; do
    echo "  Training $cat..."
    ~/.local/bin/uv run python scripts/train_federated.py \
        --config experiments/configs/federated/fedavg_dp_config.yaml \
        --data_root dataset \
        --dp_epsilon 10.0 \
        --categories $cat \
        --output_dir outputs/federated/dp_eps10/$cat
done

# 5. Federated + Robust (clean)
echo ""
echo "=== EXPERIMENT 5: Federated + Robust (clean) ==="
for cat in $CATEGORIES; do
    echo "  Training $cat..."
    ~/.local/bin/uv run python scripts/train_federated.py \
        --config experiments/configs/federated/fedavg_iid_config.yaml \
        --data_root dataset \
        --robust_aggregation coordinate_median \
        --categories $cat \
        --output_dir outputs/federated/robust_clean/$cat
done

# 6. Federated + Robust (attack)
echo ""
echo "=== EXPERIMENT 6: Federated + Robust (attack) ==="
for cat in $CATEGORIES; do
    echo "  Training $cat..."
    ~/.local/bin/uv run python scripts/train_federated.py \
        --config experiments/configs/federated/fedavg_iid_config.yaml \
        --data_root dataset \
        --robust_aggregation coordinate_median \
        --simulate_attack scaling \
        --malicious_fraction 0.2 \
        --categories $cat \
        --output_dir outputs/federated/robust_attack/$cat
done

echo ""
echo "=============================================="
echo "All experiments complete!"
echo "=============================================="

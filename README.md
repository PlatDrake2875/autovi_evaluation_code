# Evaluate your experiments on the AutoVI dataset

The evaluation scripts can be used to assess the performance of a method on the AutoVI dataset.
Given a directory with anomaly maps, the scripts compute the area under the sPRO curve for anomaly localization.

The dataset can be found at the following address: [https://doi.org/10.5281/zenodo.10459003](https://doi.org/10.5281/zenodo.10459003)

This code is adapted from the code made available by MVTec GmbH at [https://www.mvtec.com/company/research/datasets/mvtec-loco](https://www.mvtec.com/company/research/datasets/mvtec-loco)

## Installation
Our evaluation scripts require Python 3.10+ and the following packages:
- numpy
- pillow
- tqdm
- tabulate

Install dependencies using [uv](https://docs.astral.sh/uv/):
```
uv sync
```

Or with pip:
```
pip install -e .
```

## Evaluating a single experiment.
The evaluation script requires an anomaly map to be present for each test sample in our dataset in `.png` format. 
Anomaly maps must contain real-valued anomaly scores and their size must match the one of the corresponding dataset images. 
Anomaly maps must all share the same base directory and adhere to the following folder structure: 
`<anomaly_maps_dir>/<object_name>/test/<defect_name>/<image_id>.png`

To evaluate a single experiment on one of the dataset objects, the script `evaluate_experiment.py` can be used.  
It requires the following user arguments:
- `object_name`: Name of the dataset object to be evaluated.
- `dataset_base_dir`: Base directory that contains the AutoVI dataset.
- `anomaly_maps_dir`: Base directory that contains the corresponding anomaly maps.
- `output_dir`: Directory to store evaluation results as `.json` files.

A possible example call to this script would be:
```
python evaluate_experiment.py \
    --object_name pushpins \
    --dataset_base_dir 'path/to/dataset/' \
    --anomaly_maps_dir 'path/to/anomaly_maps/' \
    --output_dir 'metrics/'
```

The evaluation script computes the area under the sPRO curve up to a limited false positive rate as described in our paper. 
The integration limits are specified by the variable `MAX_FPRS`.

## Evaluate multiple experiments

If more than one experiment should be evaluated simultaneously, the script `evaluate_multiple_experiments.py` can be used. 
Multiple directories conatining anomaly maps should be specified in a `config.json` file with the following structure:
```
{
    "exp_base_dir": "/path/to/all/experiments/",
    "anomaly_maps_dirs": {
    "experiment_id_1": "eg/model_1/anomaly_maps/",
    "experiment_id_2": "eg/model_2/anomaly_maps/",
    "experiment_id_3": "eg/model_3/anomaly_maps/",
    "...": "..."
    }
}
```
- `exp_base_dir`: Base directory that contains all experimental results for each evaluated method.
- `anomaly_maps_dirs`: Dictionary that contains an identifier for each evaluated experiment and the location of its anomaly maps relative to the `exp_base_dir`.

The evaluation is run by calling `evaluate_multiple_experiments.py` with the following user arguments:
- `dataset_base_dir`: Base directory that contains the AutoVI dataset.
- `experiment_configs`: Path to the above `config.json` that contains all experiments to be evaluated.
- `output_dir`: Directory to store evaluation results as `.json` files.

A possible example call to this script would be:
```
  python evaluate_multiple_experiments.py \
    --dataset_base_dir 'path/to/dataset/' \
    --experiment_configs 'configs.json' \
    --output_dir 'metrics/'
```

## Visualize the evaluation results.
After running `evaluate_experiment.py` or `evaluate_multiple_experiments.py`, the script `print_metrics.py`  can be used to visualize all computed metrics in a table. 
In total, three tables are printed to the standard output. The first two tables display the performance for the structural and logical anomalies, respectively. 
The third table shows the mean performance over both anomaly types.

The script requires the following user arguments:
- `metrics_folder`: The base directory that contains the computed metrics for each evaluated method. This directory is usually identical to the output directory specified in `evaluate_experiment.py` or `evaluate_multiple_experiments.py`.
- `metric_type`: Select either `localization` or `classification`. When selecting `localization`,
the AUC-sPRO results for the pixelwise localization of anomalies is shown. When selecting `classification`, the image level AUC-ROC for anomaly classification is shown.
- `integration_limit`: The integration limit until which the area under the sPRO curve is computed. This parameter is only applicable when `metric_type` is set to `localization`.

---

# Stage 2: Trustworthiness Enhancements

This project extends the baseline evaluation framework with trustworthiness features for federated learning:

- **Differential Privacy (DP)**: Privacy-preserving federated learning with configurable epsilon/delta budgets
- **Robustness**: Byzantine-robust aggregation and attack detection
- **Fairness**: Client and category performance parity evaluation

## Trustworthiness Modules

### Privacy (`src/privacy/`)

Implements differential privacy for federated PatchCore with Gaussian mechanism:

```python
from src.privacy import DPConfig, EmbeddingSanitizer, PrivacyAccountant

# Configure DP
dp_config = DPConfig(
    epsilon=5.0,       # Privacy budget (lower = more private)
    delta=1e-5,        # Failure probability
    clipping_norm=1.0  # L2 norm clipping bound
)

# Create sanitizer
sanitizer = EmbeddingSanitizer(dp_config)

# Apply DP to embeddings
private_embeddings = sanitizer.sanitize(embeddings)

# Track privacy budget
accountant = PrivacyAccountant(epsilon=5.0, delta=1e-5)
accountant.step()
print(f"Remaining budget: {accountant.remaining_epsilon}")
```

### Robustness (`src/robustness/`)

Byzantine-robust aggregation with attack detection:

```python
from src.robustness import (
    RobustnessConfig,
    CoordinateMedianAggregator,
    ZScoreDetector,
    ModelPoisoningAttack,
)

# Configure robust aggregation
config = RobustnessConfig(
    enabled=True,
    aggregation_method="coordinate_median",  # Robust to 50% malicious
    client_scoring_method="zscore",          # Detect anomalous clients
    zscore_threshold=2.5,
)

# Create aggregator
aggregator = CoordinateMedianAggregator()
robust_result = aggregator.aggregate(client_updates)

# Detect malicious clients
detector = ZScoreDetector(threshold=2.5)
scores = detector.score_clients(client_updates)
outliers = [s for s in scores if s["is_outlier"]]

# Simulate attacks (for testing)
attack = ModelPoisoningAttack(attack_type="scaling", scale_factor=100.0)
attacked_data = attack.apply(client_data, malicious_indices=[0, 1])
```

### Fairness (`src/fairness/`)

Evaluate performance parity across clients and categories:

```python
from src.fairness import FairnessConfig, compute_all_metrics

# Compute fairness metrics from per-client AUC scores
client_performances = {
    "client_0": 0.85,
    "client_1": 0.82,
    "client_2": 0.78,
    "client_3": 0.88,
    "client_4": 0.80,
}

metrics = compute_all_metrics(client_performances)
print(f"Jain's Fairness Index: {metrics.jains_index:.4f}")  # 1.0 = perfect
print(f"Performance Gap: {metrics.performance_gap:.4f}")   # max - min
print(f"Worst-Case AUC: {metrics.worst_case:.4f}")         # Rawlsian fairness
print(f"Coefficient of Variation: {metrics.coefficient_of_variation:.4f}")
```

## Running Experiments

### Important: Per-Category Training

PatchCore builds a memory bank of "normal" patch features for anomaly detection. Because different object categories (e.g., engine_wiring vs tank_screw) have fundamentally different visual characteristics, **federated training must be performed separately for each category**.

Training a single model across all categories would mix patch features from different object types, making the concept of "normal" meaningless - the memory bank would contain unrelated features that don't represent normalcy for any specific object.

**Correct approach:**
```bash
# Train federated model for each category separately
for category in engine_wiring pipe_clip pipe_staple tank_screw underbody_pipes underbody_screw; do
    python scripts/train_federated.py \
        --config experiments/configs/federated/fedavg_dp_config.yaml \
        --data_root /path/to/autovi \
        --categories $category \
        --output_dir outputs/federated/dp/$category
done
```

This also naturally handles the different image sizes in the dataset:
- Small objects (400×400): engine_wiring, pipe_clip, pipe_staple
- Large objects (1000×750): tank_screw, underbody_pipes, underbody_screw

### Trade-off Analysis

Analyze accuracy vs privacy vs robustness trade-offs:

```bash
python experiments/scripts/trade_off_analysis.py \
    --metrics_dir outputs/evaluation/metrics \
    --robustness_dir results/robustness \
    --output_dir results
```

Outputs:
- `results/trade_off_table.csv` - Main comparison table
- `results/trade_off_plot.png` - Visualization
- `results/trade_off_summary.md` - Summary report

### Robustness Evaluation

Test robustness against Byzantine attacks:

```bash
python experiments/scripts/robustness_evaluation.py \
    --output_dir results/robustness \
    --num_clients 10 \
    --malicious_fractions 0.1 0.2 0.3 0.4 \
    --attack_types scaling noise sign_flip
```

### Comparison Reports

Generate comprehensive comparison reports:

```bash
python experiments/scripts/generate_comparison_report.py \
    --metrics_dir outputs/evaluation/metrics \
    --output_dir results \
    --format all
```

## Configuration Files

Example configurations in `experiments/configs/`:

| Config | Description |
|--------|-------------|
| `baseline/patchcore_config.yaml` | Centralized PatchCore |
| `federated/fedavg_iid_config.yaml` | Federated with IID partitioning |
| `federated/fedavg_dp_config.yaml` | Federated with Differential Privacy |
| `federated/fedavg_category_config.yaml` | Federated with category-based partitioning |

### DP Configuration Example

```yaml
differential_privacy:
  enabled: true
  epsilon: 5.0        # Privacy parameter (1.0-10.0)
  delta: 1e-5         # Failure probability
  clipping_norm: 1.0  # L2 norm clipping
```

## Project Structure

```
src/
├── data/                 # Dataset loading and preprocessing
├── models/               # PatchCore model and memory bank
├── training/             # Training configuration and setup
├── evaluation/           # Metrics, visualization, anomaly scoring
├── federated/            # Federated learning client/server
├── privacy/              # Differential privacy mechanisms
├── robustness/           # Byzantine-robust aggregation
└── fairness/             # Fairness metrics and evaluation

experiments/
├── configs/              # YAML configuration files
└── scripts/              # Evaluation and analysis scripts

results/                  # Generated outputs and reports
```

---

# License
The license agreement for our evaluation code is found in the accompanying
`LICENSE.txt` file.

The version of this evaluation script is: 3.0

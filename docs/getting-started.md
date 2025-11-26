# Getting Started

This guide walks you through setting up the development environment for the Federated Learning anomaly detection project.

---

## Prerequisites

- Python 3.10+
- CUDA 11.8+ (optional, for GPU acceleration)
- Git

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-team/federated-autovi.git
cd federated-autovi
```

### 2. Create Virtual Environment

Using `uv` (recommended):
```bash
uv venv
source .venv/bin/activate
uv sync
```

Using `pip`:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 3. Download AutoVI Dataset

Download from Zenodo: https://doi.org/10.5281/zenodo.10459003

```bash
# Extract to data directory
mkdir -p data/autovi
# Extract downloaded archive to data/autovi/
```

Expected structure:
```
data/autovi/
├── engine_wiring/
│   ├── train/good/
│   ├── test/{good,structural_anomalies,logical_anomalies}/
│   ├── ground_truth/
│   └── defects_config.json
├── pipe_clip/
├── pipe_staple/
├── tank_screw/
├── underbody_pipes/
└── underbody_screw/
```

### 4. Verify Installation

```bash
# Run tests
pytest tests/ -v

# Check dataset loading
python -c "from src.data import AutoVIDataset; print('OK')"
```

---

## Project Structure

```
federated-autovi/
├── src/                    # Source code
│   ├── data/              # Dataset and partitioning
│   ├── models/            # PatchCore implementation
│   ├── federated/         # FL client/server
│   └── evaluation/        # Metrics wrapper
├── experiments/           # Experiment scripts and configs
├── notebooks/             # Jupyter notebooks
├── docs/                  # Documentation (you are here)
└── tests/                 # Unit tests
```

---

## Quick Start

### Run Centralized Baseline

```bash
python experiments/scripts/train_centralized.py \
    --config experiments/configs/baseline/patchcore_config.yaml
```

### Run Federated Experiment

```bash
python experiments/scripts/train_federated.py \
    --config experiments/configs/federated/fedavg_category_config.yaml
```

### Evaluate Model

```bash
python experiments/scripts/evaluate_all.py \
    --model_path outputs/models/federated_patchcore.pt \
    --output_dir outputs/results/
```

---

## Configuration

### Environment Variables

Create `.env` file:
```bash
AUTOVI_DATA_DIR=/path/to/data/autovi
OUTPUT_DIR=/path/to/outputs
NUM_WORKERS=4
DEVICE=cuda  # or cpu
```

### Experiment Config

See `experiments/configs/` for YAML configuration files:
- `baseline/patchcore_config.yaml` - Centralized training
- `federated/fedavg_iid_config.yaml` - IID partitioning
- `federated/fedavg_category_config.yaml` - Category-based partitioning

---

## Dependencies

```toml
[project.dependencies]
torch = ">=2.0.0"
torchvision = ">=0.15.0"
numpy = ">=1.24.0"
pillow = ">=10.0.0"
tqdm = ">=4.65.0"
tabulate = ">=0.9.0"
pyyaml = ">=6.0"
matplotlib = ">=3.7.0"
scikit-learn = ">=1.3.0"
faiss-cpu = ">=1.7.0"
flwr = ">=1.5.0"
```

---

## Troubleshooting

### CUDA Out of Memory

Reduce batch size in config:
```yaml
training:
  batch_size: 16  # Reduce from 32
```

### FAISS Installation Issues

```bash
# CPU version (recommended)
pip install faiss-cpu

# GPU version (requires CUDA)
pip install faiss-gpu
```

### Dataset Not Found

Ensure `AUTOVI_DATA_DIR` environment variable is set correctly and dataset is extracted.

---

## Next Steps

1. [Phase 1: Data Preparation](phases/phase-1-data-preparation/README.md) - Start here
2. [Phase 2: Baseline Model](phases/phase-2-baseline-model/README.md)
3. [Phase 3: Federated Setup](phases/phase-3-federated-setup/README.md)
4. [Phase 4: Evaluation](phases/phase-4-evaluation/README.md)

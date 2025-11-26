# API Documentation

> Code module documentation for the Federated PatchCore project.

---

## Module Overview

```
src/
├── data/           # Data loading and partitioning
├── models/         # PatchCore model implementation
├── federated/      # FL client/server components
└── evaluation/     # Metrics and visualization
```

---

## Quick Links

| Module | Description | Documentation |
|--------|-------------|---------------|
| Data | Dataset loading, preprocessing, FL partitioning | [data-module.md](data-module.md) |
| Models | PatchCore, backbone, memory bank | [model-module.md](model-module.md) |
| Federated | Client, server, aggregation strategies | [federated-module.md](federated-module.md) |
| Evaluation | Metrics wrapper, anomaly scoring | [evaluation-module.md](evaluation-module.md) |

---

## Installation

```bash
pip install -e .
```

## Basic Usage

```python
from src.data import AutoVIDataset, CategoryPartitioner
from src.models import PatchCore
from src.federated import FederatedServer, PatchCoreClient
from src.evaluation import evaluate_model

# Load data
dataset = AutoVIDataset(root="/path/to/autovi", objects=["engine_wiring"])

# Create partitions
partitioner = CategoryPartitioner(num_clients=5)
client_data = partitioner.partition(dataset)

# Train centralized baseline
model = PatchCore(backbone="wide_resnet50_2")
model.fit(dataset)

# Or train federated
server = FederatedServer(num_clients=5)
clients = [PatchCoreClient(i, data) for i, data in enumerate(client_data)]
global_model = server.train(clients)

# Evaluate
results = evaluate_model(model, test_dataset, output_dir="outputs/")
```

---

## Configuration

All modules support YAML configuration:

```yaml
# config.yaml
data:
  root: "/path/to/autovi"
  objects: ["engine_wiring", "pipe_clip"]

model:
  backbone: "wide_resnet50_2"
  coreset_percentage: 0.1

federated:
  num_clients: 5
  partitioning: "category_based"

evaluation:
  max_fprs: [0.01, 0.05, 0.1, 0.3, 1.0]
```

```python
from src.utils.config import load_config

config = load_config("config.yaml")
```

# Federated Module API

> `src/federated/` - FL client/server implementation.

---

## Classes

### FederatedServer

Central aggregation server.

```python
class FederatedServer:
    """
    Federated learning server for PatchCore aggregation.

    Args:
        num_clients (int): Expected number of clients.
        global_bank_size (int): Target global memory bank size.
        aggregation_strategy (str): "coreset" or "random".
        weighted (bool): Weight by client data size.

    Example:
        >>> server = FederatedServer(num_clients=5, global_bank_size=10000)
        >>> global_model = server.aggregate(client_coresets, client_sizes)
    """

    def __init__(self, num_clients=5, global_bank_size=10000,
                 aggregation_strategy="coreset", weighted=True):
        ...

    def aggregate(self, client_coresets, client_sizes=None):
        """
        Aggregate client memory banks.

        Args:
            client_coresets (list[np.ndarray]): Local coresets from clients.
            client_sizes (list[int], optional): Client dataset sizes for weighting.

        Returns:
            np.ndarray: Global memory bank features.
        """
        ...

    def train_round(self, clients):
        """
        Execute one federated training round.

        Args:
            clients (list[PatchCoreClient]): Connected clients.

        Returns:
            np.ndarray: Updated global memory bank.
        """
        ...

    def get_global_model(self):
        """Return current global memory bank."""
        ...


class FederatedPatchCore:
    """
    High-level federated PatchCore trainer.

    Args:
        num_clients (int): Number of federated clients.
        backbone (str): Feature extractor backbone.
        global_bank_size (int): Target global memory bank size.

    Example:
        >>> fed_model = FederatedPatchCore(num_clients=5)
        >>> fed_model.setup_clients(client_datasets)
        >>> fed_model.train()
        >>> anomaly_maps = fed_model.predict(test_dataset)
    """

    def __init__(self, num_clients=5, backbone="wide_resnet50_2",
                 global_bank_size=10000):
        ...

    def setup_clients(self, client_datasets):
        """Initialize clients with their data partitions."""
        ...

    def train(self, num_rounds=1):
        """
        Execute federated training.

        Args:
            num_rounds (int): Number of communication rounds.
                Note: PatchCore typically needs only 1 round.

        Returns:
            self
        """
        ...

    def predict(self, dataset):
        """Generate anomaly maps using global model."""
        ...

    def save(self, path):
        """Save federated model."""
        ...
```

### PatchCoreClient

Federated client implementation.

```python
class PatchCoreClient:
    """
    Federated learning client for PatchCore.

    Args:
        client_id (int): Unique client identifier.
        dataset: Local training dataset.
        backbone (str): Feature extractor (shared across clients).
        coreset_percentage (float): Local coreset size.

    Example:
        >>> client = PatchCoreClient(0, local_dataset)
        >>> local_coreset = client.train()
        >>> client.update_global_bank(global_features)
    """

    def __init__(self, client_id, dataset, backbone="wide_resnet50_2",
                 coreset_percentage=0.1):
        ...

    def train(self):
        """
        Extract features and build local coreset.

        Returns:
            np.ndarray: Local memory bank coreset.
        """
        ...

    def get_coreset(self):
        """Return current local coreset."""
        ...

    def update_global_bank(self, global_features):
        """
        Update local model with global memory bank.

        Args:
            global_features (np.ndarray): Aggregated global features.
        """
        ...

    def predict(self, dataset):
        """Generate anomaly maps using current model."""
        ...

    @property
    def data_size(self):
        """Return local dataset size."""
        ...
```

---

## Aggregation Strategies

```python
# src/federated/strategies/

def federated_coreset_aggregation(client_coresets, global_size,
                                   client_weights=None):
    """
    Aggregate local coresets using global coreset selection.

    Args:
        client_coresets (list[np.ndarray]): Local coresets.
        global_size (int): Target global memory bank size.
        client_weights (list[float], optional): Client importance weights.

    Returns:
        np.ndarray: Global memory bank.
    """
    ...


def federated_random_aggregation(client_coresets, global_size,
                                  client_weights=None):
    """
    Aggregate by random sampling (baseline).
    """
    ...


def federated_weighted_union(client_coresets, client_weights):
    """
    Weighted union without global selection.
    Maintains all client features with importance weights.
    """
    ...
```

---

## Communication Protocol

```python
class CommunicationProtocol:
    """
    Simulated communication between server and clients.

    For actual deployment, replace with gRPC/REST implementation.
    """

    @staticmethod
    def send_to_server(client_id, data):
        """Simulate client sending data to server."""
        ...

    @staticmethod
    def broadcast_to_clients(clients, data):
        """Simulate server broadcasting to all clients."""
        ...

    @staticmethod
    def compute_communication_cost(data):
        """Estimate communication cost in bytes."""
        ...
```

---

## Flower Integration (Optional)

```python
# src/federated/flower_client.py

import flwr as fl

class FlowerPatchCoreClient(fl.client.NumPyClient):
    """
    Flower-compatible PatchCore client.

    Example:
        >>> client = FlowerPatchCoreClient(dataset, backbone)
        >>> fl.client.start_numpy_client(
        ...     server_address="localhost:8080",
        ...     client=client
        ... )
    """

    def __init__(self, dataset, backbone):
        self.model = PatchCore(backbone)
        self.dataset = dataset

    def get_parameters(self, config):
        """Return local coreset as parameters."""
        return [self.model.get_memory_bank()]

    def fit(self, parameters, config):
        """Train local model."""
        self.model.fit(self.dataset)
        return self.get_parameters(config), len(self.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate using global model."""
        global_bank = parameters[0]
        self.model.set_memory_bank(global_bank)
        # Evaluation logic...
        return loss, len(self.dataset), {"accuracy": acc}
```

---

## Usage Examples

### Basic Federated Training

```python
from src.federated import FederatedPatchCore
from src.data import AutoVIDataset, CategoryPartitioner

# Load and partition data
dataset = AutoVIDataset(root="/data/autovi", objects=all_objects, split="train")
partitioner = CategoryPartitioner(num_clients=5)
client_datasets = partitioner.partition(dataset)

# Create federated model
fed_model = FederatedPatchCore(
    num_clients=5,
    backbone="wide_resnet50_2",
    global_bank_size=10000
)

# Setup clients and train
fed_model.setup_clients(client_datasets)
fed_model.train(num_rounds=1)

# Save global model
fed_model.save("outputs/federated/global_model.pt")
```

### Manual Client/Server Control

```python
from src.federated import FederatedServer, PatchCoreClient

# Create clients
clients = []
for i, client_data in enumerate(client_datasets):
    client = PatchCoreClient(
        client_id=i,
        dataset=client_data,
        coreset_percentage=0.1
    )
    clients.append(client)

# Train locally
local_coresets = []
for client in clients:
    coreset = client.train()
    local_coresets.append(coreset)

# Server aggregation
server = FederatedServer(num_clients=5, global_bank_size=10000)
global_bank = server.aggregate(
    local_coresets,
    client_sizes=[c.data_size for c in clients]
)

# Distribute global model
for client in clients:
    client.update_global_bank(global_bank)
```

### Communication Analysis

```python
from src.federated import CommunicationProtocol

# Compute communication costs
for i, coreset in enumerate(local_coresets):
    cost = CommunicationProtocol.compute_communication_cost(coreset)
    print(f"Client {i} upload: {cost / 1e6:.2f} MB")

global_cost = CommunicationProtocol.compute_communication_cost(global_bank)
print(f"Global broadcast: {global_cost / 1e6:.2f} MB")
```

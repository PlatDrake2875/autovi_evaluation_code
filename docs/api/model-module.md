# Model Module API

> `src/models/` - PatchCore model implementation.

---

## Classes

### PatchCore

Main anomaly detection model.

```python
class PatchCore:
    """
    PatchCore anomaly detection model.

    Based on "Towards Total Recall in Industrial Anomaly Detection" (CVPR 2022).

    Args:
        backbone (str): Feature extractor backbone.
            Options: "wide_resnet50_2", "resnet50", "efficientnet_b4"
        layers (list[str]): Layers to extract features from.
            Default: ["layer2", "layer3"]
        coreset_percentage (float): Fraction of patches for memory bank.
            Default: 0.1 (10%)
        device (str): "cuda" or "cpu".

    Attributes:
        backbone (nn.Module): Feature extractor network.
        memory_bank (MemoryBank): Stored normal patch features.

    Example:
        >>> model = PatchCore(backbone="wide_resnet50_2")
        >>> model.fit(train_dataset)
        >>> anomaly_maps = model.predict(test_dataset)
    """

    def __init__(self, backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                 coreset_percentage=0.1, device="cuda"):
        ...

    def fit(self, dataset):
        """
        Fit model on normal training data.

        Args:
            dataset: PyTorch dataset of normal images.

        Returns:
            self
        """
        ...

    def predict(self, dataset):
        """
        Generate anomaly maps for test images.

        Args:
            dataset: PyTorch dataset of test images.

        Returns:
            list[np.ndarray]: Anomaly maps, one per image.
        """
        ...

    def predict_single(self, image):
        """
        Generate anomaly map for a single image.

        Args:
            image (torch.Tensor): Input image [3, H, W].

        Returns:
            np.ndarray: Anomaly map [H, W].
        """
        ...

    def save(self, path):
        """Save model (backbone config + memory bank)."""
        ...

    @classmethod
    def load(cls, path):
        """Load saved model."""
        ...

    def get_memory_bank(self):
        """Return memory bank features as numpy array."""
        ...

    def set_memory_bank(self, features):
        """Set memory bank from external features (for FL)."""
        ...
```

### FeatureExtractor

Backbone feature extraction.

```python
class FeatureExtractor(nn.Module):
    """
    Multi-scale feature extractor using pre-trained backbone.

    Args:
        backbone_name (str): Name of backbone architecture.
        layers (list[str]): Layers to extract features from.
        pretrained (bool): Use ImageNet pretrained weights.

    Example:
        >>> extractor = FeatureExtractor("wide_resnet50_2", ["layer2", "layer3"])
        >>> features = extractor(images)  # dict of layer features
    """

    def __init__(self, backbone_name, layers, pretrained=True):
        ...

    def forward(self, x):
        """
        Extract features from input images.

        Args:
            x (torch.Tensor): Input images [B, 3, H, W].

        Returns:
            dict[str, torch.Tensor]: Features per layer.
        """
        ...

    def get_feature_dim(self):
        """Return concatenated feature dimension."""
        ...
```

### MemoryBank

Coreset memory storage.

```python
class MemoryBank:
    """
    Memory bank for storing normal patch features.

    Uses FAISS for efficient nearest neighbor search.

    Args:
        feature_dim (int): Dimension of feature vectors.
        max_size (int, optional): Maximum bank size.

    Attributes:
        features (np.ndarray): Stored features [N, D].
        index (faiss.Index): FAISS search index.

    Example:
        >>> bank = MemoryBank(feature_dim=1536)
        >>> bank.add(features)
        >>> distances, indices = bank.search(query_features, k=1)
    """

    def __init__(self, feature_dim, max_size=None):
        ...

    def add(self, features):
        """
        Add features to memory bank.

        Args:
            features (np.ndarray): Features to add [N, D].
        """
        ...

    def search(self, query, k=1):
        """
        Search for nearest neighbors.

        Args:
            query (np.ndarray): Query features [M, D].
            k (int): Number of neighbors.

        Returns:
            tuple: (distances [M, k], indices [M, k])
        """
        ...

    def build_index(self):
        """Rebuild FAISS index after adding features."""
        ...

    def __len__(self):
        """Return number of stored features."""
        ...

    def save(self, path):
        """Save memory bank to file."""
        ...

    @classmethod
    def load(cls, path):
        """Load memory bank from file."""
        ...
```

---

## Functions

### Coreset Selection

```python
def greedy_coreset_selection(features, target_size, seed=42):
    """
    Greedy k-center coreset selection.

    Selects diverse subset of features maximizing coverage.

    Args:
        features (np.ndarray): All features [N, D].
        target_size (int): Number of features to select.
        seed (int): Random seed.

    Returns:
        np.ndarray: Selected features [target_size, D].

    Example:
        >>> all_features = np.random.randn(10000, 1536)
        >>> coreset = greedy_coreset_selection(all_features, 1000)
    """
    ...


def random_coreset_selection(features, target_size, seed=42):
    """
    Random coreset selection (baseline).

    Args:
        features (np.ndarray): All features [N, D].
        target_size (int): Number of features to select.

    Returns:
        np.ndarray: Selected features [target_size, D].
    """
    ...
```

### Feature Processing

```python
def aggregate_features(layer_features, target_size):
    """
    Aggregate multi-scale features.

    Upsamples smaller features and concatenates.

    Args:
        layer_features (dict): Features per layer.
        target_size (tuple): Target spatial size (H, W).

    Returns:
        torch.Tensor: Aggregated features [B, C, H, W].
    """
    ...


def extract_patch_features(features):
    """
    Reshape spatial features to patch format.

    Args:
        features (torch.Tensor): Spatial features [B, C, H, W].

    Returns:
        torch.Tensor: Patch features [B*H*W, C].
    """
    ...
```

---

## Usage Examples

### Train Centralized Model

```python
from src.models import PatchCore
from src.data import AutoVIDataset

# Load training data
train_dataset = AutoVIDataset(
    root="/data/autovi",
    objects=["engine_wiring"],
    split="train"
)

# Create and train model
model = PatchCore(
    backbone="wide_resnet50_2",
    coreset_percentage=0.1,
    device="cuda"
)
model.fit(train_dataset)

# Save model
model.save("outputs/models/patchcore_engine_wiring.pt")
```

### Generate Anomaly Maps

```python
# Load test data
test_dataset = AutoVIDataset(
    root="/data/autovi",
    objects=["engine_wiring"],
    split="test"
)

# Generate anomaly maps
anomaly_maps = model.predict(test_dataset)

# Save maps
for i, (_, label, path) in enumerate(test_dataset):
    save_anomaly_map(anomaly_maps[i], f"outputs/anomaly_maps/{path}")
```

### Access Memory Bank for FL

```python
# Get memory bank features
memory_features = model.get_memory_bank()
print(f"Memory bank shape: {memory_features.shape}")

# Set from aggregated features (federated)
aggregated_features = aggregate_client_features(client_features)
model.set_memory_bank(aggregated_features)
```

# Data Module API

> `src/data/` - Dataset loading, preprocessing, and FL partitioning.

---

## Classes

### AutoVIDataset

PyTorch Dataset for the AutoVI dataset.

```python
class AutoVIDataset(torch.utils.data.Dataset):
    """
    AutoVI dataset loader.

    Args:
        root (str): Path to AutoVI dataset root directory.
        objects (list[str]): List of object categories to load.
            Options: ["engine_wiring", "pipe_clip", "pipe_staple",
                      "tank_screw", "underbody_pipes", "underbody_screw"]
        split (str): "train" or "test".
        transform (callable, optional): Image transforms.
        include_anomalies (bool): For test split, whether to include anomalies.

    Attributes:
        image_paths (list[str]): Paths to all images.
        labels (list[str]): Category labels.
        anomaly_labels (list[int]): 0=good, 1=anomaly (test only).

    Example:
        >>> dataset = AutoVIDataset(
        ...     root="/data/autovi",
        ...     objects=["engine_wiring"],
        ...     split="train"
        ... )
        >>> image, label = dataset[0]
    """

    def __init__(self, root, objects, split="train", transform=None,
                 include_anomalies=True):
        ...

    def __len__(self):
        """Return number of samples."""
        ...

    def __getitem__(self, idx):
        """Return (image, label) tuple."""
        ...

    @property
    def object_counts(self):
        """Return dict of sample counts per object category."""
        ...
```

### Preprocessing

Image preprocessing transforms.

```python
def get_transform(object_name, augment=False):
    """
    Get preprocessing transforms for a given object category.

    Args:
        object_name (str): Object category name.
        augment (bool): Whether to include data augmentation.

    Returns:
        torchvision.transforms.Compose: Transform pipeline.

    Example:
        >>> transform = get_transform("engine_wiring", augment=True)
    """
    ...

# Resize dimensions per object type
RESIZE_SMALL = (400, 400)   # engine_wiring, pipe_clip, pipe_staple
RESIZE_LARGE = (1000, 750)  # tank_screw, underbody_pipes, underbody_screw

# ImageNet normalization stats
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
```

### Partitioner

FL data partitioning utilities.

```python
class IIDPartitioner:
    """
    IID (uniform random) data partitioner.

    Args:
        num_clients (int): Number of FL clients.
        seed (int): Random seed for reproducibility.

    Example:
        >>> partitioner = IIDPartitioner(num_clients=5)
        >>> client_indices = partitioner.partition(dataset)
        >>> # client_indices[0] = indices for client 0
    """

    def __init__(self, num_clients=5, seed=42):
        ...

    def partition(self, dataset):
        """
        Partition dataset indices.

        Args:
            dataset: PyTorch dataset.

        Returns:
            list[np.ndarray]: List of index arrays, one per client.
        """
        ...


class CategoryPartitioner:
    """
    Category-based (non-IID) data partitioner.

    Args:
        num_clients (int): Number of FL clients.
        assignments (dict, optional): Client-to-category mapping.
            Default: {0: ["engine_wiring"], 1: ["underbody_*"], ...}

    Example:
        >>> partitioner = CategoryPartitioner(num_clients=5)
        >>> client_data = partitioner.partition(dataset)
    """

    def __init__(self, num_clients=5, assignments=None):
        ...

    def partition(self, dataset):
        """Partition by object category."""
        ...


class DirichletPartitioner:
    """
    Dirichlet distribution partitioner for controlled non-IID.

    Args:
        num_clients (int): Number of FL clients.
        alpha (float): Dirichlet concentration parameter.
            Lower alpha = more non-IID.

    Example:
        >>> partitioner = DirichletPartitioner(num_clients=5, alpha=0.5)
    """

    def __init__(self, num_clients=5, alpha=0.5, seed=42):
        ...
```

---

## Functions

### Data Loading

```python
def load_defects_config(dataset_root, object_name):
    """
    Load defects configuration for an object category.

    Args:
        dataset_root (str): Path to dataset root.
        object_name (str): Object category name.

    Returns:
        list[dict]: Defect configurations with keys:
            - defect_name (str)
            - pixel_value (int)
            - saturation_threshold (float)
            - relative_saturation (bool)
    """
    ...


def get_ground_truth_paths(dataset_root, object_name, image_id):
    """
    Get ground truth mask paths for a test image.

    Args:
        dataset_root (str): Path to dataset root.
        object_name (str): Object category name.
        image_id (str): Test image identifier.

    Returns:
        list[str]: Paths to ground truth mask PNG files.
    """
    ...
```

---

## Usage Examples

### Load Full Dataset

```python
from src.data import AutoVIDataset, get_transform

# All categories
all_objects = [
    "engine_wiring", "pipe_clip", "pipe_staple",
    "tank_screw", "underbody_pipes", "underbody_screw"
]

# Training set (good images only)
train_dataset = AutoVIDataset(
    root="/data/autovi",
    objects=all_objects,
    split="train",
    transform=get_transform(all_objects[0])
)

# Test set (good + anomalies)
test_dataset = AutoVIDataset(
    root="/data/autovi",
    objects=all_objects,
    split="test"
)
```

### Create FL Partitions

```python
from src.data import AutoVIDataset, CategoryPartitioner
from torch.utils.data import Subset, DataLoader

# Load dataset
dataset = AutoVIDataset(root="/data/autovi", objects=all_objects, split="train")

# Partition for 5 clients
partitioner = CategoryPartitioner(num_clients=5)
client_indices = partitioner.partition(dataset)

# Create DataLoaders for each client
client_loaders = []
for indices in client_indices:
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=32, shuffle=True)
    client_loaders.append(loader)
```

### Custom Partitioning

```python
# Custom category assignments
custom_assignments = {
    0: ["engine_wiring", "pipe_clip"],
    1: ["pipe_staple", "tank_screw"],
    2: ["underbody_pipes", "underbody_screw"],
    3: ["engine_wiring"],  # Overlap allowed
    4: ["all"],  # Special: sample from all
}

partitioner = CategoryPartitioner(
    num_clients=5,
    assignments=custom_assignments
)
```

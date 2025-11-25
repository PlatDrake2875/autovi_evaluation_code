"""Federated Learning data partitioning strategies for AutoVI dataset."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .autovi_dataset import AutoVIDataset, AutoVISubset, CATEGORIES


# Category-based client assignments for Non-IID partitioning
CATEGORY_CLIENT_ASSIGNMENTS = {
    0: ["engine_wiring"],  # Client 0: Engine Assembly
    1: ["underbody_pipes", "underbody_screw"],  # Client 1: Underbody Line
    2: ["tank_screw", "pipe_staple"],  # Client 2: Fastener Station
    3: ["pipe_clip"],  # Client 3: Clip Inspection
    4: ["all"],  # Client 4: Quality Control (mixed)
}


class IIDPartitioner:
    """IID (Independent and Identically Distributed) partitioner.

    Randomly shuffles and uniformly distributes data across clients.
    """

    def __init__(self, num_clients: int = 5, seed: int = 42):
        """Initialize IID partitioner.

        Args:
            num_clients: Number of FL clients.
            seed: Random seed for reproducibility.
        """
        self.num_clients = num_clients
        self.seed = seed

    def partition(self, dataset: AutoVIDataset) -> Dict[int, List[int]]:
        """Partition dataset into IID client splits.

        Args:
            dataset: AutoVIDataset to partition.

        Returns:
            Dictionary mapping client_id -> list of sample indices.
        """
        np.random.seed(self.seed)

        # Get all indices and shuffle
        indices = np.random.permutation(len(dataset))

        # Split into equal parts
        splits = np.array_split(indices, self.num_clients)

        return {i: splits[i].tolist() for i in range(self.num_clients)}

    def create_subsets(self, dataset: AutoVIDataset) -> Dict[int, AutoVISubset]:
        """Create AutoVISubset objects for each client.

        Args:
            dataset: AutoVIDataset to partition.

        Returns:
            Dictionary mapping client_id -> AutoVISubset.
        """
        partition = self.partition(dataset)
        return {i: AutoVISubset(dataset, indices) for i, indices in partition.items()}


class CategoryPartitioner:
    """Category-based Non-IID partitioner.

    Assigns clients based on object categories, simulating realistic factory setup
    where different stations inspect different parts.
    """

    def __init__(
        self,
        client_assignments: Optional[Dict[int, List[str]]] = None,
        qc_sample_ratio: float = 0.1,
        seed: int = 42,
    ):
        """Initialize category-based partitioner.

        Args:
            client_assignments: Custom mapping of client_id -> list of categories.
                If None, uses default CATEGORY_CLIENT_ASSIGNMENTS.
            qc_sample_ratio: Ratio of samples for QC client from each category.
            seed: Random seed for reproducibility.
        """
        self.client_assignments = client_assignments or CATEGORY_CLIENT_ASSIGNMENTS
        self.qc_sample_ratio = qc_sample_ratio
        self.seed = seed

    def partition(self, dataset: AutoVIDataset) -> Dict[int, List[int]]:
        """Partition dataset based on categories.

        Args:
            dataset: AutoVIDataset to partition.

        Returns:
            Dictionary mapping client_id -> list of sample indices.
        """
        np.random.seed(self.seed)

        # Build category -> indices mapping
        category_indices: Dict[str, List[int]] = {cat: [] for cat in CATEGORIES}
        for idx, (_, _, category, _) in enumerate(dataset.samples):
            category_indices[category].append(idx)

        # Assign indices to clients
        partition: Dict[int, List[int]] = {i: [] for i in self.client_assignments}

        # Track assigned indices to avoid duplicates
        assigned = set()

        # First pass: handle QC client (sample from all categories first)
        for client_id, categories in self.client_assignments.items():
            if "all" in categories:
                # QC client gets a sample from each category FIRST
                for cat, indices in category_indices.items():
                    n_samples = max(1, int(len(indices) * self.qc_sample_ratio))
                    selected = np.random.choice(
                        indices, size=min(n_samples, len(indices)), replace=False
                    )
                    partition[client_id].extend(selected.tolist())
                    assigned.update(selected)

        # Second pass: assign remaining samples to specific clients
        for client_id, categories in self.client_assignments.items():
            if "all" not in categories:
                # Regular client gets remaining samples from assigned categories
                for cat in categories:
                    if cat in category_indices:
                        available = [i for i in category_indices[cat] if i not in assigned]
                        partition[client_id].extend(available)
                        assigned.update(available)

        return partition

    def create_subsets(self, dataset: AutoVIDataset) -> Dict[int, AutoVISubset]:
        """Create AutoVISubset objects for each client.

        Args:
            dataset: AutoVIDataset to partition.

        Returns:
            Dictionary mapping client_id -> AutoVISubset.
        """
        partition = self.partition(dataset)
        return {i: AutoVISubset(dataset, indices) for i, indices in partition.items()}


def create_partition(
    dataset: AutoVIDataset,
    strategy: str = "iid",
    num_clients: int = 5,
    seed: int = 42,
    **kwargs,
) -> Dict[int, List[int]]:
    """Create a partition using the specified strategy.

    Args:
        dataset: AutoVIDataset to partition.
        strategy: Either "iid" or "category".
        num_clients: Number of clients (only used for IID).
        seed: Random seed.
        **kwargs: Additional arguments for the partitioner.

    Returns:
        Dictionary mapping client_id -> list of sample indices.
    """
    if strategy == "iid":
        partitioner = IIDPartitioner(num_clients=num_clients, seed=seed)
    elif strategy == "category":
        partitioner = CategoryPartitioner(seed=seed, **kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'iid' or 'category'.")

    return partitioner.partition(dataset)


def save_partition(
    partition: Dict[int, List[int]],
    output_path: str,
    stats: Optional[Dict] = None,
) -> None:
    """Save partition to JSON file.

    Args:
        partition: Dictionary mapping client_id -> list of sample indices.
        output_path: Path to save the JSON file.
        stats: Optional statistics dictionary to include.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "partition": {str(k): v for k, v in partition.items()},
        "num_clients": len(partition),
        "total_samples": sum(len(v) for v in partition.values()),
    }

    if stats:
        data["stats"] = stats

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def load_partition(input_path: str) -> Dict[int, List[int]]:
    """Load partition from JSON file.

    Args:
        input_path: Path to the JSON file.

    Returns:
        Dictionary mapping client_id -> list of sample indices.
    """
    with open(input_path, "r") as f:
        data = json.load(f)

    return {int(k): v for k, v in data["partition"].items()}


def compute_partition_stats(
    dataset: AutoVIDataset,
    partition: Dict[int, List[int]],
) -> Dict:
    """Compute statistics for a partition.

    Args:
        dataset: The source AutoVIDataset.
        partition: Dictionary mapping client_id -> list of sample indices.

    Returns:
        Dictionary with partition statistics.
    """
    stats = {
        "num_clients": len(partition),
        "total_samples": sum(len(v) for v in partition.values()),
        "clients": {},
    }

    for client_id, indices in partition.items():
        client_stats = {
            "num_samples": len(indices),
            "by_category": {},
            "by_label": {0: 0, 1: 0},
        }

        for idx in indices:
            _, label, category, _ = dataset.samples[idx]

            if category not in client_stats["by_category"]:
                client_stats["by_category"][category] = 0
            client_stats["by_category"][category] += 1
            client_stats["by_label"][label] += 1

        stats["clients"][str(client_id)] = client_stats

    return stats

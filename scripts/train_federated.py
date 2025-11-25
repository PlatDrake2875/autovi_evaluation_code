#!/usr/bin/env python3
"""Federated PatchCore training script.

This script runs federated learning experiments with either IID or
category-based data partitioning.

Usage:
    python scripts/train_federated.py --config experiments/configs/federated/fedavg_iid_config.yaml --data_root /path/to/autovi
    python scripts/train_federated.py --config experiments/configs/federated/fedavg_category_config.yaml --data_root /path/to/autovi
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.autovi_dataset import AutoVIDataset, AutoVISubset, CATEGORIES
from src.data.partitioner import (
    IIDPartitioner,
    CategoryPartitioner,
    compute_partition_stats,
    save_partition,
)
from src.data.preprocessing import get_transforms
from src.federated import FederatedPatchCore


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_dataloaders(
    dataset: AutoVIDataset,
    partition: dict,
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict:
    """Create dataloaders for each client partition.

    Args:
        dataset: The source AutoVIDataset.
        partition: Dictionary mapping client_id -> list of sample indices.
        batch_size: Batch size for dataloaders.
        num_workers: Number of worker processes.

    Returns:
        Dictionary mapping client_id -> DataLoader.
    """
    dataloaders = {}

    for client_id, indices in partition.items():
        subset = AutoVISubset(dataset, indices)

        # Custom collate function for AutoVISubset
        def collate_fn(batch):
            images = torch.stack([item["image"] for item in batch])
            labels = torch.tensor([item["label"] for item in batch])
            categories = [item["category"] for item in batch]
            return {"image": images, "label": labels, "category": categories}

        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        dataloaders[client_id] = loader

    return dataloaders


def main():
    parser = argparse.ArgumentParser(description="Train Federated PatchCore")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to AutoVI dataset root",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=None,
        help="Number of federated training rounds (overrides config)",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=1,
        help="Save checkpoint every N rounds (default: 1)",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"\nLoaded configuration from: {args.config}")

    # Override config with command line arguments
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    output_dir = args.output_dir if args.output_dir else config.get("output", {}).get("dir", "outputs/federated")
    num_rounds = args.num_rounds if args.num_rounds is not None else config.get("federated", {}).get("num_rounds", 1)
    checkpoint_every = args.checkpoint_every

    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config to output
    config_save_path = output_path / "config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\n{'='*60}")
    print("Federated PatchCore Training")
    print(f"{'='*60}")
    print(f"Data root: {args.data_root}")
    print(f"Output directory: {output_dir}")
    print(f"Partitioning: {config['federated']['partitioning']}")
    print(f"Number of clients: {config['federated']['num_clients']}")
    print(f"Number of rounds: {num_rounds}")
    print(f"Checkpoint every: {checkpoint_every} rounds")
    print(f"Random seed: {seed}")

    # Load dataset
    print("\n--- Loading Dataset ---")

    # Determine which categories to use
    categories = CATEGORIES

    # Create dataset with appropriate transforms
    # Note: We'll apply transforms per-category during training
    dataset = AutoVIDataset(
        root_dir=args.data_root,
        categories=categories,
        split="train",
        transform=None,  # Will apply transforms in collate_fn
    )

    print(f"Dataset loaded: {len(dataset)} samples")
    stats = dataset.get_statistics()
    print(f"Categories: {list(stats['by_category'].keys())}")

    # Create partitioner based on config
    print("\n--- Creating Data Partitions ---")
    fed_config = config["federated"]
    num_clients = fed_config["num_clients"]
    partitioning = fed_config["partitioning"]

    if partitioning == "iid":
        partitioner = IIDPartitioner(num_clients=num_clients, seed=seed)
    elif partitioning == "category":
        # Parse client assignments from config
        client_assignments = fed_config.get("client_assignments", None)
        if client_assignments:
            # Convert string keys to int
            client_assignments = {int(k): v for k, v in client_assignments.items()}
        qc_sample_ratio = fed_config.get("qc_sample_ratio", 0.1)
        partitioner = CategoryPartitioner(
            client_assignments=client_assignments,
            qc_sample_ratio=qc_sample_ratio,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown partitioning: {partitioning}")

    partition = partitioner.partition(dataset)
    partition_stats = compute_partition_stats(dataset, partition)

    # Save partition
    partition_path = output_path / "partition.json"
    save_partition(partition, str(partition_path), partition_stats)
    print(f"Saved partition to {partition_path}")

    # Print partition details
    print("\nPartition Statistics:")
    for client_id, client_stats in partition_stats["clients"].items():
        print(f"  Client {client_id}: {client_stats['num_samples']} samples")
        for cat, count in client_stats["by_category"].items():
            print(f"    - {cat}: {count}")

    # Create dataset with transforms for each category
    print("\n--- Setting up Dataloaders ---")

    # We need to apply category-specific transforms
    # Create a transform wrapper
    class CategoryTransformDataset:
        def __init__(self, dataset, transforms_dict):
            self.dataset = dataset
            self.transforms = transforms_dict

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            category = item["category"]
            if category in self.transforms:
                item["image"] = self.transforms[category](item["image"])
            return item

    # Build transforms for each category
    transforms_dict = {}
    for cat in CATEGORIES:
        transforms_dict[cat] = get_transforms(cat, normalize=True, to_tensor=True)

    # Wrap dataset with transforms
    transformed_dataset = CategoryTransformDataset(dataset, transforms_dict)

    # Create AutoVISubset-like wrapper
    class TransformedSubset:
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]

    # Create dataloaders
    training_config = config.get("training", {})
    batch_size = training_config.get("batch_size", 32)
    num_workers = training_config.get("num_workers", 4)

    dataloaders = {}
    for client_id, indices in partition.items():
        subset = TransformedSubset(transformed_dataset, indices)

        def collate_fn(batch):
            images = torch.stack([item["image"] for item in batch])
            labels = torch.tensor([item["label"] for item in batch])
            categories = [item["category"] for item in batch]
            return {"image": images, "label": labels, "category": categories}

        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
        )
        dataloaders[client_id] = loader

    print(f"Created {len(dataloaders)} dataloaders (batch_size={batch_size})")

    # Initialize FederatedPatchCore
    print("\n--- Initializing Federated System ---")
    model_config = config.get("model", {})
    aggregation_config = config.get("aggregation", {})

    federated_model = FederatedPatchCore(
        num_clients=num_clients,
        backbone_name=model_config.get("backbone", "wide_resnet50_2"),
        layers=model_config.get("layers", ["layer2", "layer3"]),
        coreset_ratio=model_config.get("coreset_ratio", 0.1),
        global_bank_size=aggregation_config.get("global_bank_size", 10000),
        neighborhood_size=model_config.get("neighborhood_size", 3),
        aggregation_strategy=aggregation_config.get("strategy", "federated_coreset"),
        weighted_by_samples=aggregation_config.get("weighted_by_samples", True),
        use_faiss=model_config.get("use_faiss", True),
        device=args.device,
        num_rounds=num_rounds,
    )

    # Store partition info
    federated_model.partition = partition
    federated_model.partition_stats = partition_stats

    # Run federated training
    print("\n--- Starting Federated Training ---")
    start_time = time.time()

    # Create checkpoint directory
    checkpoint_dir = output_path / "checkpoints"

    global_memory_bank = federated_model.train(
        dataloaders,
        seed=seed,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_every=checkpoint_every,
    )

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds")

    # Save model and results
    print("\n--- Saving Results ---")
    federated_model.save(str(output_path))

    # Print summary
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Global memory bank size: {len(global_memory_bank)}")
    print(f"Feature dimension: {global_memory_bank.shape[1]}")
    print(f"Output saved to: {output_path}")

    # Print final statistics
    final_stats = federated_model.get_stats()
    print(f"\nFinal Statistics:")
    print(f"  Clients: {final_stats['num_clients']}")
    print(f"  Rounds: {final_stats['num_rounds']}")
    print(f"  Backbone: {final_stats['backbone']}")
    print(f"  Aggregation: {final_stats['aggregation_strategy']}")
    print(f"  Global bank: {final_stats['actual_global_bank_size']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Federated PatchCore training script.

This script runs federated learning experiments with either IID or
category-based data partitioning.

Usage:
    python scripts/train_federated.py --config experiments/configs/federated/fedavg_iid_config.yaml --data_root /path/to/autovi
    python scripts/train_federated.py --config experiments/configs/federated/fedavg_category_config.yaml --data_root /path/to/autovi
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    AutoVIDataset,
    CATEGORIES,
    CategoryTransformDataset,
    IIDPartitioner,
    CategoryPartitioner,
    get_transforms,
)
from src.data.partitioner import compute_partition_stats, save_partition
from src.federated import FederatedPatchCore
from src.training import (
    load_config,
    set_random_seeds,
    setup_output_directory,
    print_training_header,
    create_federated_dataloaders,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Federated PatchCore")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data_root", type=str, required=True, help="Path to AutoVI dataset root")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (overrides config)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    parser.add_argument("--num_rounds", type=int, default=None, help="Number of federated training rounds (overrides config)")
    parser.add_argument("--checkpoint_every", type=int, default=1, help="Save checkpoint every N rounds (default: 1)")
    parser.add_argument("--categories", type=str, nargs="+", default=None, help="Categories to train on (default: all)")
    return parser.parse_args()


def get_config_values(config, args):
    """Extract configuration values with CLI overrides."""
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    output_dir = args.output_dir if args.output_dir else config.get("output", {}).get("dir", "outputs/federated")
    num_rounds = args.num_rounds if args.num_rounds is not None else config.get("federated", {}).get("num_rounds", 1)
    return seed, output_dir, num_rounds


def get_dp_config(config):
    """Extract differential privacy configuration."""
    dp_config = config.get("differential_privacy", {})
    return {
        "enabled": dp_config.get("enabled", False),
        "epsilon": dp_config.get("epsilon", 1.0),
        "delta": dp_config.get("delta", 1e-5),
        "clipping_norm": dp_config.get("clipping_norm", 1.0),
    }


def validate_categories(categories):
    """Validate that all categories are known."""
    for cat in categories:
        if cat not in CATEGORIES:
            print(f"Error: Unknown category '{cat}'. Valid categories: {CATEGORIES}")
            return False
    return True


def create_partitioner(fed_config, seed):
    """Create the appropriate data partitioner."""
    num_clients = fed_config["num_clients"]
    partitioning = fed_config["partitioning"]

    if partitioning == "iid":
        return IIDPartitioner(num_clients=num_clients, seed=seed)
    elif partitioning == "category":
        client_assignments = fed_config.get("client_assignments", None)
        if client_assignments:
            client_assignments = {int(k): v for k, v in client_assignments.items()}
        qc_sample_ratio = fed_config.get("qc_sample_ratio", 0.1)
        return CategoryPartitioner(
            client_assignments=client_assignments,
            qc_sample_ratio=qc_sample_ratio,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown partitioning: {partitioning}")


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"\nLoaded configuration from: {args.config}")

    # Get config values with CLI overrides
    seed, output_dir, num_rounds = get_config_values(config, args)
    checkpoint_every = args.checkpoint_every
    dp_config = get_dp_config(config)

    # Setup
    set_random_seeds(seed)
    output_path = setup_output_directory(output_dir, config)

    # Determine categories
    categories = args.categories if args.categories else CATEGORIES
    if args.categories and not validate_categories(categories):
        return 1

    # Print header
    fed_config = config["federated"]
    header_params = {
        "Data root": args.data_root,
        "Output directory": output_dir,
        "Categories": args.categories if args.categories else "all",
        "Partitioning": fed_config["partitioning"],
        "Number of clients": fed_config["num_clients"],
        "Number of rounds": num_rounds,
        "Checkpoint every": f"{checkpoint_every} rounds",
        "Random seed": seed,
    }
    if dp_config["enabled"]:
        header_params["Differential Privacy"] = "enabled"
        header_params["  Epsilon"] = dp_config["epsilon"]
        header_params["  Delta"] = dp_config["delta"]
        header_params["  Clipping norm"] = dp_config["clipping_norm"]
    print_training_header("Federated PatchCore Training", header_params)

    # Load dataset
    print("\n--- Loading Dataset ---")
    dataset = AutoVIDataset(
        root_dir=args.data_root,
        categories=categories,
        split="train",
        transform=None,
    )
    print(f"Dataset loaded: {len(dataset)} samples")
    stats = dataset.get_statistics()
    print(f"Categories: {list(stats['by_category'].keys())}")

    # Create partitions
    print("\n--- Creating Data Partitions ---")
    partitioner = create_partitioner(fed_config, seed)
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

    # Setup transforms and dataloaders
    print("\n--- Setting up Dataloaders ---")
    transforms_dict = {cat: get_transforms(cat, normalize=True, to_tensor=True) for cat in CATEGORIES}
    transformed_dataset = CategoryTransformDataset(dataset, transforms_dict)

    training_config = config.get("training", {})
    dataloaders = create_federated_dataloaders(
        transformed_dataset,
        partition,
        batch_size=training_config.get("batch_size", 32),
        num_workers=training_config.get("num_workers", 4),
    )
    print(f"Created {len(dataloaders)} dataloaders")

    # Initialize model
    print("\n--- Initializing Federated System ---")
    model_config = config.get("model", {})
    aggregation_config = config.get("aggregation", {})

    federated_model = FederatedPatchCore(
        num_clients=fed_config["num_clients"],
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
        dp_enabled=dp_config["enabled"],
        dp_epsilon=dp_config["epsilon"],
        dp_delta=dp_config["delta"],
        dp_clipping_norm=dp_config["clipping_norm"],
    )
    federated_model.partition = partition
    federated_model.partition_stats = partition_stats

    # Train
    print("\n--- Starting Federated Training ---")
    start_time = time.time()
    checkpoint_dir = output_path / "checkpoints"

    global_memory_bank = federated_model.train(
        dataloaders,
        seed=seed,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_every=checkpoint_every,
    )

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds")

    # Save results
    print("\n--- Saving Results ---")
    federated_model.save(str(output_path))

    # Print summary
    print(f"\n{'=' * 60}")
    print("Training Complete!")
    print(f"{'=' * 60}")
    print(f"Global memory bank size: {len(global_memory_bank)}")
    print(f"Feature dimension: {global_memory_bank.shape[1]}")
    print(f"Output saved to: {output_path}")

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

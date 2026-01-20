#!/usr/bin/env python3
"""Generate FL partition files for AutoVI dataset."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import AutoVIDataset
from src.data.partitioner import (
    IIDPartitioner,
    CategoryPartitioner,
    save_partition,
    compute_partition_stats,
)


def main():
    # Configuration
    DATA_ROOT = Path("/home/ldg2875/Downloads/auto_vi")
    OUTPUT_DIR = project_root / "outputs" / "partitions"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from: {DATA_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Load training dataset
    train_dataset = AutoVIDataset(root_dir=str(DATA_ROOT), split="train")
    print(f"Training samples: {len(train_dataset)}")

    # Generate IID partition
    print("\n--- IID Partition ---")
    iid_partitioner = IIDPartitioner(num_clients=5, seed=42)
    iid_partition = iid_partitioner.partition(train_dataset)
    iid_stats = compute_partition_stats(train_dataset, iid_partition)

    save_partition(
        iid_partition,
        OUTPUT_DIR / "iid_partition.json",
        stats=iid_stats,
    )
    print(f"Saved IID partition to: {OUTPUT_DIR / 'iid_partition.json'}")

    for client_id, indices in iid_partition.items():
        print(f"  Client {client_id}: {len(indices)} samples")

    # Generate Category-based partition
    print("\n--- Category Partition (Non-IID) ---")
    category_partitioner = CategoryPartitioner(seed=42)
    category_partition = category_partitioner.partition(train_dataset)
    category_stats = compute_partition_stats(train_dataset, category_partition)

    save_partition(
        category_partition,
        OUTPUT_DIR / "category_partition.json",
        stats=category_stats,
    )
    print(f"Saved Category partition to: {OUTPUT_DIR / 'category_partition.json'}")

    client_roles = {
        0: "Engine Assembly",
        1: "Underbody Line",
        2: "Fastener Station",
        3: "Clip Inspection",
        4: "Quality Control",
    }

    for client_id, indices in category_partition.items():
        role = client_roles.get(client_id, "Unknown")
        print(f"  Client {client_id} ({role}): {len(indices)} samples")

    # Save combined stats
    import json
    combined_stats = {
        "iid": iid_stats,
        "category": category_stats,
    }
    with open(OUTPUT_DIR / "partition_stats.json", "w") as f:
        json.dump(combined_stats, f, indent=2)
    print(f"\nSaved combined stats to: {OUTPUT_DIR / 'partition_stats.json'}")

    print("\n=== Partition generation complete! ===")


if __name__ == "__main__":
    main()

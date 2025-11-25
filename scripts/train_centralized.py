#!/usr/bin/env python3
"""Centralized PatchCore training script for AutoVI dataset."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import AutoVIDataset, CATEGORIES, get_resize_shape
from src.models.patchcore import PatchCore
from src.training import (
    load_config,
    set_random_seeds,
    setup_output_directory,
    setup_logging,
)


def get_transforms_for_category(category: str, config: Dict) -> transforms.Compose:
    """Get image transforms for a category."""
    resize_shape = get_resize_shape(category)

    normalize_mean = config.get("preprocessing", {}).get(
        "normalize_mean", [0.485, 0.456, 0.406]
    )
    normalize_std = config.get("preprocessing", {}).get(
        "normalize_std", [0.229, 0.224, 0.225]
    )

    return transforms.Compose([
        transforms.Resize(resize_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
    ])


def train_single_category(
    category: str,
    data_dir: Path,
    output_dir: Path,
    config: Dict,
    logger,
) -> Dict:
    """Train PatchCore model for a single category."""
    logger.info(f"{'=' * 60}")
    logger.info(f"Training PatchCore for category: {category}")
    logger.info(f"{'=' * 60}")

    # Get transforms
    transform = get_transforms_for_category(category, config)

    # Create dataset
    dataset = AutoVIDataset(
        root_dir=str(data_dir),
        categories=[category],
        split="train",
        transform=transform,
    )
    logger.info(f"Training samples: {len(dataset)}")

    # Create dataloader
    batch_size = config.get("training", {}).get("batch_size", 32)
    num_workers = config.get("training", {}).get("num_workers", 4)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Create model
    model_config = config.get("model", {})
    model = PatchCore(
        backbone_name=model_config.get("backbone", "wide_resnet50_2"),
        layers=model_config.get("layers", ["layer2", "layer3"]),
        coreset_ratio=model_config.get("coreset_percentage", 0.1),
        neighborhood_size=model_config.get("neighborhood_size", 3),
        device="auto",
        use_faiss=True,
    )

    # Train
    seed = config.get("seed", 42)
    max_samples = model_config.get("max_memory_samples", None)
    model.fit(dataloader, max_samples=max_samples, seed=seed)

    # Save model
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"patchcore_{category}"
    model.save(str(model_path))

    # Get statistics
    stats = model.get_stats()
    stats["category"] = category
    stats["num_training_samples"] = len(dataset)

    logger.info(f"Model statistics: {json.dumps(stats, indent=2, default=str)}")

    return stats


def train_all_categories(
    data_dir: Path,
    output_dir: Path,
    config: Dict,
    categories: Optional[List[str]] = None,
    logger=None,
) -> Dict:
    """Train PatchCore models for all categories."""
    import logging
    if logger is None:
        logger = logging.getLogger(__name__)

    if categories is None or "all" in categories:
        categories = CATEGORIES

    logger.info(f"Training categories: {categories}")

    all_stats = {}
    for category in categories:
        try:
            stats = train_single_category(
                category=category,
                data_dir=data_dir,
                output_dir=output_dir,
                config=config,
                logger=logger,
            )
            all_stats[category] = stats
        except Exception as e:
            logger.error(f"Error training {category}: {e}")
            all_stats[category] = {"error": str(e)}

    # Save summary
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_stats, f, indent=2, default=str)

    logger.info(f"Training summary saved to {summary_path}")

    return all_stats


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train centralized PatchCore models on AutoVI dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/baseline/patchcore_config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to AutoVI dataset root directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/baseline",
        help="Output directory for models and logs",
    )
    parser.add_argument(
        "--objects",
        type=str,
        nargs="+",
        default=["all"],
        help="Object categories to train (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Setup output directory and logging
    setup_output_directory(str(output_dir))
    logger = setup_logging(output_dir)
    logger.info(f"Arguments: {args}")

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(str(config_path))
        logger.info(f"Loaded config from {config_path}")
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        config = {}

    # Override seed
    config["seed"] = args.seed

    # Set random seeds
    set_random_seeds(args.seed)

    # Train models
    train_all_categories(
        data_dir=data_dir,
        output_dir=output_dir,
        config=config,
        categories=args.objects,
        logger=logger,
    )

    logger.info("Training complete!")


if __name__ == "__main__":
    main()

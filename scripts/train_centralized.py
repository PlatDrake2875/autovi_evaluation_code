#!/usr/bin/env python3
"""Centralized PatchCore training script for AutoVI dataset."""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.autovi_dataset import AutoVIDataset, CATEGORIES, get_resize_shape
from src.models.patchcore import PatchCore


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_transforms(category: str, config: Dict) -> transforms.Compose:
    """Get image transforms for a category."""
    resize_shape = get_resize_shape(category)

    # ImageNet normalization
    normalize_mean = config.get("preprocessing", {}).get(
        "normalize_mean", [0.485, 0.456, 0.406]
    )
    normalize_std = config.get("preprocessing", {}).get(
        "normalize_std", [0.229, 0.224, 0.225]
    )

    transform = transforms.Compose([
        transforms.Resize(resize_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
    ])

    return transform


def train_single_category(
    category: str,
    data_dir: Path,
    output_dir: Path,
    config: Dict,
    logger: logging.Logger,
) -> Dict:
    """Train PatchCore model for a single category.

    Args:
        category: Object category name.
        data_dir: Path to AutoVI dataset.
        output_dir: Output directory for models.
        config: Configuration dictionary.
        logger: Logger instance.

    Returns:
        Training statistics dictionary.
    """
    logger.info(f"=" * 60)
    logger.info(f"Training PatchCore for category: {category}")
    logger.info(f"=" * 60)

    # Get transforms
    transform = get_transforms(category, config)

    # Create dataset (training set, good samples only)
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

    # Create PatchCore model
    model_config = config.get("model", {})
    model = PatchCore(
        backbone_name=model_config.get("backbone", "wide_resnet50_2"),
        layers=model_config.get("layers", ["layer2", "layer3"]),
        coreset_ratio=model_config.get("coreset_percentage", 0.1),
        neighborhood_size=model_config.get("neighborhood_size", 3),
        device="auto",
        use_faiss=True,
    )

    # Train (build memory bank)
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
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """Train PatchCore models for all categories.

    Args:
        data_dir: Path to AutoVI dataset.
        output_dir: Output directory.
        config: Configuration dictionary.
        categories: List of categories to train. If None, trains all.
        logger: Logger instance.

    Returns:
        Summary statistics dictionary.
    """
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


def main():
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

    args = parser.parse_args()

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
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

    # Override seed if specified
    config["seed"] = args.seed

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

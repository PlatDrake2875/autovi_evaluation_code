"""AutoVI Dataset class for federated learning experiments."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
from PIL import Image
from loguru import logger

# Dataset categories
CATEGORIES = [
    "engine_wiring",
    "pipe_clip",
    "pipe_staple",
    "tank_screw",
    "underbody_pipes",
    "underbody_screw",
]

# Image size categories based on object type
SMALL_OBJECTS = ["engine_wiring", "pipe_clip", "pipe_staple"]  # 400x400
LARGE_OBJECTS = ["tank_screw", "underbody_pipes", "underbody_screw"]  # 1000x750


def get_resize_shape(category: str) -> Tuple[int, int]:
    """Get the appropriate resize dimensions for a category.

    Args:
        category: The object category name.

    Returns:
        Tuple of (width, height) for resizing.
    """
    if category in SMALL_OBJECTS:
        return (400, 400)
    elif category in LARGE_OBJECTS:
        return (1000, 750)
    else:
        raise ValueError(f"Unknown category: {category}")


class AutoVIDataset:
    """Dataset class for the AutoVI visual inspection dataset.

    This class handles loading and accessing images from the AutoVI dataset,
    supporting both training (good samples only) and test (good + defective) sets.

    Attributes:
        root_dir: Path to the root directory of the AutoVI dataset.
        categories: List of object categories to include.
        split: Either 'train' or 'test'.
        transform: Optional transform to apply to images.
        samples: List of (image_path, label, category, defect_type) tuples.
    """

    def __init__(
        self,
        root_dir: str,
        categories: Optional[List[str]] = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        include_good_only: bool = False,
    ):
        """Initialize the AutoVI dataset.

        Args:
            root_dir: Path to the AutoVI dataset root directory.
            categories: List of categories to include. If None, includes all.
            split: Either 'train' or 'test'.
            transform: Optional callable transform for images.
            include_good_only: If True and split='test', only include good samples.
        """
        self.root_dir = Path(root_dir)
        self.categories = categories if categories else CATEGORIES
        self.split = split
        self.transform = transform
        self.include_good_only = include_good_only

        # Validate categories
        for cat in self.categories:
            if cat not in CATEGORIES:
                raise ValueError(f"Unknown category: {cat}. Valid: {CATEGORIES}")

        # Validate split
        if split not in ["train", "test"]:
            raise ValueError(f"Split must be 'train' or 'test', got: {split}")

        # Load samples
        self.samples: List[Tuple[str, int, str, Optional[str]]] = []
        self._load_samples()

        # Load defect configs
        self.defect_configs: Dict[str, List[dict]] = {}
        self._load_defect_configs()

    def _load_samples(self) -> None:
        """Load all sample paths and labels."""
        for category in self.categories:
            category_dir = self.root_dir / category / self.split

            if not category_dir.exists():
                logger.warning(f"Category directory not found: {category_dir}")
                continue

            # Load good samples (label=0)
            good_dir = category_dir / "good"
            if good_dir.exists():
                for img_path in sorted(good_dir.glob("*.png")):
                    self.samples.append((str(img_path), 0, category, None))

            # Load defective samples (label=1) - only for test split
            if self.split == "test" and not self.include_good_only:
                for defect_dir in category_dir.iterdir():
                    if defect_dir.is_dir() and defect_dir.name != "good":
                        defect_type = defect_dir.name
                        for img_path in sorted(defect_dir.glob("*.png")):
                            self.samples.append((str(img_path), 1, category, defect_type))

    def _load_defect_configs(self) -> None:
        """Load defect configuration for each category."""
        for category in self.categories:
            config_path = self.root_dir / category / "defects_config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    self.defect_configs[category] = json.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """Get a sample by index.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing:
                - image: PIL Image or transformed tensor
                - label: 0 for good, 1 for defective
                - category: Object category name
                - defect_type: Defect type name (None for good samples)
                - path: Original image path
        """
        img_path, label, category, defect_type = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transform if provided
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "category": category,
            "defect_type": defect_type if defect_type is not None else "good",
            "path": img_path,
        }

    def get_ground_truth(self, idx: int) -> Optional[np.ndarray]:
        """Get ground truth mask for a test sample.

        Args:
            idx: Sample index.

        Returns:
            Ground truth mask as numpy array, or None if not available.
        """
        img_path, label, category, defect_type = self.samples[idx]

        if label == 0 or defect_type is None:
            return None

        # Construct ground truth path
        img_name = Path(img_path).stem
        gt_dir = self.root_dir / category / "ground_truth" / defect_type / img_name

        if not gt_dir.exists():
            return None

        # Load and combine all mask channels
        masks = []
        for mask_path in sorted(gt_dir.glob("*.png")):
            mask = np.array(Image.open(mask_path))
            masks.append(mask)

        if not masks:
            return None

        # Combine masks with logical OR
        combined = np.zeros_like(masks[0], dtype=bool)
        for mask in masks:
            combined = combined | (mask > 0)

        return combined.astype(np.uint8) * 255

    def get_category_indices(self, category: str) -> List[int]:
        """Get all sample indices for a specific category.

        Args:
            category: Category name.

        Returns:
            List of indices for samples in that category.
        """
        return [i for i, (_, _, cat, _) in enumerate(self.samples) if cat == category]

    def get_defect_indices(self, defect_type: str) -> List[int]:
        """Get all sample indices for a specific defect type.

        Args:
            defect_type: Defect type name.

        Returns:
            List of indices for samples with that defect type.
        """
        return [i for i, (_, _, _, dt) in enumerate(self.samples) if dt == defect_type]

    def get_statistics(self) -> Dict:
        """Get dataset statistics.

        Returns:
            Dictionary with counts per category, defect type, and label.
        """
        stats = {
            "total": len(self.samples),
            "by_category": {},
            "by_defect_type": {},
            "by_label": {0: 0, 1: 0},
        }

        for _, label, category, defect_type in self.samples:
            # Count by category
            if category not in stats["by_category"]:
                stats["by_category"][category] = {"good": 0, "defective": 0}
            if label == 0:
                stats["by_category"][category]["good"] += 1
            else:
                stats["by_category"][category]["defective"] += 1

            # Count by defect type
            if defect_type is not None:
                if defect_type not in stats["by_defect_type"]:
                    stats["by_defect_type"][defect_type] = 0
                stats["by_defect_type"][defect_type] += 1

            # Count by label
            stats["by_label"][label] += 1

        return stats


class AutoVISubset:
    """A subset of AutoVIDataset for federated learning client data.

    Wraps the parent dataset and provides access to a subset of indices.
    """

    def __init__(self, dataset: AutoVIDataset, indices: List[int]):
        """Initialize the subset.

        Args:
            dataset: The parent AutoVIDataset.
            indices: List of indices to include in this subset.
        """
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict:
        return self.dataset[self.indices[idx]]

    def get_statistics(self) -> Dict:
        """Get statistics for this subset."""
        stats = {
            "total": len(self.indices),
            "by_category": {},
            "by_label": {0: 0, 1: 0},
        }

        for idx in self.indices:
            _, label, category, _ = self.dataset.samples[idx]

            if category not in stats["by_category"]:
                stats["by_category"][category] = 0
            stats["by_category"][category] += 1
            stats["by_label"][label] += 1

        return stats

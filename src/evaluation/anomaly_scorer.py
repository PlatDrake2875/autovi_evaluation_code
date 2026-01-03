"""Anomaly map generation for model evaluation.

This module provides functionality to generate pixel-wise anomaly maps
from trained PatchCore models (centralized or federated).
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.patchcore import PatchCore
from src.util import get_device
from src.models.backbone import (
    FeatureExtractor,
    apply_local_neighborhood_averaging,
    reshape_features_to_patches,
)
from src.models.memory_bank import MemoryBank
from src.data.autovi_dataset import AutoVIDataset, get_resize_shape, CATEGORIES


class AnomalyScorer:
    """Generates anomaly maps using a trained PatchCore model.

    This class handles the end-to-end process of generating anomaly maps:
    1. Load trained model (centralized) or memory bank (federated)
    2. Extract features from test images
    3. Query memory bank for anomaly scores
    4. Upsample and save anomaly maps as PNG

    Attributes:
        feature_extractor: The backbone feature extractor.
        memory_bank: Memory bank for anomaly scoring.
        device: PyTorch device for computation.
    """

    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        layers: List[str] = None,
        neighborhood_size: int = 3,
        device: str = "auto",
        use_faiss: bool = True,
    ):
        """Initialize the AnomalyScorer.

        Args:
            backbone_name: Name of the backbone CNN.
            layers: List of layer names to extract features from.
            neighborhood_size: Kernel size for local neighborhood averaging.
            device: Device for computation ("auto", "cuda", or "cpu").
            use_faiss: Whether to use FAISS for nearest neighbor search.
        """
        self.backbone_name = backbone_name
        self.layers = layers if layers else ["layer2", "layer3"]
        self.neighborhood_size = neighborhood_size
        self.use_faiss = use_faiss

        # Set device
        self.device = get_device(device)

        logger.info(f"AnomalyScorer using device: {self.device}")

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            backbone_name=backbone_name,
            layers=self.layers,
            pretrained=True,
        ).to(self.device)

        # Feature dimension
        self.feature_dim = self.feature_extractor.get_feature_dim()

        # Memory bank (loaded from model)
        self.memory_bank: Optional[MemoryBank] = None

    def load_centralized_model(self, model_path: str) -> None:
        """Load a centralized PatchCore model.

        Args:
            model_path: Path to the saved model (without extension).
        """
        logger.info(f"Loading centralized model from {model_path}...")

        # Load memory bank
        memory_bank_path = str(model_path) + "_memory_bank.npz"
        self.memory_bank = MemoryBank(
            feature_dim=self.feature_dim,
            use_faiss=self.use_faiss,
            use_gpu=self.device.type == "cuda",
        )
        self.memory_bank.load(memory_bank_path)

        logger.info(f"Loaded memory bank with {len(self.memory_bank)} patches")

    def load_federated_memory_bank(self, memory_bank_path: str) -> None:
        """Load a federated global memory bank.

        Args:
            memory_bank_path: Path to the global memory bank file.
        """
        logger.info(f"Loading federated memory bank from {memory_bank_path}...")

        self.memory_bank = MemoryBank(
            feature_dim=self.feature_dim,
            use_faiss=self.use_faiss,
            use_gpu=self.device.type == "cuda",
        )

        # Handle both .npz and raw numpy formats
        if memory_bank_path.endswith(".npz"):
            self.memory_bank.load(memory_bank_path)
        else:
            # Load raw numpy array
            data = np.load(memory_bank_path)
            if isinstance(data, np.ndarray):
                self.memory_bank.set_features(data)
            else:
                self.memory_bank.set_features(data["features"])

        logger.info(f"Loaded memory bank with {len(self.memory_bank)} patches")

    def load_memory_bank_from_array(self, features: np.ndarray) -> None:
        """Load memory bank from a numpy array directly.

        Args:
            features: Memory bank features of shape [N, D].
        """
        self.memory_bank = MemoryBank(
            feature_dim=self.feature_dim,
            use_faiss=self.use_faiss,
            use_gpu=self.device.type == "cuda",
        )
        self.memory_bank.set_features(features)

        logger.info(f"Loaded memory bank with {len(self.memory_bank)} patches")

    def generate_anomaly_map(
        self,
        image: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Generate anomaly map for a single image.

        Args:
            image: Input image tensor [1, 3, H, W] or [3, H, W].
            output_size: Target output size (H, W). If None, uses input size.

        Returns:
            Anomaly map as numpy array [H, W] with values in [0, 255].
        """
        if self.memory_bank is None:
            raise RuntimeError("Memory bank not loaded. Call load_*() first.")

        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        # Store original size
        original_size = (image.shape[2], image.shape[3]) if output_size is None else output_size

        with torch.no_grad():
            # Extract features
            features = self.feature_extractor(image)

            # Apply local neighborhood averaging
            if self.neighborhood_size > 1:
                features = apply_local_neighborhood_averaging(
                    features, self.neighborhood_size
                )

            feature_h, feature_w = features.shape[2], features.shape[3]

            # Reshape to patches
            patches = reshape_features_to_patches(features)

        # Get anomaly scores from memory bank
        anomaly_scores = self.memory_bank.get_anomaly_scores(patches.cpu().numpy())

        # Reshape to spatial map
        anomaly_map = anomaly_scores.reshape(feature_h, feature_w)

        # Upsample to original size
        anomaly_map_tensor = torch.from_numpy(anomaly_map).unsqueeze(0).unsqueeze(0).float()
        anomaly_map_upsampled = F.interpolate(
            anomaly_map_tensor,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )
        anomaly_map = anomaly_map_upsampled.squeeze().numpy()

        # Normalize to [0, 255]
        if anomaly_map.max() > anomaly_map.min():
            anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
        else:
            anomaly_map = np.zeros_like(anomaly_map)

        anomaly_map_uint8 = (anomaly_map * 255).astype(np.uint8)

        return anomaly_map_uint8

    def generate_anomaly_maps_for_dataset(
        self,
        dataset: AutoVIDataset,
        output_dir: str,
        batch_size: int = 1,
        num_workers: int = 0,
    ) -> Dict[str, List[str]]:
        """Generate anomaly maps for all images in a dataset.

        Args:
            dataset: AutoVI test dataset.
            output_dir: Directory to save anomaly maps.
            batch_size: Batch size for processing.
            num_workers: Number of data loading workers.

        Returns:
            Dictionary mapping category to list of saved file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: Dict[str, List[str]] = {}

        logger.info(f"Generating anomaly maps for {len(dataset)} images...")

        for idx in tqdm(range(len(dataset)), desc="Generating anomaly maps"):
            sample = dataset[idx]
            image = sample["image"]
            category = sample["category"]
            defect_type = sample["defect_type"]
            img_path = Path(sample["path"])

            # Get appropriate output size for category
            resize_shape = get_resize_shape(category)

            # Ensure image is tensor
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(np.array(image).transpose(2, 0, 1).astype(np.float32) / 255.0)

            # Generate anomaly map
            anomaly_map = self.generate_anomaly_map(image, output_size=resize_shape)

            # Construct output path: output_dir/defect_type/image_name.png
            rel_path = Path(defect_type) / f"{img_path.stem}.png"
            save_path = output_dir / rel_path
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Save anomaly map
            Image.fromarray(anomaly_map).save(save_path)

            # Track saved paths
            if category not in saved_paths:
                saved_paths[category] = []
            saved_paths[category].append(str(save_path))

        logger.info(f"Saved {sum(len(v) for v in saved_paths.values())} anomaly maps to {output_dir}")

        return saved_paths


def generate_anomaly_maps(
    model_path: str,
    model_type: str,
    dataset_dir: str,
    output_dir: str,
    categories: Optional[List[str]] = None,
    device: str = "auto",
    transform=None,
) -> Dict[str, Dict[str, List[str]]]:
    """Generate anomaly maps for all categories using a trained model.

    This is the main entry point for anomaly map generation.

    Args:
        model_path: Path to the trained model or memory bank.
        model_type: Type of model ("centralized" or "federated").
        dataset_dir: Path to the AutoVI dataset root.
        output_dir: Directory to save anomaly maps.
        categories: List of categories to process. If None, processes all.
        device: Device for computation.
        transform: Optional transform for images.

    Returns:
        Nested dictionary: {category: {defect_type: [paths]}}.
    """
    categories = categories if categories else CATEGORIES
    output_dir = Path(output_dir)
    all_saved_paths: Dict[str, Dict[str, List[str]]] = {}

    for category in categories:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing category: {category}")
        logger.info(f"{'='*50}")

        # Create scorer for this category
        scorer = AnomalyScorer(device=device)

        # Load model
        if model_type == "centralized":
            # For centralized, might have per-category models or single model
            if Path(model_path).is_dir():
                cat_model_path = Path(model_path) / category / "patchcore"
            else:
                cat_model_path = model_path
            scorer.load_centralized_model(str(cat_model_path))
        else:  # federated
            scorer.load_federated_memory_bank(model_path)

        # Create test dataset for this category
        test_dataset = AutoVIDataset(
            root_dir=dataset_dir,
            categories=[category],
            split="test",
            transform=transform,
            include_good_only=False,
        )

        # Generate anomaly maps
        cat_output_dir = output_dir / category
        saved_paths = scorer.generate_anomaly_maps_for_dataset(
            dataset=test_dataset,
            output_dir=str(cat_output_dir),
        )

        all_saved_paths[category] = saved_paths

    return all_saved_paths


def generate_anomaly_maps_from_patchcore(
    model: PatchCore,
    dataset: AutoVIDataset,
    output_dir: str,
) -> Dict[str, List[str]]:
    """Generate anomaly maps using an existing PatchCore model instance.

    Args:
        model: Trained PatchCore model.
        dataset: AutoVI test dataset.
        output_dir: Directory to save anomaly maps.

    Returns:
        Dictionary mapping defect_type to list of saved paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: Dict[str, List[str]] = {}

    logger.info(f"Generating anomaly maps for {len(dataset)} images...")

    for idx in tqdm(range(len(dataset)), desc="Generating anomaly maps"):
        sample = dataset[idx]
        image = sample["image"]
        defect_type = sample["defect_type"]
        img_path = Path(sample["path"])
        category = sample["category"]

        # Ensure image is tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.array(image).transpose(2, 0, 1).astype(np.float32) / 255.0)

        # Generate anomaly map using model's predict method
        anomaly_map, _ = model.predict_single(image)

        # Normalize to [0, 255]
        if anomaly_map.max() > anomaly_map.min():
            anomaly_map_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
        else:
            anomaly_map_norm = np.zeros_like(anomaly_map)

        anomaly_map_uint8 = (anomaly_map_norm * 255).astype(np.uint8)

        # Construct output path
        rel_path = Path(defect_type) / f"{img_path.stem}.png"
        save_path = output_dir / rel_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save anomaly map
        Image.fromarray(anomaly_map_uint8).save(save_path)

        # Track saved paths
        if defect_type not in saved_paths:
            saved_paths[defect_type] = []
        saved_paths[defect_type].append(str(save_path))

    logger.info(f"Saved {sum(len(v) for v in saved_paths.values())} anomaly maps to {output_dir}")

    return saved_paths

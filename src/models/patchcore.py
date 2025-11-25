"""PatchCore anomaly detection model implementation."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from .backbone import (
    FeatureExtractor,
    apply_local_neighborhood_averaging,
    reshape_features_to_patches,
)
from .memory_bank import MemoryBank


class PatchCore:
    """PatchCore anomaly detection model.

    PatchCore uses a pre-trained CNN backbone to extract patch-level features,
    stores representative features in a memory bank, and detects anomalies
    by computing distances to the nearest neighbors in the memory bank.

    Attributes:
        feature_extractor: The backbone feature extractor.
        memory_bank: Memory bank storing normal patch features.
        device: PyTorch device for computation.
    """

    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        layers: List[str] = None,
        coreset_ratio: float = 0.1,
        neighborhood_size: int = 3,
        device: str = "auto",
        use_faiss: bool = True,
    ):
        """Initialize PatchCore model.

        Args:
            backbone_name: Name of the backbone CNN.
            layers: List of layer names to extract features from.
            coreset_ratio: Fraction of patches to keep in memory bank.
            neighborhood_size: Kernel size for local neighborhood averaging.
            device: Device for computation ("auto", "cuda", or "cpu").
            use_faiss: Whether to use FAISS for nearest neighbor search.
        """
        self.backbone_name = backbone_name
        self.layers = layers if layers else ["layer2", "layer3"]
        self.coreset_ratio = coreset_ratio
        self.neighborhood_size = neighborhood_size
        self.use_faiss = use_faiss

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"PatchCore using device: {self.device}")

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            backbone_name=backbone_name,
            layers=self.layers,
            pretrained=True,
        ).to(self.device)

        # Feature dimension
        self.feature_dim = self.feature_extractor.get_feature_dim()

        # Memory bank (initialized during training)
        self.memory_bank: Optional[MemoryBank] = None

        # Store image size for anomaly map generation
        self.image_size: Optional[Tuple[int, int]] = None
        self.feature_map_size: Optional[Tuple[int, int]] = None

    def fit(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        """Train PatchCore by building the memory bank.

        Args:
            dataloader: DataLoader providing training images (normal samples only).
            max_samples: Maximum memory bank size (overrides coreset_ratio).
            seed: Random seed for coreset selection.
        """
        print("Extracting features from training images...")

        all_features = []
        self.feature_extractor.eval()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Feature extraction"):
                # Handle different batch formats
                if isinstance(batch, dict):
                    images = batch["image"]
                elif isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch

                images = images.to(self.device)

                # Store image size from first batch
                if self.image_size is None:
                    self.image_size = (images.shape[2], images.shape[3])

                # Extract features
                features = self.feature_extractor(images)

                # Store feature map size
                if self.feature_map_size is None:
                    self.feature_map_size = (features.shape[2], features.shape[3])

                # Apply local neighborhood averaging
                if self.neighborhood_size > 1:
                    features = apply_local_neighborhood_averaging(
                        features, self.neighborhood_size
                    )

                # Reshape to patch vectors
                patches = reshape_features_to_patches(features)
                all_features.append(patches.cpu().numpy())

        # Concatenate all features
        all_features = np.concatenate(all_features, axis=0)
        print(f"Total patches extracted: {all_features.shape[0]}")

        # Build memory bank with coreset subsampling
        self.memory_bank = MemoryBank(
            feature_dim=self.feature_dim,
            use_faiss=self.use_faiss,
            use_gpu=self.device.type == "cuda",
        )
        self.memory_bank.fit(
            features=all_features,
            coreset_ratio=self.coreset_ratio,
            max_samples=max_samples,
            seed=seed,
        )

    def predict(
        self,
        images: Union[torch.Tensor, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomaly scores for images.

        Args:
            images: Input images as tensor [B, 3, H, W] or numpy array.

        Returns:
            Tuple of:
                - anomaly_maps: Pixel-wise anomaly scores [B, H, W]
                - image_scores: Image-level anomaly scores [B]
        """
        if self.memory_bank is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        # Convert to tensor if needed
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)

        images = images.to(self.device)
        batch_size = images.shape[0]

        with torch.no_grad():
            # Extract features
            features = self.feature_extractor(images)

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
        anomaly_maps = anomaly_scores.reshape(batch_size, feature_h, feature_w)

        # Upsample to original image size
        if self.image_size is not None:
            anomaly_maps_tensor = torch.from_numpy(anomaly_maps).unsqueeze(1).float()
            anomaly_maps_upsampled = F.interpolate(
                anomaly_maps_tensor,
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            )
            anomaly_maps = anomaly_maps_upsampled.squeeze(1).numpy()

        # Compute image-level scores (max of anomaly map)
        image_scores = anomaly_maps.reshape(batch_size, -1).max(axis=1)

        return anomaly_maps, image_scores

    def predict_single(
        self,
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        transform=None,
    ) -> Tuple[np.ndarray, float]:
        """Predict anomaly score for a single image.

        Args:
            image: Input image as tensor, numpy array, or PIL Image.
            transform: Optional transform to apply to PIL images.

        Returns:
            Tuple of:
                - anomaly_map: Pixel-wise anomaly scores [H, W]
                - image_score: Image-level anomaly score
        """
        # Handle PIL Image
        if isinstance(image, Image.Image):
            if transform is not None:
                image = transform(image)
            else:
                image = np.array(image).astype(np.float32) / 255.0
                image = image.transpose(2, 0, 1)
                image = torch.from_numpy(image)

        # Add batch dimension
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
        elif isinstance(image, np.ndarray):
            if image.ndim == 3:
                image = image[np.newaxis, ...]

        anomaly_maps, image_scores = self.predict(image)

        return anomaly_maps[0], float(image_scores[0])

    def save(self, path: str) -> None:
        """Save model to file.

        Args:
            path: Output file path (without extension).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save memory bank
        memory_bank_path = str(path) + "_memory_bank.npz"
        self.memory_bank.save(memory_bank_path)

        # Save model config
        config = {
            "backbone_name": self.backbone_name,
            "layers": self.layers,
            "coreset_ratio": self.coreset_ratio,
            "neighborhood_size": self.neighborhood_size,
            "feature_dim": self.feature_dim,
            "image_size": self.image_size,
            "feature_map_size": self.feature_map_size,
        }
        config_path = str(path) + "_config.npz"
        np.savez(config_path, **config)

        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model from file.

        Args:
            path: Input file path (without extension).
        """
        # Load config
        config_path = str(path) + "_config.npz"
        config = np.load(config_path, allow_pickle=True)

        self.backbone_name = str(config["backbone_name"])
        self.layers = list(config["layers"])
        self.coreset_ratio = float(config["coreset_ratio"])
        self.neighborhood_size = int(config["neighborhood_size"])
        self.feature_dim = int(config["feature_dim"])
        self.image_size = tuple(config["image_size"]) if config["image_size"] is not None else None
        self.feature_map_size = tuple(config["feature_map_size"]) if config["feature_map_size"] is not None else None

        # Reinitialize feature extractor
        self.feature_extractor = FeatureExtractor(
            backbone_name=self.backbone_name,
            layers=self.layers,
            pretrained=True,
        ).to(self.device)

        # Load memory bank
        memory_bank_path = str(path) + "_memory_bank.npz"
        self.memory_bank = MemoryBank(
            feature_dim=self.feature_dim,
            use_faiss=self.use_faiss,
            use_gpu=self.device.type == "cuda",
        )
        self.memory_bank.load(memory_bank_path)

        print(f"Model loaded from {path}")

    def get_stats(self) -> Dict:
        """Get model statistics.

        Returns:
            Dictionary with model statistics.
        """
        stats = {
            "backbone": self.backbone_name,
            "layers": self.layers,
            "feature_dim": self.feature_dim,
            "coreset_ratio": self.coreset_ratio,
            "neighborhood_size": self.neighborhood_size,
            "image_size": self.image_size,
            "feature_map_size": self.feature_map_size,
            "memory_bank_size": len(self.memory_bank) if self.memory_bank else 0,
            "device": str(self.device),
        }
        return stats


def create_patchcore(
    config: Dict,
    device: str = "auto",
) -> PatchCore:
    """Create PatchCore model from configuration dictionary.

    Args:
        config: Configuration dictionary with model parameters.
        device: Device for computation.

    Returns:
        Initialized PatchCore model.
    """
    return PatchCore(
        backbone_name=config.get("backbone", "wide_resnet50_2"),
        layers=config.get("layers", ["layer2", "layer3"]),
        coreset_ratio=config.get("coreset_percentage", 0.1),
        neighborhood_size=config.get("neighborhood_size", 3),
        device=device,
        use_faiss=config.get("use_faiss", True),
    )

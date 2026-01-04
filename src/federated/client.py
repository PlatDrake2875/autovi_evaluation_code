"""PatchCore client for federated learning."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.backbone import FeatureExtractor
from src.models.memory_bank import greedy_coreset_selection
from src.privacy import EmbeddingSanitizer, DPConfig
from src.util import get_device, extract_images_from_batch


class PatchCoreClient:
    """Federated learning client for PatchCore anomaly detection.

    Each client holds a local data partition and extracts patch features
    from its local data. The client builds a local coreset which is then
    sent to the server for aggregation.

    Attributes:
        client_id: Unique identifier for this client.
        feature_extractor: Shared backbone for feature extraction.
        local_coreset: Local memory bank coreset after training.
        device: PyTorch device for computation.
    """

    def __init__(
        self,
        client_id: int,
        backbone_name: str = "wide_resnet50_2",
        layers: List[str] = None,
        neighborhood_size: int = 3,
        coreset_ratio: float = 0.1,
        device: str = "auto",
        dp_config: Optional[DPConfig] = None,
    ):
        """Initialize the PatchCore client.

        Args:
            client_id: Unique identifier for this client.
            backbone_name: Name of the backbone CNN.
            layers: List of layer names to extract features from.
            neighborhood_size: Kernel size for local neighborhood averaging.
            coreset_ratio: Fraction of local patches to keep in local coreset.
            device: Device for computation ("auto", "cuda", or "cpu").
            dp_config: Optional differential privacy configuration.
        """
        self.client_id = client_id
        self.backbone_name = backbone_name
        self.layers = layers if layers else ["layer2", "layer3"]
        self.neighborhood_size = neighborhood_size
        self.coreset_ratio = coreset_ratio

        # Set device
        self.device = get_device(device)

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            backbone_name=backbone_name,
            layers=self.layers,
            pretrained=True,
        ).to(self.device)

        # Feature dimension
        self.feature_dim = self.feature_extractor.get_feature_dim()

        # Local coreset (populated after training)
        self.local_coreset: Optional[np.ndarray] = None

        # Statistics
        self.stats: Dict = {
            "client_id": client_id,
            "num_samples": 0,
            "num_patches": 0,
            "coreset_size": 0,
        }

        # Differential privacy
        self.dp_config = dp_config
        self.sanitizer: Optional[EmbeddingSanitizer] = None
        if dp_config and dp_config.enabled:
            self.sanitizer = EmbeddingSanitizer(dp_config)
            logger.info(f"Client {client_id}: DP enabled with epsilon={dp_config.epsilon}")

    def extract_features(
        self,
        dataloader: DataLoader,
    ) -> np.ndarray:
        """Extract patch features from local data.

        Args:
            dataloader: DataLoader providing local training images.

        Returns:
            All extracted patch features as numpy array [N, D].
        """
        logger.info(f"Client {self.client_id}: Extracting features...")

        all_features = []
        self.feature_extractor.eval()
        num_samples = 0

        with torch.no_grad():
            for batch in tqdm(
                dataloader,
                desc=f"Client {self.client_id} feature extraction",
                leave=False,
            ):
                images = extract_images_from_batch(batch)
                images = images.to(self.device)
                num_samples += images.shape[0]

                # Extract patch features
                patches = self.feature_extractor.extract_patches(images, self.neighborhood_size)
                all_features.append(patches.cpu().numpy())

        # Concatenate all features
        all_features = np.concatenate(all_features, axis=0)

        # Update statistics
        self.stats["num_samples"] = num_samples
        self.stats["num_patches"] = all_features.shape[0]

        logger.info(
            f"Client {self.client_id}: Extracted {all_features.shape[0]} patches "
            f"from {num_samples} images"
        )

        return all_features

    def build_local_coreset(
        self,
        features: np.ndarray,
        target_size: Optional[int] = None,
        seed: int = 42,
    ) -> np.ndarray:
        """Build local coreset from extracted features.

        Args:
            features: Extracted patch features [N, D].
            target_size: Target coreset size. If None, uses coreset_ratio.
            seed: Random seed for coreset selection.

        Returns:
            Local coreset as numpy array [M, D] where M << N.
        """
        n_samples = features.shape[0]

        if target_size is None:
            target_size = max(1, int(n_samples * self.coreset_ratio))

        logger.info(
            f"Client {self.client_id}: Building local coreset "
            f"({target_size} from {n_samples} patches)..."
        )

        # Use client-specific seed for reproducibility
        client_seed = seed + self.client_id

        if target_size < n_samples:
            selected_indices = greedy_coreset_selection(
                features, target_size=target_size, seed=client_seed
            )
            self.local_coreset = features[selected_indices].copy()
        else:
            self.local_coreset = features.copy()

        # Apply differential privacy sanitization if enabled
        if self.sanitizer is not None:
            self.local_coreset = self.sanitizer.sanitize(
                self.local_coreset, seed=client_seed
            )

        # Update statistics
        self.stats["coreset_size"] = len(self.local_coreset)

        logger.info(f"Client {self.client_id}: Local coreset size: {len(self.local_coreset)}")

        return self.local_coreset

    def extract_and_build_coreset(
        self,
        dataloader: DataLoader,
        target_size: Optional[int] = None,
        seed: int = 42,
    ) -> np.ndarray:
        """Extract features and build local coreset in one step.

        Args:
            dataloader: DataLoader providing local training images.
            target_size: Target coreset size. If None, uses coreset_ratio.
            seed: Random seed for coreset selection.

        Returns:
            Local coreset as numpy array.
        """
        features = self.extract_features(dataloader)
        return self.build_local_coreset(features, target_size=target_size, seed=seed)

    def get_local_coreset(self) -> Optional[np.ndarray]:
        """Get the local coreset.

        Returns:
            Local coreset or None if not yet built.
        """
        return self.local_coreset

    def get_stats(self) -> Dict:
        """Get client statistics.

        Returns:
            Dictionary with client statistics.
        """
        return self.stats.copy()

    def get_privacy_spent(self) -> Tuple[float, float]:
        """Get privacy budget spent.

        Returns:
            Tuple of (epsilon, delta). Returns (0.0, 0.0) if DP is disabled.
        """
        if self.sanitizer is None:
            return (0.0, 0.0)
        return self.sanitizer.get_privacy_spent()

    def set_global_memory_bank(self, global_memory: np.ndarray) -> None:
        """Receive and store the global memory bank from server.

        After aggregation, the server broadcasts the global memory bank
        to all clients for local inference.

        Args:
            global_memory: Global aggregated memory bank.
        """
        # For now, we store it in local_coreset for inference
        # In a full implementation, this would update a separate
        # inference memory bank
        self._global_memory = global_memory
        logger.info(
            f"Client {self.client_id}: Received global memory bank "
            f"with {len(global_memory)} patches"
        )

    def __repr__(self) -> str:
        dp_status = "enabled" if self.sanitizer is not None else "disabled"
        return (
            f"PatchCoreClient(id={self.client_id}, "
            f"coreset_size={self.stats.get('coreset_size', 0)}, "
            f"device={self.device}, dp={dp_status})"
        )

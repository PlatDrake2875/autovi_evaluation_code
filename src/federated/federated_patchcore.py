"""FederatedPatchCore orchestrator for federated anomaly detection."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from src.data.autovi_dataset import AutoVIDataset, AutoVISubset
from src.data.partitioner import (
    IIDPartitioner,
    CategoryPartitioner,
    compute_partition_stats,
)
from src.models.backbone import (
    FeatureExtractor,
    apply_local_neighborhood_averaging,
    reshape_features_to_patches,
)
from src.models.memory_bank import MemoryBank

from .client import PatchCoreClient
from .server import FederatedServer


class FederatedPatchCore:
    """Orchestrator for federated PatchCore anomaly detection.

    This class coordinates the entire federated learning process:
    1. Setting up clients with data partitions
    2. Running the federated training round
    3. Providing inference using the global memory bank

    Attributes:
        num_clients: Number of federated clients.
        clients: List of PatchCoreClient objects.
        server: FederatedServer for aggregation.
        global_memory_bank: Global memory bank after training.
    """

    def __init__(
        self,
        num_clients: int = 5,
        backbone_name: str = "wide_resnet50_2",
        layers: List[str] = None,
        coreset_ratio: float = 0.1,
        global_bank_size: int = 10000,
        neighborhood_size: int = 3,
        aggregation_strategy: str = "federated_coreset",
        weighted_by_samples: bool = True,
        use_faiss: bool = True,
        device: str = "auto",
        num_rounds: int = 1,
    ):
        """Initialize the FederatedPatchCore system.

        Args:
            num_clients: Number of federated clients.
            backbone_name: Name of the backbone CNN.
            layers: List of layer names to extract features from.
            coreset_ratio: Fraction of patches to keep in local coresets.
            global_bank_size: Target size for the global memory bank.
            neighborhood_size: Kernel size for local neighborhood averaging.
            aggregation_strategy: Strategy for aggregating client coresets.
            weighted_by_samples: If True, weight client contributions by data size.
            use_faiss: Whether to use FAISS for nearest neighbor search.
            device: Device for computation.
            num_rounds: Number of federated training rounds.
        """
        self.num_clients = num_clients
        self.backbone_name = backbone_name
        self.layers = layers if layers else ["layer2", "layer3"]
        self.coreset_ratio = coreset_ratio
        self.global_bank_size = global_bank_size
        self.neighborhood_size = neighborhood_size
        self.aggregation_strategy = aggregation_strategy
        self.weighted_by_samples = weighted_by_samples
        self.use_faiss = use_faiss
        self.num_rounds = num_rounds

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"FederatedPatchCore using device: {self.device}")

        # Initialize clients
        self.clients: List[PatchCoreClient] = []
        for i in range(num_clients):
            client = PatchCoreClient(
                client_id=i,
                backbone_name=backbone_name,
                layers=self.layers,
                neighborhood_size=neighborhood_size,
                coreset_ratio=coreset_ratio,
                device=str(self.device),
            )
            self.clients.append(client)

        # Initialize server
        self.server = FederatedServer(
            global_bank_size=global_bank_size,
            aggregation_strategy=aggregation_strategy,
            weighted_by_samples=weighted_by_samples,
            use_faiss=use_faiss,
            use_gpu=self.device.type == "cuda",
        )

        # Shared feature extractor for inference
        self.feature_extractor = FeatureExtractor(
            backbone_name=backbone_name,
            layers=self.layers,
            pretrained=True,
        ).to(self.device)

        self.feature_dim = self.feature_extractor.get_feature_dim()

        # Global memory bank (populated after training)
        self.global_memory_bank: Optional[MemoryBank] = None

        # Image/feature map sizes (set during training)
        self.image_size: Optional[Tuple[int, int]] = None
        self.feature_map_size: Optional[Tuple[int, int]] = None

        # Training statistics
        self.training_stats: Dict = {}

    def setup_clients(
        self,
        dataset: AutoVIDataset,
        partitioning: str = "iid",
        seed: int = 42,
        **partitioner_kwargs,
    ) -> Dict[int, List[int]]:
        """Setup clients with data partitions.

        Args:
            dataset: AutoVIDataset to partition.
            partitioning: Partitioning strategy ("iid" or "category").
            seed: Random seed for partitioning.
            **partitioner_kwargs: Additional arguments for the partitioner.

        Returns:
            Dictionary mapping client_id -> list of sample indices.
        """
        print(f"\nSetting up {self.num_clients} clients with {partitioning} partitioning...")

        if partitioning == "iid":
            partitioner = IIDPartitioner(num_clients=self.num_clients, seed=seed)
        elif partitioning == "category":
            partitioner = CategoryPartitioner(seed=seed, **partitioner_kwargs)
        else:
            raise ValueError(f"Unknown partitioning: {partitioning}")

        self.partition = partitioner.partition(dataset)
        self.partition_stats = compute_partition_stats(dataset, self.partition)

        # Print partition statistics
        print("\nPartition statistics:")
        for client_id, stats in self.partition_stats["clients"].items():
            print(f"  Client {client_id}: {stats['num_samples']} samples")
            for cat, count in stats["by_category"].items():
                print(f"    - {cat}: {count}")

        return self.partition

    def train(
        self,
        dataloaders: Dict[int, DataLoader],
        seed: int = 42,
        checkpoint_dir: Optional[str] = None,
        checkpoint_every: int = 1,
    ) -> np.ndarray:
        """Run federated training for multiple rounds with optional checkpointing.

        This is the main training method that:
        1. Has each client extract features and build local coresets
        2. Aggregates coresets on the server
        3. Broadcasts the global memory bank to all clients
        4. Saves checkpoints after each round (if checkpoint_dir is provided)

        Args:
            dataloaders: Dictionary mapping client_id -> DataLoader.
            seed: Random seed for reproducibility.
            checkpoint_dir: Directory to save checkpoints. If None, no checkpoints are saved.
            checkpoint_every: Save checkpoint every N rounds (default: 1, meaning every round).

        Returns:
            Global memory bank as numpy array.
        """
        total_start_time = time.time()
        print("\n" + "=" * 60)
        print(f"Starting Federated Training ({self.num_rounds} rounds)")
        print("=" * 60)

        global_features = None

        for round_num in range(1, self.num_rounds + 1):
            print(f"\n{'='*60}")
            print(f"Round {round_num}/{self.num_rounds}")
            print(f"{'='*60}")

            # Run single round
            round_seed = seed + round_num - 1
            global_features = self._train_single_round(dataloaders, seed=round_seed, round_num=round_num)

            # Save checkpoint if needed
            if checkpoint_dir and round_num % checkpoint_every == 0:
                round_dir = Path(checkpoint_dir) / f"round_{round_num:03d}"
                self.save(str(round_dir))
                print(f"Saved checkpoint to {round_dir}")

        total_elapsed_time = time.time() - total_start_time
        print("\n" + "=" * 60)
        print(f"Federated Training Complete ({total_elapsed_time:.2f}s, {self.num_rounds} rounds)")
        print(f"Global memory bank size: {len(global_features)}")
        print("=" * 60)

        return global_features

    def _train_single_round(
        self,
        dataloaders: Dict[int, DataLoader],
        seed: int = 42,
        round_num: int = 1,
    ) -> np.ndarray:
        """Run a single federated training round.

        Args:
            dataloaders: Dictionary mapping client_id -> DataLoader.
            seed: Random seed for reproducibility.
            round_num: Current round number (for logging).

        Returns:
            Global memory bank as numpy array.
        """
        start_time = time.time()

        # Phase 1: Local feature extraction and coreset building
        print("\n--- Phase 1: Local Client Processing ---")
        client_coresets = []
        client_stats = []

        for client_id, client in enumerate(self.clients):
            if client_id not in dataloaders:
                print(f"Warning: No dataloader for client {client_id}")
                continue

            dataloader = dataloaders[client_id]

            # Store image size from first batch
            if self.image_size is None:
                for batch in dataloader:
                    if isinstance(batch, dict):
                        images = batch["image"]
                    elif isinstance(batch, (list, tuple)):
                        images = batch[0]
                    else:
                        images = batch
                    self.image_size = (images.shape[2], images.shape[3])
                    break

            # Extract features and build coreset
            coreset = client.extract_and_build_coreset(dataloader, seed=seed)
            client_coresets.append(coreset)
            client_stats.append(client.get_stats())

        # Phase 2: Server aggregation
        print("\n--- Phase 2: Server Aggregation ---")
        self.server.receive_client_coresets(client_coresets, client_stats)
        global_features = self.server.aggregate(seed=seed)

        # Build global memory bank
        self.global_memory_bank = self.server.get_global_memory_bank()

        # Get feature map size from feature extractor
        if self.feature_map_size is None and self.image_size is not None:
            # Estimate feature map size based on backbone
            # For WideResNet50, layer3 output is H/8, W/8
            self.feature_map_size = (
                self.image_size[0] // 8,
                self.image_size[1] // 8,
            )

        # Phase 3: Broadcast to clients
        print("\n--- Phase 3: Broadcasting Global Model ---")
        self.server.broadcast_to_clients(self.clients)

        # Compile training statistics
        elapsed_time = time.time() - start_time
        self.training_stats = {
            "elapsed_time_seconds": elapsed_time,
            "num_clients": len(self.clients),
            "num_rounds": self.num_rounds,
            "current_round": round_num,
            "partition_stats": self.partition_stats if hasattr(self, "partition_stats") else {},
            "server_stats": self.server.get_stats(),
            "client_stats": client_stats,
        }

        print(f"\nRound {round_num} complete ({elapsed_time:.2f}s)")
        print(f"Global memory bank size: {len(global_features)}")

        return global_features

    def predict(
        self,
        images: Union[torch.Tensor, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomaly scores for images using the global memory bank.

        Args:
            images: Input images as tensor [B, 3, H, W] or numpy array.

        Returns:
            Tuple of:
                - anomaly_maps: Pixel-wise anomaly scores [B, H, W]
                - image_scores: Image-level anomaly scores [B]
        """
        if self.global_memory_bank is None:
            raise RuntimeError("Model not trained. Call train() first.")

        # Convert to tensor if needed
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)

        images = images.to(self.device)
        batch_size = images.shape[0]

        # Store image size if not set
        if self.image_size is None:
            self.image_size = (images.shape[2], images.shape[3])

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

        # Get anomaly scores from global memory bank
        anomaly_scores = self.global_memory_bank.get_anomaly_scores(patches.cpu().numpy())

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

    def save(self, output_dir: str) -> None:
        """Save the federated model and statistics.

        Args:
            output_dir: Output directory path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save server (includes global memory bank)
        self.server.save(output_dir)

        # Save model config
        config = {
            "num_clients": self.num_clients,
            "backbone_name": self.backbone_name,
            "layers": self.layers,
            "coreset_ratio": self.coreset_ratio,
            "global_bank_size": self.global_bank_size,
            "neighborhood_size": self.neighborhood_size,
            "aggregation_strategy": self.aggregation_strategy,
            "weighted_by_samples": self.weighted_by_samples,
            "num_rounds": self.num_rounds,
            "feature_dim": self.feature_dim,
            "image_size": self.image_size,
            "feature_map_size": self.feature_map_size,
        }
        config_path = output_dir / "federated_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved federated config to {config_path}")

        # Save training statistics
        if self.training_stats:
            stats_path = output_dir / "training_log.json"
            with open(stats_path, "w") as f:
                json.dump(_convert_to_serializable(self.training_stats), f, indent=2)
            print(f"Saved training log to {stats_path}")

    def load(self, input_dir: str) -> None:
        """Load a previously saved federated model.

        Args:
            input_dir: Input directory path.
        """
        input_dir = Path(input_dir)

        # Load config
        config_path = input_dir / "federated_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)

            self.num_clients = config.get("num_clients", self.num_clients)
            self.backbone_name = config.get("backbone_name", self.backbone_name)
            self.layers = config.get("layers", self.layers)
            self.coreset_ratio = config.get("coreset_ratio", self.coreset_ratio)
            self.global_bank_size = config.get("global_bank_size", self.global_bank_size)
            self.neighborhood_size = config.get("neighborhood_size", self.neighborhood_size)
            self.aggregation_strategy = config.get("aggregation_strategy", self.aggregation_strategy)
            self.weighted_by_samples = config.get("weighted_by_samples", self.weighted_by_samples)
            self.num_rounds = config.get("num_rounds", self.num_rounds)
            self.feature_dim = config.get("feature_dim", self.feature_dim)
            self.image_size = tuple(config["image_size"]) if config.get("image_size") else None
            self.feature_map_size = tuple(config["feature_map_size"]) if config.get("feature_map_size") else None

        # Load server
        self.server.load(input_dir)
        self.global_memory_bank = self.server.get_global_memory_bank()

        # Reinitialize feature extractor
        self.feature_extractor = FeatureExtractor(
            backbone_name=self.backbone_name,
            layers=self.layers,
            pretrained=True,
        ).to(self.device)

        print(f"Loaded federated model from {input_dir}")

    def get_stats(self) -> Dict:
        """Get comprehensive statistics about the federated system.

        Returns:
            Dictionary with system statistics.
        """
        stats = {
            "num_clients": self.num_clients,
            "backbone": self.backbone_name,
            "layers": self.layers,
            "coreset_ratio": self.coreset_ratio,
            "global_bank_size": self.global_bank_size,
            "neighborhood_size": self.neighborhood_size,
            "aggregation_strategy": self.aggregation_strategy,
            "num_rounds": self.num_rounds,
            "device": str(self.device),
            "feature_dim": self.feature_dim,
            "image_size": self.image_size,
            "actual_global_bank_size": (
                len(self.global_memory_bank) if self.global_memory_bank else 0
            ),
        }

        if self.training_stats:
            stats["training"] = self.training_stats

        return stats

    def __repr__(self) -> str:
        bank_size = len(self.global_memory_bank) if self.global_memory_bank else 0
        return (
            f"FederatedPatchCore(clients={self.num_clients}, "
            f"strategy={self.aggregation_strategy}, "
            f"global_bank={bank_size}/{self.global_bank_size})"
        )


def _convert_to_serializable(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

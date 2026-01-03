"""Federated server for PatchCore memory bank aggregation."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

from src.models.memory_bank import MemoryBank
from .strategies.federated_memory import STRATEGY_REGISTRY


class FederatedServer:
    """Central server for federated PatchCore training.

    The server coordinates the federated training process:
    1. Receives local coresets from all clients
    2. Aggregates them into a global memory bank
    3. Broadcasts the global memory bank back to clients

    Attributes:
        global_memory_bank: The aggregated global memory bank.
        aggregation_stats: Statistics from the last aggregation.
    """

    def __init__(
        self,
        global_bank_size: int = 10000,
        aggregation_strategy: str = "federated_coreset",
        weighted_by_samples: bool = True,
        use_faiss: bool = True,
        use_gpu: bool = False,
    ):
        """Initialize the federated server.

        Args:
            global_bank_size: Target size for the global memory bank.
            aggregation_strategy: Strategy for aggregating client coresets.
                Options: "federated_coreset", "simple_concatenate", "diversity_preserving"
            weighted_by_samples: If True, weight client contributions by data size.
            use_faiss: Whether to use FAISS for the global memory bank.
            use_gpu: Whether to use GPU for FAISS.
        """
        self.global_bank_size = global_bank_size
        self.aggregation_strategy = aggregation_strategy
        self.weighted_by_samples = weighted_by_samples
        self.use_faiss = use_faiss
        self.use_gpu = use_gpu

        # Global memory bank (populated after aggregation)
        self.global_memory_bank: Optional[MemoryBank] = None
        self.global_features: Optional[np.ndarray] = None

        # Statistics
        self.aggregation_stats: Dict = {}
        self.client_stats: List[Dict] = []

    def receive_client_coresets(
        self,
        client_coresets: List[np.ndarray],
        client_stats: Optional[List[Dict]] = None,
    ) -> None:
        """Receive local coresets from all clients.

        Args:
            client_coresets: List of local coreset arrays from each client.
            client_stats: Optional list of statistics from each client.
        """
        self._pending_coresets = client_coresets
        if client_stats:
            self.client_stats = client_stats

        logger.info(f"Server: Received coresets from {len(client_coresets)} clients")
        for i, coreset in enumerate(client_coresets):
            logger.debug(f"  Client {i}: {len(coreset)} patches")

    def aggregate(self, seed: int = 42) -> np.ndarray:
        """Aggregate client coresets into a global memory bank.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            Global memory bank as numpy array.
        """
        if not hasattr(self, "_pending_coresets") or not self._pending_coresets:
            raise RuntimeError("No client coresets received. Call receive_client_coresets first.")

        logger.info(f"Server: Aggregating using strategy: {self.aggregation_strategy}")

        # Select aggregation strategy from registry
        strategy_fn = STRATEGY_REGISTRY.get(self.aggregation_strategy)
        if strategy_fn is None:
            raise ValueError(
                f"Unknown aggregation strategy: {self.aggregation_strategy}. "
                f"Available: {list(STRATEGY_REGISTRY.keys())}"
            )

        # Execute the strategy
        self.global_features, self.aggregation_stats = strategy_fn(
            client_coresets=self._pending_coresets,
            global_bank_size=self.global_bank_size,
            seed=seed,
        )

        # Build memory bank for inference
        feature_dim = self.global_features.shape[1]
        self.global_memory_bank = MemoryBank(
            feature_dim=feature_dim,
            use_faiss=self.use_faiss,
            use_gpu=self.use_gpu,
        )
        # Set features using public API
        self.global_memory_bank.set_features(self.global_features)

        logger.info(f"Server: Global memory bank built with {len(self.global_features)} patches")

        return self.global_features

    def get_global_memory_bank(self) -> Optional[MemoryBank]:
        """Get the global memory bank for inference.

        Returns:
            Global MemoryBank object or None if not yet aggregated.
        """
        return self.global_memory_bank

    def get_global_features(self) -> Optional[np.ndarray]:
        """Get the raw global feature array.

        Returns:
            Global features as numpy array or None if not yet aggregated.
        """
        return self.global_features

    def broadcast_to_clients(self, clients: List) -> None:
        """Broadcast global memory bank to all clients.

        Args:
            clients: List of PatchCoreClient objects.
        """
        if self.global_features is None:
            raise RuntimeError("No global memory bank. Call aggregate first.")

        logger.info(f"Server: Broadcasting global memory bank to {len(clients)} clients")
        for client in clients:
            client.set_global_memory_bank(self.global_features)

    def get_stats(self) -> Dict:
        """Get server statistics including aggregation details.

        Returns:
            Dictionary with server and aggregation statistics.
        """
        stats = {
            "global_bank_size": self.global_bank_size,
            "aggregation_strategy": self.aggregation_strategy,
            "weighted_by_samples": self.weighted_by_samples,
            "aggregation_stats": self.aggregation_stats,
            "client_stats": self.client_stats,
        }

        if self.global_features is not None:
            stats["actual_global_bank_size"] = len(self.global_features)
            stats["feature_dim"] = self.global_features.shape[1]

        return stats

    def save(self, output_dir: str) -> None:
        """Save the global memory bank and statistics.

        Args:
            output_dir: Output directory path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save global features
        if self.global_features is not None:
            features_path = output_dir / "global_memory_bank.npz"
            np.savez(
                features_path,
                features=self.global_features,
                feature_dim=self.global_features.shape[1],
            )
            logger.info(f"Saved global memory bank to {features_path}")

        # Save statistics
        stats = self.get_stats()
        stats_path = output_dir / "server_stats.json"
        with open(stats_path, "w") as f:
            # Convert numpy types to Python types for JSON serialization
            json.dump(_convert_to_serializable(stats), f, indent=2)
        logger.info(f"Saved server stats to {stats_path}")

        # Save client statistics
        if self.client_stats:
            client_stats_path = output_dir / "client_stats.json"
            with open(client_stats_path, "w") as f:
                json.dump(_convert_to_serializable(self.client_stats), f, indent=2)
            logger.info(f"Saved client stats to {client_stats_path}")

    def load(self, input_dir: str) -> None:
        """Load a previously saved global memory bank.

        Args:
            input_dir: Input directory path.
        """
        input_dir = Path(input_dir)

        # Load global features
        features_path = input_dir / "global_memory_bank.npz"
        if features_path.exists():
            data = np.load(features_path)
            self.global_features = data["features"]
            feature_dim = int(data["feature_dim"])

            # Rebuild memory bank
            self.global_memory_bank = MemoryBank(
                feature_dim=feature_dim,
                use_faiss=self.use_faiss,
                use_gpu=self.use_gpu,
            )
            self.global_memory_bank.set_features(self.global_features)

            logger.info(f"Loaded global memory bank from {features_path}")
            logger.debug(f"  Shape: {self.global_features.shape}")

        # Load statistics
        stats_path = input_dir / "server_stats.json"
        if stats_path.exists():
            with open(stats_path, "r") as f:
                stats = json.load(f)
                self.aggregation_stats = stats.get("aggregation_stats", {})

        client_stats_path = input_dir / "client_stats.json"
        if client_stats_path.exists():
            with open(client_stats_path, "r") as f:
                self.client_stats = json.load(f)

    def __repr__(self) -> str:
        size = len(self.global_features) if self.global_features is not None else 0
        return (
            f"FederatedServer(strategy={self.aggregation_strategy}, "
            f"global_bank_size={size}/{self.global_bank_size})"
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

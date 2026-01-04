"""Robust aggregation methods for Byzantine-resilient federated learning."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


class RobustAggregator(ABC):
    """Base class for robust aggregation methods."""

    @abstractmethod
    def aggregate(
        self,
        client_updates: List[np.ndarray],
        weights: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Aggregate client updates using a robust method.

        Args:
            client_updates: List of client feature arrays, each of shape [n_i, d].
            weights: Optional weights for each client.

        Returns:
            Tuple of (aggregated_features, stats_dict).
        """
        pass


class CoordinateMedianAggregator(RobustAggregator):
    """Coordinate-wise median aggregator for Byzantine-resilient aggregation.

    This aggregator computes the median across clients for each feature dimension,
    which is robust to up to 50% malicious clients (Byzantine fault tolerance).

    The algorithm:
    1. Sample a fixed number of vectors from each client's coreset
    2. Stack samples into a tensor of shape [num_clients, num_samples, feature_dim]
    3. Compute the median across the client dimension (axis=0)
    4. Return the median features

    Attributes:
        num_samples_per_client: Number of samples to take from each client.
        seed: Random seed for reproducibility.
    """

    def __init__(self, num_samples_per_client: int = 100, seed: int = 42):
        """Initialize the coordinate median aggregator.

        Args:
            num_samples_per_client: Number of samples to take from each client.
            seed: Random seed for sampling.
        """
        self.num_samples_per_client = num_samples_per_client
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def aggregate(
        self,
        client_updates: List[np.ndarray],
        weights: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Aggregate client coresets using coordinate-wise median.

        Args:
            client_updates: List of client feature arrays, each of shape [n_i, d].
            weights: Optional weights (ignored for median aggregation).

        Returns:
            Tuple of (median_features, stats_dict).
        """
        if not client_updates:
            raise ValueError("No client updates provided")

        num_clients = len(client_updates)
        feature_dim = client_updates[0].shape[1]

        # Sample from each client
        sampled = []
        client_sizes = []
        for i, update in enumerate(client_updates):
            n_samples = update.shape[0]
            client_sizes.append(n_samples)

            if n_samples == 0:
                logger.warning(f"Client {i} has no samples, skipping")
                continue

            # Sample with replacement if needed
            if n_samples >= self.num_samples_per_client:
                indices = self._rng.choice(
                    n_samples, size=self.num_samples_per_client, replace=False
                )
            else:
                indices = self._rng.choice(
                    n_samples, size=self.num_samples_per_client, replace=True
                )

            sampled.append(update[indices])

        if not sampled:
            raise ValueError("All clients have empty updates")

        # Stack into [num_clients, num_samples, feature_dim]
        stacked = np.stack(sampled, axis=0)

        # Compute median across clients (axis=0)
        median_features = np.median(stacked, axis=0)

        stats = {
            "aggregation_method": "coordinate_median",
            "num_clients": num_clients,
            "num_clients_used": len(sampled),
            "samples_per_client": self.num_samples_per_client,
            "feature_dim": feature_dim,
            "output_shape": list(median_features.shape),
            "client_sizes": client_sizes,
        }

        logger.debug(
            f"CoordinateMedian: aggregated {len(sampled)} clients, "
            f"output shape {median_features.shape}"
        )

        return median_features, stats

    def __repr__(self) -> str:
        return (
            f"CoordinateMedianAggregator("
            f"num_samples_per_client={self.num_samples_per_client}, "
            f"seed={self.seed})"
        )

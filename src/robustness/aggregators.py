"""Robust aggregation methods for Byzantine-resilient federated learning."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


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

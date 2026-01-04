"""Client scoring and anomaly detection for federated learning."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
from loguru import logger


@dataclass
class ClientScore:
    """Score and metadata for a single client.

    Attributes:
        client_id: Identifier for the client.
        score: Anomaly score (higher = more anomalous).
        is_outlier: Whether the client is flagged as an outlier.
        details: Additional details about the scoring.
    """

    client_id: int
    score: float
    is_outlier: bool
    details: Dict[str, Any] = field(default_factory=dict)


class ZScoreDetector:
    """Detect anomalous clients using Z-score on update statistics.

    This detector computes statistics (mean norm, std, max norm) for each
    client's feature vectors, then calculates Z-scores across all clients
    to identify outliers.

    A client is flagged as an outlier if its Z-score exceeds the threshold
    for any of the tracked metrics.

    Attributes:
        threshold: Z-score threshold for outlier detection.
    """

    def __init__(self, threshold: float = 3.0):
        """Initialize the Z-score detector.

        Args:
            threshold: Z-score threshold above which a client is flagged
                as an outlier. Default is 3.0 (standard practice).
        """
        if threshold <= 0:
            raise ValueError(f"Threshold must be positive, got {threshold}")
        self.threshold = threshold

    def score_clients(self, client_updates: List[np.ndarray]) -> List[ClientScore]:
        """Score clients based on their update statistics.

        Args:
            client_updates: List of client feature arrays, each of shape [n_i, d].

        Returns:
            List of ClientScore objects, one per client.
        """
        if not client_updates:
            return []

        num_clients = len(client_updates)

        # Compute per-client statistics
        client_stats = []
        for i, update in enumerate(client_updates):
            if update.shape[0] == 0:
                # Handle empty updates
                client_stats.append({
                    "mean_norm": 0.0,
                    "std_norm": 0.0,
                    "max_norm": 0.0,
                })
                continue

            norms = np.linalg.norm(update, axis=1)
            client_stats.append({
                "mean_norm": float(np.mean(norms)),
                "std_norm": float(np.std(norms)),
                "max_norm": float(np.max(norms)),
            })

        # Convert to arrays for Z-score computation
        mean_norms = np.array([s["mean_norm"] for s in client_stats])
        std_norms = np.array([s["std_norm"] for s in client_stats])
        max_norms = np.array([s["max_norm"] for s in client_stats])

        # Compute Z-scores for each metric
        z_scores = {}
        for name, values in [
            ("mean_norm", mean_norms),
            ("std_norm", std_norms),
            ("max_norm", max_norms),
        ]:
            mean = np.mean(values)
            std = np.std(values)
            if std > 1e-10:  # Avoid division by zero
                z_scores[name] = np.abs((values - mean) / std)
            else:
                z_scores[name] = np.zeros_like(values)

        # Create ClientScore objects
        scores = []
        num_outliers = 0
        for i in range(num_clients):
            # Maximum Z-score across all metrics
            max_z = max(
                z_scores["mean_norm"][i],
                z_scores["std_norm"][i],
                z_scores["max_norm"][i],
            )
            is_outlier = max_z >= self.threshold

            if is_outlier:
                num_outliers += 1

            scores.append(ClientScore(
                client_id=i,
                score=float(max_z),
                is_outlier=is_outlier,
                details={
                    "mean_norm": client_stats[i]["mean_norm"],
                    "std_norm": client_stats[i]["std_norm"],
                    "max_norm": client_stats[i]["max_norm"],
                    "z_mean_norm": float(z_scores["mean_norm"][i]),
                    "z_std_norm": float(z_scores["std_norm"][i]),
                    "z_max_norm": float(z_scores["max_norm"][i]),
                },
            ))

        logger.debug(
            f"ZScoreDetector: scored {num_clients} clients, "
            f"{num_outliers} outliers detected (threshold={self.threshold})"
        )

        return scores

    def __repr__(self) -> str:
        return f"ZScoreDetector(threshold={self.threshold})"

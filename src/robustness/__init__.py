"""Robustness module for Byzantine-resilient federated learning.

This module provides mechanisms to defend against Byzantine/malicious
clients in federated learning, including:
- Robust aggregation methods (coordinate-wise median, trimmed mean)
- Client anomaly detection (Z-score based)
- Attack simulations for evaluation
"""

from .config import RobustnessConfig
from .aggregators import RobustAggregator, CoordinateMedianAggregator
from .client_scoring import ClientScore

__all__ = [
    "RobustnessConfig",
    "RobustAggregator",
    "CoordinateMedianAggregator",
    "ClientScore",
]

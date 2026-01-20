"""Fairness evaluation module for federated learning.

This module provides tools to evaluate fairness across different groups
(clients, categories, defect types) in federated anomaly detection.

Key metrics implemented:
- Jain's Fairness Index: Measures distribution equality (1.0 = perfect)
- Performance Variance: Lower variance = more fair
- Performance Gap: Max - min difference
- Worst-Case Performance: Rawlsian fairness (min performance)
- Coefficient of Variation: Normalized disparity (std/mean)

References:
- Li et al. (2021). "New Metrics to Evaluate the Performance and Fairness
  of Personalized Federated Learning" - https://arxiv.org/abs/2107.13173
"""

from .config import FairnessConfig
from .metrics import (
    FairnessMetrics,
    compute_jains_index,
    compute_performance_variance,
    compute_performance_gap,
    compute_worst_case,
    compute_coefficient_of_variation,
    compute_all_metrics,
)
from .evaluator import (
    GroupEvaluationResult,
    FairnessEvaluationResult,
    FairnessEvaluator,
)

__all__ = [
    # Config
    "FairnessConfig",
    # Metrics
    "FairnessMetrics",
    "compute_jains_index",
    "compute_performance_variance",
    "compute_performance_gap",
    "compute_worst_case",
    "compute_coefficient_of_variation",
    "compute_all_metrics",
    # Evaluator
    "GroupEvaluationResult",
    "FairnessEvaluationResult",
    "FairnessEvaluator",
]

"""Core fairness metrics for federated learning evaluation.

References:
- Li et al. (2021). "New Metrics to Evaluate the Performance and Fairness
  of Personalized Federated Learning" - https://arxiv.org/abs/2107.13173
- Nguyen et al. (2025). "Fairness in Federated Learning: Fairness for Whom?"
  - https://arxiv.org/abs/2505.21584
- Salazar et al. (2024). "Federated Fairness Analytics: Quantifying Fairness
  in Federated Learning" - https://arxiv.org/abs/2408.08214
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class FairnessMetrics:
    """Container for computed fairness metrics.

    Attributes:
        jains_index: Jain's Fairness Index (1.0 = perfect fairness, 1/n = worst).
        variance: Variance of performance across groups.
        performance_gap: Max - min performance gap.
        worst_case: Worst-case (minimum) performance (Rawlsian fairness).
        coefficient_of_variation: Std / mean (normalized disparity).
        mean: Mean performance across groups.
        std: Standard deviation of performance.
        n_groups: Number of groups evaluated.
        group_performances: Individual group performances.
    """

    jains_index: float
    variance: float
    performance_gap: float
    worst_case: float
    coefficient_of_variation: float
    mean: float
    std: float
    n_groups: int
    group_performances: dict[str, float]


def compute_jains_index(performances: Sequence[float]) -> float:
    """Compute Jain's Fairness Index.

    Jain's Fairness Index measures how fairly resources/performance are
    distributed. The formula is: (sum(x))^2 / (n * sum(x^2))

    Args:
        performances: Sequence of performance values (e.g., AUC scores per group).

    Returns:
        Jain's Fairness Index in range [1/n, 1.0].
        1.0 indicates perfect fairness (all groups have equal performance).
        1/n indicates worst fairness (one group has all performance).

    Raises:
        ValueError: If performances is empty or all values are zero.
    """
    if len(performances) == 0:
        raise ValueError("performances cannot be empty")

    performances_arr = np.array(performances, dtype=np.float64)
    n = len(performances_arr)

    sum_x = np.sum(performances_arr)
    sum_x_squared = np.sum(performances_arr**2)

    if sum_x_squared == 0:
        raise ValueError("Cannot compute Jain's index when all performances are zero")

    return float((sum_x**2) / (n * sum_x_squared))


def compute_performance_variance(performances: Sequence[float]) -> float:
    """Compute variance of performance across groups.

    Lower variance indicates more fair distribution of performance.

    Args:
        performances: Sequence of performance values.

    Returns:
        Variance of performances.

    Raises:
        ValueError: If performances is empty.
    """
    if len(performances) == 0:
        raise ValueError("performances cannot be empty")

    return float(np.var(performances))


def compute_performance_gap(performances: Sequence[float]) -> float:
    """Compute performance gap (max - min).

    The performance gap measures the disparity between best and worst
    performing groups. Lower gap indicates more fair distribution.

    Args:
        performances: Sequence of performance values.

    Returns:
        Performance gap (max - min).

    Raises:
        ValueError: If performances is empty.
    """
    if len(performances) == 0:
        raise ValueError("performances cannot be empty")

    performances_arr = np.array(performances)
    return float(np.max(performances_arr) - np.min(performances_arr))


def compute_worst_case(performances: Sequence[float]) -> float:
    """Compute worst-case (minimum) performance.

    This metric follows Rawlsian fairness principle - maximizing the
    minimum performance across groups.

    Args:
        performances: Sequence of performance values.

    Returns:
        Minimum performance value.

    Raises:
        ValueError: If performances is empty.
    """
    if len(performances) == 0:
        raise ValueError("performances cannot be empty")

    return float(np.min(performances))


def compute_coefficient_of_variation(performances: Sequence[float]) -> float:
    """Compute coefficient of variation (std / mean).

    CV is a normalized measure of dispersion. Lower CV indicates more
    fair distribution. Returns 0 if mean is 0 (all performances are 0).

    Args:
        performances: Sequence of performance values.

    Returns:
        Coefficient of variation (std / mean), or 0 if mean is 0.

    Raises:
        ValueError: If performances is empty.
    """
    if len(performances) == 0:
        raise ValueError("performances cannot be empty")

    performances_arr = np.array(performances, dtype=np.float64)
    mean = np.mean(performances_arr)
    std = np.std(performances_arr)

    if mean == 0:
        return 0.0

    return float(std / mean)


def compute_all_metrics(
    group_performances: dict[str, float],
) -> FairnessMetrics:
    """Compute all fairness metrics for a set of group performances.

    Args:
        group_performances: Dictionary mapping group names to performance values.

    Returns:
        FairnessMetrics object containing all computed metrics.

    Raises:
        ValueError: If group_performances is empty.
    """
    if not group_performances:
        raise ValueError("group_performances cannot be empty")

    performances = list(group_performances.values())

    return FairnessMetrics(
        jains_index=compute_jains_index(performances),
        variance=compute_performance_variance(performances),
        performance_gap=compute_performance_gap(performances),
        worst_case=compute_worst_case(performances),
        coefficient_of_variation=compute_coefficient_of_variation(performances),
        mean=float(np.mean(performances)),
        std=float(np.std(performances)),
        n_groups=len(performances),
        group_performances=group_performances,
    )

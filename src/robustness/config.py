"""Configuration for robustness mechanisms in federated learning."""

from dataclasses import dataclass


@dataclass
class RobustnessConfig:
    """Configuration for Byzantine-resilient federated learning.

    Attributes:
        enabled: Whether robustness mechanisms are enabled.
        aggregation_method: Robust aggregation method ("coordinate_median" or "trimmed_mean").
        num_byzantine: Expected number of Byzantine/malicious clients.
        trim_fraction: Fraction of values to trim for trimmed mean (0, 0.5).
        client_scoring_method: Method for scoring clients ("zscore" or "none").
        zscore_threshold: Z-score threshold for outlier detection.
    """

    enabled: bool = False
    aggregation_method: str = "coordinate_median"
    num_byzantine: int = 0
    trim_fraction: float = 0.1
    client_scoring_method: str = "zscore"
    zscore_threshold: float = 3.0

    def __post_init__(self):
        """Validate configuration values."""
        if self.enabled:
            valid_methods = ["coordinate_median", "trimmed_mean"]
            if self.aggregation_method not in valid_methods:
                raise ValueError(
                    f"aggregation_method must be one of {valid_methods}, "
                    f"got {self.aggregation_method}"
                )
            if self.num_byzantine < 0:
                raise ValueError(
                    f"num_byzantine must be non-negative, got {self.num_byzantine}"
                )
            if not 0 < self.trim_fraction < 0.5:
                raise ValueError(
                    f"trim_fraction must be in (0, 0.5), got {self.trim_fraction}"
                )
            valid_scoring = ["zscore", "none"]
            if self.client_scoring_method not in valid_scoring:
                raise ValueError(
                    f"client_scoring_method must be one of {valid_scoring}, "
                    f"got {self.client_scoring_method}"
                )
            if self.zscore_threshold <= 0:
                raise ValueError(
                    f"zscore_threshold must be positive, got {self.zscore_threshold}"
                )

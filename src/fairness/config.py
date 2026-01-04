"""Configuration for fairness evaluation in federated learning."""

from dataclasses import dataclass, field


@dataclass
class FairnessConfig:
    """Configuration for fairness evaluation in federated learning.

    Attributes:
        enabled: Whether fairness evaluation is enabled.
        evaluation_dimensions: Which dimensions to evaluate fairness on.
            Options: "client", "category", "defect_type".
        metrics: Which fairness metrics to compute.
        min_samples_per_group: Minimum samples required per group for evaluation.
        max_fpr: Maximum false positive rate for AUC-sPRO calculation.
        export_detailed_results: Whether to export per-group detailed results.
    """

    enabled: bool = False
    evaluation_dimensions: list[str] = field(
        default_factory=lambda: ["client", "category"]
    )
    metrics: list[str] = field(
        default_factory=lambda: [
            "jains_index",
            "variance",
            "performance_gap",
            "worst_case",
            "coefficient_of_variation",
        ]
    )
    min_samples_per_group: int = 1
    max_fpr: float = 0.05
    export_detailed_results: bool = True

    # Valid options for validation
    VALID_DIMENSIONS: tuple[str, ...] = ("client", "category", "defect_type")
    VALID_METRICS: tuple[str, ...] = (
        "jains_index",
        "variance",
        "performance_gap",
        "worst_case",
        "coefficient_of_variation",
        "mean",
        "std",
    )

    def __post_init__(self):
        """Validate configuration values."""
        if self.enabled:
            # Validate dimensions
            for dim in self.evaluation_dimensions:
                if dim not in self.VALID_DIMENSIONS:
                    raise ValueError(
                        f"evaluation_dimensions must be from {self.VALID_DIMENSIONS}, "
                        f"got '{dim}'"
                    )

            # Validate metrics
            for metric in self.metrics:
                if metric not in self.VALID_METRICS:
                    raise ValueError(
                        f"metrics must be from {self.VALID_METRICS}, got '{metric}'"
                    )

            # Validate min_samples_per_group
            if self.min_samples_per_group < 1:
                raise ValueError(
                    f"min_samples_per_group must be >= 1, got {self.min_samples_per_group}"
                )

            # Validate max_fpr
            if not 0 < self.max_fpr <= 1:
                raise ValueError(
                    f"max_fpr must be in (0, 1], got {self.max_fpr}"
                )

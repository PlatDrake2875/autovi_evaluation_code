"""Evaluation module for Phase 4: Model evaluation and comparison."""

from .anomaly_scorer import AnomalyScorer, generate_anomaly_maps
from .metrics_wrapper import MetricsWrapper, evaluate_object, evaluate_all_objects
from .visualization import (
    plot_fpr_spro_curves,
    plot_comparison_bar_chart,
    plot_performance_heatmap,
    create_comparison_report,
)

__all__ = [
    "AnomalyScorer",
    "generate_anomaly_maps",
    "MetricsWrapper",
    "evaluate_object",
    "evaluate_all_objects",
    "plot_fpr_spro_curves",
    "plot_comparison_bar_chart",
    "plot_performance_heatmap",
    "create_comparison_report",
]

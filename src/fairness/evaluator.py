"""Per-client and per-category fairness evaluation for federated learning.

This module provides tools to evaluate fairness across different groups
(clients, categories, defect types) in federated anomaly detection.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

from src.aggregation import ThresholdMetrics
from src.data.partitioner import CATEGORY_CLIENT_ASSIGNMENTS
from src.image import AnomalyMap
from src.util import get_auc_for_max_fpr

from .config import FairnessConfig
from .metrics import FairnessMetrics, compute_all_metrics


# Default client-to-category mapping for the AutoVI dataset
DEFAULT_CLIENT_MAPPING: dict[int, list[str]] = CATEGORY_CLIENT_ASSIGNMENTS


@dataclass
class GroupEvaluationResult:
    """Evaluation result for a single group (client/category/defect_type).

    Attributes:
        group_name: Identifier for the group.
        auc_spro: Area under the sPRO-FPR curve.
        num_images: Number of images in this group.
        num_defects: Number of defect instances in this group.
        details: Additional details (e.g., per-defect-type breakdown).
    """

    group_name: str
    auc_spro: float
    num_images: int
    num_defects: int
    details: dict = field(default_factory=dict)


@dataclass
class FairnessEvaluationResult:
    """Complete fairness evaluation result.

    Attributes:
        dimension: The evaluation dimension ("client", "category", "defect_type").
        group_results: Per-group evaluation results.
        fairness_metrics: Computed fairness metrics across groups.
    """

    dimension: str
    group_results: dict[str, GroupEvaluationResult]
    fairness_metrics: FairnessMetrics


class FairnessEvaluator:
    """Evaluates fairness across different groups in federated learning."""

    def __init__(
        self,
        config: FairnessConfig,
        client_mapping: Optional[dict[int, list[str]]] = None,
    ):
        """Initialize fairness evaluator.

        Args:
            config: Fairness evaluation configuration.
            client_mapping: Mapping of client_id -> list of categories.
                Defaults to CATEGORY_CLIENT_ASSIGNMENTS.
        """
        self.config = config
        self.client_mapping = client_mapping or DEFAULT_CLIENT_MAPPING

        # Build reverse mapping: category -> client_id
        self._category_to_client: dict[str, int] = {}
        for client_id, categories in self.client_mapping.items():
            for cat in categories:
                if cat != "all":  # Skip QC client's "all" marker
                    self._category_to_client[cat] = client_id

    def evaluate_by_category(
        self,
        metrics: ThresholdMetrics,
    ) -> FairnessEvaluationResult:
        """Evaluate fairness across categories.

        Args:
            metrics: ThresholdMetrics object from evaluation.

        Returns:
            FairnessEvaluationResult with per-category performance and fairness metrics.
        """
        logger.info("Evaluating fairness by category...")

        # Group anomaly maps by category
        category_maps = self._group_anomaly_maps_by_category(metrics.anomaly_maps)

        # Evaluate each category
        group_results: dict[str, GroupEvaluationResult] = {}
        group_performances: dict[str, float] = {}

        for category, anomaly_maps in category_maps.items():
            if len(anomaly_maps) < self.config.min_samples_per_group:
                logger.warning(
                    f"Category '{category}' has only {len(anomaly_maps)} images, "
                    f"skipping (min_samples_per_group={self.config.min_samples_per_group})"
                )
                continue

            result = self._evaluate_group(metrics, anomaly_maps, category)
            group_results[category] = result
            group_performances[category] = result.auc_spro

        if not group_performances:
            raise ValueError("No groups met the minimum sample requirement")

        # Compute fairness metrics
        fairness_metrics = compute_all_metrics(group_performances)

        logger.info(
            f"Category fairness: Jain's Index={fairness_metrics.jains_index:.4f}, "
            f"Gap={fairness_metrics.performance_gap:.4f}, "
            f"Worst={fairness_metrics.worst_case:.4f}"
        )

        return FairnessEvaluationResult(
            dimension="category",
            group_results=group_results,
            fairness_metrics=fairness_metrics,
        )

    def evaluate_by_client(
        self,
        metrics: ThresholdMetrics,
    ) -> FairnessEvaluationResult:
        """Evaluate fairness across clients.

        Args:
            metrics: ThresholdMetrics object from evaluation.

        Returns:
            FairnessEvaluationResult with per-client performance and fairness metrics.
        """
        logger.info("Evaluating fairness by client...")

        # Group anomaly maps by client
        client_maps = self._group_anomaly_maps_by_client(metrics.anomaly_maps)

        # Evaluate each client
        group_results: dict[str, GroupEvaluationResult] = {}
        group_performances: dict[str, float] = {}

        for client_id, anomaly_maps in client_maps.items():
            client_name = f"client_{client_id}"

            if len(anomaly_maps) < self.config.min_samples_per_group:
                logger.warning(
                    f"Client {client_id} has only {len(anomaly_maps)} images, "
                    f"skipping (min_samples_per_group={self.config.min_samples_per_group})"
                )
                continue

            result = self._evaluate_group(metrics, anomaly_maps, client_name)
            group_results[client_name] = result
            group_performances[client_name] = result.auc_spro

        if not group_performances:
            raise ValueError("No groups met the minimum sample requirement")

        # Compute fairness metrics
        fairness_metrics = compute_all_metrics(group_performances)

        logger.info(
            f"Client fairness: Jain's Index={fairness_metrics.jains_index:.4f}, "
            f"Gap={fairness_metrics.performance_gap:.4f}, "
            f"Worst={fairness_metrics.worst_case:.4f}"
        )

        return FairnessEvaluationResult(
            dimension="client",
            group_results=group_results,
            fairness_metrics=fairness_metrics,
        )

    def evaluate_by_defect_type(
        self,
        metrics: ThresholdMetrics,
    ) -> FairnessEvaluationResult:
        """Evaluate fairness across defect types.

        Args:
            metrics: ThresholdMetrics object from evaluation.

        Returns:
            FairnessEvaluationResult with per-defect-type performance and fairness metrics.
        """
        logger.info("Evaluating fairness by defect type...")

        # Group anomaly maps by defect type
        defect_maps = self._group_anomaly_maps_by_defect_type(metrics.anomaly_maps)

        # Evaluate each defect type
        group_results: dict[str, GroupEvaluationResult] = {}
        group_performances: dict[str, float] = {}

        for defect_type, anomaly_maps in defect_maps.items():
            if defect_type == "good":  # Skip normal images
                continue

            if len(anomaly_maps) < self.config.min_samples_per_group:
                logger.warning(
                    f"Defect type '{defect_type}' has only {len(anomaly_maps)} images, "
                    f"skipping (min_samples_per_group={self.config.min_samples_per_group})"
                )
                continue

            result = self._evaluate_group(metrics, anomaly_maps, defect_type)
            group_results[defect_type] = result
            group_performances[defect_type] = result.auc_spro

        if not group_performances:
            raise ValueError("No groups met the minimum sample requirement")

        # Compute fairness metrics
        fairness_metrics = compute_all_metrics(group_performances)

        logger.info(
            f"Defect-type fairness: Jain's Index={fairness_metrics.jains_index:.4f}, "
            f"Gap={fairness_metrics.performance_gap:.4f}, "
            f"Worst={fairness_metrics.worst_case:.4f}"
        )

        return FairnessEvaluationResult(
            dimension="defect_type",
            group_results=group_results,
            fairness_metrics=fairness_metrics,
        )

    def evaluate_all_dimensions(
        self,
        metrics: ThresholdMetrics,
    ) -> dict[str, FairnessEvaluationResult]:
        """Evaluate fairness across all configured dimensions.

        Args:
            metrics: ThresholdMetrics object from evaluation.

        Returns:
            Dictionary mapping dimension name to FairnessEvaluationResult.
        """
        results: dict[str, FairnessEvaluationResult] = {}

        for dimension in self.config.evaluation_dimensions:
            if dimension == "client":
                results["client"] = self.evaluate_by_client(metrics)
            elif dimension == "category":
                results["category"] = self.evaluate_by_category(metrics)
            elif dimension == "defect_type":
                results["defect_type"] = self.evaluate_by_defect_type(metrics)

        return results

    def _evaluate_group(
        self,
        metrics: ThresholdMetrics,
        anomaly_maps: list[AnomalyMap],
        group_name: str,
    ) -> GroupEvaluationResult:
        """Evaluate a single group of anomaly maps.

        Args:
            metrics: Full ThresholdMetrics object.
            anomaly_maps: Anomaly maps belonging to this group.
            group_name: Name of the group.

        Returns:
            GroupEvaluationResult for this group.
        """
        # Filter metrics to this group
        group_metrics = metrics.reduce_to_images(anomaly_maps)

        # Compute AUC-sPRO
        try:
            fp_rates = group_metrics.get_fp_rates()
            mean_spros = group_metrics.get_mean_spros()
            auc_spro = get_auc_for_max_fpr(
                fprs=fp_rates,
                y_values=mean_spros,
                max_fpr=self.config.max_fpr,
                scale_to_one=True,
            )
        except (ZeroDivisionError, IndexError) as e:
            logger.warning(f"Could not compute AUC-sPRO for group '{group_name}': {e}")
            auc_spro = 0.0

        # Count defects
        num_defects = sum(
            len(gt.channels) if gt is not None else 0
            for gt in group_metrics.gt_maps
        )

        logger.debug(
            f"Group '{group_name}': AUC-sPRO={auc_spro:.4f}, "
            f"images={len(anomaly_maps)}, defects={num_defects}"
        )

        return GroupEvaluationResult(
            group_name=group_name,
            auc_spro=auc_spro,
            num_images=len(anomaly_maps),
            num_defects=num_defects,
        )

    def _group_anomaly_maps_by_category(
        self,
        anomaly_maps: list[AnomalyMap],
    ) -> dict[str, list[AnomalyMap]]:
        """Group anomaly maps by category (extracted from file path).

        The category is extracted from the file path structure:
        .../output_dir/category/defect_type/image.png

        Args:
            anomaly_maps: List of AnomalyMap objects.

        Returns:
            Dictionary mapping category name to list of AnomalyMaps.
        """
        grouped: dict[str, list[AnomalyMap]] = {}

        for amap in anomaly_maps:
            # Extract category from path: .../category/defect_type/image.png
            path = Path(amap.file_path)
            category = path.parent.parent.name

            if category not in grouped:
                grouped[category] = []
            grouped[category].append(amap)

        return grouped

    def _group_anomaly_maps_by_client(
        self,
        anomaly_maps: list[AnomalyMap],
    ) -> dict[int, list[AnomalyMap]]:
        """Group anomaly maps by client (via category mapping).

        Args:
            anomaly_maps: List of AnomalyMap objects.

        Returns:
            Dictionary mapping client_id to list of AnomalyMaps.
        """
        # First group by category
        category_maps = self._group_anomaly_maps_by_category(anomaly_maps)

        # Then map categories to clients
        grouped: dict[int, list[AnomalyMap]] = {}

        for category, maps in category_maps.items():
            client_id = self._category_to_client.get(category)
            if client_id is None:
                logger.warning(f"Category '{category}' not mapped to any client")
                continue

            if client_id not in grouped:
                grouped[client_id] = []
            grouped[client_id].extend(maps)

        return grouped

    def _group_anomaly_maps_by_defect_type(
        self,
        anomaly_maps: list[AnomalyMap],
    ) -> dict[str, list[AnomalyMap]]:
        """Group anomaly maps by defect type (extracted from file path).

        The defect type is extracted from the file path structure:
        .../output_dir/category/defect_type/image.png

        Args:
            anomaly_maps: List of AnomalyMap objects.

        Returns:
            Dictionary mapping defect type name to list of AnomalyMaps.
        """
        grouped: dict[str, list[AnomalyMap]] = {}

        for amap in anomaly_maps:
            # Extract defect type from path: .../category/defect_type/image.png
            path = Path(amap.file_path)
            defect_type = path.parent.name

            if defect_type not in grouped:
                grouped[defect_type] = []
            grouped[defect_type].append(amap)

        return grouped

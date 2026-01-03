"""Metrics wrapper for evaluating anomaly detection models.

This module provides a high-level interface to the existing evaluation code,
making it easy to compute metrics for centralized and federated models.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger

from src.aggregation import MetricsAggregator, ThresholdMetrics
from src.image import GroundTruthMap, AnomalyMap, DefectsConfig
from src.util import get_auc_for_max_fpr, listdir, compute_classification_auc_roc
from src.data.autovi_dataset import CATEGORIES

# Standard FPR limits for AUC-sPRO computation
MAX_FPRS = [0.01, 0.05, 0.1, 0.3, 1.0]

# PNG extensions
PNG_EXTS = ['.png', '.PNG']


class MetricsWrapper:
    """High-level wrapper for computing evaluation metrics.

    This class wraps the existing evaluation code and provides a simplified
    interface for computing AUC-sPRO and AUC-ROC metrics.

    Attributes:
        dataset_base_dir: Path to the AutoVI dataset root.
        defects_configs: Dictionary of defect configurations per object.
    """

    def __init__(self, dataset_base_dir: str):
        """Initialize the metrics wrapper.

        Args:
            dataset_base_dir: Path to the AutoVI dataset root directory.
        """
        self.dataset_base_dir = Path(dataset_base_dir)
        self.defects_configs: Dict[str, DefectsConfig] = {}

        # Load defect configs for all objects
        for obj_name in CATEGORIES:
            config_path = self.dataset_base_dir / obj_name / "defects_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    defects_list = json.load(f)
                self.defects_configs[obj_name] = DefectsConfig.create_from_list(defects_list)

    def evaluate_object(
        self,
        object_name: str,
        anomaly_maps_dir: str,
        output_dir: Optional[str] = None,
        curve_max_distance: float = 0.001,
        num_parallel_workers: Optional[int] = None,
    ) -> Dict:
        """Evaluate anomaly detection on a single object.

        Args:
            object_name: Name of the object category.
            anomaly_maps_dir: Directory containing anomaly maps for this object.
            output_dir: Optional directory to save results.
            curve_max_distance: Maximum distance for threshold refinement.
            num_parallel_workers: Number of workers for parallel processing.

        Returns:
            Dictionary containing evaluation results.
        """
        logger.info(f"Evaluating object: {object_name}")
        logger.info(f"Anomaly maps dir: {anomaly_maps_dir}")

        # Get resize shape for this object
        resize_shape = self._get_resize_shape(object_name)

        # Load defects config
        if object_name not in self.defects_configs:
            raise ValueError(f"No defects config found for {object_name}")
        defects_config = self.defects_configs[object_name]

        # Construct paths
        gt_dir = self.dataset_base_dir / object_name / "ground_truth"

        # Read ground truth and anomaly maps
        gt_maps, anomaly_maps = self._read_maps(
            gt_dir=str(gt_dir),
            anomaly_maps_dir=anomaly_maps_dir,
            defects_config=defects_config,
            resize_shape=resize_shape,
        )

        # Run metrics aggregation
        metrics_aggregator = MetricsAggregator(
            gt_maps=gt_maps,
            anomaly_maps=anomaly_maps,
            parallel_workers=num_parallel_workers,
        )
        metrics = metrics_aggregator.run(curve_max_distance=curve_max_distance)

        # Compute AUC-sPRO at different FPR limits
        auc_spro = self._get_auc_spros(metrics)

        # Compute per-defect-type AUC-sPRO
        per_defect_auc_spro = self._get_auc_spros_per_subdir(
            metrics=metrics,
            anomaly_maps_dir=anomaly_maps_dir,
        )

        # Compute image-level classification metrics
        classification = self._get_image_level_metrics(gt_maps, anomaly_maps)

        # Build results dictionary
        results = {
            "object_name": object_name,
            "localization": {
                "auc_spro": auc_spro,
                "per_defect_type": per_defect_auc_spro,
            },
            "classification": classification,
            "num_images": len(anomaly_maps),
            "num_defects": sum(1 for gt in gt_maps if gt is not None),
        }

        # Save results if output directory specified
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            metrics_path = output_path / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(results, f, indent=4, sort_keys=True)
            logger.info(f"Results saved to {metrics_path}")

        return results

    def _get_resize_shape(self, object_name: str) -> Tuple[int, int]:
        """Get resize shape for an object category."""
        if object_name in ["engine_wiring", "pipe_clip", "pipe_staple"]:
            return (400, 400)
        elif object_name in ["tank_screw", "underbody_pipes", "underbody_screw"]:
            return (1000, 750)
        else:
            raise ValueError(f"Unknown object: {object_name}")

    def _read_maps(
        self,
        gt_dir: str,
        anomaly_maps_dir: str,
        defects_config: DefectsConfig,
        resize_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[List, List]:
        """Read ground truth and anomaly maps."""
        logger.debug("Reading ground truth and corresponding anomaly maps...")

        gt_maps = []
        anomaly_maps = []

        # Get available ground truth relative paths
        gt_map_rel_paths = set(self._get_gt_map_rel_paths(gt_dir))

        # Get available anomaly map relative paths
        anomaly_rel_paths = list(self._get_anomaly_map_rel_paths(anomaly_maps_dir))
        anomaly_rel_paths_no_ext = [os.path.splitext(p)[0] for p in anomaly_rel_paths]

        # Check no duplicates
        assert len(set(anomaly_rel_paths_no_ext)) == len(anomaly_rel_paths_no_ext)

        # Every ground truth must have corresponding anomaly map
        skipped = gt_map_rel_paths.difference(anomaly_rel_paths_no_ext)
        if skipped:
            logger.warning(f"{len(skipped)} ground truth maps have no anomaly maps")

        # Load maps
        for rel_path, rel_path_no_ext in zip(anomaly_rel_paths, anomaly_rel_paths_no_ext):
            anomaly_map_path = os.path.join(anomaly_maps_dir, rel_path)
            anomaly_map = AnomalyMap.read_from_png(anomaly_map_path, resize_shape)
            anomaly_maps.append(anomaly_map)

            if rel_path_no_ext in gt_map_rel_paths:
                gt_map_path = os.path.join(gt_dir, rel_path_no_ext)
                gt_map = GroundTruthMap.read_from_png_dir(
                    png_dir=gt_map_path,
                    defects_config=defects_config,
                )
                gt_maps.append(gt_map)
            else:
                gt_maps.append(None)

        logger.info(f"Loaded {len(anomaly_maps)} anomaly maps, {sum(1 for g in gt_maps if g)} with GT")

        return gt_maps, anomaly_maps

    def _get_gt_map_rel_paths(self, gt_dir: str):
        """Get relative paths to ground truth maps."""
        for defect_type_name in listdir(gt_dir):
            defect_type_dir = os.path.join(gt_dir, defect_type_name)
            if not os.path.isdir(defect_type_dir):
                continue

            for image_dir_name in listdir(defect_type_dir):
                image_dir_path = os.path.join(defect_type_dir, image_dir_name)
                if os.path.isdir(image_dir_path):
                    yield os.path.join(defect_type_name, image_dir_name)

    def _get_anomaly_map_rel_paths(self, anomaly_maps_dir: str):
        """Get relative paths to anomaly maps."""
        for defect_dir_name in listdir(anomaly_maps_dir):
            defect_dir = os.path.join(anomaly_maps_dir, defect_dir_name)
            if not os.path.isdir(defect_dir):
                continue

            for file_name in listdir(defect_dir):
                _, ext = os.path.splitext(file_name)
                if ext in PNG_EXTS:
                    yield os.path.join(defect_dir_name, file_name)

    def _get_auc_spros(
        self,
        metrics: ThresholdMetrics,
        filter_defect_names: Optional[List[str]] = None,
    ) -> Dict[str, Optional[float]]:
        """Compute AUC-sPRO at different FPR limits."""
        auc_spros = {}
        for max_fpr in MAX_FPRS:
            try:
                fp_rates = metrics.get_fp_rates()
                mean_spros = metrics.get_mean_spros(filter_defect_names=filter_defect_names)
                if mean_spros is None:
                    auc = None
                else:
                    auc = get_auc_for_max_fpr(
                        fprs=fp_rates,
                        y_values=mean_spros,
                        max_fpr=max_fpr,
                        scale_to_one=True,
                    )
            except ZeroDivisionError:
                auc = None
            auc_spros[str(max_fpr)] = auc
        return auc_spros

    def _get_auc_spros_per_subdir(
        self,
        metrics: ThresholdMetrics,
        anomaly_maps_dir: str,
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """Compute AUC-sPRO for each defect type subdirectory."""
        results = {}
        subdir_names = listdir(anomaly_maps_dir)

        # Get good images for inclusion
        good_images = []
        if "good" in subdir_names:
            good_subdir = os.path.realpath(os.path.join(anomaly_maps_dir, "good"))
            good_images = [
                a for a in metrics.anomaly_maps
                if os.path.realpath(a.file_path).startswith(good_subdir)
            ]
            subdir_names.remove("good")

        # Filter out non-directories
        subdir_names = [
            s for s in subdir_names
            if os.path.isdir(os.path.join(anomaly_maps_dir, s))
        ]

        for subdir_name in subdir_names:
            subdir = os.path.realpath(os.path.join(anomaly_maps_dir, subdir_name))
            subdir_anomaly_maps = [
                a for a in metrics.anomaly_maps
                if os.path.realpath(a.file_path).startswith(subdir)
            ]
            subdir_anomaly_maps += good_images

            if not subdir_anomaly_maps:
                continue

            try:
                subdir_metrics = metrics.reduce_to_images(subdir_anomaly_maps)
                results[subdir_name] = self._get_auc_spros(subdir_metrics)
            except Exception as e:
                logger.warning(f"Could not compute metrics for {subdir_name}: {e}")
                results[subdir_name] = {str(fpr): None for fpr in MAX_FPRS}

        return results

    def _get_image_level_metrics(
        self,
        gt_maps: List,
        anomaly_maps: List,
    ) -> Dict:
        """Compute image-level classification metrics."""
        # Collect anomaly scores by image type
        scores_by_type = {}

        for gt_map, anomaly_map in zip(gt_maps, anomaly_maps):
            # Get image type from path
            parent_dir = os.path.dirname(anomaly_map.file_path)
            image_type = os.path.basename(parent_dir)

            # Compute image-level score (max of anomaly map)
            score = float(np.max(anomaly_map.np_array))

            if image_type not in scores_by_type:
                scores_by_type[image_type] = []
            scores_by_type[image_type].append(score)

        # Compute AUC-ROC for each defect type vs good
        auc_roc = {}
        if "good" in scores_by_type:
            good_scores = scores_by_type["good"]
            for defect_type, defect_scores in scores_by_type.items():
                if defect_type == "good":
                    continue
                try:
                    auc = compute_classification_auc_roc(good_scores, defect_scores)
                    auc_roc[defect_type] = auc
                except Exception:
                    auc_roc[defect_type] = None

            # Compute mean AUC-ROC
            valid_aucs = [v for v in auc_roc.values() if v is not None]
            if valid_aucs:
                auc_roc["mean"] = sum(valid_aucs) / len(valid_aucs)
            else:
                auc_roc["mean"] = None

        return {"auc_roc": auc_roc}


def evaluate_object(
    object_name: str,
    dataset_base_dir: str,
    anomaly_maps_dir: str,
    output_dir: Optional[str] = None,
    curve_max_distance: float = 0.001,
    num_parallel_workers: Optional[int] = None,
) -> Dict:
    """Convenience function to evaluate a single object.

    Args:
        object_name: Name of the object category.
        dataset_base_dir: Path to the AutoVI dataset root.
        anomaly_maps_dir: Directory containing anomaly maps.
        output_dir: Optional directory to save results.
        curve_max_distance: Maximum distance for threshold refinement.
        num_parallel_workers: Number of workers for parallel processing.

    Returns:
        Dictionary containing evaluation results.
    """
    wrapper = MetricsWrapper(dataset_base_dir)
    return wrapper.evaluate_object(
        object_name=object_name,
        anomaly_maps_dir=anomaly_maps_dir,
        output_dir=output_dir,
        curve_max_distance=curve_max_distance,
        num_parallel_workers=num_parallel_workers,
    )


def evaluate_all_objects(
    dataset_base_dir: str,
    anomaly_maps_base_dir: str,
    output_base_dir: str,
    objects: Optional[List[str]] = None,
    curve_max_distance: float = 0.001,
    num_parallel_workers: Optional[int] = None,
) -> Dict[str, Dict]:
    """Evaluate all objects for a method.

    Args:
        dataset_base_dir: Path to the AutoVI dataset root.
        anomaly_maps_base_dir: Base directory containing anomaly maps per object.
        output_base_dir: Base directory for saving results.
        objects: List of objects to evaluate. If None, evaluates all.
        curve_max_distance: Maximum distance for threshold refinement.
        num_parallel_workers: Number of workers for parallel processing.

    Returns:
        Dictionary mapping object names to their results.
    """
    objects = objects if objects else CATEGORIES
    all_results = {}

    wrapper = MetricsWrapper(dataset_base_dir)

    for obj_name in objects:
        anomaly_maps_dir = os.path.join(anomaly_maps_base_dir, obj_name)
        output_dir = os.path.join(output_base_dir, obj_name)

        if not os.path.exists(anomaly_maps_dir):
            logger.warning(f"Anomaly maps directory not found: {anomaly_maps_dir}")
            continue

        try:
            results = wrapper.evaluate_object(
                object_name=obj_name,
                anomaly_maps_dir=anomaly_maps_dir,
                output_dir=output_dir,
                curve_max_distance=curve_max_distance,
                num_parallel_workers=num_parallel_workers,
            )
            all_results[obj_name] = results
        except Exception as e:
            logger.error(f"Error evaluating {obj_name}: {e}")
            all_results[obj_name] = {"error": str(e)}

    return all_results


def compute_aggregate_metrics(
    results: Dict[str, Dict],
    fpr_limit: float = 0.05,
) -> Dict:
    """Compute aggregate metrics across all objects.

    Args:
        results: Dictionary mapping object names to their results.
        fpr_limit: FPR limit for AUC-sPRO aggregation.

    Returns:
        Dictionary with aggregate metrics.
    """
    auc_spros = []
    auc_rocs = []

    for obj_name, obj_results in results.items():
        if "error" in obj_results:
            continue

        # Get AUC-sPRO at specified FPR limit
        auc_spro = obj_results.get("localization", {}).get("auc_spro", {}).get(str(fpr_limit))
        if auc_spro is not None:
            auc_spros.append(auc_spro)

        # Get mean AUC-ROC
        auc_roc = obj_results.get("classification", {}).get("auc_roc", {}).get("mean")
        if auc_roc is not None:
            auc_rocs.append(auc_roc)

    aggregate = {
        "auc_spro": {
            "mean": np.mean(auc_spros) if auc_spros else None,
            "std": np.std(auc_spros) if auc_spros else None,
            "min": np.min(auc_spros) if auc_spros else None,
            "max": np.max(auc_spros) if auc_spros else None,
            "n": len(auc_spros),
        },
        "auc_roc": {
            "mean": np.mean(auc_rocs) if auc_rocs else None,
            "std": np.std(auc_rocs) if auc_rocs else None,
            "min": np.min(auc_rocs) if auc_rocs else None,
            "max": np.max(auc_rocs) if auc_rocs else None,
            "n": len(auc_rocs),
        },
        "fpr_limit": fpr_limit,
    }

    return aggregate

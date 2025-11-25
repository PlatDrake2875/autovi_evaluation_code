#!/usr/bin/env python3
"""Evaluation script for centralized and federated models.

This script generates anomaly maps and computes evaluation metrics for
centralized and federated PatchCore models on the AutoVI dataset.

Usage:
    python experiments/scripts/evaluate_all.py \
        --dataset_dir /path/to/autovi \
        --models_dir outputs/models \
        --output_dir outputs/evaluation

Example:
    # Evaluate all models
    python experiments/scripts/evaluate_all.py \
        --dataset_dir /data/autovi \
        --models_dir outputs/training \
        --output_dir outputs/evaluation

    # Evaluate only centralized model
    python experiments/scripts/evaluate_all.py \
        --dataset_dir /data/autovi \
        --models_dir outputs/training \
        --output_dir outputs/evaluation \
        --methods centralized
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.autovi_dataset import AutoVIDataset, CATEGORIES, get_resize_shape
from src.data.preprocessing import get_test_transform
from src.evaluation.anomaly_scorer import AnomalyScorer, generate_anomaly_maps_from_patchcore
from src.evaluation.metrics_wrapper import MetricsWrapper, evaluate_all_objects, compute_aggregate_metrics
from src.evaluation.visualization import create_comparison_report
from src.models.patchcore import PatchCore


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate centralized and federated PatchCore models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the AutoVI dataset root directory",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        required=True,
        help="Path to directory containing trained models",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/evaluation",
        help="Directory for evaluation outputs",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["centralized", "federated_iid", "federated_category"],
        help="Methods to evaluate",
    )
    parser.add_argument(
        "--objects",
        type=str,
        nargs="+",
        default=None,
        help="Objects to evaluate (default: all)",
    )
    parser.add_argument(
        "--skip_anomaly_maps",
        action="store_true",
        help="Skip anomaly map generation (use existing)",
    )
    parser.add_argument(
        "--skip_metrics",
        action="store_true",
        help="Skip metrics computation (use existing)",
    )
    parser.add_argument(
        "--curve_max_distance",
        type=float,
        default=0.001,
        help="Maximum distance for threshold refinement",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers for metrics computation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for computation",
    )
    parser.add_argument(
        "--fpr_limit",
        type=float,
        default=0.05,
        help="FPR limit for comparison report",
    )

    return parser.parse_args()


def find_model_path(models_dir: str, method: str, object_name: str) -> Optional[str]:
    """Find the model path for a given method and object.

    Args:
        models_dir: Base directory containing models.
        method: Method name (centralized, federated_iid, federated_category).
        object_name: Object category name.

    Returns:
        Path to the model or None if not found.
    """
    models_dir = Path(models_dir)

    # Try different path patterns
    patterns = [
        # Pattern 1: models_dir/method/object/patchcore
        models_dir / method / object_name / "patchcore",
        # Pattern 2: models_dir/method/object/model
        models_dir / method / object_name / "model",
        # Pattern 3: models_dir/baseline/object/patchcore (for centralized)
        models_dir / "baseline" / object_name / "patchcore",
        # Pattern 4: models_dir/federated/iid/global_memory_bank.npz
        models_dir / "federated" / "iid" / "global_memory_bank.npz",
        models_dir / "federated" / "category" / "global_memory_bank.npz",
        # Pattern 5: Check if it's a memory bank file
        models_dir / method / "global_memory_bank.npz",
        models_dir / method / f"{object_name}_memory_bank.npz",
    ]

    for pattern in patterns:
        # Check for model files with extensions
        if pattern.suffix == ".npz":
            if pattern.exists():
                return str(pattern)
        else:
            # Check for config file (indicates saved model)
            config_path = Path(str(pattern) + "_config.npz")
            if config_path.exists():
                return str(pattern)

    return None


def generate_anomaly_maps_for_method(
    method: str,
    models_dir: str,
    dataset_dir: str,
    output_dir: str,
    objects: List[str],
    device: str = "auto",
) -> Dict[str, str]:
    """Generate anomaly maps for a method across all objects.

    Args:
        method: Method name.
        models_dir: Directory containing models.
        dataset_dir: Path to dataset.
        output_dir: Output directory for anomaly maps.
        objects: List of objects to process.
        device: Computation device.

    Returns:
        Dictionary mapping object names to their anomaly map directories.
    """
    output_dir = Path(output_dir)
    anomaly_dirs = {}

    print(f"\n{'='*60}")
    print(f"Generating anomaly maps for method: {method}")
    print(f"{'='*60}")

    for obj_name in objects:
        print(f"\n--- Processing {obj_name} ---")

        # Find model path
        model_path = find_model_path(models_dir, method, obj_name)

        if model_path is None:
            print(f"Warning: No model found for {method}/{obj_name}")
            continue

        print(f"Model path: {model_path}")

        # Create output directory
        obj_output_dir = output_dir / method / obj_name
        obj_output_dir.mkdir(parents=True, exist_ok=True)

        # Get transform for this object
        resize_shape = get_resize_shape(obj_name)
        transform = get_test_transform(resize_shape)

        # Load test dataset
        test_dataset = AutoVIDataset(
            root_dir=dataset_dir,
            categories=[obj_name],
            split="test",
            transform=transform,
            include_good_only=False,
        )

        print(f"Test dataset: {len(test_dataset)} images")

        # Create scorer and load model
        scorer = AnomalyScorer(device=device)

        if method == "centralized":
            scorer.load_centralized_model(model_path)
        else:
            scorer.load_federated_memory_bank(model_path)

        # Generate anomaly maps
        scorer.generate_anomaly_maps_for_dataset(
            dataset=test_dataset,
            output_dir=str(obj_output_dir),
        )

        anomaly_dirs[obj_name] = str(obj_output_dir)

    return anomaly_dirs


def evaluate_method(
    method: str,
    dataset_dir: str,
    anomaly_maps_base_dir: str,
    output_dir: str,
    objects: List[str],
    curve_max_distance: float = 0.001,
    num_workers: Optional[int] = None,
) -> Dict[str, Dict]:
    """Evaluate a method across all objects.

    Args:
        method: Method name.
        dataset_dir: Path to dataset.
        anomaly_maps_base_dir: Base directory containing anomaly maps.
        output_dir: Output directory for metrics.
        objects: List of objects to evaluate.
        curve_max_distance: Maximum distance for threshold refinement.
        num_workers: Number of parallel workers.

    Returns:
        Dictionary mapping object names to their results.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating method: {method}")
    print(f"{'='*60}")

    anomaly_maps_dir = Path(anomaly_maps_base_dir) / method
    metrics_output_dir = Path(output_dir) / method

    results = evaluate_all_objects(
        dataset_base_dir=dataset_dir,
        anomaly_maps_base_dir=str(anomaly_maps_dir),
        output_base_dir=str(metrics_output_dir),
        objects=objects,
        curve_max_distance=curve_max_distance,
        num_parallel_workers=num_workers,
    )

    # Compute aggregate metrics
    aggregate = compute_aggregate_metrics(results)

    # Save aggregate results
    aggregate_path = metrics_output_dir / "aggregate_metrics.json"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    with open(aggregate_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nAggregate metrics saved to {aggregate_path}")

    return results


def main():
    """Main evaluation pipeline."""
    args = parse_args()

    objects = args.objects if args.objects else CATEGORIES
    output_dir = Path(args.output_dir)

    print("="*70)
    print("AutoVI Evaluation Pipeline")
    print("="*70)
    print(f"Dataset: {args.dataset_dir}")
    print(f"Models: {args.models_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Methods: {args.methods}")
    print(f"Objects: {objects}")
    print(f"Device: {args.device}")

    # Store all results for comparison
    all_results: Dict[str, Dict[str, Dict]] = {}

    # Phase 1: Generate anomaly maps
    if not args.skip_anomaly_maps:
        print("\n" + "="*70)
        print("PHASE 1: Anomaly Map Generation")
        print("="*70)

        anomaly_maps_dir = output_dir / "anomaly_maps"

        for method in args.methods:
            generate_anomaly_maps_for_method(
                method=method,
                models_dir=args.models_dir,
                dataset_dir=args.dataset_dir,
                output_dir=str(anomaly_maps_dir),
                objects=objects,
                device=args.device,
            )
    else:
        print("\nSkipping anomaly map generation (--skip_anomaly_maps)")

    # Phase 2: Compute metrics
    if not args.skip_metrics:
        print("\n" + "="*70)
        print("PHASE 2: Metrics Computation")
        print("="*70)

        anomaly_maps_dir = output_dir / "anomaly_maps"
        metrics_dir = output_dir / "metrics"

        for method in args.methods:
            results = evaluate_method(
                method=method,
                dataset_dir=args.dataset_dir,
                anomaly_maps_base_dir=str(anomaly_maps_dir),
                output_dir=str(metrics_dir),
                objects=objects,
                curve_max_distance=args.curve_max_distance,
                num_workers=args.num_workers,
            )
            all_results[method] = results
    else:
        print("\nSkipping metrics computation (--skip_metrics)")

        # Load existing metrics
        metrics_dir = output_dir / "metrics"
        for method in args.methods:
            method_dir = metrics_dir / method
            results = {}
            for obj in objects:
                obj_metrics_path = method_dir / obj / "metrics.json"
                if obj_metrics_path.exists():
                    with open(obj_metrics_path) as f:
                        results[obj] = json.load(f)
            all_results[method] = results

    # Phase 3: Generate comparison report
    print("\n" + "="*70)
    print("PHASE 3: Comparison Report Generation")
    print("="*70)

    reports_dir = output_dir / "reports"
    artifacts = create_comparison_report(
        results_by_method=all_results,
        output_dir=str(reports_dir),
        fpr_limit=args.fpr_limit,
    )

    # Print summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\nGenerated artifacts:")
    for name, path in artifacts.items():
        print(f"  - {name}: {path}")

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Robustness evaluation script for federated learning.

This script evaluates the robustness of federated learning under various
attack scenarios, comparing robust aggregation (coordinate median) against
baseline aggregation (mean).

Experiments:
- Malicious fractions: 10%, 20%, 30%, 40%
- Attack types: scaling, noise, sign_flip
- With/without client scoring (anomaly detection)

Metrics:
- AUC degradation (relative to no-attack baseline)
- Detection rate (true positive rate for malicious clients)
- False positive rate (honest clients incorrectly flagged)

Usage:
    python experiments/scripts/robustness_evaluation.py \
        --output_dir results/robustness \
        --num_clients 10 \
        --num_runs 5

Example:
    python experiments/scripts/robustness_evaluation.py \
        --output_dir results/robustness \
        --num_clients 10 \
        --feature_dim 64 \
        --samples_per_client 100
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.robustness import (
    RobustnessConfig,
    CoordinateMedianAggregator,
    ZScoreDetector,
    ModelPoisoningAttack,
)
from src.federated.server import FederatedServer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate robustness of federated learning under attacks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/robustness",
        help="Directory for evaluation outputs",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=10,
        help="Number of clients in federated learning",
    )
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=64,
        help="Dimension of feature embeddings",
    )
    parser.add_argument(
        "--samples_per_client",
        type=int,
        default=100,
        help="Number of samples per client",
    )
    parser.add_argument(
        "--global_bank_size",
        type=int,
        default=500,
        help="Target size for global memory bank",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of runs for averaging results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--malicious_fractions",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4],
        help="Fractions of malicious clients to test",
    )
    parser.add_argument(
        "--attack_types",
        type=str,
        nargs="+",
        default=["scaling", "noise", "sign_flip"],
        help="Attack types to test",
    )
    parser.add_argument(
        "--zscore_threshold",
        type=float,
        default=2.5,
        help="Z-score threshold for outlier detection",
    )

    return parser.parse_args()


def generate_synthetic_client_data(
    num_clients: int,
    samples_per_client: int,
    feature_dim: int,
    honest_mean: float = 0.0,
    honest_std: float = 1.0,
    seed: int = 42,
) -> List[np.ndarray]:
    """Generate synthetic client data for experiments.

    Args:
        num_clients: Number of clients.
        samples_per_client: Samples per client.
        feature_dim: Feature dimension.
        honest_mean: Mean for honest data.
        honest_std: Std for honest data.
        seed: Random seed.

    Returns:
        List of client data arrays.
    """
    rng = np.random.default_rng(seed)
    return [
        rng.normal(honest_mean, honest_std, size=(samples_per_client, feature_dim))
        for _ in range(num_clients)
    ]


def compute_aggregation_quality(
    aggregated: np.ndarray,
    honest_mean: float = 0.0,
    honest_std: float = 1.0,
) -> Dict[str, float]:
    """Compute quality metrics for aggregated result.

    Args:
        aggregated: Aggregated feature array.
        honest_mean: Expected mean for honest data.
        honest_std: Expected std for honest data.

    Returns:
        Dictionary with quality metrics.
    """
    mean_diff = abs(np.mean(aggregated) - honest_mean)
    std_diff = abs(np.std(aggregated) - honest_std)
    max_deviation = np.max(np.abs(aggregated - honest_mean))

    return {
        "mean_deviation": float(mean_diff),
        "std_deviation": float(std_diff),
        "max_deviation": float(max_deviation),
    }


def compute_detection_metrics(
    client_scores: List[Dict],
    malicious_indices: List[int],
) -> Dict[str, float]:
    """Compute detection metrics for client scoring.

    Args:
        client_scores: List of client score dictionaries.
        malicious_indices: Indices of truly malicious clients.

    Returns:
        Dictionary with detection metrics.
    """
    if not client_scores:
        return {"tpr": 0.0, "fpr": 0.0, "precision": 0.0, "f1": 0.0}

    malicious_set = set(malicious_indices)
    honest_set = set(range(len(client_scores))) - malicious_set

    # Count detections
    true_positives = sum(
        1 for s in client_scores
        if s["is_outlier"] and s["client_id"] in malicious_set
    )
    false_positives = sum(
        1 for s in client_scores
        if s["is_outlier"] and s["client_id"] in honest_set
    )
    false_negatives = sum(
        1 for s in client_scores
        if not s["is_outlier"] and s["client_id"] in malicious_set
    )

    # Compute metrics
    tpr = true_positives / len(malicious_set) if malicious_set else 0.0
    fpr = false_positives / len(honest_set) if honest_set else 0.0
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    f1 = (
        2 * precision * tpr / (precision + tpr)
        if (precision + tpr) > 0
        else 0.0
    )

    return {
        "tpr": float(tpr),  # True positive rate (detection rate)
        "fpr": float(fpr),  # False positive rate
        "precision": float(precision),
        "f1": float(f1),
    }


def run_single_experiment(
    num_clients: int,
    feature_dim: int,
    samples_per_client: int,
    global_bank_size: int,
    malicious_fraction: float,
    attack_type: str,
    use_robust: bool,
    use_client_scoring: bool,
    zscore_threshold: float,
    seed: int,
) -> Dict:
    """Run a single robustness experiment.

    Args:
        num_clients: Number of clients.
        feature_dim: Feature dimension.
        samples_per_client: Samples per client.
        global_bank_size: Target global bank size.
        malicious_fraction: Fraction of malicious clients.
        attack_type: Type of attack.
        use_robust: Whether to use robust aggregation.
        use_client_scoring: Whether to use client scoring.
        zscore_threshold: Z-score threshold.
        seed: Random seed.

    Returns:
        Dictionary with experiment results.
    """
    # Generate synthetic data
    client_data = generate_synthetic_client_data(
        num_clients=num_clients,
        samples_per_client=samples_per_client,
        feature_dim=feature_dim,
        honest_mean=0.0,
        honest_std=1.0,
        seed=seed,
    )

    # Determine malicious clients
    num_malicious = int(num_clients * malicious_fraction)
    malicious_indices = list(range(num_malicious))

    # Apply attack
    attack = ModelPoisoningAttack(
        attack_type=attack_type,
        scale_factor=100.0,
        noise_std=10.0,
        seed=seed,
    )
    attacked_data = attack.apply(client_data, malicious_indices)

    # Configure server
    if use_robust:
        config = RobustnessConfig(
            enabled=True,
            aggregation_method="coordinate_median",
            client_scoring_method="zscore" if use_client_scoring else "none",
            zscore_threshold=zscore_threshold,
        )
    else:
        config = None

    server = FederatedServer(
        global_bank_size=global_bank_size,
        robustness_config=config,
        use_faiss=False,
    )

    # Run aggregation
    server.receive_client_coresets(attacked_data)
    aggregated = server.aggregate(seed=seed)

    # Compute quality metrics
    quality = compute_aggregation_quality(aggregated, honest_mean=0.0, honest_std=1.0)

    # Compute detection metrics if client scoring enabled
    stats = server.get_stats()
    agg_stats = stats.get("aggregation_stats", {})
    client_scores = agg_stats.get("client_scores", [])
    detection = compute_detection_metrics(client_scores, malicious_indices)

    return {
        "quality": quality,
        "detection": detection,
        "num_outliers_detected": agg_stats.get("num_outliers", 0),
        "outlier_indices": agg_stats.get("outlier_indices", []),
    }


def run_all_experiments(args) -> List[Dict]:
    """Run all robustness experiments.

    Args:
        args: Command line arguments.

    Returns:
        List of experiment results.
    """
    results = []

    # Also run baseline (no attack) for comparison
    all_fractions = [0.0] + list(args.malicious_fractions)

    total_experiments = (
        len(all_fractions) *
        len(args.attack_types) *
        2 *  # robust vs baseline
        2 *  # with/without client scoring
        args.num_runs
    )

    logger.info(f"Running {total_experiments} experiments...")

    experiment_id = 0
    for malicious_fraction in all_fractions:
        for attack_type in args.attack_types:
            # Skip attack type variations when no malicious clients
            if malicious_fraction == 0.0 and attack_type != args.attack_types[0]:
                continue

            for use_robust in [True, False]:
                for use_client_scoring in [True, False]:
                    # Skip client scoring when not using robust aggregation
                    if use_client_scoring and not use_robust:
                        continue

                    # Run multiple times for averaging
                    run_results = []
                    for run in range(args.num_runs):
                        seed = args.seed + run
                        result = run_single_experiment(
                            num_clients=args.num_clients,
                            feature_dim=args.feature_dim,
                            samples_per_client=args.samples_per_client,
                            global_bank_size=args.global_bank_size,
                            malicious_fraction=malicious_fraction,
                            attack_type=attack_type,
                            use_robust=use_robust,
                            use_client_scoring=use_client_scoring,
                            zscore_threshold=args.zscore_threshold,
                            seed=seed,
                        )
                        run_results.append(result)

                    # Average results
                    avg_quality = {
                        k: np.mean([r["quality"][k] for r in run_results])
                        for k in run_results[0]["quality"]
                    }
                    avg_detection = {
                        k: np.mean([r["detection"][k] for r in run_results])
                        for k in run_results[0]["detection"]
                    }

                    experiment_result = {
                        "experiment_id": experiment_id,
                        "malicious_fraction": malicious_fraction,
                        "attack_type": attack_type if malicious_fraction > 0 else "none",
                        "use_robust": use_robust,
                        "use_client_scoring": use_client_scoring,
                        "num_clients": args.num_clients,
                        "feature_dim": args.feature_dim,
                        "num_runs": args.num_runs,
                        **avg_quality,
                        **avg_detection,
                    }
                    results.append(experiment_result)

                    experiment_id += 1
                    logger.info(
                        f"Experiment {experiment_id}: "
                        f"malicious={malicious_fraction:.0%}, "
                        f"attack={attack_type if malicious_fraction > 0 else 'none'}, "
                        f"robust={use_robust}, "
                        f"scoring={use_client_scoring} -> "
                        f"mean_dev={avg_quality['mean_deviation']:.4f}, "
                        f"tpr={avg_detection['tpr']:.2f}"
                    )

    return results


def save_results(results: List[Dict], output_dir: Path) -> None:
    """Save experiment results to CSV and JSON.

    Args:
        results: List of experiment results.
        output_dir: Output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    csv_path = output_dir / "robustness_results.csv"
    fieldnames = list(results[0].keys())

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Saved results to {csv_path}")

    # Save to JSON (with metadata)
    json_path = output_dir / "robustness_results.json"
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "num_experiments": len(results),
        "results": results,
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    logger.info(f"Saved JSON to {json_path}")


def generate_summary_report(results: List[Dict], output_dir: Path) -> None:
    """Generate a summary report of the experiments.

    Args:
        results: List of experiment results.
        output_dir: Output directory.
    """
    report_lines = [
        "# Robustness Evaluation Summary",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Key Findings",
        "",
    ]

    # Group results by attack scenario
    baseline_results = [r for r in results if r["malicious_fraction"] == 0.0]
    attack_results = [r for r in results if r["malicious_fraction"] > 0.0]

    # Baseline performance
    if baseline_results:
        baseline_robust = next(
            (r for r in baseline_results if r["use_robust"]),
            None
        )
        baseline_std = next(
            (r for r in baseline_results if not r["use_robust"]),
            None
        )

        report_lines.extend([
            "### Baseline (No Attack)",
            "",
            "| Method | Mean Deviation |",
            "|--------|----------------|",
        ])
        if baseline_robust:
            report_lines.append(
                f"| Robust (Median) | {baseline_robust['mean_deviation']:.4f} |"
            )
        if baseline_std:
            report_lines.append(
                f"| Standard (Mean) | {baseline_std['mean_deviation']:.4f} |"
            )
        report_lines.append("")

    # Attack performance comparison
    report_lines.extend([
        "### Attack Resistance",
        "",
        "| Attack | Mal. % | Method | Mean Dev | TPR | FPR |",
        "|--------|--------|--------|----------|-----|-----|",
    ])

    for r in sorted(attack_results, key=lambda x: (x["attack_type"], x["malicious_fraction"])):
        method = "Robust+Scoring" if r["use_robust"] and r["use_client_scoring"] else \
                 "Robust" if r["use_robust"] else "Standard"
        report_lines.append(
            f"| {r['attack_type']} | {r['malicious_fraction']:.0%} | "
            f"{method} | {r['mean_deviation']:.4f} | "
            f"{r['tpr']:.2f} | {r['fpr']:.2f} |"
        )

    report_lines.extend([
        "",
        "## Interpretation",
        "",
        "- **Mean Deviation**: Lower is better (closer to honest data)",
        "- **TPR (True Positive Rate)**: Higher is better (detects more malicious clients)",
        "- **FPR (False Positive Rate)**: Lower is better (fewer false alarms)",
        "",
    ])

    # Write report
    report_path = output_dir / "robustness_summary.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Saved summary report to {report_path}")


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    output_dir = Path(args.output_dir)

    logger.info("Starting robustness evaluation...")
    logger.info(f"  Clients: {args.num_clients}")
    logger.info(f"  Feature dim: {args.feature_dim}")
    logger.info(f"  Samples/client: {args.samples_per_client}")
    logger.info(f"  Malicious fractions: {args.malicious_fractions}")
    logger.info(f"  Attack types: {args.attack_types}")
    logger.info(f"  Runs per experiment: {args.num_runs}")

    # Run experiments
    results = run_all_experiments(args)

    # Save results
    save_results(results, output_dir)

    # Generate summary report
    generate_summary_report(results, output_dir)

    logger.info("Robustness evaluation complete!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

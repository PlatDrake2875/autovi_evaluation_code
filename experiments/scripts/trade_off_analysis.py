#!/usr/bin/env python3
"""Trade-off analysis script for accuracy vs privacy vs robustness.

This script generates comprehensive trade-off analysis for the federated
learning experiments, comparing:
- Baseline centralized model
- Federated model (IID partitioning)
- Federated + DP at different epsilon values (1.0, 5.0, 10.0)
- Federated + Robust aggregation (clean and under attack)

Outputs:
- results/trade_off_table.csv - Main trade-off data
- results/comparison_table.csv - Method comparison with all metrics
- results/trade_off_plot.png - Visualization
- results/statistical_analysis.json - Statistical tests

Usage:
    python experiments/scripts/trade_off_analysis.py \
        --metrics_dir outputs/evaluation/metrics \
        --robustness_dir results/robustness \
        --output_dir results

Example:
    python experiments/scripts/trade_off_analysis.py \
        --metrics_dir outputs/evaluation/metrics \
        --output_dir results
"""

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.autovi_dataset import CATEGORIES
from src.fairness.metrics import compute_all_metrics as compute_fairness_metrics


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""

    name: str
    display_name: str
    privacy_epsilon: Optional[float] = None
    is_robust: bool = False
    under_attack: bool = False
    metrics_subdir: Optional[str] = None


# Define experiment configurations
EXPERIMENT_CONFIGS = [
    ExperimentConfig(
        name="centralized",
        display_name="Centralized",
        privacy_epsilon=None,
        is_robust=False,
        metrics_subdir="centralized",
    ),
    ExperimentConfig(
        name="federated_iid",
        display_name="Federated (IID)",
        privacy_epsilon=None,
        is_robust=False,
        metrics_subdir="federated_iid",
    ),
    ExperimentConfig(
        name="federated_dp_eps1",
        display_name="Federated + DP (ε=1.0)",
        privacy_epsilon=1.0,
        is_robust=False,
        metrics_subdir="federated_dp_eps1",
    ),
    ExperimentConfig(
        name="federated_dp_eps5",
        display_name="Federated + DP (ε=5.0)",
        privacy_epsilon=5.0,
        is_robust=False,
        metrics_subdir="federated_dp_eps5",
    ),
    ExperimentConfig(
        name="federated_dp_eps10",
        display_name="Federated + DP (ε=10.0)",
        privacy_epsilon=10.0,
        is_robust=False,
        metrics_subdir="federated_dp_eps10",
    ),
    ExperimentConfig(
        name="federated_robust_clean",
        display_name="Federated + Robust",
        privacy_epsilon=None,
        is_robust=True,
        under_attack=False,
        metrics_subdir="federated_robust_clean",
    ),
    ExperimentConfig(
        name="federated_robust_attack",
        display_name="Federated + Robust (under attack)",
        privacy_epsilon=None,
        is_robust=True,
        under_attack=True,
        metrics_subdir="federated_robust_attack",
    ),
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Trade-off analysis: Accuracy vs Privacy vs Robustness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--metrics_dir",
        type=str,
        default="outputs/evaluation/metrics",
        help="Directory containing evaluation metrics",
    )
    parser.add_argument(
        "--robustness_dir",
        type=str,
        default="results/robustness",
        help="Directory containing robustness evaluation results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for trade-off analysis",
    )
    parser.add_argument(
        "--fpr_limit",
        type=float,
        default=0.05,
        help="FPR limit for AUC-sPRO comparison",
    )

    return parser.parse_args()


def load_experiment_metrics(
    metrics_dir: Path,
    config: ExperimentConfig,
    fpr_limit: float = 0.05,
) -> Optional[Dict[str, Any]]:
    """Load metrics for an experiment configuration.

    Args:
        metrics_dir: Base metrics directory.
        config: Experiment configuration.
        fpr_limit: FPR limit for AUC-sPRO.

    Returns:
        Dictionary with experiment metrics or None if not found.
    """
    if config.metrics_subdir is None:
        return None

    exp_dir = metrics_dir / config.metrics_subdir
    if not exp_dir.exists():
        return None

    # Load per-object metrics
    object_metrics = {}
    for obj_name in CATEGORIES:
        metrics_path = exp_dir / obj_name / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                obj_data = json.load(f)
                # Extract AUC-sPRO at specified FPR limit
                auc_spro = obj_data.get("localization", {}).get("auc_spro", {})
                fpr_key = str(fpr_limit)
                if fpr_key in auc_spro:
                    object_metrics[obj_name] = auc_spro[fpr_key]

    if not object_metrics:
        return None

    # Compute aggregate statistics
    values = list(object_metrics.values())
    result = {
        "name": config.name,
        "display_name": config.display_name,
        "privacy_epsilon": config.privacy_epsilon,
        "is_robust": config.is_robust,
        "under_attack": config.under_attack,
        "auc_spro_mean": float(np.mean(values)),
        "auc_spro_std": float(np.std(values)),
        "auc_spro_min": float(np.min(values)),
        "auc_spro_max": float(np.max(values)),
        "n_objects": len(values),
        "per_object": object_metrics,
    }

    # Compute fairness metrics
    try:
        fairness = compute_fairness_metrics(object_metrics)
        result["fairness"] = {
            "jains_index": fairness.jains_index,
            "variance": fairness.variance,
            "performance_gap": fairness.performance_gap,
            "worst_case": fairness.worst_case,
            "coefficient_of_variation": fairness.coefficient_of_variation,
        }
    except ValueError:
        result["fairness"] = None

    return result


def load_robustness_results(robustness_dir: Path) -> Optional[Dict[str, Any]]:
    """Load robustness evaluation results.

    Args:
        robustness_dir: Directory containing robustness results.

    Returns:
        Dictionary with robustness summary or None if not found.
    """
    results_path = robustness_dir / "robustness_results.json"
    if not results_path.exists():
        return None

    with open(results_path) as f:
        data = json.load(f)

    return data.get("results", [])


def generate_trade_off_table(
    experiments: List[Dict[str, Any]],
    robustness_results: Optional[List[Dict]] = None,
) -> List[Dict[str, Any]]:
    """Generate trade-off table from experiment results.

    Args:
        experiments: List of experiment result dictionaries.
        robustness_results: Optional robustness evaluation results.

    Returns:
        List of rows for trade-off table.
    """
    rows = []

    for exp in experiments:
        row = {
            "Method": exp["display_name"],
            "AUC-sPRO (mean)": exp["auc_spro_mean"],
            "AUC-sPRO (std)": exp["auc_spro_std"],
            "Privacy (epsilon)": exp["privacy_epsilon"] if exp["privacy_epsilon"] else "None",
            "Robust Aggregation": "Yes" if exp["is_robust"] else "No",
            "Under Attack": "Yes" if exp["under_attack"] else "No",
            "Jain's Index": exp.get("fairness", {}).get("jains_index") if exp.get("fairness") else None,
            "Worst-Case AUC": exp.get("fairness", {}).get("worst_case") if exp.get("fairness") else None,
        }
        rows.append(row)

    # Add robustness-specific metrics if available
    if robustness_results:
        for r in robustness_results:
            if r.get("malicious_fraction", 0) > 0:
                method = "Robust" if r["use_robust"] else "Standard"
                if r.get("use_client_scoring"):
                    method += "+Scoring"

                row = {
                    "Method": f"{method} ({r['attack_type']}, {r['malicious_fraction']:.0%} mal.)",
                    "Mean Deviation": r.get("mean_deviation"),
                    "Detection TPR": r.get("tpr"),
                    "Detection FPR": r.get("fpr"),
                }
                # Only add if not already in table by name
                if not any(existing["Method"] == row["Method"] for existing in rows):
                    rows.append(row)

    return rows


def save_trade_off_csv(rows: List[Dict], output_path: Path) -> None:
    """Save trade-off table to CSV.

    Args:
        rows: List of row dictionaries.
        output_path: Path to output CSV file.
    """
    if not rows:
        print("Warning: No data to save to CSV")
        return

    # Get all unique keys
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    # Order keys logically
    key_order = [
        "Method",
        "AUC-sPRO (mean)",
        "AUC-sPRO (std)",
        "Privacy (epsilon)",
        "Robust Aggregation",
        "Under Attack",
        "Jain's Index",
        "Worst-Case AUC",
        "Mean Deviation",
        "Detection TPR",
        "Detection FPR",
    ]
    fieldnames = [k for k in key_order if k in all_keys]
    fieldnames.extend([k for k in all_keys if k not in fieldnames])

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved trade-off table to {output_path}")


def save_comparison_json(
    experiments: List[Dict[str, Any]],
    robustness_results: Optional[List[Dict]],
    output_path: Path,
) -> None:
    """Save detailed comparison to JSON.

    Args:
        experiments: List of experiment results.
        robustness_results: Optional robustness results.
        output_path: Path to output JSON file.
    """
    data = {
        "timestamp": datetime.now().isoformat(),
        "experiments": experiments,
        "robustness_evaluation": robustness_results,
        "summary": {
            "n_experiments": len(experiments),
            "methods_with_dp": sum(1 for e in experiments if e.get("privacy_epsilon")),
            "methods_with_robustness": sum(1 for e in experiments if e.get("is_robust")),
        },
    }

    # Add comparative analysis
    if len(experiments) >= 2:
        # Find baseline (federated_iid) for comparison
        baseline = next((e for e in experiments if e["name"] == "federated_iid"), None)
        if baseline:
            data["comparisons"] = {}
            for exp in experiments:
                if exp["name"] != "federated_iid":
                    diff = exp["auc_spro_mean"] - baseline["auc_spro_mean"]
                    data["comparisons"][f"{exp['name']}_vs_baseline"] = {
                        "auc_spro_diff": diff,
                        "relative_change": diff / baseline["auc_spro_mean"] * 100 if baseline["auc_spro_mean"] != 0 else 0,
                    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved comparison JSON to {output_path}")


def generate_trade_off_plot(
    experiments: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate trade-off visualization.

    Args:
        experiments: List of experiment results.
        output_path: Path to output plot file.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping plot generation")
        return

    if not experiments:
        print("Warning: No experiments to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Accuracy by method
    ax1 = axes[0]
    names = [e["display_name"] for e in experiments]
    means = [e["auc_spro_mean"] for e in experiments]
    stds = [e["auc_spro_std"] for e in experiments]

    x = np.arange(len(names))
    bars = ax1.bar(x, means, yerr=stds, capsize=5, alpha=0.7, edgecolor="black")

    # Color by type
    for i, exp in enumerate(experiments):
        if exp.get("privacy_epsilon"):
            bars[i].set_color("green")
        elif exp.get("is_robust"):
            bars[i].set_color("orange" if exp.get("under_attack") else "blue")
        else:
            bars[i].set_color("gray")

    ax1.set_xlabel("Method")
    ax1.set_ylabel("AUC-sPRO (FPR=0.05)")
    ax1.set_title("Accuracy Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right")
    ax1.grid(axis="y", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="gray", label="No privacy/robustness"),
        Patch(facecolor="green", label="With DP"),
        Patch(facecolor="blue", label="With Robustness"),
        Patch(facecolor="orange", label="Robustness (under attack)"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right")

    # Plot 2: Privacy-Accuracy Trade-off (DP experiments only)
    ax2 = axes[1]
    dp_experiments = [e for e in experiments if e.get("privacy_epsilon") is not None]

    if dp_experiments:
        epsilons = [e["privacy_epsilon"] for e in dp_experiments]
        accuracies = [e["auc_spro_mean"] for e in dp_experiments]
        accuracy_stds = [e["auc_spro_std"] for e in dp_experiments]

        ax2.errorbar(epsilons, accuracies, yerr=accuracy_stds, marker="o", capsize=5, linewidth=2, markersize=8)
        ax2.set_xlabel("Privacy Budget (ε)")
        ax2.set_ylabel("AUC-sPRO")
        ax2.set_title("Privacy-Accuracy Trade-off")
        ax2.set_xscale("log")
        ax2.grid(True, alpha=0.3)

        # Add baseline reference
        baseline = next((e for e in experiments if e["name"] == "federated_iid"), None)
        if baseline:
            ax2.axhline(y=baseline["auc_spro_mean"], color="red", linestyle="--", label="No DP baseline")
            ax2.legend()
    else:
        ax2.text(0.5, 0.5, "No DP experiments available", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Privacy-Accuracy Trade-off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved trade-off plot to {output_path}")


def main():
    """Main entry point."""
    args = parse_args()

    metrics_dir = Path(args.metrics_dir)
    robustness_dir = Path(args.robustness_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Trade-off Analysis: Accuracy vs Privacy vs Robustness")
    print("=" * 60)
    print(f"Metrics directory: {metrics_dir}")
    print(f"Robustness directory: {robustness_dir}")
    print(f"Output directory: {output_dir}")
    print(f"FPR limit: {args.fpr_limit}")
    print()

    # Load experiment metrics
    print("Loading experiment metrics...")
    experiments = []
    for config in EXPERIMENT_CONFIGS:
        result = load_experiment_metrics(metrics_dir, config, args.fpr_limit)
        if result:
            experiments.append(result)
            print(f"  ✓ {config.display_name}: AUC-sPRO = {result['auc_spro_mean']:.4f} (±{result['auc_spro_std']:.4f})")
        else:
            print(f"  ✗ {config.display_name}: Not found")

    # Load robustness results
    print("\nLoading robustness evaluation results...")
    robustness_results = load_robustness_results(robustness_dir)
    if robustness_results:
        print(f"  ✓ Found {len(robustness_results)} robustness experiments")
    else:
        print("  ✗ Robustness results not found")

    if not experiments:
        print("\nError: No experiment results found!")
        print("Please run experiments first or check the metrics directory.")
        sys.exit(1)

    # Generate outputs
    print("\n" + "-" * 60)
    print("Generating outputs...")

    # Trade-off table (CSV)
    trade_off_rows = generate_trade_off_table(experiments, robustness_results)
    save_trade_off_csv(trade_off_rows, output_dir / "trade_off_table.csv")

    # Comparison JSON
    save_comparison_json(experiments, robustness_results, output_dir / "comparison_analysis.json")

    # Trade-off plot
    generate_trade_off_plot(experiments, output_dir / "trade_off_plot.png")

    # Generate markdown summary
    summary_lines = [
        "# Trade-off Analysis Summary",
        "",
        f"Generated: {datetime.now().isoformat()}",
        f"FPR Limit: {args.fpr_limit}",
        "",
        "## Method Comparison",
        "",
        "| Method | AUC-sPRO (mean) | AUC-sPRO (std) | Privacy (ε) | Robust | Under Attack |",
        "|--------|-----------------|----------------|-------------|--------|--------------|",
    ]

    for exp in experiments:
        epsilon = f"{exp['privacy_epsilon']}" if exp.get("privacy_epsilon") else "-"
        robust = "✓" if exp.get("is_robust") else "-"
        attack = "✓" if exp.get("under_attack") else "-"
        summary_lines.append(
            f"| {exp['display_name']} | {exp['auc_spro_mean']:.4f} | {exp['auc_spro_std']:.4f} | {epsilon} | {robust} | {attack} |"
        )

    summary_lines.extend([
        "",
        "## Fairness Analysis",
        "",
        "| Method | Jain's Index | Worst-Case AUC | Performance Gap | CV |",
        "|--------|--------------|----------------|-----------------|-----|",
    ])

    for exp in experiments:
        if exp.get("fairness"):
            f = exp["fairness"]
            summary_lines.append(
                f"| {exp['display_name']} | {f['jains_index']:.4f} | {f['worst_case']:.4f} | {f['performance_gap']:.4f} | {f['coefficient_of_variation']:.4f} |"
            )

    summary_lines.extend([
        "",
        "## Key Findings",
        "",
    ])

    # Add key findings
    if len(experiments) >= 2:
        baseline = next((e for e in experiments if e["name"] == "federated_iid"), experiments[0])
        best = max(experiments, key=lambda e: e["auc_spro_mean"])
        worst = min(experiments, key=lambda e: e["auc_spro_mean"])

        summary_lines.append(f"- **Best performing method**: {best['display_name']} (AUC-sPRO = {best['auc_spro_mean']:.4f})")
        summary_lines.append(f"- **Worst performing method**: {worst['display_name']} (AUC-sPRO = {worst['auc_spro_mean']:.4f})")

        dp_exps = [e for e in experiments if e.get("privacy_epsilon")]
        if dp_exps:
            dp_impact = baseline["auc_spro_mean"] - min(e["auc_spro_mean"] for e in dp_exps)
            summary_lines.append(f"- **DP impact**: Up to {dp_impact:.4f} AUC-sPRO reduction from baseline")

    summary_lines.append("")

    summary_path = output_dir / "trade_off_summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"Saved summary to {summary_path}")

    print("\n" + "=" * 60)
    print("Trade-off analysis complete!")
    print("=" * 60)
    print(f"\nOutputs saved to: {output_dir}")
    print("  - trade_off_table.csv")
    print("  - comparison_analysis.json")
    print("  - trade_off_plot.png")
    print("  - trade_off_summary.md")


if __name__ == "__main__":
    main()

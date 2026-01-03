#!/usr/bin/env python3
"""Generate comparison report from existing evaluation metrics.

This script loads evaluation metrics from JSON files and generates
comparison tables, plots, and statistical analysis.

Usage:
    python experiments/scripts/generate_comparison_report.py \
        --metrics_dir outputs/evaluation/metrics \
        --output_dir outputs/evaluation/reports

Example:
    python experiments/scripts/generate_comparison_report.py \
        --metrics_dir outputs/metrics \
        --output_dir outputs/reports \
        --fpr_limit 0.05
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.autovi_dataset import CATEGORIES
from src.evaluation.visualization import (
    create_comparison_report,
    create_comparison_table,
    compute_statistical_analysis,
    plot_comparison_bar_chart,
    plot_performance_heatmap,
    plot_box_comparison,
    plot_fpr_spro_curves,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate comparison report from evaluation metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--metrics_dir",
        type=str,
        required=True,
        help="Directory containing metrics (organized by method/object)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/reports",
        help="Output directory for report",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["centralized", "federated_iid", "federated_category"],
        help="Methods to include in comparison",
    )
    parser.add_argument(
        "--objects",
        type=str,
        nargs="+",
        default=None,
        help="Objects to include (default: all available)",
    )
    parser.add_argument(
        "--fpr_limit",
        type=float,
        default=0.05,
        help="FPR limit for AUC-sPRO comparison",
    )
    parser.add_argument(
        "--format",
        type=str,
        nargs="+",
        default=["csv", "json", "markdown", "plots"],
        choices=["csv", "json", "markdown", "plots", "all"],
        help="Output formats to generate",
    )

    return parser.parse_args()


def load_metrics(
    metrics_dir: str,
    methods: List[str],
    objects: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Dict]]:
    """Load metrics from directory structure.

    Args:
        metrics_dir: Base metrics directory.
        methods: List of methods to load.
        objects: List of objects to load (default: all available).

    Returns:
        Nested dictionary {method: {object: results}}.
    """
    metrics_dir = Path(metrics_dir)
    all_results = {}

    objects_to_load = objects if objects else CATEGORIES

    for method in methods:
        method_dir = metrics_dir / method
        if not method_dir.exists():
            print(f"Warning: Method directory not found: {method_dir}")
            continue

        method_results = {}
        for obj_name in objects_to_load:
            metrics_path = method_dir / obj_name / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    method_results[obj_name] = json.load(f)
            else:
                print(f"Warning: Metrics not found: {metrics_path}")

        if method_results:
            all_results[method] = method_results

    return all_results


def generate_csv_report(
    results: Dict[str, Dict[str, Dict]],
    output_dir: Path,
    fpr_limit: float,
) -> None:
    """Generate CSV comparison tables."""
    print("\nGenerating CSV reports...")

    # Main comparison table
    table_path = output_dir / "comparison_table.csv"
    df = create_comparison_table(results, fpr_limit, str(table_path))

    # Per-FPR tables
    for fpr in [0.01, 0.05, 0.1, 0.3, 1.0]:
        fpr_table_path = output_dir / f"comparison_table_fpr{fpr}.csv"
        create_comparison_table(results, fpr, str(fpr_table_path))


def generate_json_report(
    results: Dict[str, Dict[str, Dict]],
    output_dir: Path,
    fpr_limit: float,
) -> None:
    """Generate JSON statistical analysis."""
    print("\nGenerating JSON reports...")

    # Statistical analysis
    stats = compute_statistical_analysis(results, fpr_limit)
    stats_path = output_dir / "statistical_analysis.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: {stats_path}")

    # Combined results
    combined = {
        "results_by_method": {},
        "statistical_analysis": stats,
        "fpr_limit": fpr_limit,
    }
    for method, obj_results in results.items():
        combined["results_by_method"][method] = obj_results

    combined_path = output_dir / "combined_results.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"  Saved: {combined_path}")


def generate_plots(
    results: Dict[str, Dict[str, Dict]],
    output_dir: Path,
    fpr_limit: float,
) -> None:
    """Generate comparison plots."""
    print("\nGenerating plots...")

    # Bar chart
    bar_path = output_dir / "bar_comparison.png"
    plot_comparison_bar_chart(results, fpr_limit, str(bar_path))

    # Heatmap
    heatmap_path = output_dir / "performance_heatmap.png"
    plot_performance_heatmap(results, fpr_limit, str(heatmap_path))

    # Box plot
    box_path = output_dir / "box_comparison.png"
    plot_box_comparison(results, fpr_limit, str(box_path))

    # Per-object FPR-sPRO curves
    curves_dir = output_dir / "fpr_spro_curves"
    curves_dir.mkdir(exist_ok=True)

    objects = set()
    for method_results in results.values():
        objects.update(method_results.keys())

    for obj_name in objects:
        obj_results = {
            method: method_results.get(obj_name, {})
            for method, method_results in results.items()
        }
        curve_path = curves_dir / f"{obj_name}.png"
        plot_fpr_spro_curves(obj_results, obj_name, str(curve_path))


def generate_markdown_report(
    results: Dict[str, Dict[str, Dict]],
    output_dir: Path,
    fpr_limit: float,
) -> None:
    """Generate Markdown summary report."""
    print("\nGenerating Markdown report...")

    stats = compute_statistical_analysis(results, fpr_limit)
    df = create_comparison_table(results, fpr_limit)

    methods = list(results.keys())

    lines = [
        "# Evaluation Comparison Report",
        "",
        f"**FPR Limit**: {fpr_limit}",
        f"**Methods**: {', '.join(methods)}",
        "",
        "## Executive Summary",
        "",
    ]

    # Summary statistics
    if stats.get("descriptive"):
        lines.append("### Method Performance")
        lines.append("")

        for method, desc in stats["descriptive"].items():
            lines.append(f"- **{method}**: Mean AUC-sPRO = {desc['mean']:.4f} (±{desc['std']:.4f})")

        lines.append("")

    # Key findings
    if stats.get("comparisons"):
        lines.append("### Key Findings")
        lines.append("")

        for comp_key, comp_data in stats["comparisons"].items():
            method1, method2 = comp_key.split("_vs_")
            diff = comp_data["mean_diff"]
            p_val = comp_data["paired_t_test"]["p_value"]

            if diff > 0:
                better = method1
                worse = method2
            else:
                better = method2
                worse = method1

            significance = "significantly" if p_val < 0.05 else "not significantly"
            lines.append(
                f"- {better} outperforms {worse} by {abs(diff):.4f} "
                f"({significance} different, p={p_val:.4f})"
            )

        lines.append("")

    # Per-object results
    lines.append("## Per-Object Results")
    lines.append("")
    lines.append("| Object | " + " | ".join(methods) + " |")
    lines.append("|--------|" + "|".join(["------"] * len(methods)) + "|")

    for _, row in df.iterrows():
        if row["object"] == "Mean":
            continue
        values = [f"{row.get(m, 0):.4f}" if row.get(m) is not None else "N/A" for m in methods]
        lines.append(f"| {row['object']} | " + " | ".join(values) + " |")

    # Mean row
    mean_row = df[df["object"] == "Mean"].iloc[0] if len(df[df["object"] == "Mean"]) > 0 else None
    if mean_row is not None:
        mean_values = [f"**{mean_row.get(m, 0):.4f}**" if mean_row.get(m) is not None else "N/A" for m in methods]
        lines.append(f"| **Mean** | " + " | ".join(mean_values) + " |")

    lines.extend([
        "",
        "## Statistical Analysis",
        "",
    ])

    if stats.get("comparisons"):
        for comp_key, comp_data in stats["comparisons"].items():
            lines.append(f"### {comp_key.replace('_vs_', ' vs ')}")
            lines.append("")
            lines.append(f"| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Mean Difference | {comp_data['mean_diff']:.4f} |")
            lines.append(f"| Cohen's d | {comp_data['cohens_d']:.4f} |")
            lines.append(f"| Paired t-test p-value | {comp_data['paired_t_test']['p_value']:.4f} |")

            if comp_data["wilcoxon"]["p_value"] is not None:
                lines.append(f"| Wilcoxon p-value | {comp_data['wilcoxon']['p_value']:.4f} |")

            lines.append("")

    # Write file
    report_path = output_dir / "summary_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {report_path}")


def main():
    """Main entry point."""
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Comparison Report Generator")
    print("="*60)
    print(f"Metrics directory: {args.metrics_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Methods: {args.methods}")
    print(f"FPR limit: {args.fpr_limit}")
    print(f"Formats: {args.format}")

    # Load metrics
    print("\nLoading metrics...")
    results = load_metrics(args.metrics_dir, args.methods, args.objects)

    if not results:
        print("Error: No metrics found!")
        sys.exit(1)

    print(f"Loaded metrics for {len(results)} methods")
    for method, obj_results in results.items():
        print(f"  - {method}: {len(obj_results)} objects")

    # Expand "all" format
    formats = args.format
    if "all" in formats:
        formats = ["csv", "json", "markdown", "plots"]

    # Generate outputs
    if "csv" in formats:
        generate_csv_report(results, output_dir, args.fpr_limit)

    if "json" in formats:
        generate_json_report(results, output_dir, args.fpr_limit)

    if "markdown" in formats:
        generate_markdown_report(results, output_dir, args.fpr_limit)

    if "plots" in formats:
        generate_plots(results, output_dir, args.fpr_limit)

    print("\n" + "="*60)
    print("Report generation complete!")
    print("="*60)
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()

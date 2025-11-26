"""Visualization utilities for evaluation results.

This module provides functions for creating comparison plots and reports
for centralized vs federated model evaluation.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.data.autovi_dataset import CATEGORIES

# Default FPR limits
DEFAULT_FPR_LIMITS = [0.01, 0.05, 0.1, 0.3, 1.0]

# Method display names
METHOD_NAMES = {
    "centralized": "Centralized",
    "federated_iid": "Fed (IID)",
    "federated_category": "Fed (Category)",
}

# Color scheme for methods
METHOD_COLORS = {
    "centralized": "#2196F3",  # Blue
    "federated_iid": "#4CAF50",  # Green
    "federated_category": "#FF9800",  # Orange
}


def plot_fpr_spro_curves(
    results_by_method: Dict[str, Dict],
    object_name: str,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    xlim: Tuple[float, float] = (0, 0.3),
) -> plt.Figure:
    """Plot FPR-sPRO curves comparing multiple methods.

    Args:
        results_by_method: Dictionary mapping method names to their metrics.
        object_name: Name of the object category.
        output_path: Optional path to save the figure.
        figsize: Figure size in inches.
        xlim: X-axis limits.

    Returns:
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for method, results in results_by_method.items():
        if "error" in results:
            continue

        auc_spro = results.get("localization", {}).get("auc_spro", {})

        # Extract FPR and sPRO values
        fprs = []
        spros = []
        for fpr_limit in DEFAULT_FPR_LIMITS:
            fpr_key = str(fpr_limit)
            if fpr_key in auc_spro and auc_spro[fpr_key] is not None:
                fprs.append(fpr_limit)
                spros.append(auc_spro[fpr_key])

        if fprs:
            label = METHOD_NAMES.get(method, method)
            color = METHOD_COLORS.get(method, None)
            ax.plot(fprs, spros, 'o-', label=label, color=color, linewidth=2, markersize=8)

    ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
    ax.set_ylabel("AUC-sPRO", fontsize=12)
    ax.set_title(f"FPR vs AUC-sPRO: {object_name}", fontsize=14)
    ax.set_xlim(xlim)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    return fig


def plot_comparison_bar_chart(
    results_by_method: Dict[str, Dict[str, Dict]],
    fpr_limit: float = 0.05,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """Plot bar chart comparing methods across all objects.

    Args:
        results_by_method: Nested dict {method: {object: results}}.
        fpr_limit: FPR limit for AUC-sPRO comparison.
        output_path: Optional path to save the figure.
        figsize: Figure size in inches.

    Returns:
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    methods = list(results_by_method.keys())
    objects = CATEGORIES
    n_methods = len(methods)
    n_objects = len(objects)

    # Set up bar positions
    x = np.arange(n_objects)
    width = 0.8 / n_methods

    for i, method in enumerate(methods):
        values = []
        for obj in objects:
            obj_results = results_by_method.get(method, {}).get(obj, {})
            if "error" in obj_results:
                values.append(0)
            else:
                auc_spro = obj_results.get("localization", {}).get("auc_spro", {})
                val = auc_spro.get(str(fpr_limit))
                values.append(val if val is not None else 0)

        offset = (i - n_methods / 2 + 0.5) * width
        label = METHOD_NAMES.get(method, method)
        color = METHOD_COLORS.get(method, None)
        bars = ax.bar(x + offset, values, width, label=label, color=color, alpha=0.8)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.annotate(
                    f'{val:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8,
                    rotation=45,
                )

    # Customize plot
    ax.set_xlabel("Object Category", fontsize=12)
    ax.set_ylabel(f"AUC-sPRO @ FPR={fpr_limit}", fontsize=12)
    ax.set_title(f"Method Comparison: AUC-sPRO @ FPR={fpr_limit}", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([obj.replace('_', '\n') for obj in objects], fontsize=10)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    return fig


def plot_performance_heatmap(
    results_by_method: Dict[str, Dict[str, Dict]],
    fpr_limit: float = 0.05,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """Plot heatmap of performance across methods and objects.

    Args:
        results_by_method: Nested dict {method: {object: results}}.
        fpr_limit: FPR limit for AUC-sPRO comparison.
        output_path: Optional path to save the figure.
        figsize: Figure size in inches.

    Returns:
        Matplotlib figure object.
    """
    methods = list(results_by_method.keys())
    objects = CATEGORIES

    # Build data matrix
    data = np.zeros((len(methods), len(objects)))
    for i, method in enumerate(methods):
        for j, obj in enumerate(objects):
            obj_results = results_by_method.get(method, {}).get(obj, {})
            if "error" not in obj_results:
                auc_spro = obj_results.get("localization", {}).get("auc_spro", {})
                val = auc_spro.get(str(fpr_limit))
                data[i, j] = val if val is not None else np.nan
            else:
                data[i, j] = np.nan

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(f"AUC-sPRO @ FPR={fpr_limit}", rotation=-90, va="bottom", fontsize=11)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(objects)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels([obj.replace('_', '\n') for obj in objects], fontsize=10)
    ax.set_yticklabels([METHOD_NAMES.get(m, m) for m in methods], fontsize=10)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(objects)):
            val = data[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 0.5 else "black"
                ax.text(j, i, f'{val:.2f}', ha="center", va="center", color=text_color, fontsize=10)

    ax.set_title(f"Performance Heatmap: AUC-sPRO @ FPR={fpr_limit}", fontsize=14)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    return fig


def plot_box_comparison(
    results_by_method: Dict[str, Dict[str, Dict]],
    fpr_limit: float = 0.05,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot box plot comparing method performance distributions.

    Args:
        results_by_method: Nested dict {method: {object: results}}.
        fpr_limit: FPR limit for AUC-sPRO comparison.
        output_path: Optional path to save the figure.
        figsize: Figure size in inches.

    Returns:
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    methods = list(results_by_method.keys())
    data = []
    labels = []

    for method in methods:
        values = []
        for obj in CATEGORIES:
            obj_results = results_by_method.get(method, {}).get(obj, {})
            if "error" not in obj_results:
                auc_spro = obj_results.get("localization", {}).get("auc_spro", {})
                val = auc_spro.get(str(fpr_limit))
                if val is not None:
                    values.append(val)
        data.append(values)
        labels.append(METHOD_NAMES.get(method, method))

    # Create box plot
    bp = ax.boxplot(data, labels=labels, patch_artist=True)

    # Color the boxes
    for i, (patch, method) in enumerate(zip(bp['boxes'], methods)):
        color = METHOD_COLORS.get(method, '#888888')
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel(f"AUC-sPRO @ FPR={fpr_limit}", fontsize=12)
    ax.set_title("Performance Distribution by Method", fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    return fig


def compute_statistical_analysis(
    results_by_method: Dict[str, Dict[str, Dict]],
    fpr_limit: float = 0.05,
) -> Dict:
    """Compute statistical analysis comparing methods.

    Args:
        results_by_method: Nested dict {method: {object: results}}.
        fpr_limit: FPR limit for AUC-sPRO comparison.

    Returns:
        Dictionary containing statistical analysis results.
    """
    methods = list(results_by_method.keys())
    scores_by_method = {}

    # Collect scores
    for method in methods:
        scores = []
        for obj in CATEGORIES:
            obj_results = results_by_method.get(method, {}).get(obj, {})
            if "error" not in obj_results:
                auc_spro = obj_results.get("localization", {}).get("auc_spro", {})
                val = auc_spro.get(str(fpr_limit))
                if val is not None:
                    scores.append(val)
        scores_by_method[method] = np.array(scores)

    analysis = {
        "fpr_limit": fpr_limit,
        "descriptive": {},
        "comparisons": {},
    }

    # Descriptive statistics
    for method, scores in scores_by_method.items():
        if len(scores) > 0:
            analysis["descriptive"][method] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "median": float(np.median(scores)),
                "n": len(scores),
            }

    # Pairwise comparisons
    method_list = list(scores_by_method.keys())
    for i, method1 in enumerate(method_list):
        for method2 in method_list[i + 1:]:
            scores1 = scores_by_method[method1]
            scores2 = scores_by_method[method2]

            if len(scores1) > 1 and len(scores2) > 1 and len(scores1) == len(scores2):
                # Paired t-test
                t_stat, t_pval = stats.ttest_rel(scores1, scores2)

                # Wilcoxon signed-rank test
                try:
                    w_stat, w_pval = stats.wilcoxon(scores1, scores2)
                except ValueError:
                    w_stat, w_pval = None, None

                # Effect size (Cohen's d for paired samples)
                diff = scores1 - scores2
                cohens_d = float(np.mean(diff) / np.std(diff)) if np.std(diff) > 0 else 0.0

                # Mean difference
                mean_diff = float(np.mean(scores1) - np.mean(scores2))

                comparison_key = f"{method1}_vs_{method2}"
                analysis["comparisons"][comparison_key] = {
                    "mean_diff": mean_diff,
                    "cohens_d": cohens_d,
                    "paired_t_test": {
                        "statistic": float(t_stat),
                        "p_value": float(t_pval),
                    },
                    "wilcoxon": {
                        "statistic": float(w_stat) if w_stat is not None else None,
                        "p_value": float(w_pval) if w_pval is not None else None,
                    },
                }

    return analysis


def create_comparison_table(
    results_by_method: Dict[str, Dict[str, Dict]],
    fpr_limit: float = 0.05,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Create comparison table as DataFrame.

    Args:
        results_by_method: Nested dict {method: {object: results}}.
        fpr_limit: FPR limit for AUC-sPRO comparison.
        output_path: Optional path to save as CSV.

    Returns:
        Pandas DataFrame with comparison data.
    """
    methods = list(results_by_method.keys())
    rows = []

    for obj in CATEGORIES:
        row = {"object": obj}
        for method in methods:
            obj_results = results_by_method.get(method, {}).get(obj, {})
            if "error" not in obj_results:
                auc_spro = obj_results.get("localization", {}).get("auc_spro", {})
                val = auc_spro.get(str(fpr_limit))
                row[method] = val
            else:
                row[method] = None
        rows.append(row)

    # Add mean row
    mean_row = {"object": "Mean"}
    for method in methods:
        values = [r[method] for r in rows if r[method] is not None]
        mean_row[method] = np.mean(values) if values else None
    rows.append(mean_row)

    df = pd.DataFrame(rows)

    # Add gap columns if centralized exists
    if "centralized" in methods:
        for method in methods:
            if method != "centralized":
                gap_col = f"gap_{method}"
                df[gap_col] = df.apply(
                    lambda r: (r[method] - r["centralized"]) / r["centralized"] * 100
                    if r["centralized"] is not None and r[method] is not None
                    else None,
                    axis=1,
                )

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved comparison table to {output_path}")

    return df


def create_comparison_report(
    results_by_method: Dict[str, Dict[str, Dict]],
    output_dir: str,
    fpr_limit: float = 0.05,
) -> Dict:
    """Create comprehensive comparison report with all visualizations.

    Args:
        results_by_method: Nested dict {method: {object: results}}.
        output_dir: Directory to save all outputs.
        fpr_limit: FPR limit for comparison.

    Returns:
        Dictionary containing all generated artifacts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {}

    print("\nGenerating comparison report...")

    # 1. Comparison table
    print("  Creating comparison table...")
    table_path = output_dir / "comparison_table.csv"
    df = create_comparison_table(results_by_method, fpr_limit, str(table_path))
    artifacts["comparison_table"] = str(table_path)

    # 2. Bar chart
    print("  Creating bar chart...")
    bar_path = output_dir / "bar_comparison.png"
    plot_comparison_bar_chart(results_by_method, fpr_limit, str(bar_path))
    artifacts["bar_chart"] = str(bar_path)

    # 3. Heatmap
    print("  Creating heatmap...")
    heatmap_path = output_dir / "performance_heatmap.png"
    plot_performance_heatmap(results_by_method, fpr_limit, str(heatmap_path))
    artifacts["heatmap"] = str(heatmap_path)

    # 4. Box plot
    print("  Creating box plot...")
    box_path = output_dir / "box_comparison.png"
    plot_box_comparison(results_by_method, fpr_limit, str(box_path))
    artifacts["box_plot"] = str(box_path)

    # 5. Per-object FPR-sPRO curves
    print("  Creating per-object curves...")
    curves_dir = output_dir / "fpr_spro_curves"
    curves_dir.mkdir(exist_ok=True)

    for obj in CATEGORIES:
        obj_results = {
            method: results.get(obj, {})
            for method, results in results_by_method.items()
        }
        curve_path = curves_dir / f"{obj}.png"
        plot_fpr_spro_curves(obj_results, obj, str(curve_path))

    artifacts["fpr_spro_curves_dir"] = str(curves_dir)

    # 6. Statistical analysis
    print("  Computing statistical analysis...")
    stats_analysis = compute_statistical_analysis(results_by_method, fpr_limit)
    stats_path = output_dir / "statistical_analysis.json"
    with open(stats_path, "w") as f:
        json.dump(stats_analysis, f, indent=2)
    artifacts["statistical_analysis"] = str(stats_path)

    # 7. Summary report (Markdown)
    print("  Generating summary report...")
    summary_path = output_dir / "summary_report.md"
    _generate_summary_markdown(
        results_by_method,
        stats_analysis,
        df,
        fpr_limit,
        str(summary_path),
    )
    artifacts["summary_report"] = str(summary_path)

    print(f"\nReport generated in {output_dir}")

    return artifacts


def _generate_summary_markdown(
    results_by_method: Dict,
    stats_analysis: Dict,
    comparison_df: pd.DataFrame,
    fpr_limit: float,
    output_path: str,
) -> None:
    """Generate summary report in Markdown format."""
    methods = list(results_by_method.keys())

    lines = [
        "# Evaluation Summary Report",
        "",
        f"**FPR Limit**: {fpr_limit}",
        "",
        "## Performance Summary",
        "",
    ]

    # Descriptive statistics table
    lines.append("### Descriptive Statistics")
    lines.append("")
    lines.append("| Method | Mean | Std | Min | Max | Median |")
    lines.append("|--------|------|-----|-----|-----|--------|")

    for method in methods:
        desc = stats_analysis.get("descriptive", {}).get(method, {})
        if desc:
            lines.append(
                f"| {METHOD_NAMES.get(method, method)} | "
                f"{desc.get('mean', 0):.4f} | "
                f"{desc.get('std', 0):.4f} | "
                f"{desc.get('min', 0):.4f} | "
                f"{desc.get('max', 0):.4f} | "
                f"{desc.get('median', 0):.4f} |"
            )

    lines.extend(["", "### Per-Object Results", ""])

    # Per-object table
    lines.append("| Object | " + " | ".join(METHOD_NAMES.get(m, m) for m in methods) + " |")
    lines.append("|--------|" + "|".join(["------"] * len(methods)) + "|")

    for _, row in comparison_df.iterrows():
        if row["object"] == "Mean":
            continue
        values = [f"{row.get(m, 0):.4f}" if row.get(m) is not None else "N/A" for m in methods]
        lines.append(f"| {row['object']} | " + " | ".join(values) + " |")

    # Mean row
    mean_values = [f"{comparison_df[comparison_df['object'] == 'Mean'][m].values[0]:.4f}"
                   if comparison_df[comparison_df['object'] == 'Mean'][m].values[0] is not None else "N/A"
                   for m in methods]
    lines.append(f"| **Mean** | " + " | ".join(f"**{v}**" for v in mean_values) + " |")

    # Statistical comparisons
    if stats_analysis.get("comparisons"):
        lines.extend(["", "## Statistical Analysis", ""])

        for comp_key, comp_data in stats_analysis["comparisons"].items():
            lines.append(f"### {comp_key.replace('_vs_', ' vs ')}")
            lines.append("")
            lines.append(f"- **Mean Difference**: {comp_data.get('mean_diff', 0):.4f}")
            lines.append(f"- **Cohen's d**: {comp_data.get('cohens_d', 0):.4f}")

            t_test = comp_data.get("paired_t_test", {})
            lines.append(f"- **Paired t-test**: t={t_test.get('statistic', 0):.4f}, p={t_test.get('p_value', 0):.4f}")

            wilcoxon = comp_data.get("wilcoxon", {})
            if wilcoxon.get("statistic") is not None:
                lines.append(f"- **Wilcoxon**: W={wilcoxon.get('statistic', 0):.4f}, p={wilcoxon.get('p_value', 0):.4f}")

            lines.append("")

    # Write file
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved summary report to {output_path}")

# Evaluation Module API

> `src/evaluation/` - Metrics computation and visualization.

---

## Classes

### MetricsWrapper

Wrapper around existing AutoVI evaluation code.

```python
class MetricsWrapper:
    """
    Wrapper for AutoVI evaluation metrics.

    Integrates with existing evaluation code in the repository.

    Args:
        dataset_root (str): Path to AutoVI dataset.
        output_dir (str): Directory for saving results.

    Example:
        >>> wrapper = MetricsWrapper("/data/autovi", "outputs/metrics")
        >>> results = wrapper.evaluate(
        ...     object_name="engine_wiring",
        ...     anomaly_maps_dir="outputs/anomaly_maps/engine_wiring"
        ... )
    """

    def __init__(self, dataset_root, output_dir):
        ...

    def evaluate(self, object_name, anomaly_maps_dir, max_fprs=None):
        """
        Evaluate anomaly maps for a single object category.

        Args:
            object_name (str): Object category to evaluate.
            anomaly_maps_dir (str): Directory containing anomaly map PNGs.
            max_fprs (list[float]): FPR limits for AUC-sPRO.
                Default: [0.01, 0.05, 0.1, 0.3, 1.0]

        Returns:
            dict: Evaluation results including:
                - auc_spro: dict of AUC-sPRO at each FPR limit
                - auc_roc: image-level classification AUC
                - per_defect: per-defect-type metrics
        """
        ...

    def evaluate_all(self, anomaly_maps_root):
        """
        Evaluate all object categories.

        Args:
            anomaly_maps_root (str): Root directory with per-object subdirs.

        Returns:
            dict: Results per object category.
        """
        ...

    def save_results(self, results, filename):
        """Save results to JSON file."""
        ...
```

### AnomalyScorer

Generate anomaly maps from models.

```python
class AnomalyScorer:
    """
    Generate and save anomaly maps from trained models.

    Args:
        model: Trained PatchCore model.
        output_dir (str): Directory for saving anomaly map PNGs.
        resize_shape (tuple, optional): Output resolution.

    Example:
        >>> scorer = AnomalyScorer(model, "outputs/anomaly_maps")
        >>> scorer.generate(test_dataset, save=True)
    """

    def __init__(self, model, output_dir, resize_shape=None):
        ...

    def generate(self, dataset, save=True):
        """
        Generate anomaly maps for dataset.

        Args:
            dataset: Test dataset.
            save (bool): Whether to save as PNG files.

        Returns:
            list[np.ndarray]: Anomaly maps.
        """
        ...

    def generate_single(self, image):
        """
        Generate anomaly map for single image.

        Args:
            image (torch.Tensor): Input image [3, H, W].

        Returns:
            np.ndarray: Anomaly map [H, W] in range [0, 255].
        """
        ...

    @staticmethod
    def normalize_map(anomaly_map):
        """Normalize map to [0, 255] uint8."""
        ...

    @staticmethod
    def save_map(anomaly_map, path):
        """Save anomaly map as PNG."""
        ...
```

### ComparisonAnalyzer

Compare multiple methods.

```python
class ComparisonAnalyzer:
    """
    Compare evaluation results across methods.

    Args:
        methods (list[str]): Method names to compare.
        metrics_root (str): Root directory with per-method results.

    Example:
        >>> analyzer = ComparisonAnalyzer(
        ...     methods=["centralized", "federated_iid", "federated_category"],
        ...     metrics_root="outputs/metrics"
        ... )
        >>> comparison = analyzer.compare()
        >>> analyzer.generate_report("outputs/reports")
    """

    def __init__(self, methods, metrics_root):
        ...

    def compare(self):
        """
        Generate comparison dataframe.

        Returns:
            pd.DataFrame: Comparison table with columns:
                [object, method, auc_spro_0.01, ..., auc_roc, ...]
        """
        ...

    def compute_gaps(self, baseline="centralized"):
        """
        Compute performance gaps vs baseline.

        Args:
            baseline (str): Baseline method name.

        Returns:
            pd.DataFrame: Gap percentages.
        """
        ...

    def statistical_tests(self, method1, method2):
        """
        Run statistical significance tests.

        Args:
            method1, method2 (str): Methods to compare.

        Returns:
            dict: Test results (t-stat, p-value, effect size).
        """
        ...

    def generate_report(self, output_dir):
        """
        Generate comparison report with tables and figures.

        Creates:
            - comparison_table.csv
            - fpr_spro_curves.pdf
            - bar_charts.pdf
            - statistical_analysis.json
        """
        ...
```

---

## Visualization Functions

```python
# src/evaluation/visualization.py

def plot_fpr_spro_curve(metrics, title="FPR-sPRO Curve"):
    """
    Plot FPR-sPRO curve.

    Args:
        metrics: Metrics object with fprs and spros.
        title (str): Plot title.

    Returns:
        matplotlib.figure.Figure
    """
    ...


def plot_comparison_curves(metrics_dict, object_name, output_path=None):
    """
    Plot FPR-sPRO curves for multiple methods.

    Args:
        metrics_dict (dict): {method_name: metrics_object}
        object_name (str): Object category name.
        output_path (str, optional): Save path.

    Returns:
        matplotlib.figure.Figure
    """
    ...


def plot_comparison_bars(comparison_df, metric="auc_spro_0.05", output_path=None):
    """
    Plot grouped bar chart comparing methods.

    Args:
        comparison_df (pd.DataFrame): Comparison dataframe.
        metric (str): Metric column to plot.
        output_path (str, optional): Save path.

    Returns:
        matplotlib.figure.Figure
    """
    ...


def plot_heatmap(comparison_df, metric="auc_spro_0.05", output_path=None):
    """
    Plot heatmap of object x method performance.

    Args:
        comparison_df (pd.DataFrame): Comparison dataframe.
        metric (str): Metric to visualize.
        output_path (str, optional): Save path.

    Returns:
        matplotlib.figure.Figure
    """
    ...


def visualize_anomaly_map(image, anomaly_map, ground_truth=None, output_path=None):
    """
    Visualize anomaly map overlaid on original image.

    Args:
        image (np.ndarray): Original image [H, W, 3].
        anomaly_map (np.ndarray): Anomaly scores [H, W].
        ground_truth (np.ndarray, optional): GT mask [H, W].
        output_path (str, optional): Save path.

    Returns:
        matplotlib.figure.Figure
    """
    ...
```

---

## Usage Examples

### Evaluate Single Model

```python
from src.evaluation import MetricsWrapper, AnomalyScorer

# Generate anomaly maps
scorer = AnomalyScorer(model, "outputs/anomaly_maps/engine_wiring")
scorer.generate(test_dataset)

# Evaluate
wrapper = MetricsWrapper("/data/autovi", "outputs/metrics")
results = wrapper.evaluate(
    object_name="engine_wiring",
    anomaly_maps_dir="outputs/anomaly_maps/engine_wiring"
)

print(f"AUC-sPRO@0.05: {results['auc_spro']['0.05']:.3f}")
print(f"AUC-ROC: {results['auc_roc']:.3f}")
```

### Compare Methods

```python
from src.evaluation import ComparisonAnalyzer

# Run comparison
analyzer = ComparisonAnalyzer(
    methods=["centralized", "federated_iid", "federated_category"],
    metrics_root="outputs/metrics"
)

# Generate comparison table
comparison_df = analyzer.compare()
print(comparison_df.to_markdown())

# Compute performance gaps
gaps = analyzer.compute_gaps(baseline="centralized")
print(f"Mean gap (IID): {gaps['federated_iid'].mean():.1%}")
print(f"Mean gap (Category): {gaps['federated_category'].mean():.1%}")

# Statistical tests
stats = analyzer.statistical_tests("centralized", "federated_category")
print(f"p-value: {stats['p_value']:.4f}")
print(f"Effect size (Cohen's d): {stats['cohens_d']:.2f}")

# Generate full report
analyzer.generate_report("outputs/reports")
```

### Visualizations

```python
from src.evaluation.visualization import (
    plot_comparison_curves,
    plot_comparison_bars,
    visualize_anomaly_map
)

# FPR-sPRO curves
fig = plot_comparison_curves(
    {"Centralized": central_metrics, "Federated": fed_metrics},
    object_name="engine_wiring",
    output_path="outputs/reports/fpr_spro_engine.pdf"
)

# Bar chart comparison
fig = plot_comparison_bars(
    comparison_df,
    metric="auc_spro_0.05",
    output_path="outputs/reports/comparison_bars.pdf"
)

# Anomaly map visualization
fig = visualize_anomaly_map(
    image=test_image,
    anomaly_map=anomaly_map,
    ground_truth=gt_mask,
    output_path="outputs/visualizations/sample_001.png"
)
```

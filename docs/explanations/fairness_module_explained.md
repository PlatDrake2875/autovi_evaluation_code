# The Fairness Module: A Comprehensive Deep Dive

## Understanding the Fundamental Problem: Why Fairness Matters

In the context of federated learning for anomaly detection, achieving high average performance is not sufficient. A model that performs excellently on engine wiring defects but poorly on pipe clip defects is not truly useful for a comprehensive quality control system. The fairness module addresses a critical question that average metrics fail to capture: does the model perform well for everyone, or does it systematically disadvantage certain groups?

Consider a real-world scenario in the AutoVI federated system. Factory A specializes in engine wiring and contributes 500 training images. Factory B handles pipe clips and contributes only 100 images. When the federated model is trained and evaluated, the average AUC-sPRO might be an impressive 0.85. However, this average could hide a troubling disparity: 0.92 for engine wiring but only 0.68 for pipe clips. Factory B's quality control would be significantly compromised, potentially allowing defective products to reach customers.

This disparity is not merely a statistical curiosity. It represents a failure of the federated system to serve all participants equitably. The fairness module provides the tools to detect, measure, and ultimately address such disparities.

---

## The Philosophical Foundations of Fairness

Before examining the code, understanding the different philosophical perspectives on fairness helps explain why multiple metrics exist and what each captures.

### Egalitarian Fairness

The egalitarian perspective holds that fairness means equal outcomes for all groups. Under this view, a fair system would give every factory, every category, and every defect type identical performance. Jain's Fairness Index directly measures this notion of equality, reaching its maximum value of 1.0 only when all groups have exactly equal performance.

### Rawlsian Fairness

The philosopher John Rawls proposed that a just society should maximize the welfare of its worst-off members. Applied to machine learning, Rawlsian fairness focuses on the minimum performance across groups. A system optimized for Rawlsian fairness might accept slightly lower average performance if it significantly improves outcomes for the worst-performing group. The worst-case metric in this module captures this perspective.

### Utilitarian Fairness

The utilitarian perspective focuses on maximizing total welfare, which in machine learning typically translates to optimizing average performance. While average performance is important, purely utilitarian optimization can lead to neglecting minority groups if their small numbers make them statistically insignificant. The module includes mean performance but balances it against distributional metrics.

### The Practical Balance

Real-world systems must balance these competing perspectives. A manufacturing quality control system cannot afford catastrophic failures for any product category (Rawlsian concern), but it also cannot sacrifice overall performance excessively (utilitarian concern), and large disparities between categories create organizational and ethical problems (egalitarian concern). The fairness module provides metrics that capture each perspective, allowing practitioners to make informed tradeoffs.

---

## File 1: config.py — The Configuration Foundation

The configuration file establishes the parameters that control fairness evaluation. Each parameter represents a decision about what aspects of fairness to measure and how to measure them.

### The FairnessConfig Class

```python
@dataclass
class FairnessConfig:
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
```

Let us examine each parameter and the reasoning behind its design.

### The enabled Parameter

This boolean flag controls whether fairness evaluation runs at all. The default of False reflects that fairness evaluation adds computational overhead by requiring per-group metric calculations. In early development or when fairness is not a concern, disabling it streamlines the workflow. However, for production systems serving multiple stakeholders, enabling fairness evaluation is essential for responsible deployment.

### The evaluation_dimensions Parameter

This parameter specifies which groupings to analyze for fairness. The system supports three dimensions, each capturing a different perspective on fairness.

The "client" dimension groups test images by which federated client's data partition they belong to. This measures whether the federated model serves all participating factories equally. A system that performs well for large contributors but poorly for small ones has a client fairness problem.

The "category" dimension groups images by object category (engine_wiring, pipe_clip, tank_screw, and so forth). This measures whether the model handles all product types equally. Category fairness is important because different categories may have different visual characteristics, training data quantities, or defect patterns.

The "defect_type" dimension groups images by the type of defect present. This measures whether the model detects all defect types equally well. Some defects may be subtle while others are obvious, and fairness across defect types ensures no defect category is systematically missed.

The default value includes "client" and "category" because these are most directly relevant to federated learning scenarios. Defect type fairness is also important but may require more specialized analysis.

### The metrics Parameter

This parameter specifies which fairness metrics to compute. The default includes all five primary metrics because each captures a different aspect of fairness, and computing all of them provides a comprehensive picture. The available metrics are jains_index for measuring equality of distribution, variance for measuring spread of performance, performance_gap for measuring the range from best to worst, worst_case for measuring the minimum performance (Rawlsian fairness), and coefficient_of_variation for measuring normalized dispersion that allows comparison across different scales.

### The min_samples_per_group Parameter

This parameter sets the minimum number of test samples required for a group to be included in fairness analysis. The default of 1 is permissive, including all groups with any data. However, groups with very few samples produce unreliable performance estimates due to high statistical variance.

In practice, setting this to a higher value like 10 or 20 ensures that reported fairness metrics are based on statistically meaningful performance estimates. The tradeoff is that small groups may be excluded from the analysis, potentially hiding fairness issues affecting minority groups.

### The max_fpr Parameter

This parameter specifies the maximum false positive rate for computing AUC-sPRO, the primary performance metric. The default of 0.05 (5%) focuses evaluation on the operationally relevant regime where false positive rates are low enough for practical deployment.

The choice of 0.05 reflects a common operational requirement: a quality control system that generates false alarms on 5% of good products is tolerable in many settings, but higher rates become burdensome. By computing AUC-sPRO up to this threshold, we measure performance in the regime that matters most for deployment decisions.

### The Validation Logic

The __post_init__ method ensures configuration validity only when fairness evaluation is enabled. This allows users to set up configurations in advance without triggering errors for disabled features.

The validation checks that evaluation_dimensions only contains valid dimension names, that metrics only contains valid metric names, that min_samples_per_group is at least 1 (because you cannot evaluate with zero samples), and that max_fpr is in the valid range (0, 1] where 0 is excluded because computing AUC up to zero false positives is undefined.

---

## File 2: metrics.py — The Mathematical Core

The metrics file implements the mathematical functions that quantify fairness. Understanding these metrics requires understanding both their mathematical definitions and their intuitive interpretations.

### The FairnessMetrics Data Class

```python
@dataclass
class FairnessMetrics:
    jains_index: float
    variance: float
    performance_gap: float
    worst_case: float
    coefficient_of_variation: float
    mean: float
    std: float
    n_groups: int
    group_performances: dict[str, float]
```

This dataclass serves as a container for all computed metrics. Including both the summary metrics and the raw group_performances allows downstream analysis to examine both aggregate fairness measures and individual group results.

### Jain's Fairness Index

Jain's Fairness Index, developed by Raj Jain and colleagues for evaluating network resource allocation, has become a standard measure of distribution equality. The mathematical formula is:

```
J(x₁, x₂, ..., xₙ) = (Σxᵢ)² / (n × Σxᵢ²)
```

Let us understand why this formula captures fairness through several examples.

Consider first the case of perfect equality where all n groups have the same performance value c. The sum is Σxᵢ = nc, and the sum of squares is Σxᵢ² = nc². The index becomes (nc)² / (n × nc²) = n²c² / n²c² = 1.0. Thus perfect equality always yields the maximum index of 1.0, regardless of what the common value is.

Consider next extreme inequality where one group has all the performance (value c) and all other groups have zero. The sum is Σxᵢ = c, and the sum of squares is Σxᵢ² = c². The index becomes c² / (n × c²) = 1/n. Thus extreme inequality yields the minimum index of 1/n, which decreases as the number of groups increases.

Consider an intermediate case with three groups having performances [0.8, 0.7, 0.6]. The sum is 2.1 and the sum of squares is 0.64 + 0.49 + 0.36 = 1.49. The index is 2.1² / (3 × 1.49) = 4.41 / 4.47 = 0.987. This high value (close to 1.0) indicates relatively fair distribution despite the differences.

Consider another intermediate case with three groups having performances [0.9, 0.5, 0.2]. The sum is 1.6 and the sum of squares is 0.81 + 0.25 + 0.04 = 1.10. The index is 1.6² / (3 × 1.10) = 2.56 / 3.30 = 0.776. This lower value indicates more unfair distribution.

The implementation is straightforward:

```python
def compute_jains_index(performances: Sequence[float]) -> float:
    performances_arr = np.array(performances, dtype=np.float64)
    n = len(performances_arr)
    sum_x = np.sum(performances_arr)
    sum_x_squared = np.sum(performances_arr**2)
    if sum_x_squared == 0:
        raise ValueError("Cannot compute Jain's index when all performances are zero")
    return float((sum_x**2) / (n * sum_x_squared))
```

The check for zero sum of squares prevents division by zero, which would occur only if all performances are exactly zero, a degenerate case indicating complete system failure.

### Performance Variance

Variance measures how spread out the performance values are from their mean. The formula is:

```
Var(X) = (1/n) × Σ(xᵢ - μ)²
```

where μ is the mean of the values. Lower variance indicates performances are clustered together, suggesting fairness. Higher variance indicates performances are spread apart, suggesting unfairness.

The implementation uses NumPy's built-in variance function:

```python
def compute_performance_variance(performances: Sequence[float]) -> float:
    if len(performances) == 0:
        raise ValueError("performances cannot be empty")
    return float(np.var(performances))
```

Variance has a useful interpretation: it represents the average squared deviation from the mean. For performances measured as proportions (like AUC values between 0 and 1), a variance of 0.01 means the typical deviation from mean performance is about 0.1 (the square root of 0.01).

### Performance Gap

The performance gap is simply the difference between the best and worst performing groups:

```
Gap = max(xᵢ) - min(xᵢ)
```

This metric directly captures the range of disparities in the system. A gap of 0 means all groups perform identically. A gap of 0.2 means the best group outperforms the worst group by 0.2 in AUC-sPRO terms.

```python
def compute_performance_gap(performances: Sequence[float]) -> float:
    if len(performances) == 0:
        raise ValueError("performances cannot be empty")
    performances_arr = np.array(performances)
    return float(np.max(performances_arr) - np.min(performances_arr))
```

The performance gap is intuitive and easily communicated to stakeholders. Saying "the worst category is 20 percentage points behind the best" is more immediately understandable than reporting a Jain's index of 0.85.

### Worst-Case Performance

The worst-case metric simply returns the minimum performance across all groups:

```
Worst = min(xᵢ)
```

This metric embodies Rawlsian fairness by focusing on the group that benefits least from the system. A system with high average performance but low worst-case performance has a fairness problem that this metric exposes.

```python
def compute_worst_case(performances: Sequence[float]) -> float:
    if len(performances) == 0:
        raise ValueError("performances cannot be empty")
    return float(np.min(performances))
```

In practice, worst-case performance often determines system viability. A quality control system with 0.95 average AUC-sPRO but 0.50 worst-case AUC-sPRO for one category is effectively failing for that category. Improving the worst case, even at some cost to the average, may be the right engineering tradeoff.

### Coefficient of Variation

The coefficient of variation (CV) normalizes dispersion by dividing the standard deviation by the mean:

```
CV = σ / μ
```

This normalization makes CV scale-invariant. If all performances doubled, the raw variance would quadruple, but CV would remain unchanged. This property makes CV useful for comparing fairness across different performance regimes.

```python
def compute_coefficient_of_variation(performances: Sequence[float]) -> float:
    if len(performances) == 0:
        raise ValueError("performances cannot be empty")
    performances_arr = np.array(performances, dtype=np.float64)
    mean = np.mean(performances_arr)
    std = np.std(performances_arr)
    if mean == 0:
        return 0.0
    return float(std / mean)
```

The handling of zero mean returns 0.0 rather than raising an error. This choice reflects that when all performances are zero, there is no dispersion (all groups are equally terrible), so CV of 0 is arguably correct.

A CV below 0.10 (10%) generally indicates good uniformity. CV above 0.25 (25%) indicates substantial dispersion warranting investigation. These thresholds are rules of thumb rather than hard boundaries.

### The compute_all_metrics Function

This convenience function computes all metrics at once and packages them into a FairnessMetrics object:

```python
def compute_all_metrics(group_performances: dict[str, float]) -> FairnessMetrics:
    if not group_performances:
        raise ValueError("group_performances cannot be empty")

    performances = list(group_performances.values())

    return FairnessMetrics(
        jains_index=compute_jains_index(performances),
        variance=compute_performance_variance(performances),
        performance_gap=compute_performance_gap(performances),
        worst_case=compute_worst_case(performances),
        coefficient_of_variation=compute_coefficient_of_variation(performances),
        mean=float(np.mean(performances)),
        std=float(np.std(performances)),
        n_groups=len(performances),
        group_performances=group_performances,
    )
```

The function preserves the original group_performances dictionary in the result, allowing downstream analysis to examine which specific groups are performing well or poorly.

---

## File 3: evaluator.py — Applying Fairness to Anomaly Detection

The evaluator file bridges the abstract fairness metrics with the concrete reality of anomaly detection evaluation. It handles the practical challenges of grouping test images, computing per-group performance, and assembling comprehensive fairness reports.

### The GroupEvaluationResult Data Class

```python
@dataclass
class GroupEvaluationResult:
    group_name: str
    auc_spro: float
    num_images: int
    num_defects: int
    details: dict = field(default_factory=dict)
```

This class captures everything relevant about a single group's performance. The group_name identifies the group (a category name, client identifier, or defect type). The auc_spro is the primary performance metric for that group. The num_images and num_defects provide context about the group's size and composition. The details dictionary allows storing additional information like per-defect-type breakdowns within a category.

### The FairnessEvaluationResult Data Class

```python
@dataclass
class FairnessEvaluationResult:
    dimension: str
    group_results: dict[str, GroupEvaluationResult]
    fairness_metrics: FairnessMetrics
```

This class packages a complete fairness evaluation for one dimension. The dimension identifies what grouping was used (client, category, or defect_type). The group_results contains detailed results for each group. The fairness_metrics contains the aggregate fairness measures computed across all groups.

### The FairnessEvaluator Class

The FairnessEvaluator orchestrates the entire fairness evaluation process. Understanding its design requires following the flow from configuration through grouping to metric computation.

#### Initialization and Client Mapping

```python
def __init__(
    self,
    config: FairnessConfig,
    client_mapping: Optional[dict[int, list[str]]] = None,
):
    self.config = config
    self.client_mapping = client_mapping or DEFAULT_CLIENT_MAPPING

    self._category_to_client: dict[str, int] = {}
    for client_id, categories in self.client_mapping.items():
        for cat in categories:
            if cat != "all":
                self._category_to_client[cat] = client_id
```

The constructor takes a configuration and an optional client mapping. The client mapping specifies which categories each client handles. In the AutoVI federated setup, Client 0 handles engine_wiring, Client 1 handles underbody components, and so forth.

The constructor builds a reverse mapping from category to client_id. This reverse mapping is essential because the evaluation data is organized by category, but fairness-by-client requires knowing which client each category belongs to. The special value "all" is skipped because it represents the quality control client that samples from all categories rather than owning any specific category.

#### The Category Evaluation Workflow

The evaluate_by_category method demonstrates the complete evaluation workflow for one dimension.

```python
def evaluate_by_category(self, metrics: ThresholdMetrics) -> FairnessEvaluationResult:
    category_maps = self._group_anomaly_maps_by_category(metrics.anomaly_maps)

    group_results: dict[str, GroupEvaluationResult] = {}
    group_performances: dict[str, float] = {}

    for category, anomaly_maps in category_maps.items():
        if len(anomaly_maps) < self.config.min_samples_per_group:
            logger.warning(...)
            continue

        result = self._evaluate_group(metrics, anomaly_maps, category)
        group_results[category] = result
        group_performances[category] = result.auc_spro

    fairness_metrics = compute_all_metrics(group_performances)

    return FairnessEvaluationResult(
        dimension="category",
        group_results=group_results,
        fairness_metrics=fairness_metrics,
    )
```

The workflow proceeds through several stages. First, all anomaly maps are grouped by category using file path parsing. Second, groups below the minimum sample threshold are filtered out with a warning. Third, each remaining group is evaluated using the _evaluate_group helper. Fourth, fairness metrics are computed from the per-group performances. Finally, everything is packaged into a FairnessEvaluationResult.

#### The Grouping Methods

The three grouping methods share a common pattern of parsing file paths to extract the relevant grouping key.

```python
def _group_anomaly_maps_by_category(self, anomaly_maps: list[AnomalyMap]) -> dict[str, list[AnomalyMap]]:
    grouped: dict[str, list[AnomalyMap]] = {}

    for amap in anomaly_maps:
        path = Path(amap.file_path)
        category = path.parent.parent.name

        if category not in grouped:
            grouped[category] = []
        grouped[category].append(amap)

    return grouped
```

The path parsing assumes a specific directory structure where images are organized as .../category/defect_type/image.png. The parent.parent extracts the category from this structure. This coupling to directory structure is a design choice that keeps the code simple but requires consistent file organization.

The client grouping method builds on category grouping:

```python
def _group_anomaly_maps_by_client(self, anomaly_maps: list[AnomalyMap]) -> dict[int, list[AnomalyMap]]:
    category_maps = self._group_anomaly_maps_by_category(anomaly_maps)

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
```

This two-stage approach first groups by category (which is directly available from file paths) and then maps categories to clients using the pre-built reverse mapping. Categories not in the mapping are logged as warnings and excluded, preventing silent failures.

#### The Per-Group Evaluation

The _evaluate_group method computes performance metrics for a single group:

```python
def _evaluate_group(
    self,
    metrics: ThresholdMetrics,
    anomaly_maps: list[AnomalyMap],
    group_name: str,
) -> GroupEvaluationResult:
    group_metrics = metrics.reduce_to_images(anomaly_maps)

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

    num_defects = sum(
        len(gt.channels) if gt is not None else 0
        for gt in group_metrics.gt_maps
    )

    return GroupEvaluationResult(
        group_name=group_name,
        auc_spro=auc_spro,
        num_images=len(anomaly_maps),
        num_defects=num_defects,
    )
```

The method first reduces the full metrics to only the images in this group. This filtering is essential because the overall metrics object contains results for all test images, but we need group-specific calculations.

The AUC-sPRO computation can fail for groups with unusual characteristics (for example, all normal images with no defects), so it is wrapped in a try-except block that gracefully degrades to 0.0 performance. This prevents one problematic group from crashing the entire evaluation.

The defect counting provides context for interpreting the AUC-sPRO value. A group with only 2 defects will have high variance in its AUC-sPRO estimate, while a group with 50 defects provides a more reliable measurement.

---

## Understanding the Performance Metric: AUC-sPRO

The fairness module uses AUC-sPRO (Area Under the saturated Per-Region Overlap curve) as its performance metric. Understanding this choice requires understanding what AUC-sPRO measures and why it is appropriate for anomaly detection fairness.

### What sPRO Measures

Traditional metrics like pixel-level accuracy treat each pixel independently, which can be misleading for anomaly detection. A model that perfectly detects 9 defects but completely misses the 10th would score 90% pixel accuracy if defects are equal-sized, but operationally this could mean 10% of defective products reach customers.

The Per-Region Overlap (PRO) metric addresses this by treating each defect region as a unit. For each defect region, PRO computes what fraction of that region the model correctly identifies. The mean PRO across all defect regions captures the model's ability to find defects regardless of their size.

The "saturated" variant (sPRO) adds a cap at 100% overlap, preventing a single well-detected defect from dominating the metric.

### Why AUC-sPRO for Fairness

Using AUC-sPRO for fairness evaluation has several advantages. First, it focuses on defect detection ability rather than pixel-level accuracy, which aligns with operational goals. Second, computing it per-group reveals whether certain groups (categories, clients, defect types) are systematically harder for the model. Third, the area-under-curve formulation integrates performance across different operating points, providing a single number for comparison.

### The max_fpr Parameter

The max_fpr parameter in the configuration controls the integration limit for AUC computation. Setting max_fpr to 0.05 means we compute the area under the sPRO curve only for false positive rates up to 5%.

This focus on low FPR regions is important because high FPR values are not operationally relevant. A quality control system that raises false alarms on 50% of good products is useless regardless of its defect detection rate. By limiting to low FPR, we measure performance in the regime that matters.

---

## How Fairness Integrates with the Pipeline

Understanding how the fairness module connects to the broader system requires tracing the data flow from training through evaluation to fairness analysis.

### The Evaluation Data Flow

After training completes (whether centralized or federated), the system runs inference on test images to produce anomaly maps. These anomaly maps, combined with ground truth annotations, form the input to metric computation.

```
Training (PatchCore/FederatedPatchCore)
         │
         ▼
Inference on test set
         │
         ▼
ThresholdMetrics computation
         │
         ▼
FairnessEvaluator.evaluate_all_dimensions()
         │
         ▼
Per-dimension FairnessEvaluationResult
```

The ThresholdMetrics object contains anomaly maps and ground truth for all test images. The FairnessEvaluator groups these by the configured dimensions and computes per-group performance.

### Configuration at System Level

Fairness evaluation is typically configured alongside other evaluation parameters:

```python
fairness_config = FairnessConfig(
    enabled=True,
    evaluation_dimensions=["client", "category"],
    metrics=["jains_index", "variance", "performance_gap", "worst_case"],
    min_samples_per_group=10,
    max_fpr=0.05,
)
```

The configuration specifies that fairness evaluation is enabled, that we want to analyze fairness across clients and categories, which metrics to compute, that groups need at least 10 samples, and that AUC-sPRO should be computed up to 5% FPR.

### Interpreting Results

The fairness evaluation produces rich output that supports multiple levels of analysis.

At the aggregate level, Jain's Fairness Index provides a single number summarizing equality. Values above 0.95 generally indicate good fairness. Values below 0.8 warrant investigation.

At the distributional level, performance gap reveals the spread between best and worst groups. Worst-case performance reveals whether any group is being left behind.

At the detailed level, per-group results allow drilling down into exactly which groups are underperforming and by how much.

---

## Practical Scenarios and Interpretations

To make the abstract metrics concrete, let us consider several realistic scenarios and their fairness implications.

### Scenario 1: The Small Client Problem

A federated system has 5 clients. Client 0 contributed 500 images and achieves AUC-sPRO of 0.90. Client 4 contributed only 50 images and achieves AUC-sPRO of 0.65.

The fairness metrics would show a significant performance gap (0.25), a concerning worst-case (0.65), and a Jain's index below 0.9. These metrics correctly identify a fairness problem where the small client is underserved.

The root cause might be that the small client's data is underrepresented in the global memory bank, causing the model to poorly capture patterns specific to that client's products. Solutions might include fairness-aware aggregation that ensures minimum representation from all clients.

### Scenario 2: The Difficult Category Problem

Analysis by category reveals that "pipe_clip" achieves AUC-sPRO of 0.70 while all other categories achieve above 0.85.

The fairness metrics would highlight pipe_clip as the worst case and show elevated variance and gap. Jain's index might still be reasonable (above 0.85) if most categories perform well, but the worst-case metric reveals the problem.

The root cause might be inherent difficulty (pipe clips have subtle defects), insufficient training data (few pipe clip images in training), or category-specific visual characteristics that the model handles poorly. Investigation of these possibilities guides remediation.

### Scenario 3: The Defect Type Problem

Analysis by defect type reveals that "structural" defects (physical damage) achieve AUC-sPRO of 0.90 while "logical" defects (missing components) achieve only 0.72.

This disparity makes operational sense: physical damage changes local texture patterns that PatchCore detects well, while missing components require more global reasoning about what should be present. The fairness metrics quantify this known limitation.

The response might not be algorithmic, as improving logical defect detection might require fundamentally different approaches. Instead, the fairness analysis informs deployment decisions: perhaps structural defect detection is automated while logical defects require human inspection augmentation.

---

## Visual Summary of the Fairness Module

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       FAIRNESS MODULE ARCHITECTURE                           │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────────┐
                              │   FairnessConfig    │
                              │                     │
                              │  enabled            │
                              │  evaluation_dims    │
                              │  metrics            │
                              │  min_samples        │
                              │  max_fpr            │
                              └──────────┬──────────┘
                                         │
                                         │ configures
                                         ▼
                              ┌─────────────────────┐
                              │  FairnessEvaluator  │
                              │                     │
                              │  evaluate_by_client │
                              │  evaluate_by_cat    │
                              │  evaluate_by_defect │
                              └──────────┬──────────┘
                                         │
                          ┌──────────────┼──────────────┐
                          │              │              │
                          ▼              ▼              ▼
                    ┌──────────┐   ┌──────────┐   ┌──────────┐
                    │ Client   │   │ Category │   │ Defect   │
                    │ Grouping │   │ Grouping │   │ Grouping │
                    └────┬─────┘   └────┬─────┘   └────┬─────┘
                         │              │              │
                         └──────────────┼──────────────┘
                                        │
                                        ▼
                              ┌─────────────────────┐
                              │   Per-Group         │
                              │   AUC-sPRO          │
                              │   Computation       │
                              └──────────┬──────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │  Fairness Metrics   │
                              │                     │
                              │  jains_index        │
                              │  variance           │
                              │  performance_gap    │
                              │  worst_case         │
                              │  coeff_of_variation │
                              └─────────────────────┘
```

---

## The Mathematics in Detail

For those seeking deeper understanding, let us explore the mathematical properties of the key metrics.

### Jain's Fairness Index: Derivation and Properties

Jain's index can be understood as measuring the "effective number" of groups relative to the actual number. Define:

```
J = (Σxᵢ)² / (n × Σxᵢ²)
```

This can be rewritten as:

```
J = (μ²) / (μ² + σ²/n × n) = 1 / (1 + (CV)²)
```

where μ is the mean, σ is the standard deviation, and CV = σ/μ is the coefficient of variation.

This reformulation reveals that Jain's index is a monotonic transformation of CV. As CV increases (more dispersion), J decreases. When CV = 0 (perfect equality), J = 1. As CV approaches infinity (extreme inequality), J approaches 0.

The index has the attractive property that it is scale-invariant: multiplying all performances by a constant does not change J. It is also independent of the performance level, focusing purely on the distribution shape.

### Variance and Standard Deviation

The variance formula uses the population variance (dividing by n rather than n-1):

```
Var(X) = (1/n) × Σ(xᵢ - μ)²
```

This choice is appropriate because we are computing dispersion for all groups in our evaluation, not estimating a population variance from a sample. The standard deviation is simply the square root of variance, providing a measure in the same units as the original performances.

### The Relationship Between Metrics

The five fairness metrics are not independent. Given the mean, standard deviation, minimum, and maximum, we can derive:

```
Jain's Index ≈ 1 / (1 + (std/mean)²)
Variance = std²
Performance Gap = max - min
Worst Case = min
CV = std / mean
```

Despite these relationships, reporting multiple metrics is valuable because each presents the information in a different form that may resonate with different stakeholders or use cases.

---

## Recommendations for Practitioners

When deploying fairness evaluation in practice, several considerations guide effective use.

Setting min_samples_per_group appropriately is important for reliable metrics. With fewer than 10 samples per group, performance estimates have high variance, and fairness metrics may be dominated by noise rather than signal. For reliable fairness assessment, aim for at least 20-30 samples per group.

Choosing evaluation dimensions should match the fairness concerns relevant to your deployment. Client fairness matters when different organizations contribute to federated training and expect equitable service. Category fairness matters when the system must handle diverse product types. Defect type fairness matters when missing any defect category has severe consequences.

Interpreting results requires context. A Jain's index of 0.92 is excellent in general, but if the worst-case performance is 0.50, the system has a serious problem for that group regardless of the aggregate measure. Always examine both aggregate metrics and worst-case specifically.

Taking action on fairness findings might involve rebalancing training data, using fairness-aware aggregation strategies, deploying separate specialized models for underperforming groups, or accepting limitations and augmenting with human inspection for problematic categories.

---

## References

The fairness metrics and evaluation approaches in this module draw from established research in machine learning fairness and federated learning.

Jain, R., Chiu, D., & Hawe, W. (1984). "A Quantitative Measure of Fairness and Discrimination for Resource Allocation in Shared Computer Systems." DEC Technical Report.

Li, T., Sanjabi, M., Beirami, A., & Smith, V. (2021). "Fair Resource Allocation in Federated Learning." ICLR.

Rawls, J. (1971). "A Theory of Justice." Harvard University Press.

Nguyen, A. T., et al. (2025). "Fairness in Federated Learning: Fairness for Whom?" arXiv:2505.21584.

Salazar, A., et al. (2024). "Federated Fairness Analytics: Quantifying Fairness in Federated Learning." arXiv:2408.08214.

---

*This document explains the fairness module in the AutoVI Federated PatchCore project.*

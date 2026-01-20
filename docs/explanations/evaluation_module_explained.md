# Evaluation Module Explained

## The Purpose of Evaluation in Anomaly Detection

After training an anomaly detection model, whether centralized or federated, the natural question arises: how well does it actually work? Evaluation provides the answer by comparing model predictions against ground truth annotations to produce quantitative performance metrics. This comparison is more nuanced than typical classification tasks because anomaly detection operates at two distinct levels: the image level (is this image anomalous?) and the pixel level (where exactly is the anomaly located?).

The evaluation module provides the infrastructure for comprehensive model assessment. The anomaly_scorer.py file generates pixel-wise anomaly maps from trained models. The metrics_wrapper.py file computes standardized performance metrics by comparing anomaly maps against ground truth. The visualization.py file creates plots, tables, and reports for understanding and communicating results.

Together, these components transform raw model outputs into actionable insights about detection quality, enabling fair comparison between methods and identification of areas for improvement.

---

## Understanding Anomaly Detection Metrics

### The Two Levels of Evaluation

Anomaly detection evaluation occurs at two complementary levels that address different practical questions.

Image-level evaluation asks whether the model correctly identifies which images contain defects. This is a binary classification problem: given an image, predict normal (0) or anomalous (1). The standard metric here is Area Under the Receiver Operating Characteristic curve (AUC-ROC), which measures how well the model separates normal from anomalous images across all possible decision thresholds.

Pixel-level evaluation asks whether the model correctly localizes defects within images. This is a segmentation problem: for each pixel, predict whether it belongs to a defective region. The challenge is that defects are often small, making simple pixel-wise accuracy misleading. A model that predicts "normal" for every pixel would achieve high accuracy on images where defects occupy only 1% of pixels, yet be completely useless for quality control.

### AUC-ROC for Classification

The Receiver Operating Characteristic (ROC) curve plots the True Positive Rate (TPR, also called recall or sensitivity) against the False Positive Rate (FPR, also called fall-out) as the classification threshold varies. Each point on the curve represents a different threshold choice: high thresholds produce few false positives but may miss some anomalies, while low thresholds catch more anomalies but also flag more normal images incorrectly.

The Area Under this Curve (AUC) summarizes classification performance in a single number between 0 and 1. An AUC of 0.5 indicates random guessing (the diagonal line), while 1.0 indicates perfect separation. Importantly, AUC is threshold-independent: it measures the probability that a randomly chosen anomalous image receives a higher score than a randomly chosen normal image.

For computing image-level scores from pixel-wise anomaly maps, the standard approach takes the maximum value in each image's anomaly map. The intuition is that the presence of any highly anomalous region should flag the entire image as potentially defective.

### AUC-sPRO for Localization

The Saturated Per-Region Overlap (sPRO) metric is designed specifically for industrial defect localization. Unlike pixel-wise metrics that treat all pixels equally, sPRO evaluates detection quality at the region level, addressing the challenge that defects come in vastly different sizes.

The core insight of sPRO is that successfully detecting a defect means achieving sufficient overlap with the ground truth region, not necessarily predicting every pixel correctly. A defect that occupies 100 pixels is "detected" if the predicted anomaly region overlaps sufficiently, even if some pixels are missed. The "saturation" aspect means that exceeding a threshold overlap does not provide additional benefit; once a region is detected, it is detected.

The sPRO curve plots the mean per-region overlap against the false positive rate, similar to a ROC curve. At each threshold, the metric computes what fraction of each ground truth defect region is covered by predictions exceeding that threshold, then averages across all defect regions. Simultaneously, it measures the false positive rate as the fraction of normal pixels incorrectly flagged.

AUC-sPRO integrates this curve, but typically only up to a specified FPR limit. This reflects the practical reality that industrial inspection systems must operate with very low false positive rates. A system that flags 30% of normal products for human review would be economically infeasible regardless of how well it catches actual defects.

### FPR Limits and Their Significance

The evaluation module computes AUC-sPRO at multiple FPR limits: 0.01 (1%), 0.05 (5%), 0.1 (10%), 0.3 (30%), and 1.0 (100%). Each limit addresses different operational scenarios.

The 0.01 limit represents an extremely stringent false positive requirement. In high-volume manufacturing, even 1% false positives can create overwhelming inspection queues. A good AUC-sPRO at this limit indicates the model can precisely localize defects without flagging normal regions.

The 0.05 limit (5% FPR) is commonly used as the primary comparison metric. It balances sensitivity against practicality, allowing some false positives while still requiring good precision.

The 0.3 limit is more permissive, useful for understanding model behavior in scenarios where false positives are acceptable, perhaps for initial screening before human review.

The 1.0 limit computes AUC over the entire curve, providing a comprehensive measure but one less relevant to practical deployment where false positive control matters.

---

## The AnomalyScorer Class

### Purpose and Architecture

The AnomalyScorer class generates pixel-wise anomaly maps from trained PatchCore models. Given a test image, it produces a heatmap where pixel intensity indicates the estimated degree of anomaly at that location. These maps serve as both the primary model output for evaluation and a visualization aid for understanding model behavior.

The class is designed to work with both centralized PatchCore models and federated global memory banks. This flexibility allows consistent evaluation across training paradigms.

### Initialization

During initialization, the scorer creates a FeatureExtractor matching the configuration used during training. The backbone_name parameter specifies the CNN architecture (defaulting to WideResNet-50-2), and the layers parameter indicates which intermediate layers to use for feature extraction (defaulting to layer2 and layer3 for multi-scale representation).

The neighborhood_size parameter controls the spatial smoothing applied to extracted features, matching the setting used during training to ensure consistent behavior. The use_faiss parameter enables FAISS-accelerated nearest neighbor search for computing anomaly scores.

The scorer maintains a reference to a memory bank that will be populated when a model is loaded. This memory bank contains the representative normal patch features against which test patches are compared.

### Loading Trained Models

The class provides three methods for loading trained models, accommodating different model sources.

The load_centralized_model method loads a PatchCore model trained in the standard centralized manner. It expects a path to the saved model (without file extension) and loads the associated memory bank from the corresponding .npz file.

The load_federated_memory_bank method loads a global memory bank produced by federated training. It handles both the structured .npz format and raw numpy arrays, providing flexibility for different saving conventions.

The load_memory_bank_from_array method accepts a numpy array directly, useful for programmatic evaluation without file I/O. This is particularly convenient for evaluation during training or for testing with dynamically constructed memory banks.

### Anomaly Map Generation

The generate_anomaly_map method transforms a single image into a pixel-wise anomaly map through several stages.

First, the image tensor is passed through the feature extractor backbone. The pretrained CNN processes the image through its convolutional layers, with hooks capturing the intermediate representations at layer2 and layer3. These features are concatenated to form a multi-scale representation.

If neighborhood_size is greater than 1, local neighborhood averaging is applied. This smoothing makes features more robust by incorporating context from surrounding spatial positions, consistent with the processing applied during training.

The spatial feature map is then reshaped into individual patch feature vectors. For a 224×224 input image processed through the backbone, this produces 784 patch vectors (28×28 spatial positions) each of dimension 1536.

Each patch is queried against the memory bank to find its nearest neighbor among the stored normal patches. The distance to this nearest neighbor serves as the patch's anomaly score. Patches similar to normal training patterns have small distances; patches representing novel (potentially anomalous) patterns have large distances.

The flat array of anomaly scores is reshaped back into a spatial map matching the feature map dimensions (28×28 for the default configuration). This map is then upsampled to the original image resolution using bilinear interpolation, producing a dense anomaly heatmap where each pixel has an associated score.

Finally, the anomaly map is normalized to the range [0, 255] for storage as an 8-bit PNG image. The normalization applies min-max scaling: the minimum score becomes 0 (least anomalous) and the maximum becomes 255 (most anomalous).

### Dataset-Level Processing

The generate_anomaly_maps_for_dataset method processes an entire AutoVIDataset, generating and saving anomaly maps for all test images. It iterates through the dataset, applying generate_anomaly_map to each image and saving the result to an organized directory structure.

The output is organized by defect type, mirroring the ground truth directory structure. This organization facilitates subsequent evaluation by making it easy to match anomaly maps with their corresponding ground truth annotations.

The method returns a dictionary mapping categories to lists of saved file paths, providing a record of what was generated for downstream processing.

---

## The MetricsWrapper Class

### Purpose and Design

The MetricsWrapper class provides a high-level interface for computing evaluation metrics. It wraps lower-level metric computation code and provides convenient methods for evaluating single objects or entire datasets.

The wrapper handles the complexity of loading ground truth annotations, matching them with generated anomaly maps, and orchestrating the metric computation pipeline. This abstraction allows users to focus on what they want to evaluate rather than the mechanics of how.

### Initialization and Defect Configuration

During initialization, the wrapper loads defect configuration files for all object categories. These JSON files describe the types of defects present in each category and how they are annotated. The DefectsConfig objects provide metadata needed for proper interpretation of ground truth masks.

The dataset_base_dir parameter points to the AutoVI dataset root, allowing the wrapper to locate ground truth annotations and configuration files for any category.

### Object Evaluation

The evaluate_object method performs comprehensive evaluation for a single object category. It accepts the object name, path to generated anomaly maps, and optional output directory for saving results.

The evaluation proceeds through several stages. First, the method determines the appropriate image size for this category (400×400 for small objects, 1000×750 for large objects). It then loads the defect configuration and constructs paths to ground truth directories.

The _read_maps method loads corresponding ground truth and anomaly maps. For each anomaly map found in the specified directory, it attempts to locate matching ground truth. Not all anomaly maps have ground truth (good samples do not have defect annotations), so the method handles missing ground truth gracefully by storing None for those entries.

After loading, a MetricsAggregator processes the paired maps to compute threshold-based metrics. The aggregator sweeps through possible thresholds, computing sPRO and FPR at each, producing the data needed for AUC computation.

The _get_auc_spros method computes AUC-sPRO at each of the standard FPR limits. For each limit, it integrates the sPRO curve from FPR=0 to the specified limit, scaling the result to [0, 1] for interpretability.

The _get_auc_spros_per_subdir method computes metrics separately for each defect type, enabling analysis of model performance on different defect categories. Some defects may be easier to detect than others, and this breakdown reveals such variations.

The _get_image_level_metrics method computes classification AUC-ROC. It extracts image-level scores (maximum of anomaly map) for each image, groups them by type (good vs. each defect type), and computes the area under the ROC curve for each comparison.

### Results Structure

The method returns a structured dictionary containing all computed metrics. The localization section contains AUC-sPRO values at each FPR limit and per-defect-type breakdowns. The classification section contains AUC-ROC values for each defect type and the mean across types.

If an output directory is specified, results are also saved as JSON for persistent storage and later analysis.

---

## Visualization Functions

### Purpose of Visualization

Numerical metrics tell only part of the story. Visualization transforms abstract numbers into intuitive displays that reveal patterns, enable comparisons, and support communication of findings. The visualization module provides a suite of plotting functions designed for anomaly detection evaluation.

### FPR-sPRO Curves

The plot_fpr_spro_curves function visualizes the fundamental tradeoff between detection sensitivity and false positive rate. For each method being compared, it plots AUC-sPRO values at different FPR limits, showing how performance degrades as false positive requirements tighten.

The resulting plot reveals method characteristics. A curve that stays high even at low FPR limits indicates a model that achieves good detection with few false positives. A curve that drops sharply as FPR decreases suggests a model that requires tolerating more false positives to achieve good detection.

The function accepts results from multiple methods, overlaying their curves for direct comparison. Color coding and legends distinguish methods, while the shared axes enable visual comparison of relative performance.

### Bar Chart Comparisons

The plot_comparison_bar_chart function creates grouped bar charts comparing methods across all object categories. Each category gets a cluster of bars (one per method), with bar height indicating AUC-sPRO at the specified FPR limit.

This visualization immediately reveals which methods excel on which categories and whether any method dominates across the board. Performance variations across categories may indicate dataset characteristics (some objects are harder to inspect) or method biases (some approaches work better for certain defect types).

Value labels on each bar provide precise numbers while the visual layout supports quick pattern recognition.

### Performance Heatmaps

The plot_performance_heatmap function provides a matrix view of performance. Rows represent methods, columns represent object categories, and cell colors indicate AUC-sPRO values. A color scale from red (poor) through yellow (moderate) to green (good) makes performance patterns immediately visible.

This visualization excels at revealing systematic patterns. If one row is consistently greener than others, that method outperforms overall. If one column is consistently redder, that category is challenging for all methods. Off-diagonal patterns might reveal method-category interactions where specific methods excel on specific categories.

### Box Plot Comparisons

The plot_box_comparison function shows the distribution of performance across categories for each method. Rather than showing individual category values, box plots summarize the spread: median, quartiles, and outliers.

This visualization emphasizes consistency. A method with high median but large spread performs well on average but unreliably. A method with slightly lower median but tight box performs more consistently. The choice between these might depend on operational priorities.

### Statistical Analysis

The compute_statistical_analysis function performs formal statistical tests comparing methods. Beyond descriptive statistics (mean, standard deviation, median, min, max), it conducts pairwise comparisons using appropriate statistical tests.

The paired t-test assumes normally distributed differences and tests whether the mean difference between two methods is significantly different from zero. For n=6 categories, this test has limited power but provides a baseline significance assessment.

The Wilcoxon signed-rank test is a non-parametric alternative that makes fewer distributional assumptions. It tests whether the median difference is significantly different from zero, providing a robust check when normality cannot be assumed.

Cohen's d measures effect size, quantifying the magnitude of differences in standard deviation units. Values of 0.2 indicate small effects, 0.5 medium effects, and 0.8 large effects. This contextualizes statistical significance: a significant p-value with tiny effect size may not be practically meaningful.

### Comparison Tables

The create_comparison_table function generates a pandas DataFrame summarizing performance across methods and categories. This tabular format facilitates precise numerical comparison and can be exported to CSV for use in reports or further analysis.

When a centralized baseline exists, the table includes "gap" columns showing the percentage difference between federated methods and the centralized baseline. This directly answers the question of how much performance is sacrificed for the privacy benefits of federation.

### Comprehensive Reports

The create_comparison_report function orchestrates all visualization components, generating a complete evaluation report. It creates the comparison table, bar chart, heatmap, box plot, per-category FPR-sPRO curves, statistical analysis, and a summary markdown report.

The markdown summary provides a human-readable document suitable for sharing results. It includes tables of descriptive statistics, per-category performance, and statistical test results. This report serves as a standalone artifact documenting the evaluation findings.

---

## Parameter Reference

### AnomalyScorer Parameters

**backbone_name** (default: "wide_resnet50_2"): The CNN architecture for feature extraction. Must match the backbone used during training.

**layers** (default: ["layer2", "layer3"]): Which intermediate layers to extract features from. Must match training configuration.

**neighborhood_size** (default: 3): Kernel size for local neighborhood averaging. Must match training configuration.

**device** (default: "auto"): Computation device. "auto" selects GPU if available, otherwise CPU.

**use_faiss** (default: True): Whether to use FAISS for accelerated nearest neighbor search.

### MetricsWrapper Parameters

**dataset_base_dir**: Path to the AutoVI dataset root directory. Used to locate ground truth and configuration files.

**curve_max_distance** (default: 0.001): Controls the granularity of threshold sampling for curve computation. Smaller values produce smoother curves but increase computation time.

**num_parallel_workers**: Number of parallel workers for metric computation. None uses system default.

### Visualization Parameters

**fpr_limit** (default: 0.05): The FPR limit for AUC-sPRO comparison. 0.05 (5%) is standard for primary comparisons.

**figsize**: Figure dimensions as (width, height) in inches. Defaults vary by plot type.

**xlim**: X-axis limits for FPR-sPRO curves. Default (0, 0.3) focuses on the practical low-FPR range.

**output_path**: Optional path to save generated figures. If None, figures are returned but not saved.

### Standard FPR Limits

The MAX_FPRS constant defines the standard FPR limits for AUC-sPRO computation: [0.01, 0.05, 0.1, 0.3, 1.0]. These span from stringent (1%) to permissive (100%), enabling analysis across operational scenarios.

---

## The Evaluation Workflow

Understanding how evaluation integrates with the overall pipeline clarifies the role of each component.

After training completes (whether centralized or federated), the trained model exists as a memory bank of representative normal patch features. The first evaluation step is generating anomaly maps for all test images using the AnomalyScorer.

For each test image, the scorer extracts features, queries the memory bank, and produces a normalized anomaly map saved as a PNG file. The output directory structure mirrors the test set structure, with subdirectories for each defect type.

The MetricsWrapper then processes these anomaly maps along with ground truth annotations. It loads both sets of data, matches them by filename, and computes metrics. The aggregator sweeps through thresholds, computing sPRO and FPR at each, then integrates to produce AUC values.

For method comparison, results from multiple evaluations are collected into the nested dictionary structure expected by visualization functions. The create_comparison_report function then generates all visualization artifacts, providing a comprehensive view of relative performance.

The statistical analysis adds formal hypothesis testing, determining whether observed differences are statistically significant or might be attributable to chance variation.

---

## Interpretation Guidance

### What Makes a Good Result?

AUC-sPRO values approaching 1.0 indicate excellent localization performance. At FPR=0.05, values above 0.7 are generally good, above 0.8 are excellent. Values below 0.5 suggest the model struggles to localize defects better than random.

AUC-ROC values for classification follow similar interpretation. Values above 0.9 indicate reliable image-level detection; values near 0.5 indicate performance no better than chance.

### Comparing Methods

When comparing centralized versus federated models, the key question is how much performance is sacrificed for privacy benefits. Small gaps (under 5% relative difference) suggest federated learning achieves nearly the same performance while protecting data privacy. Larger gaps may indicate areas for improvement in the federated approach.

Non-IID partitioning typically produces larger gaps than IID partitioning. This reflects the challenge of learning from heterogeneous client data where different clients see different portions of the feature space.

### Understanding Variance

Performance variance across categories is informative. Large variance suggests the model or dataset has category-specific challenges. Some categories may have distinctive defect types that are easier or harder to detect, or image characteristics that affect feature extraction.

The statistical analysis quantifies whether observed differences exceed what might arise from this variance alone. Non-significant p-values caution against overinterpreting small differences; significant p-values suggest genuine method differences.

---

## Integration with Other Modules

The evaluation module consumes outputs from other modules and produces the final assessment of system performance.

From the models module, it uses the same feature extraction architecture (FeatureExtractor) and memory bank operations (MemoryBank) employed during training. This ensures consistency between training and evaluation.

From the data module, it uses AutoVIDataset to load test images and the category definitions to organize results appropriately.

From the federated module, it receives trained global memory banks that represent the collective learning of distributed clients.

The evaluation module produces the final verdict on whether the privacy-preserving federated approach achieves acceptable performance compared to centralized training. It also enables analysis of how robustness and fairness modifications affect detection accuracy.

---

## Summary

The evaluation module transforms trained models into quantified assessments of detection performance. The AnomalyScorer bridges models to predictions, generating pixel-wise anomaly maps that reveal where the model perceives anomalies. The MetricsWrapper compares these predictions against ground truth, computing industry-standard metrics that capture both localization quality (AUC-sPRO) and classification accuracy (AUC-ROC).

The visualization components transform numerical metrics into interpretable displays that support understanding, comparison, and communication. Statistical analysis adds rigor by testing whether observed differences are significant.

Together, these components complete the anomaly detection pipeline. Training produces a model; evaluation determines its quality. The structured metrics and comprehensive reports enable fair comparison between approaches, identification of strengths and weaknesses, and informed decisions about deployment readiness.

For the federated learning context, evaluation plays a particularly crucial role in answering whether privacy-preserving training sacrifices detection quality. The comparison tools directly address this question, enabling researchers and practitioners to make informed tradeoffs between privacy protection and inspection accuracy.

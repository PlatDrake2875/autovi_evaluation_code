# Jupyter Notebooks Explained

This document explains the four Jupyter notebooks in the notebooks/ folder, covering their purpose, the theoretical and mathematical concepts they employ, and their relationship to the source code modules. Understanding these notebooks is essential for reproducing experiments and comprehending the complete experimental workflow.

---

## The Role of Jupyter Notebooks in This Project

Jupyter notebooks (.ipynb files) serve a fundamentally different purpose than Python source files (.py files) in this project. The source code in src/ provides reusable, modular components designed for production use, testing, and integration into larger pipelines. The notebooks provide an interactive experimentation environment where researchers can explore data, run experiments step-by-step, visualize intermediate results, and iterate on analysis without modifying the core codebase.

This separation follows a common pattern in machine learning research. The .py files encapsulate stable, tested implementations of algorithms and data processing pipelines. The .ipynb files serve as executable documentation that demonstrates how to use these components, allows for exploratory analysis, and produces the figures and statistics that appear in publications. When a researcher wants to understand what the code does, they read the source files. When they want to see how to use the code and what results it produces, they run the notebooks.

The notebooks in this project follow a logical progression that mirrors the scientific method: first explore and understand the data (01), then establish a baseline (02), run the experimental variations (03), and finally analyze and compare results (04). Each notebook builds upon the previous ones, creating a coherent narrative from raw data to scientific conclusions.

---

## notebooks/01_data_exploration.ipynb

### Purpose and Overview

This notebook provides a comprehensive exploration of the AutoVI visual inspection dataset before any model training begins. Understanding the data distribution, image characteristics, and class imbalances is crucial for designing appropriate experiments and interpreting results. The notebook answers fundamental questions: How many samples exist? What do the images look like? How are defects distributed? What will the federated partitions contain?

### Dataset Loading and Structure

The notebook begins by loading the AutoVIDataset class from src/data/autovi_dataset.py and examining the six object categories: engine_wiring, pipe_clip, pipe_staple, tank_screw, underbody_pipes, and underbody_screw. The dataset is organized into train and test splits, where training data contains only normal (good) samples and test data contains both good and defective samples. This one-class classification setup reflects real industrial scenarios where defective examples are scarce or impossible to collect comprehensively.

The get_statistics() method returns dictionaries summarizing sample counts by category and label. The notebook displays these statistics in tabular form, revealing the class distribution across categories. Understanding which categories have more or fewer samples informs expectations about model performance—categories with more training data typically yield better-generalized memory banks.

### Image Dimensions Analysis

A critical aspect of the AutoVI dataset is the category-dependent image resolution. The notebook iterates through categories, loads sample images, and reports their dimensions. Small objects (engine_wiring, pipe_clip, pipe_staple) have images sized at 400×400 pixels, while large objects (tank_screw, underbody_pipes, underbody_screw) use 1000×750 pixels. This size distinction reflects the physical scale of components and the field of view required to capture them during inspection.

The visualization displays sample images from each category in a 2×3 grid, allowing visual inspection of image quality, lighting conditions, and object appearance. This qualitative assessment helps identify potential challenges such as reflections, occlusions, or texture variations that might affect feature extraction.

### Class Distribution Visualization

The notebook generates bar charts showing the distribution of samples across categories for both training and test sets. The training set bar chart uses a single color since all training samples are good. The test set uses a grouped bar chart with green for good samples and red for defective samples, immediately revealing the test set composition.

Class imbalance is a common challenge in anomaly detection. If certain categories have significantly fewer training samples, the corresponding memory banks will be smaller and potentially less representative. The visualizations make these imbalances immediately apparent, informing decisions about evaluation strategies and result interpretation.

### Defect Type Analysis

Beyond binary good/defective labels, the AutoVI dataset provides fine-grained defect type annotations. The notebook uses Python's Counter class to enumerate defect types within each category, printing counts for each type. Defect types fall into two broad categories: structural anomalies (physical damage, missing components, misalignment) and logical anomalies (incorrect assembly, wrong parts, sequence errors).

For the engine_wiring category, the notebook displays example images for different defect types, providing visual reference for what the model must detect. This qualitative understanding of defect appearances helps interpret later evaluation results—if a model struggles with logical anomalies, the visualizations explain why (logical anomalies often appear visually similar to normal samples).

### Federated Learning Partitioning Preview

The final section previews how data will be distributed across federated clients. Two partitioning strategies are demonstrated:

The IIDPartitioner creates a uniform random distribution where each of the 5 clients receives approximately 20% of the total data. After shuffling all sample indices, they are divided into equal portions. This represents an idealized scenario where all clients see statistically similar data distributions. The printed statistics show that each client receives samples from all categories in roughly equal proportions.

The CategoryPartitioner creates a non-IID distribution based on product categories, simulating realistic factory deployments. Client 0 (Engine Assembly) receives only engine_wiring. Client 1 (Underbody Line) receives underbody_pipes and underbody_screw. Client 2 (Fastener Station) receives tank_screw and pipe_staple. Client 3 (Clip Inspection) receives only pipe_clip. Client 4 (Quality Control) receives 10% samples from each category, simulating a QC station that performs random sampling across all products.

The stacked bar chart visualization makes the stark difference between IID and category-based partitioning immediately apparent. In IID, all bars have similar heights with rainbow-colored segments representing all categories. In category-based, most clients have single-color bars (one category) while the QC client has a smaller rainbow bar.

### Key Takeaways

The exploration reveals that the AutoVI dataset contains approximately 1,500 training samples and 2,400 test samples across 6 categories. Image sizes vary by category (400×400 vs 1000×750), requiring category-specific preprocessing. The federated partitioning preview demonstrates that category-based distribution creates highly heterogeneous client data, presenting a significant challenge for federated learning algorithms that must aggregate knowledge from specialized local models.

---

## notebooks/02_baseline_training.ipynb

### Purpose and Overview

This notebook trains centralized PatchCore models that serve as the upper-bound reference for federated experiments. By training on all available data at once, the centralized baseline represents the best possible performance achievable without privacy or communication constraints. All subsequent federated experiments are evaluated relative to this baseline.

### Theoretical Background: PatchCore Algorithm

PatchCore is a state-of-the-art anomaly detection method based on the observation that pre-trained deep neural networks encode rich semantic features that transfer well to new domains. Rather than training a network from scratch on the target domain, PatchCore uses a frozen backbone (WideResNet-50-2) pre-trained on ImageNet to extract patch-level features from normal training images.

The algorithm builds a memory bank containing representative feature vectors from normal samples. During inference, test image features are compared against this memory bank using nearest neighbor search. Anomalous regions produce features that are distant from any normal features in the memory bank, yielding high anomaly scores.

The mathematical formulation is straightforward. Let φ(x) denote the feature extractor that maps an image x to a set of patch features {p₁, p₂, ..., pₙ}. The memory bank M is a subset of all patch features from training images, selected via coreset sampling. For a test patch p, the anomaly score is s(p) = min_{m∈M} ||p - m||₂, the distance to the nearest neighbor in M.

### Configuration and Setup

The notebook defines configuration parameters matching those in experiments/configs/baseline/patchcore_config.yaml. The backbone is "wide_resnet50_2", layers are ["layer2", "layer3"], coreset_percentage is 0.1, and neighborhood_size is 3. The batch size of 32 and 4 data loading workers balance throughput with memory constraints.

Reproducibility is ensured by setting random seeds (torch.manual_seed(42) and np.random.seed(42)) before any stochastic operations. This guarantees that coreset selection produces identical memory banks across runs.

### Image Preprocessing Pipeline

The get_transforms() function creates a torchvision.transforms.Compose pipeline for each category. Images are first resized to category-specific dimensions (400×400 or 1000×750). Then ToTensor() converts PIL images to PyTorch tensors with values in [0, 1]. Finally, Normalize() applies ImageNet statistics: mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].

This normalization is mathematically expressed as: x_normalized = (x - mean) / std, applied independently to each RGB channel. Since WideResNet-50-2 was trained on ImageNet with these statistics, input images must be normalized identically for the pre-trained features to be meaningful.

### Single-Category Training Demonstration

The notebook first trains on a single category (engine_wiring) to verify the implementation before scaling to all categories. The AutoVIDataset is instantiated with categories=[DEMO_CATEGORY] to filter to that single category. A DataLoader wraps the dataset with batching, shuffling disabled (unnecessary for memory bank construction), and parallel data loading.

The PatchCore model is initialized with the configured parameters and device="auto" for automatic GPU detection. The fit() method processes all training batches, extracts features, and performs coreset selection. The get_stats() method returns a dictionary with memory_bank_size, feature_dim, and other diagnostics.

### Anomaly Map Visualization

The visualize_anomaly_map() function displays three panels: the original image (denormalized for viewing), the raw anomaly map as a heatmap, and an overlay combining both. The denormalization reverses the preprocessing: img = img * std + mean, converting normalized tensor values back to displayable [0, 1] range.

The anomaly map is a 2D array of scores, one per spatial location. The jet colormap maps low scores (normal) to blue and high scores (anomalous) to red. The overlay uses alpha=0.5 transparency to show the anomaly map superimposed on the original image, highlighting which regions triggered high scores.

### Inference Testing

The notebook tests the trained model on test samples, including both good and defective images. For each sample, predict_single() returns the anomaly map and a scalar image-level score (typically the maximum or percentile of pixel scores). The visualization shows whether the model correctly identifies defective regions and produces low scores for normal images.

### Training All Categories

The train_category() function encapsulates the complete training pipeline for a single category: create transforms, load dataset, create dataloader, initialize model, fit, and save. The notebook iterates through all CATEGORIES, calling this function for each.

Error handling with try/except ensures that failures in one category don't prevent training of others. The all_stats dictionary accumulates statistics from each category, saved as training_summary.json for later reference.

### Relation to Source Code

This notebook demonstrates the PatchCore class from src/models/patchcore.py and FeatureExtractor from src/models/backbone.py. The dataset loading uses AutoVIDataset from src/data/autovi_dataset.py. The transforms are created using get_transforms from src/data/preprocessing.py. Understanding the notebook requires familiarity with these modules, documented in models_module_explained.md and data_module_explained.md.

---

## notebooks/03_federated_experiments.ipynb

### Purpose and Overview

This notebook runs the core federated learning experiments, comparing IID and category-based data partitioning. It demonstrates how the federated infrastructure distributes training across clients and aggregates their local memory banks into a global model. The experiments simulate realistic factory scenarios where inspection stations cannot share raw data but can collaboratively build a shared anomaly detection model.

### Theoretical Background: Federated Learning for Anomaly Detection

Traditional federated learning (FedAvg) aggregates model gradients or weights from clients training local neural networks. PatchCore presents a unique challenge because it has no trainable parameters—the memory bank is a collection of feature vectors, not learned weights. Federated PatchCore therefore aggregates memory banks rather than gradients.

The federated coreset strategy works as follows. Each client extracts features from local normal images and performs local coreset selection, producing a compressed local memory bank. Clients send their local banks to the server, which concatenates all received features and performs global coreset selection to produce the final global memory bank. This approach balances representation across clients while maintaining a bounded memory footprint.

### Data Partitioning Comparison

The notebook creates both IIDPartitioner and CategoryPartitioner instances with the same seed for reproducibility. The partition() method returns a dictionary mapping client IDs to lists of sample indices. The compute_partition_stats() function analyzes these partitions, counting samples per category per client.

The plot_partition_distribution() function creates grouped bar charts showing how categories are distributed across clients. For IID partitioning, each client's bar group contains roughly equal amounts of each category. For category-based partitioning, most clients have only 1-2 categories represented, creating visually striking differences in the distribution.

### Dataset Wrapping for Federated Training

The CategoryTransformDataset wrapper applies category-specific transforms on-the-fly. Since different categories require different resize dimensions, this wrapper examines each sample's category and applies the appropriate transform. The TransformedSubset class creates views into the transformed dataset for specific index subsets, enabling client-specific dataloaders.

The create_dataloaders() function creates one DataLoader per client, each containing only that client's assigned samples. A custom collate_fn batches samples into dictionaries with stacked image tensors, label tensors, and category lists.

### IID Federated Experiment

The FederatedPatchCore class is instantiated with num_clients=5 and aggregation_strategy="federated_coreset". The global_bank_size=10000 limits the final aggregated memory bank regardless of how many features clients contribute. The weighted_by_samples=True flag ensures clients with more data have proportionally more representation.

The train() method orchestrates the federated training round. Each client processes its dataloader, extracts features, and performs local coreset selection. The server aggregates all client banks, applies global coreset selection, and returns the final global memory bank. The output shape confirms the target size (e.g., [10000, 1536] for 10,000 patches of 1536-dimensional features).

### Category-Based Federated Experiment

The same FederatedPatchCore class trains on the category-based partition. Despite the highly non-IID data distribution—where most clients see only 1-2 categories—the federated aggregation must produce a global model that performs well across all categories.

The challenge is ensuring adequate representation. If Client 3 (Clip Inspection) has fewer samples than Client 1 (Underbody Line), the weighted aggregation ensures proportional contribution. However, if a category has inherently fewer samples across all clients, it may be underrepresented in the global bank.

### Client Contribution Analysis

The plot_client_contributions() function visualizes how many samples each client processed and how large their local coresets were. Two bar charts compare samples per client (before coreset) and coreset size per client (after compression). These visualizations reveal whether the partitioning created balanced or imbalanced client workloads.

For IID partitioning, all clients have similar sample counts and coreset sizes. For category-based partitioning, significant variation appears—clients assigned to categories with more data contribute larger coresets.

### Quick Inference Test

The notebook performs a brief inference test on the test dataset using both trained models. The test_inference() function applies category-specific transforms to test images and calls predict_single() to obtain anomaly maps and scores. Results are printed showing index, category, ground truth label, and predicted score.

Comparing scores across methods provides immediate feedback on model quality. If defective samples consistently score higher than good samples, the model is discriminating effectively. Large differences between IID and category-based scores suggest the partitioning strategy affects performance.

### Relation to Source Code

This notebook exercises FederatedPatchCore from src/federated/federated_patchcore.py, which internally uses PatchCoreClient from src/federated/client.py and FederatedServer from src/federated/server.py. The partitioning uses IIDPartitioner and CategoryPartitioner from src/data/partitioner.py. These relationships are documented in federated_module_explained.md.

---

## notebooks/04_results_analysis.ipynb

### Purpose and Overview

This notebook performs comprehensive statistical analysis comparing centralized and federated PatchCore models. It loads evaluation results produced by the evaluation pipeline, computes aggregate statistics, tests for statistical significance, and generates publication-quality visualizations. The analysis answers the core research question: How much performance is lost by training federatedly instead of centrally?

### Evaluation Metrics Background

Two primary metrics assess anomaly detection performance:

**AUC-sPRO (Area Under the Saturated Per-Region Overlap curve)** measures pixel-level localization accuracy. For each defect region in the ground truth mask, sPRO computes the overlap between the predicted anomaly map and the ground truth at various thresholds. The "saturated" variant clips the per-region overlap at 1.0, preventing large defects from dominating the score. The AUC integrates this curve up to a specified false positive rate (FPR) limit.

**AUC-ROC (Area Under the Receiver Operating Characteristic curve)** measures image-level classification accuracy. The ROC curve plots true positive rate against false positive rate as the decision threshold varies. AUC-ROC of 1.0 indicates perfect separation between good and defective images; 0.5 indicates random guessing.

The FPR limit parameter (typically 0.05) bounds the false positive rate on normal pixels. In industrial settings, low FPR limits are crucial—too many false alarms overwhelm human operators. AUC-sPRO at FPR=0.05 captures how well the model localizes defects while keeping false positives below 5%.

### Loading and Organizing Results

The load_metrics() function traverses the metrics directory structure, loading JSON files for each method-category combination. The expected structure is outputs/evaluation/metrics/{method}/{category}/metrics.json. The function handles missing files gracefully, printing warnings and continuing.

Results are organized into a nested dictionary: results[method][category] = metrics_dict. This structure enables easy iteration over methods or categories for comparison.

### Per-Object Performance Analysis

The create_comparison_table() function from src/evaluation/visualization.py constructs a pandas DataFrame with categories as rows and methods as columns. Each cell contains the AUC-sPRO value at the specified FPR limit. This table provides an immediate overview of how methods compare across categories.

The extract_auc_spro() helper function extracts values from the nested JSON structure. For each object, it navigates to results["localization"]["auc_spro"][fpr_limit]. Missing values are handled by returning empty dictionaries.

### Aggregate Statistics

For each method, the notebook computes descriptive statistics across categories: mean, standard deviation, minimum, maximum, and median. These aggregates summarize overall performance in single numbers. The mean AUC-sPRO is the primary comparison metric—higher is better.

The performance gap analysis computes relative differences between federated and centralized methods. For each category, gap_pct = (federated - centralized) / centralized × 100 expresses the difference as a percentage. Negative gaps indicate federated underperformance; positive gaps indicate (rare) federated improvement.

### Statistical Significance Testing

The compute_statistical_analysis() function performs rigorous hypothesis testing:

**Paired t-test** compares means across paired observations (same categories, different methods). The null hypothesis is that the mean difference is zero. A p-value below 0.05 indicates statistically significant difference. The test statistic t = (mean_diff) / (std_diff / √n) follows a t-distribution with n-1 degrees of freedom.

**Wilcoxon signed-rank test** is a non-parametric alternative that makes no distributional assumptions. It tests whether the median difference is zero by ranking absolute differences and comparing positive vs negative rank sums. With small sample sizes (6 categories), the Wilcoxon test provides robustness against non-normality.

**Cohen's d effect size** quantifies the magnitude of difference in standard deviation units: d = mean_diff / pooled_std. Interpretation guidelines: |d| < 0.2 is negligible, 0.2-0.5 is small, 0.5-0.8 is medium, > 0.8 is large. Effect size complements p-values by indicating practical significance—a statistically significant but tiny effect may not matter in practice.

### Visualization Methods

**Bar comparison chart** (plot_comparison_bar_chart) groups bars by category with different colors for each method. This visualization enables side-by-side comparison at the category level, immediately revealing which methods excel or struggle on specific categories.

**Performance heatmap** (plot_performance_heatmap) displays a matrix with categories as rows and methods as columns, color-coded by AUC-sPRO values. Heatmaps reveal patterns—if a column is consistently darker (lower values), that method underperforms across the board.

**Box plot comparison** (plot_box_comparison) shows the distribution of AUC-sPRO values across categories for each method. The box spans the interquartile range (25th to 75th percentile), the line marks the median, and whiskers extend to extreme values. Box plots reveal both central tendency and variability.

**FPR-sPRO curves** (plot_fpr_spro_curves) plot sPRO values against FPR limits for each method on a single category. These curves show how performance changes as the FPR constraint is relaxed. Methods that maintain high sPRO at low FPR limits are preferred for high-precision applications.

### Per-Defect Type Analysis

The notebook disaggregates results by defect type, comparing performance on structural vs logical anomalies. Structural anomalies (physical damage) typically produce visually distinctive features that PatchCore detects well. Logical anomalies (assembly errors) may appear visually similar to normal samples, challenging the detection algorithm.

By computing separate means for structural and logical defect types, the analysis reveals whether certain methods handle one type better than the other. This information guides practical deployment decisions—if logical anomalies are critical in a factory, method selection should prioritize performance on that category.

### Image-Level Classification

Beyond localization, the notebook analyzes AUC-ROC for image-level classification. High AUC-ROC indicates that the model's image-level score (derived from the anomaly map) correctly ranks defective images above good images. This metric matters for deployment scenarios where binary pass/fail decisions are needed.

### Key Findings Generation

The final section synthesizes results into actionable conclusions:

1. **Best performing method**: Which approach achieves highest mean AUC-sPRO?
2. **Federated vs centralized gap**: How much performance is sacrificed for privacy/communication benefits?
3. **Statistical significance**: Are observed differences likely due to chance or genuine effects?
4. **Effect sizes**: Are the differences practically meaningful or negligibly small?

These findings directly address the research questions motivating the federated learning experiments.

### Relation to Source Code

This notebook uses visualization functions from src/evaluation/visualization.py including plot_fpr_spro_curves, plot_comparison_bar_chart, plot_performance_heatmap, plot_box_comparison, compute_statistical_analysis, and create_comparison_table. The metrics were computed by AnomalyScorer and MetricsWrapper from src/evaluation/, documented in evaluation_module_explained.md.

---

## How Notebooks and Source Code Complement Each Other

The relationship between notebooks and source code follows the principle of separation of concerns. Source code in src/ implements algorithms with clean interfaces, comprehensive error handling, and unit test coverage. Notebooks consume these implementations to run experiments and analyze results.

When modifying the project, changes typically flow in one direction: update source code first, then use notebooks to verify behavior. If a notebook reveals unexpected results, the investigation may lead back to source code modifications. This workflow keeps the codebase stable while allowing rapid experimentation.

The notebooks also serve as integration tests. If all four notebooks run successfully from start to finish, the major code paths are exercised. Any breaking change to src/ modules will likely cause notebook failures, providing early warning of regressions.

For new researchers joining the project, the recommended learning path is: read the source code documentation (the _explained.md files) to understand algorithm implementations, then run the notebooks sequentially to see the implementations in action. The combination of conceptual understanding from documentation and practical execution from notebooks provides comprehensive project comprehension.

---

## Summary Table

| Notebook | Primary Purpose | Key Outputs | Source Dependencies |
|----------|----------------|-------------|---------------------|
| 01_data_exploration | Understand dataset characteristics | Distribution tables, sample visualizations, partition previews | AutoVIDataset, partitioners |
| 02_baseline_training | Train centralized reference models | Saved PatchCore models, training statistics | PatchCore, preprocessing |
| 03_federated_experiments | Run federated training variations | Federated models, client contribution analysis | FederatedPatchCore, partitioners |
| 04_results_analysis | Statistical comparison of methods | Significance tests, visualizations, key findings | visualization module |

---

## Reproducing Results

To reproduce the experimental results:

1. Run 01_data_exploration.ipynb to verify dataset loading and understand data distribution
2. Run 02_baseline_training.ipynb to train centralized baselines for all categories
3. Run 03_federated_experiments.ipynb to train IID and category-based federated models
4. Run the evaluation scripts (evaluate_experiment.py) on all trained models
5. Run 04_results_analysis.ipynb to generate comparison statistics and visualizations

Each notebook contains configuration variables (DATA_ROOT, OUTPUT_DIR) that must be updated to match the local environment. The notebooks assume the AutoVI dataset has been downloaded and extracted to the specified DATA_ROOT path.

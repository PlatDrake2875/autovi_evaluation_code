# Project Assets Explained

This document provides explanations for various project folders and their contents, organized by folder with subtitles for each section. It serves as a reference for understanding the supplementary materials, diagrams, configurations, and other assets that support the main codebase.

---

## docs/assets/diagrams/

This folder contains four Mermaid diagram files (.mmd) that visually represent the key workflows and architecture of the federated anomaly detection system. Mermaid is a text-based diagramming language that renders flowcharts, sequence diagrams, and other visualizations from simple markup. These diagrams can be rendered in any Mermaid-compatible viewer, including GitHub, VS Code with Mermaid extensions, or online tools like mermaid.live.

### data_pipeline.mmd

This diagram illustrates the complete data preparation pipeline from raw dataset acquisition through to federated learning partitioning. The flow is organized into three sequential phases, each color-coded for clarity.

The Data Acquisition phase begins with downloading the AutoVI dataset from Zenodo, extracting the six object categories (engine_wiring, pipe_clip, pipe_staple, tank_screw, underbody_pipes, underbody_screw), and verifying that defects_config.json files are present for each category. These configuration files contain metadata about defect types and are essential for proper evaluation.

The Preprocessing phase handles image loading and normalization. A critical decision node branches based on object type: small objects (engine_wiring, pipe_clip, pipe_staple) are resized to 400×400 pixels, while large objects (tank_screw, underbody_pipes, underbody_screw) are resized to 1000×750 pixels. After resizing, all images are normalized using ImageNet statistics (mean and standard deviation values from the ImageNet dataset). Features are then extracted using the WideResNet-50 backbone and cached for efficient reuse.

The FL Partitioning phase prepares data for federated learning. Another decision node selects between IID partitioning (random uniform split to 5 clients) or category-based partitioning (assigning specific categories to specific clients to simulate non-IID distributions). The output is client-specific dataloaders and saved partition metadata for reproducibility.

### system_architecture.mmd

This diagram depicts the federated learning system architecture, showing the relationship between the central server and the five client stations.

The Central Server contains three components: the Global Memory Bank Coordinator that maintains the aggregated model, the Aggregation Engine that implements the federated coreset strategy, and the Communication Handler that manages client synchronization.

Each of the five clients represents a simulated factory inspection station with a specific role. Client 1 (Engine Assembly) handles engine_wiring data. Client 2 (Underbody Line) handles underbody_pipes and underbody_screw. Client 3 (Fastener Station) handles tank_screw and pipe_staple. Client 4 (Clip Inspection) handles only pipe_clip. Client 5 (Quality Control) receives a mixed sample of 10% from each category, simulating a QC station that performs random sampling across all product types.

Each client has the same internal structure: local data flows to a feature extractor, which feeds a local memory bank. The diagram shows bidirectional communication for initialization/synchronization (step 1), unidirectional flow of local coresets to the server (step 2), and aggregation into the global memory bank (step 3).

The color coding helps distinguish the server (blue) from each client (green, yellow, orange, purple, teal), making the distributed nature of the system immediately apparent.

### training_workflow.mmd

This diagram compares the centralized baseline training with the federated PatchCore training, showing both workflows side by side before they converge at a comparison analysis phase.

The Centralized Baseline workflow is a linear pipeline: load all training data, initialize the WideResNet-50 backbone, extract patch features, perform coreset subsampling to reduce the feature set to representative patches, build the memory bank from these patches, and save the model.

The Federated PatchCore workflow is more complex, involving iterative communication rounds. It begins by initializing the global memory bank and broadcasting to clients. Within each communication round, clients train in parallel: each client loads local normal images, extracts local patch features, performs local coreset selection, and sends results to the server. The server then aggregates the received memory banks, performs global coreset selection to compress the combined features, and updates the global memory bank. A convergence check determines whether to continue with another round or save the final federated model.

The Comparison Analysis phase receives the saved models from both approaches and generates a metrics report comparing their performance. This visualization emphasizes that the project's goal is understanding how federated training compares to the centralized ideal.

### evaluation_workflow.mmd

This diagram shows the complete evaluation pipeline from test data preparation through final report generation.

The Evaluation Setup phase loads test images, ground truth masks (pixel-level annotations of defect locations), and the trained model's memory bank.

The Anomaly Map Generation phase processes each test image: extract features using the same backbone as training, compute distances to the memory bank (nearest neighbor lookup), generate pixel-wise anomaly scores, upsample to original image resolution (since features are at reduced resolution), and save the resulting anomaly maps as PNG files.

The Metrics Computation phase uses the evaluate_experiment.py script to load results and feed them to the MetricsAggregator. This computes sPRO (saturated per-region overlap) for each defect type, determines FPR thresholds, applies binary refinement for precise threshold selection, calculates AUC-sPRO at multiple FPR limits (0.01, 0.05, 0.1, 0.3, 1.0), and computes AUC-ROC for image-level classification.

The Results Output phase produces per-object metrics JSON files, comparison tables showing how different methods perform, visualization plots for publication or analysis, and exports everything into a final report.

### How to Use These Diagrams

To render these diagrams, you can use any of the following approaches. In VS Code, install the "Markdown Preview Mermaid Support" extension and open a markdown file that includes the Mermaid code within triple-backtick mermaid blocks. On GitHub, the diagrams will render automatically when viewed in files that embed them properly. For standalone rendering, paste the contents into mermaid.live to see the visualization and export as SVG or PNG. The diagrams serve both as documentation for understanding the system and as presentation materials for explaining the project to others.

### Relating Diagrams to Code

The data_pipeline.mmd corresponds to the code in src/data/ (AutoVIDataset, Preprocessor, partitioners) and explains the flow implemented by those classes. The system_architecture.mmd maps to src/federated/ (PatchCoreClient, FederatedServer, FederatedPatchCore) and shows how the components interact at runtime. The training_workflow.mmd covers both src/models/patchcore.py for centralized training and src/federated/federated_patchcore.py for federated training. The evaluation_workflow.mmd corresponds to src/evaluation/ (AnomalyScorer, MetricsWrapper) and the evaluate_experiment.py script.

Understanding these visual representations helps you grasp the big picture before diving into code details. When reading source files, referring back to these diagrams clarifies where each component fits in the overall workflow.

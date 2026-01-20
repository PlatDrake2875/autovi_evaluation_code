# Federated Module Explained

## The Case for Federated Learning in Industrial Anomaly Detection

Manufacturing environments generate vast amounts of visual inspection data across geographically distributed facilities. A car manufacturer might operate plants on multiple continents, each producing components with slightly different characteristics due to local tooling, materials, or environmental conditions. Centralizing all this inspection data for training presents serious challenges. Data transfer costs can be enormous when dealing with high-resolution imagery. Privacy regulations like GDPR may restrict cross-border data movement. Proprietary manufacturing processes visible in images constitute trade secrets. Perhaps most critically, individual facilities may simply refuse to share their data with corporate headquarters or sister plants due to competitive concerns within the organization.

Federated learning provides an elegant solution to this dilemma. Instead of moving data to a central location for training, federated learning moves the computation to where the data resides. Each facility trains locally on its own data and shares only the learned model parameters (or in this case, representative feature vectors) with a central server. The server aggregates these contributions into a global model that benefits from the collective knowledge of all facilities without any facility ever seeing another's raw data.

The federated module implements this paradigm for PatchCore anomaly detection. The three files work together to create a complete distributed learning system. The client.py file defines PatchCoreClient, representing an individual facility that extracts features from local images and builds a local coreset. The server.py file defines FederatedServer, which receives coresets from all clients and aggregates them into a global memory bank. The federated_patchcore.py file defines FederatedPatchCore, which orchestrates the entire process from data partitioning through training to inference.

---

## Why Memory Bank Aggregation Differs from Traditional Federated Learning

Most federated learning implementations follow the FedAvg paradigm introduced by McMahan et al. at Google. In FedAvg, each client trains a neural network locally using gradient descent, then sends its updated model weights to a central server. The server averages these weights to produce a global model, which is broadcast back to clients for the next round of training. This process repeats until convergence.

PatchCore's architecture requires a fundamentally different approach because it does not use gradient-based training at all. As explained in the models module documentation, PatchCore builds a memory bank of representative normal patch features and computes anomaly scores as distances to the nearest neighbor in this bank. There are no trainable parameters to optimize with gradients. Instead, training consists of extracting features using a frozen pretrained backbone and selecting a representative subset via coreset sampling.

In the federated setting, each client extracts features from its local normal images and builds a local coreset representing its local notion of normality. These local coresets are then sent to the server, which must combine them into a global memory bank that represents the combined notion of normality across all facilities. This aggregation is not a simple average of weights but rather a careful selection or merging of feature vectors from all clients.

This memory-bank-based approach to federated learning has several advantages for anomaly detection. First, communication costs are relatively low because only coreset vectors need to be transmitted rather than entire model weights or gradients. Second, the approach naturally accommodates non-IID data distributions where different clients see different product categories or defect patterns. Third, there is no concern about gradient leakage attacks that can reconstruct training data from shared gradients in traditional federated learning.

---

## The PatchCoreClient Class: Local Learning

### Initialization and Configuration

The PatchCoreClient class represents a single participant in the federated learning process, typically corresponding to one manufacturing facility or inspection station. During initialization, the client receives a unique client_id that identifies it throughout the training process. This identifier is used for logging, statistics tracking, and ensuring reproducible random number generation through client-specific seeds.

The backbone_name parameter specifies which pretrained CNN to use for feature extraction, defaulting to WideResNet-50-2. The layers parameter determines which intermediate layers to extract features from, defaulting to layer2 and layer3 for multi-scale representation. The neighborhood_size parameter controls the spatial smoothing applied to extracted features. These parameters mirror those in the centralized PatchCore implementation, ensuring consistency between federated and non-federated training.

The coreset_ratio parameter determines what fraction of extracted patches to retain in the local coreset. A value of 0.1 means each client keeps 10% of its patches, applying the greedy k-center selection algorithm locally before sending features to the server. This local compression reduces communication costs and ensures that each client sends a diverse, representative subset rather than potentially redundant raw features.

The device parameter controls whether computation happens on CPU or GPU. Each client maintains its own feature extractor instance, which might seem wasteful compared to sharing a single extractor. However, in a real federated deployment, each client would be a separate machine with its own hardware, so this design accurately reflects the deployment scenario.

### Differential Privacy Integration

The dp_config parameter enables differential privacy protection for the local coreset before transmission to the server. When a DPConfig object with enabled=True is provided, the client creates an EmbeddingSanitizer instance that will process the coreset after selection. This sanitizer clips feature vector norms and adds calibrated Gaussian noise, ensuring that the presence or absence of any single training image cannot be detected from the transmitted coreset.

The integration happens automatically and transparently. After coreset selection in build_local_coreset, if the sanitizer exists, it processes the coreset in place. The client also tracks privacy expenditure through the get_privacy_spent method, allowing the system to monitor cumulative privacy budgets across multiple training rounds.

### Feature Extraction Process

The extract_features method processes a local dataloader of normal images and returns all extracted patch features. The method iterates through the dataloader batches, extracting images using the utility function extract_images_from_batch that handles various batch formats (dictionaries with "image" keys, tuples, or plain tensors).

For each batch, the feature extractor processes images through the pretrained backbone with neighborhood averaging applied, producing patch features of dimension 1536 (512 from layer2 concatenated with 1024 from layer3). These patches are accumulated in a list and concatenated into a single numpy array at the end. The method tracks statistics including the number of images processed and total patches extracted.

The computation happens under torch.no_grad() context since the pretrained backbone is frozen and no gradients are needed. This significantly reduces memory consumption and speeds up processing compared to training mode.

### Local Coreset Construction

The build_local_coreset method transforms raw features into a compressed representative subset using greedy coreset selection. If target_size is not explicitly provided, it is computed from coreset_ratio as approximately 10% of the total patches. The method uses a client-specific seed (base seed plus client_id) to ensure reproducibility while allowing different clients to make different selections.

The actual selection delegates to the greedy_coreset_selection function from the memory bank module, which implements the k-center algorithm. This algorithm iteratively selects the point furthest from the current selection set, ensuring maximal coverage of the feature space. After selection, the chosen features are copied to local_coreset.

If differential privacy is enabled, the sanitizer processes the coreset immediately after selection. This order is important: applying DP before coreset selection would add noise to features that might then be discarded, wasting privacy budget. By sanitizing only the selected features, privacy budget is spent efficiently on the data actually transmitted.

### Communication Methods

The extract_and_build_coreset method provides a convenient single-call interface that performs both feature extraction and coreset construction. This is the typical entry point used during federated training.

The get_local_coreset method returns the current local coreset for transmission to the server. The set_global_memory_bank method receives the aggregated global memory bank from the server after aggregation. In the current implementation, this global bank is stored for potential local inference, allowing each client to detect anomalies using the collectively learned model.

---

## The FederatedServer Class: Central Coordination

### Server Responsibilities

The FederatedServer class implements the central coordinator in the federated learning architecture. Its responsibilities include receiving local coresets from all participating clients, aggregating these coresets into a unified global memory bank, and broadcasting the result back to clients. The server also tracks statistics about the aggregation process and optionally monitors privacy budget consumption and Byzantine robustness.

### Initialization Parameters

The global_bank_size parameter specifies the target number of patches in the aggregated global memory bank. This is typically larger than individual client coresets since it must represent the combined feature space of all clients. A value of 10000 means the server will produce a memory bank containing 10000 representative patches regardless of how many total patches were received from clients.

The aggregation_strategy parameter selects the algorithm used to combine client coresets. The default "federated_coreset" strategy applies coreset selection to the concatenation of all client coresets, ensuring the global bank is a representative subset of the combined space. Alternative strategies include "simple_concatenate" which just merges all coresets without further selection, and "diversity_preserving" which uses weighted sampling to balance client contributions.

The weighted_by_samples parameter determines whether client contributions are weighted by their data sizes during aggregation. When True, a client with twice as many samples will have approximately twice as much representation in the global memory bank. This can be desirable when larger clients represent more of the true data distribution, or undesirable when it might allow one large client to dominate the model.

### Privacy Tracking

When track_privacy is True, the server creates a PrivacyAccountant to monitor cumulative privacy expenditure across the federated system. The target_epsilon parameter sets the privacy budget ceiling. As clients report their privacy expenditure through their statistics, the server records these expenditures and can generate reports showing how much of the budget has been consumed and what remains.

This centralized privacy tracking is essential for understanding the system-wide privacy guarantees. Even if each client individually satisfies (ε, δ)-differential privacy, the composition of multiple clients and multiple rounds affects the overall guarantee. The accountant uses composition theorems to compute the cumulative privacy loss.

### Robustness Configuration

When a RobustnessConfig object is provided with enabled=True, the server initializes components for Byzantine-resilient aggregation. The robust_aggregator implements a coordinate-wise median aggregation that tolerates up to 50% malicious clients. The client_scorer (typically ZScoreDetector) analyzes incoming coresets to identify statistical outliers that might indicate malicious behavior.

The initialization computes the number of samples to use for robust aggregation as global_bank_size divided by 10, balancing computational cost against statistical power. The threshold for outlier detection (zscore_threshold) defaults to 3.0, flagging clients whose contributions deviate by more than 3 standard deviations from the mean.

### Receiving Client Contributions

The receive_client_coresets method accepts the list of local coresets from all clients along with optional statistics. The coresets are stored as pending data awaiting aggregation. Client statistics, if provided, are recorded for later analysis and reporting.

When privacy tracking is enabled, the method extracts privacy expenditure information from client statistics and records it with the privacy accountant. This ensures that the server's privacy report reflects actual client-side noise addition rather than merely planned parameters.

### Robust Aggregation Process

The _robust_aggregate method implements Byzantine-resilient aggregation when robustness is enabled. If client scoring is active, it first analyzes all client coresets to identify outliers. The ZScoreDetector computes statistics about each client's contribution (such as mean feature values and norms) and flags those that deviate significantly from the group.

Detected outliers are logged as warnings but not automatically excluded. The current implementation logs outlier information for analysis while still including all contributions in aggregation. This design allows post-hoc analysis of whether detected outliers corresponded to actual attacks, useful for tuning detection thresholds.

The actual aggregation then proceeds using the robust aggregator, which computes coordinate-wise medians rather than means. This makes the aggregation resistant to extreme values from potentially malicious clients. The method returns both the aggregated features and statistics about the aggregation process.

### Standard Aggregation

When robustness is not enabled, the aggregate method uses the strategy registry to select and execute the aggregation algorithm. The STRATEGY_REGISTRY maps strategy names to functions that implement the aggregation logic. The selected function receives the list of client coresets, the target global bank size, and a random seed, returning the aggregated features and statistics.

After aggregation by either method, the server builds a MemoryBank object around the aggregated features, creating a FAISS index for efficient nearest neighbor search during inference. This memory bank is stored and can be retrieved via get_global_memory_bank for inference or broadcast to clients.

### Broadcasting Results

The broadcast_to_clients method sends the aggregated global memory bank to all participating clients by calling each client's set_global_memory_bank method. In a real distributed deployment, this would involve network communication, but the simulation treats it as direct method calls between objects.

### Persistence and Statistics

The save method writes the global memory bank and all statistics to disk. The global features are stored in numpy format, while statistics are serialized as JSON. The helper function _convert_to_serializable handles numpy types that are not directly JSON-serializable. The load method restores a previously saved state, rebuilding the memory bank and FAISS index.

---

## The FederatedPatchCore Orchestrator

### System Overview

The FederatedPatchCore class serves as the main entry point and orchestrator for the entire federated learning system. It manages the lifecycle of clients and server, coordinates training rounds, and provides inference capabilities using the trained global model. While the client and server classes implement specific functionalities, the orchestrator ties everything together into a cohesive workflow.

### Comprehensive Configuration

The constructor accepts a large number of parameters that configure every aspect of the federated system. Understanding these parameters is essential for properly deploying and tuning the system.

The num_clients parameter specifies how many federated participants will be created. Each client represents an independent data holder, typically corresponding to a manufacturing facility or inspection station. The system creates this many PatchCoreClient instances during initialization.

The backbone_name, layers, neighborhood_size, and coreset_ratio parameters are passed through to each client and control the feature extraction and local coreset construction. These match the parameters used in centralized PatchCore training to ensure comparable behavior.

The global_bank_size and aggregation_strategy parameters are passed to the server and control the aggregation process. The weighted_by_samples parameter determines whether larger clients have proportionally more influence on the global model.

### Multi-Round Training Configuration

The num_rounds parameter enables iterative refinement of the global model through multiple rounds of federated training. In each round, clients extract features and build coresets, the server aggregates them, and the result is broadcast back to clients. Multiple rounds might be useful when clients have streaming data or when the aggregation process benefits from iterative refinement.

For PatchCore with static training data, a single round is typically sufficient since there is no iterative optimization happening. However, the multi-round capability supports scenarios where new data arrives at clients over time or where clients participate in only subsets of rounds.

### Differential Privacy Parameters

The dp_enabled boolean activates differential privacy protection. When True, each client's coreset is sanitized before transmission to the server. The dp_epsilon, dp_delta, and dp_clipping_norm parameters control the privacy guarantee.

The epsilon value (default 1.0) quantifies privacy loss, with smaller values providing stronger privacy but requiring more noise. The delta value (default 1e-5) represents the probability of a catastrophic privacy failure. The clipping_norm (default 1.0) bounds the L2 norm of feature vectors before noise addition, with smaller values enabling less noise for the same privacy guarantee but potentially distorting large features.

These parameters create a DPConfig object that is passed to each client during initialization. The server tracks cumulative privacy expenditure when track_privacy is enabled.

### Robustness and Attack Simulation

The robustness_enabled boolean activates Byzantine-resilient aggregation on the server. The robustness_aggregation parameter selects the aggregation method (currently "coordinate_median"), and robustness_zscore_threshold sets the outlier detection sensitivity.

For testing robustness, the system includes attack simulation capabilities. When attack_enabled is True, the attack_type parameter specifies what kind of attack to simulate ("scaling", "noise", or "sign_flip"). The malicious_fraction parameter determines what fraction of clients are treated as adversarial.

During initialization with attacks enabled, the system creates a ModelPoisoningAttack object and designates the first malicious_fraction of clients as attackers. The malicious_indices list stores which clients will have their coresets corrupted during training.

### Client and Server Initialization

The constructor creates all PatchCoreClient instances in a loop, each receiving the same configuration but a unique client_id. It then creates a single FederatedServer with aggregation configuration and optional privacy/robustness settings.

Additionally, the orchestrator maintains its own feature extractor for inference purposes. While each client has its own extractor for training, the orchestrator's extractor is used when calling predict on the trained model. This design reflects that inference might happen at a different location than training.

### Data Partitioning

The setup_clients method partitions a dataset among the federated clients. This simulates the real-world scenario where each client has access to only a portion of the total data. The partitioning parameter selects the strategy: "iid" distributes samples uniformly at random across clients, while "category" assigns entire product categories to specific clients to simulate non-IID distributions.

The IID partitioner randomly shuffles all sample indices and divides them into roughly equal portions for each client. This produces statistically similar data distributions across clients, representing a best-case scenario for federated learning.

The category partitioner assigns complete product categories to clients. For example, with 6 categories and 5 clients, some clients might receive data from multiple categories while others might specialize in a single category. This non-IID distribution is more realistic for manufacturing scenarios where different plants produce different products.

The method stores the partition (a dictionary mapping client IDs to lists of sample indices) and computes statistics about the partition that are logged for visibility into the data distribution.

### The Training Pipeline

The train method is the main entry point for federated training. It accepts dataloaders for each client (keyed by client_id), a random seed, and optional checkpointing configuration.

The method implements a multi-round training loop. For each round, it calls _train_single_round to execute one complete cycle of local training and global aggregation. If checkpointing is enabled, it saves the model state at specified intervals.

The _train_single_round method implements the core federated learning protocol in three phases.

Phase 1 performs local client processing. For each client with an associated dataloader, the method calls extract_and_build_coreset to have the client extract features from its local data and build a compressed coreset. The resulting coreset and client statistics are collected into lists. The method also tracks image size from the first batch to enable proper anomaly map sizing later.

If attack simulation is enabled, Phase 1 concludes by applying the attack to corrupt the coresets of malicious clients. The attack.apply method modifies the coresets according to the attack type (scaling, adding noise, or flipping signs).

Phase 2 performs server aggregation. The collected coresets and statistics are passed to the server via receive_client_coresets, then aggregate is called to produce the global memory bank. The orchestrator stores a reference to this memory bank for inference.

Phase 3 broadcasts results. The server's broadcast_to_clients method distributes the global memory bank back to all clients, completing the round.

### Inference Using the Global Model

The predict method uses the trained global memory bank to compute anomaly scores for new images. The process mirrors centralized PatchCore inference: extract patch features using the feature extractor, query the global memory bank for nearest neighbor distances, reshape these distances into a spatial map, upsample to the original image resolution, and compute image-level scores as the maximum of each anomaly map.

The predict_single method provides a convenient interface for single-image inference, handling various input formats including PIL Images, PyTorch tensors, and numpy arrays.

### Model Persistence

The save method writes the complete federated model state to disk. This includes the server's global memory bank and statistics, a configuration JSON file capturing all model parameters, the training log with timing and statistics, and the privacy report if differential privacy was enabled.

The load method restores a previously saved model, allowing the system to be deployed for inference without retraining. It reads the configuration, reconstructs the feature extractor, and loads the server state including the global memory bank.

---

## Parameter Reference

### num_clients (default: 5)

The number of federated learning participants. Each client represents an independent data holder with its own local training data. More clients typically means more diverse data representation in the global model but also higher communication costs and longer training time.

### coreset_ratio (default: 0.1)

The fraction of local patches each client retains in its coreset. A value of 0.1 means 10% of extracted patches are sent to the server. Lower values reduce communication costs but risk missing important patterns. Higher values increase communication overhead and may include redundant features.

### global_bank_size (default: 10000)

The target number of patches in the aggregated global memory bank. This should be large enough to represent the combined feature space of all clients. Larger values improve coverage but increase memory usage and inference time.

### aggregation_strategy (default: "federated_coreset")

The algorithm used to combine client coresets. "federated_coreset" applies k-center selection to the concatenated coresets, ensuring diverse representation. "simple_concatenate" merges all coresets without further selection. "diversity_preserving" uses weighted sampling to balance client contributions.

### weighted_by_samples (default: True)

Whether to weight client contributions by their data sizes during aggregation. When True, clients with more data have proportionally more representation in the global model. This can be appropriate when larger datasets represent more of the true distribution, or inappropriate when it risks one client dominating.

### num_rounds (default: 1)

The number of federated training rounds. Each round involves local training, aggregation, and broadcast. For static data with single-shot coreset construction, one round typically suffices. Multiple rounds support streaming data or iterative refinement scenarios.

### dp_enabled (default: False)

Whether to enable differential privacy protection. When True, client coresets are sanitized with calibrated noise before transmission, providing formal privacy guarantees for individual training samples.

### dp_epsilon (default: 1.0)

The privacy parameter epsilon controlling the strength of privacy protection. Smaller values mean stronger privacy but more noise, potentially degrading model utility. The value 1.0 represents a reasonable balance for many applications.

### dp_delta (default: 1e-5)

The probability of catastrophic privacy failure. This is typically set to be negligible, much smaller than 1/N where N is the number of training samples.

### dp_clipping_norm (default: 1.0)

The maximum L2 norm for feature vectors after clipping. Features with larger norms are scaled down to this bound before noise addition. The choice depends on the typical magnitude of features from the backbone; 1.0 works well for normalized embeddings.

### robustness_enabled (default: False)

Whether to enable Byzantine-resilient aggregation. When True, the server uses robust aggregation methods that tolerate malicious client contributions.

### robustness_aggregation (default: "coordinate_median")

The robust aggregation method to use. "coordinate_median" computes element-wise medians across client contributions, tolerating up to 50% malicious clients.

### robustness_zscore_threshold (default: 3.0)

The threshold for flagging outlier clients. Clients whose contributions deviate by more than this many standard deviations from the mean are flagged as potential outliers. Higher values are more permissive, lower values more sensitive.

### attack_enabled (default: False)

Whether to simulate Byzantine attacks during training. Used for testing robustness mechanisms.

### attack_type (default: None)

The type of attack to simulate: "scaling" multiplies features by a large factor, "noise" adds random noise, "sign_flip" negates feature values.

### malicious_fraction (default: 0.2)

The fraction of clients designated as malicious when attack simulation is enabled. 0.2 means 20% of clients will have their contributions corrupted.

---

## Data Flow Through the System

Understanding the complete data flow illuminates how the components work together.

The process begins with data partitioning. A complete AutoVIDataset containing normal images from all categories is partitioned among clients using either IID or category-based strategies. Each client receives a list of sample indices identifying which images it can access.

During local training, each client creates a dataloader from its partition and iterates through batches of images. Each batch (typically 16-32 images) is processed through the feature extractor backbone, producing patch features of shape [batch_size × 784, 1536] where 784 is the number of spatial positions and 1536 is the feature dimension. These patches are accumulated across all batches.

After feature extraction, the client applies coreset selection to compress potentially hundreds of thousands of patches down to typically several thousand representative patches. The k-center algorithm ensures this subset covers the local feature space well.

If differential privacy is enabled, the client sanitizes the coreset by clipping feature vector norms and adding Gaussian noise. The sanitized coreset is what leaves the client's local environment.

The server receives coresets from all clients. With 5 clients each contributing 5000 patches, the server has 25000 total patches to aggregate. If robustness is enabled, the server first analyzes these contributions to detect potential outliers.

Aggregation produces a global memory bank of the specified size (e.g., 10000 patches). The federated_coreset strategy applies k-center selection to the concatenated coresets, ensuring the global bank represents the diverse feature space across all clients.

The global memory bank is then broadcast back to all clients. Each client can now perform local inference using the collectively learned model, enabling anomaly detection that benefits from the knowledge of all facilities without any facility having accessed another's raw data.

During inference, a test image is processed through the same feature extraction pipeline. The resulting patch features are compared against the global memory bank using nearest neighbor search. Distances to the nearest normal patches become anomaly scores, which are reshaped into a spatial map, upsampled to the original image resolution, and used to identify potentially anomalous regions.

---

## Integration with Privacy and Robustness

The federated module serves as the integration point for the privacy and robustness modules. Rather than duplicating privacy or robustness logic, it creates instances of the specialized classes and invokes them at appropriate points in the pipeline.

Privacy integration happens at the client level. Each PatchCoreClient receives a DPConfig object and creates an EmbeddingSanitizer if privacy is enabled. The sanitizer is invoked automatically after coreset selection in build_local_coreset. The server's PrivacyAccountant tracks cumulative privacy expenditure by reading privacy statistics from client reports.

Robustness integration happens at the server level. The FederatedServer receives a RobustnessConfig object and creates a CoordinateMedianAggregator and ZScoreDetector if robustness is enabled. These components are invoked during aggregation, with the detector analyzing contributions before the aggregator combines them.

Attack simulation is coordinated by the orchestrator. FederatedPatchCore creates a ModelPoisoningAttack object if attacks are enabled and applies it to client coresets between local training and server aggregation. This placement ensures that attacks affect the data actually transmitted to the server, mimicking real-world Byzantine behavior.

This modular design allows each concern to be developed and tested independently while providing clean integration points for combined deployment.

---

## Summary

The federated module transforms PatchCore from a centralized algorithm into a privacy-preserving distributed system. By sharing representative feature vectors rather than raw data or model gradients, it enables collaborative anomaly detection across organizational boundaries.

The PatchCoreClient class encapsulates local computation, extracting features from private data and compressing them into representative coresets that can be safely shared. Integration with differential privacy adds formal guarantees that individual training samples cannot be identified from transmitted coresets.

The FederatedServer class implements central coordination, combining contributions from all clients into a unified global model. Integration with robust aggregation ensures that the system tolerates malicious participants without catastrophic degradation.

The FederatedPatchCore orchestrator ties everything together, managing the complete lifecycle from data partitioning through multi-round training to inference. Its comprehensive configuration options expose all the knobs needed to tune the system for different deployment scenarios.

Together, these components enable industrial anomaly detection systems that learn from distributed data while respecting privacy constraints, tolerating adversarial participants, and maintaining detection accuracy competitive with centralized training.

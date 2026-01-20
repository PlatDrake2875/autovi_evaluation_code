# Experiment Configurations Explained

This document explains the YAML configuration files used to run experiments in this project. The configurations are organized into two categories: baseline (centralized training) and federated (distributed training). Understanding these parameters is essential for reproducing experiments and adapting the system to new scenarios.

---

## experiments/configs/baseline/

### patchcore_config.yaml

This configuration defines the centralized PatchCore baseline, where all training data is available at a single location. This serves as the upper-bound reference for federated experiments.

#### model section

The **backbone** parameter specifies the CNN architecture for feature extraction. The value "wide_resnet50_2" refers to WideResNet-50 with a width multiplier of 2, meaning convolutional channels are doubled compared to standard ResNet-50. This architecture achieves 78.5% ImageNet top-1 accuracy and provides rich feature representations that transfer well to anomaly detection.

The **layers** parameter ["layer2", "layer3"] indicates which intermediate layers to extract features from. In ResNet architecture, layer2 outputs 512 channels at 1/4 spatial resolution, while layer3 outputs 1024 channels at 1/8 resolution. Using both captures multi-scale information: layer2 preserves finer spatial detail, layer3 provides richer semantic features. The concatenated 1536-dimensional features (512+1024) balance localization precision with semantic richness.

The **coreset_percentage** of 0.1 means 10% of extracted patches are retained in the memory bank. If training produces 100,000 patches, 10,000 are selected via greedy k-center coreset sampling. This compression dramatically reduces memory and inference time while preserving coverage of the normal feature space. The k-center algorithm provides a 2-approximation guarantee that no normal pattern is far from some representative.

The **neighborhood_size** of 3 defines a 3×3 averaging kernel applied to extracted features. Each patch feature is averaged with its 8 spatial neighbors, making representations more robust by incorporating local context. The value must be odd for symmetric padding.

The **use_faiss** flag enables Facebook AI Similarity Search for efficient nearest neighbor lookup. FAISS provides optimized implementations that scale well to large memory banks.

The **max_memory_samples** when set to null allows coreset_percentage to determine memory bank size. Setting a specific integer would cap the memory bank regardless of data size, useful for memory-constrained deployments.

#### preprocessing section

The **resize_small** [400, 400] applies to small object categories (engine_wiring, pipe_clip, pipe_staple). These components are physically compact and captured at closer range, so 400×400 resolution suffices.

The **resize_large** [1000, 750] applies to large object categories (tank_screw, underbody_pipes, underbody_screw). These require wider field of view or higher detail, hence larger images with asymmetric aspect ratio matching typical inspection viewpoints.

The **normalize_mean** [0.485, 0.456, 0.406] and **normalize_std** [0.229, 0.224, 0.225] are the ImageNet statistics used for image normalization. Since the backbone was pretrained on ImageNet, input images must be normalized to match the statistical properties the network expects. These specific values are the channel-wise means and standard deviations computed across the entire ImageNet training set.

#### training section

The **batch_size** of 32 determines how many images are processed together. Larger batches improve GPU utilization but require more memory. 32 is a common choice balancing efficiency and memory constraints.

The **num_workers** of 4 specifies parallel data loading threads. More workers can hide data loading latency but consume additional CPU and memory.

#### inference section

The **anomaly_threshold** set to null means the threshold will be computed from validation data rather than predetermined. In practice, thresholds are often determined by desired false positive rates.

The **num_neighbors** of 1 means anomaly scores are based on distance to the single nearest neighbor in the memory bank. Using k=1 maximizes sensitivity to novel patterns; larger k would smooth the response but might mask small anomalies.

#### seed

The value 42 ensures reproducibility. All random operations (data shuffling, coreset selection) use this seed so experiments can be exactly replicated.

---

## experiments/configs/federated/

### fedavg_iid_config.yaml

This configuration implements federated learning with IID (Independent and Identically Distributed) data partitioning. Each client receives a random uniform subset of the total data, representing the idealized scenario where all clients see statistically similar data distributions.

#### federated section

The **num_clients** of 5 creates five federated participants. This simulates five inspection stations or facilities collaborating without sharing raw data.

The **partitioning** "iid" specifies random uniform distribution. All sample indices are shuffled and divided into 5 roughly equal portions, so each client's subset is statistically representative of the whole dataset.

The **num_rounds** of 1 indicates a single communication round. For PatchCore with static training data, one round typically suffices since there is no iterative optimization. Multiple rounds would be relevant for streaming data scenarios.

#### aggregation section

The **strategy** "federated_coreset" applies k-center coreset selection to the concatenation of all client coresets. This ensures the global memory bank is a diverse, representative subset of the combined feature space.

The **global_bank_size** of 10000 sets the target for the aggregated memory bank. Regardless of how many patches clients contribute, the final global bank will contain 10,000 representative patches.

The **weighted_by_samples** being true means clients with more data have proportionally more representation in the global model. A client with 2000 samples contributes more to the aggregation than one with 500 samples.

The **oversample_factor** of 2.0 causes the aggregation to initially select twice the target size before final coreset selection. This provides more candidates for the final selection, improving diversity. The algorithm selects 20,000 patches first, then applies coreset to reduce to 10,000.

#### output section

The **dir** "outputs/federated/iid" specifies where results are saved.

The **save_memory_bank** flag ensures the global memory bank is persisted for later evaluation.

The **save_client_stats** flag saves per-client statistics (number of samples, coreset sizes, timing) for analysis.

### fedavg_category_config.yaml

This configuration implements non-IID partitioning based on product categories, simulating a realistic factory scenario where different stations inspect different components.

#### federated section (unique parameters)

The **partitioning** "category" activates category-based assignment instead of random distribution.

The **client_assignments** dictionary explicitly maps each client to specific categories. Client 0 receives only engine_wiring (Engine Assembly station). Client 1 receives underbody_pipes and underbody_screw (Underbody Line). Client 2 receives tank_screw and pipe_staple (Fastener Station). Client 3 receives only pipe_clip (Clip Inspection). Client 4 receives "all", indicating it samples from every category.

This creates highly non-IID distributions where most clients see only 1-2 categories while the QC client sees a mix. Such heterogeneity is challenging for federated learning since the global model must generalize across all categories despite clients having limited local views.

The **qc_sample_ratio** of 0.1 means the QC client (client 4) receives 10% of samples from each category before other clients receive their assignments. This ensures QC has representative coverage across all products for comprehensive quality control.

### fedavg_dp_config.yaml

This configuration adds differential privacy to the IID federated setup, providing formal privacy guarantees for the training data.

#### differential_privacy section

The **enabled** flag activates the privacy mechanism. When true, client coresets are sanitized before transmission to the server.

The **epsilon** of 1.0 is the primary privacy parameter. Mathematically, ε-differential privacy guarantees that for any two neighboring datasets differing by one sample, the probability ratio of any output is bounded by e^ε. Lower epsilon means stronger privacy but requires more noise, potentially degrading utility. The value 1.0 represents moderate privacy; stricter applications might use 0.1-0.5, while relaxed settings might use 5-10.

The **delta** of 0.00001 (1e-5) represents the probability of a catastrophic privacy failure. The (ε,δ)-differential privacy guarantee allows a δ probability that the ε bound is violated. This should be negligible, typically much smaller than 1/N where N is the number of training samples. With thousands of images, 1e-5 ensures the failure probability is vanishingly small.

The **clipping_norm** of 1.0 bounds the L2 norm of feature vectors before noise addition. The sensitivity of the mechanism depends on this bound: vectors with norms exceeding 1.0 are scaled down. The Gaussian noise standard deviation is then calibrated as σ = (clipping_norm × √(2 × ln(1.25/δ))) / ε. With the configured values, σ ≈ 1.0 × √(2 × ln(125000)) / 1.0 ≈ 4.8, meaning substantial noise is added to achieve the privacy guarantee.

The privacy mechanism works as follows: each client extracts features and builds a local coreset. Before sending to the server, feature vectors are clipped to have maximum L2 norm of 1.0, then Gaussian noise with the calibrated σ is added to each dimension. The resulting noisy coreset provides plausible deniability about whether any specific image was in the training set.

---

## Parameter Summary Table

| Parameter | Baseline | Fed IID | Fed Category | Fed DP |
|-----------|----------|---------|--------------|--------|
| backbone | wide_resnet50_2 | wide_resnet50_2 | wide_resnet50_2 | wide_resnet50_2 |
| coreset_ratio | 0.1 | 0.1 | 0.1 | 0.1 |
| num_clients | 1 | 5 | 5 | 5 |
| partitioning | N/A | iid | category | iid |
| global_bank_size | N/A | 10000 | 10000 | 10000 |
| DP enabled | No | No | No | Yes (ε=1.0) |
| num_rounds | N/A | 1 | 1 | 1 |

---

## How Configurations Relate to Code

The model section parameters are consumed by PatchCore (src/models/patchcore.py) and FeatureExtractor (src/models/backbone.py). The backbone name selects the pretrained network, layers determines which hooks are registered, coreset_ratio controls the greedy_coreset_selection call.

The federated section parameters are consumed by FederatedPatchCore (src/federated/federated_patchcore.py). The num_clients determines how many PatchCoreClient instances are created, partitioning selects IIDPartitioner vs CategoryPartitioner, client_assignments configures the CategoryPartitioner.

The aggregation section parameters are passed to FederatedServer (src/federated/server.py). The strategy selects from STRATEGY_REGISTRY, global_bank_size sets the target memory bank size, weighted_by_samples and oversample_factor tune the aggregation behavior.

The differential_privacy section parameters create a DPConfig dataclass passed to each client. The EmbeddingSanitizer (src/privacy/embedding_sanitizer.py) uses epsilon, delta, and clipping_norm to compute noise levels via the GaussianMechanism.

The preprocessing section parameters are used when creating transforms via get_transforms (src/data/preprocessing.py). The resize dimensions are category-specific, and normalization uses the specified ImageNet statistics.

---

## Choosing Configuration Values

When adapting these configurations for new experiments, consider the following guidance.

For **coreset_ratio**, lower values (0.05) reduce memory but risk missing normal patterns. Higher values (0.2) improve coverage but increase storage and inference time. The default 0.1 is empirically validated for AutoVI.

For **global_bank_size**, consider the expected feature diversity across clients. More categories or more clients may warrant larger banks (15000-20000). Memory-constrained deployments might reduce to 5000.

For **epsilon**, the choice depends on privacy requirements. Medical or financial data might require 0.1-0.5. Industrial data with less sensitivity might tolerate 2.0-5.0. Values below 0.1 typically add so much noise that utility degrades significantly.

For **num_rounds**, single-round training works for static datasets. If clients have streaming data or the aggregation benefits from iterative refinement, increase to 3-5 rounds.

For **oversample_factor**, higher values (3.0-4.0) provide more candidates for final selection but increase computation. Lower values (1.5) speed up aggregation but may reduce diversity.

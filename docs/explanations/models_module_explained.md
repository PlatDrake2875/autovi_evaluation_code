# Models Module Explained

## The Foundational Challenge of Anomaly Detection

The models module sits at the heart of this project and implements PatchCore, an algorithm designed to solve a fundamental paradox in industrial quality control: how do you teach a system to recognize defects when defects are rare and varied, while normal products are abundant? Traditional classification approaches require examples of both normal and anomalous samples, but in manufacturing, collecting comprehensive examples of every possible defect is impractical. A scratch on a car door could appear in countless locations, orientations, and severities. A dent could be shallow or deep, circular or elongated. The space of possible defects is essentially infinite.

PatchCore elegantly sidesteps this problem by learning only what "normal" looks like. During training, the system builds a comprehensive understanding of the visual patterns present in defect-free products. During inference, anything that deviates significantly from this learned normality is flagged as potentially anomalous. This approach, known as one-class classification or novelty detection, transforms the impossible task of enumerating all defects into the tractable task of characterizing normal appearance.

The three files in this module work together to implement this vision. The backbone.py file handles feature extraction, transforming raw images into rich numerical representations. The memory_bank.py file manages the storage and retrieval of representative normal features. The patchcore.py file orchestrates these components into a cohesive anomaly detection system.

---

## Feature Extraction: The backbone.py File

### Why Pre-trained Neural Networks Matter

The first insight underlying PatchCore is that modern deep neural networks, when trained on large-scale image classification tasks, learn features that transfer remarkably well to other visual tasks. A network trained to distinguish dogs from cats on ImageNet has learned, in its intermediate layers, general-purpose visual features like edges, textures, shapes, and object parts that are useful far beyond pet classification.

The code leverages this through a pre-trained WideResNet-50-2, a variant of the ResNet architecture that was introduced by researchers at Facebook AI. The "Wide" designation indicates that this network has channels widened by a factor of 2 compared to the original ResNet-50, meaning each convolutional layer processes more feature channels. The "50" refers to the depth of 50 layers, and the "2" confirms the widening factor. This architecture achieves strong performance on ImageNet while being computationally reasonable, making it an excellent choice for transfer learning in anomaly detection.

The decision to use ImageNet-pretrained weights rather than training from scratch is deliberate. Training a deep network from scratch requires millions of labeled examples, but transfer learning allows us to benefit from the visual knowledge encoded in the pretrained weights even when our target domain (automotive parts) differs from the source domain (ImageNet objects). Research has shown that lower-level features (edges, textures) transfer almost universally across visual domains, while higher-level features (object parts, semantic concepts) may require fine-tuning. PatchCore uses intermediate layers specifically because they capture the right level of abstraction for detecting manufacturing defects.

### The FeatureExtractor Class Architecture

The FeatureExtractor class wraps the pretrained WideResNet-50-2 and extracts features from its intermediate layers. The class inherits from PyTorch's nn.Module, following standard practice for neural network components.

During initialization, the class loads the pretrained backbone with ImageNet weights. Immediately after loading, all parameters are frozen by setting requires_grad to False. This freezing serves two purposes. First, it ensures that the carefully learned ImageNet features remain intact rather than being corrupted by gradients during any subsequent operations. Second, it significantly reduces memory consumption and computation during the forward pass since PyTorch need not maintain computation graphs for gradient calculation.

The layers parameter specifies which intermediate layers to extract features from, defaulting to layer2 and layer3. In ResNet terminology, the network is divided into four main stages (layer1 through layer4), each containing multiple residual blocks. Layer2 produces features at one-quarter of the original spatial resolution with 512 channels, while layer3 produces features at one-eighth resolution with 1024 channels. The choice to use these specific layers represents a carefully considered trade-off between semantic richness and spatial precision.

Layer1 features are too low-level, capturing mostly edges and simple textures that might be present in both normal and anomalous regions. Layer4 features are highly semantic but spatially coarse, having been downsampled to one-thirty-second of the original resolution, which makes precise localization of small defects difficult. Layers 2 and 3 occupy the sweet spot where features are abstract enough to capture meaningful visual patterns but spatially detailed enough to localize anomalies precisely.

### Forward Hooks for Intermediate Feature Capture

PyTorch's forward hook mechanism provides an elegant solution for extracting features from intermediate layers without modifying the original network architecture. A hook is a callback function that gets invoked automatically whenever a specified layer processes data during a forward pass.

The _register_hooks method iterates through the specified layer names and attaches a hook to each corresponding layer in the backbone. The _get_hook method creates a closure that captures the layer name and returns a hook function. When the hook is called, it receives the module, input, and output tensors, storing the output in the features dictionary keyed by the layer name.

During the forward pass, the code first clears the features dictionary to remove stale values from previous invocations. It then runs the input through the backbone within a torch.no_grad context, which disables gradient computation for efficiency since we are only performing inference. After the forward pass completes, the hook functions have populated the features dictionary with the intermediate representations.

### Multi-scale Feature Concatenation

The extracted features from layer2 and layer3 have different spatial resolutions due to the progressive downsampling in the ResNet architecture. Layer2 features have shape [B, 512, H/4, W/4] where B is the batch size, 512 is the channel count, and H/4 and W/4 indicate the spatial dimensions are one-quarter of the original image dimensions. Layer3 features have shape [B, 1024, H/8, W/8], with more channels but coarser spatial resolution.

To combine these multi-scale features, the code upsamples the layer2 features to match the spatial resolution of layer3 using bilinear interpolation. The interpolate function with mode="bilinear" computes new pixel values as weighted averages of the four nearest neighbors in the original feature map. The align_corners=False setting ensures consistent behavior where corner pixels of the input and output are not aligned, which is the recommended setting for most modern applications.

After spatial alignment, the features are concatenated along the channel dimension, producing a combined representation with 512 + 1024 = 1536 channels at the layer3 spatial resolution. This multi-scale concatenation allows the model to leverage both the finer spatial detail preserved in layer2 and the richer semantic information captured in layer3.

### Local Neighborhood Averaging

The apply_local_neighborhood_averaging function implements a crucial smoothing operation that makes patch features more robust by incorporating context from neighboring spatial locations. The intuition is that an isolated patch feature might be noisy or overly specific to minor pixel variations, but averaging with neighbors creates a more stable representation that captures local structure.

The function creates a uniform averaging kernel of size neighborhood_size × neighborhood_size, defaulting to 3×3. Each element in this kernel has value 1/9 (for the 3×3 case), ensuring that the output is a proper average rather than a scaled sum. The padding is computed as neighborhood_size // 2, which for a 3×3 kernel equals 1, ensuring that the output feature map has the same spatial dimensions as the input.

The implementation applies this averaging kernel to each channel independently using 2D convolution. The features tensor is reshaped from [B, C, H, W] to [B*C, 1, H, W] so that each channel can be processed as an independent single-channel image. After convolution with the averaging kernel, the result is reshaped back to [B, C, H, W].

This neighborhood averaging effectively increases the receptive field of each patch feature, allowing it to capture information from a slightly larger region of the image. For defect detection, this is valuable because defects often have spatial extent that might span multiple patches, and averaging helps create features that are aware of this local context.

### Patch Extraction

The reshape_features_to_patches function transforms the spatial feature map into a collection of individual patch feature vectors. Starting from the feature tensor of shape [B, C, H, W], it permutes the dimensions to [B, H, W, C] and then reshapes to [B*H*W, C]. Each row in the output represents a single spatial location's feature vector.

For a typical input image of 224×224 pixels processed through the backbone, the layer3 resolution would be 28×28 (one-eighth of the original), producing 784 patch features per image. With the default 1536-dimensional feature vectors, each image yields a patch feature matrix of shape [784, 1536].

The get_patch_locations helper function generates corresponding spatial coordinates for each patch, which can be useful for later visualization or localization of anomalies. It creates y and x coordinate tensors, stacks them, and expands across the batch dimension to produce location vectors for every patch.

---

## Memory Bank Management: The memory_bank.py File

### The Concept of a Memory Bank

The memory bank is PatchCore's mechanism for storing and retrieving representative normal patch features. During training, the model extracts patch features from all normal training images, but storing every single patch would be computationally prohibitive and largely redundant since many patches are visually similar. The memory bank uses coreset subsampling to select a diverse, representative subset that covers the feature space efficiently.

During inference, the memory bank provides the reference against which test patches are compared. An anomaly score for each test patch is computed as its distance to the nearest neighbor in the memory bank. If a test patch is far from all stored normal patches, it likely represents an anomalous visual pattern.

### The MemoryBank Class

The MemoryBank class manages the storage, indexing, and querying of patch features. It maintains the features as a NumPy array and optionally uses FAISS (Facebook AI Similarity Search) for efficient nearest neighbor lookup.

The constructor accepts three parameters. The feature_dim specifies the dimensionality of feature vectors (1536 in the default configuration). The use_faiss boolean determines whether to use FAISS for accelerated search, falling back to NumPy-based computation if FAISS is unavailable. The use_gpu boolean enables GPU acceleration within FAISS when hardware permits.

The class design separates the concerns of storage (the features array), indexing (the FAISS index), and querying (the search operations). This separation allows flexibility in how features are populated, whether through the fit method with coreset selection or directly via set_features for federated learning scenarios.

### The fit Method and Coreset Selection

The fit method is the primary entry point for building the memory bank from training features. It receives the full set of extracted patch features and reduces them to a manageable representative subset through coreset subsampling.

The target size for the memory bank can be specified in two ways. If max_samples is provided, it directly sets the maximum number of patches to retain. Otherwise, the coreset_ratio parameter determines what fraction of the input features to keep, with a default of 0.1 (10%). For a dataset producing 100,000 patches, this would select 10,000 representative patches.

The coreset subsampling is performed by the greedy_coreset_selection function (detailed below), which implements the k-center algorithm to select points that maximize coverage of the feature space. After selection, the chosen features are stored and the FAISS index is built.

### FAISS Indexing for Efficient Search

The _build_index method constructs a FAISS index for fast nearest neighbor search. FAISS, developed by Facebook AI Research, provides optimized implementations of similarity search algorithms that can handle millions of vectors efficiently.

The code creates an IndexFlatL2, which is FAISS's basic index performing exact L2 (Euclidean) distance search. While FAISS offers approximate search indices that are even faster for very large datasets, the exact search index ensures accuracy at scales typical for this application. The features must be contiguous float32 arrays for FAISS compatibility.

If GPU acceleration is requested and available, the index is transferred to the GPU using index_cpu_to_gpu. GPU-accelerated FAISS can achieve dramatic speedups for large-scale search operations, though for memory banks of tens of thousands of patches, the CPU implementation is often sufficient.

### Query Operations

The query method finds the k nearest neighbors for each query patch in the memory bank. It ensures the query features are properly formatted, then delegates to either the FAISS index or a NumPy fallback implementation.

The FAISS search returns two arrays: distances containing the L2 distances to the k nearest neighbors, and indices containing the corresponding indices in the memory bank. These arrays have shape [M, k] where M is the number of query patches.

The NumPy fallback implementation (_numpy_knn) computes all pairwise distances using the algebraic identity that ||a - b||² = ||a||² + ||b||² - 2⟨a, b⟩. This avoids explicit looping by leveraging broadcasting and matrix multiplication. After computing the full distance matrix, it uses argpartition for efficient partial sorting to identify the k smallest distances.

### Anomaly Score Computation

The get_anomaly_scores method provides a convenient interface for computing anomaly scores. It calls query with k=1 and returns only the distances to the nearest neighbors. These distances serve as anomaly scores: patches similar to normal training patches will have small distances, while anomalous patches will be far from any stored normal pattern.

The choice of using just the nearest neighbor (k=1) rather than averaging over multiple neighbors is deliberate. In anomaly detection, we want to identify patches that are different from any normal pattern. Using k=1 maximizes sensitivity to novel patterns, while larger k values would smooth out the response and potentially mask small anomalies.

---

## Greedy Coreset Selection

### The k-Center Problem

Coreset selection addresses the question of how to choose a small subset of points that best represents a larger dataset. The goal is to select m points from n candidates such that every candidate point is close to at least one selected point. This is formalized as the k-center problem: find a set S of k points that minimizes the maximum distance from any point to its nearest neighbor in S.

The k-center problem is NP-hard, meaning no polynomial-time algorithm is known to find the optimal solution. However, a simple greedy algorithm achieves a 2-approximation guarantee, meaning its solution is at most twice the optimal radius. This greedy algorithm is what the code implements.

### The Greedy Algorithm

The greedy_coreset_selection function implements the k-center greedy algorithm. It starts by selecting an arbitrary initial point (randomly chosen for reproducibility with the given seed). Then, it iteratively selects the point that is furthest from the current set of selected points, continuing until the target number of points is reached.

The algorithm maintains a min_distances array tracking each point's distance to its nearest selected neighbor. Initially, all distances are infinite. In each iteration, the algorithm computes distances from the most recently selected point to all candidates and updates min_distances to reflect any shorter paths discovered. The next point selected is the one with the maximum value in min_distances, meaning the point least well covered by current selections.

This greedy selection ensures good coverage of the feature space. Points in dense regions of the space will be represented by nearby selections, while outlying points in sparse regions will be selected early (when they have large distances to existing selections). The result is a diverse subset that spans the full range of normal visual patterns.

### GPU Acceleration with PyTorch

The _greedy_coreset_pytorch_gpu function provides GPU-accelerated coreset selection when CUDA is available. Moving the feature matrix to GPU and using torch.cdist for distance computation can dramatically speed up the selection process for large feature sets.

The PyTorch implementation follows the same logic as the CPU version but leverages GPU parallelism for the distance computations. The torch.cdist function efficiently computes pairwise distances between two sets of vectors using optimized CUDA kernels. The torch.minimum function performs element-wise minimum operations on GPU tensors.

The CPU fallback (_greedy_coreset_cpu) uses NumPy operations with explicit progress logging every 1000 iterations. For very large datasets, coreset selection can take significant time, so progress feedback helps users understand that computation is proceeding normally.

---

## The PatchCore Orchestrator: patchcore.py

### System Architecture

The PatchCore class serves as the main interface, orchestrating feature extraction and memory bank operations into a cohesive anomaly detection system. It exposes simple fit and predict methods that hide the complexity of the underlying pipeline.

The constructor initializes the feature extractor with specified configuration and prepares for memory bank creation. Key parameters include the backbone_name (defaulting to wide_resnet50_2), the layers to extract features from, the coreset_ratio controlling memory bank compression, the neighborhood_size for local averaging, and whether to use FAISS for nearest neighbor search.

The device parameter supports three modes: "auto" automatically selects GPU if available, falling back to CPU; "cuda" forces GPU usage; and "cpu" forces CPU usage. Automatic device selection is typically the best choice, allowing the system to leverage available hardware without manual configuration.

### The Training Pipeline: fit Method

The fit method builds the memory bank from a dataloader of normal training images. The dataloader should yield batches of defect-free images that represent the normal appearance of the product being inspected.

The method iterates through all training batches, extracting patch features from each. The extract_images_from_batch utility handles the case where dataloader batches might include labels or other metadata along with images, ensuring only the image tensors are processed. For each batch, the method stores the image size (for later anomaly map upsampling) and the feature map size (for reshaping anomaly scores), then extracts patches with neighborhood averaging.

All extracted patches are accumulated in a list and concatenated into a single array. This array is then passed to a new MemoryBank instance's fit method, which performs coreset selection and FAISS indexing. After fit completes, the PatchCore model is ready for inference.

### The Inference Pipeline: predict Method

The predict method computes anomaly scores for input images, returning both pixel-wise anomaly maps and image-level anomaly scores. The method accepts either PyTorch tensors or NumPy arrays.

First, it extracts patch features from the input images using the same process as training. These patches are then queried against the memory bank to obtain anomaly scores (distances to nearest neighbors). The scores are reshaped from a flat array into a spatial map matching the feature map dimensions.

The spatial anomaly map is then upsampled to the original image resolution using bilinear interpolation. This upsampling allows the anomaly map to be overlaid on the original image for visualization and ensures that anomaly localization is at pixel-level resolution despite the coarser resolution of the feature extraction process.

Finally, image-level anomaly scores are computed by taking the maximum value in each image's anomaly map. The maximum rather than average is used because a single highly anomalous patch is sufficient to indicate a defect, and averaging would dilute the signal from small, localized anomalies.

### Single Image Prediction

The predict_single method provides a convenient interface for processing individual images rather than batches. It handles various input formats including PIL Images, PyTorch tensors, and NumPy arrays. For PIL Images, it can optionally apply a transform for preprocessing; otherwise, it performs basic normalization (scaling to [0, 1] and transposing to channel-first format).

After ensuring proper tensor format, the method delegates to predict and extracts the single image's results from the batch output. This method is particularly useful for interactive applications or processing images one at a time.

---

## Parameter Definitions and Mathematical Details

### backbone_name (default: "wide_resnet50_2")

This parameter specifies the pretrained neural network architecture used for feature extraction. WideResNet-50-2 is a ResNet variant with 50 layers and width multiplier 2, meaning convolutional layer channel counts are doubled compared to the original ResNet. The architecture achieves 78.5% top-1 accuracy on ImageNet while remaining computationally tractable. Alternative backbones could be supported by extending the conditional logic in FeatureExtractor, but WideResNet-50-2 is standard in PatchCore implementations due to its favorable accuracy-efficiency trade-off.

### layers (default: ["layer2", "layer3"])

These are the ResNet stages from which features are extracted. In ResNet architecture, layers are grouped into four stages with progressive downsampling. Layer2 outputs features at 1/4 spatial resolution with 512 channels for WideResNet-50-2. Layer3 outputs at 1/8 resolution with 1024 channels. Using both layers captures multi-scale information: layer2 provides finer spatial detail while layer3 provides richer semantic features. The concatenated 1536-dimensional features balance localization precision with semantic richness.

### coreset_ratio (default: 0.1)

This parameter controls what fraction of extracted patches are retained in the memory bank. A value of 0.1 means 10% of patches are kept. Lower values reduce memory usage and query time but may miss important normal patterns. Higher values increase coverage but add computational overhead. The default of 0.1 is empirically found to preserve most normal patterns while achieving good compression. For a training set producing 100,000 patches, this yields a 10,000-patch memory bank.

### max_samples (optional)

When specified, this directly limits the memory bank size regardless of the coreset_ratio. This is useful for ensuring consistent memory usage across datasets of different sizes. If both max_samples and coreset_ratio would apply, the effective limit is the minimum of max_samples and the ratio-derived count.

### neighborhood_size (default: 3)

This specifies the kernel size for local neighborhood averaging applied to extracted features. A value of 3 means each patch feature is averaged with its 8 immediate spatial neighbors in a 3×3 region. This smoothing makes features more robust by incorporating local context, reducing sensitivity to minor pixel-level variations. The value must be odd to ensure symmetric padding. Larger values (like 5 or 7) would incorporate wider context but might blur fine-grained anomaly boundaries.

### seed (default: 42)

The random seed ensures reproducibility of coreset selection. The greedy algorithm starts from a randomly chosen initial point, and this seed controls that randomness. Using a fixed seed ensures that the same training data produces the same memory bank, which is important for reproducible experiments.

### use_faiss (default: True)

This boolean determines whether to use FAISS for nearest neighbor search. FAISS provides optimized implementations that are significantly faster than naive NumPy implementations, especially for large memory banks. The fallback to NumPy search is provided for environments where FAISS installation is difficult.

### use_gpu (default: False for MemoryBank, device-dependent for PatchCore)

Controls GPU acceleration for FAISS operations. GPU-accelerated FAISS can provide substantial speedups for very large memory banks but requires appropriate hardware and FAISS built with GPU support. For typical memory bank sizes in this application, CPU FAISS is often sufficient.

---

## The Complete Pipeline Flow

Understanding how data flows through the system illuminates how these components work together to detect anomalies.

During training, images of normal products are loaded through a dataloader in batches. Each image batch, typically of shape [B, 3, 224, 224] for B images with 3 color channels at 224×224 resolution, is passed to the feature extractor. The backbone processes these images through its layers, with hooks capturing the intermediate representations at layer2 and layer3. Layer2 features have shape [B, 512, 56, 56] and layer3 features have shape [B, 1024, 28, 28]. After upsampling layer2 to 28×28 and concatenating, the combined features have shape [B, 1536, 28, 28].

Neighborhood averaging smooths these features, maintaining the same shape but making each spatial location's feature vector incorporate information from its 3×3 neighborhood. The features are then reshaped to [B×784, 1536], where 784 = 28×28 is the number of spatial positions. After processing all training batches, the accumulated features might number in the hundreds of thousands.

Coreset selection reduces this to a manageable size. The greedy k-center algorithm selects points that maximize coverage of the feature space, producing a diverse subset. These selected features are stored in the memory bank, and a FAISS index is built for efficient retrieval.

During inference, a test image follows the same feature extraction path, producing 784 patch features of dimension 1536. Each patch is queried against the memory bank's FAISS index to find its nearest neighbor among the stored normal patches. The L2 distance to this nearest neighbor serves as the patch's anomaly score.

The 784 anomaly scores are reshaped to a 28×28 spatial map, then upsampled to 224×224 to match the input resolution. This creates a pixel-wise anomaly heatmap where high values indicate potentially anomalous regions. The maximum value across this map serves as the image-level anomaly score: if any patch is highly anomalous, the entire image is flagged.

---

## Integration with the Broader System

The models module provides the core anomaly detection capability upon which the rest of the system builds. The federated learning components in src/federated/ use these classes to perform distributed training, where multiple clients extract features locally and share their memory banks (or coreset subsets thereof) with a central server for aggregation.

The privacy module in src/privacy/ applies differential privacy mechanisms to the patch features before they leave each client, protecting sensitive information while preserving anomaly detection utility. The robustness module in src/robustness/ ensures that the aggregation process is resilient to malicious or corrupted client contributions. The fairness module in src/fairness/ evaluates whether the aggregated model performs equitably across different clients, product categories, and defect types.

All of these higher-level concerns ultimately depend on the models module's ability to extract meaningful features, store representative patterns, and compute anomaly scores. The clean separation of feature extraction (backbone.py), storage management (memory_bank.py), and orchestration (patchcore.py) facilitates this integration by providing well-defined interfaces that other modules can build upon.

---

## Summary of Key Insights

The PatchCore approach transforms anomaly detection from a classification problem requiring anomaly examples into a representation learning problem requiring only normal examples. By leveraging pretrained features from WideResNet-50-2, the system inherits rich visual understanding without task-specific training. Multi-scale feature extraction from layers 2 and 3 balances semantic richness with spatial precision.

Coreset selection using the greedy k-center algorithm efficiently compresses potentially hundreds of thousands of training patches to a manageable memory bank while preserving coverage of the normal feature space. The 2-approximation guarantee ensures that no normal pattern is far from some representative in the memory bank.

FAISS indexing enables efficient nearest neighbor search, making inference practical even with large memory banks. The simple distance-to-nearest-neighbor scoring provides interpretable anomaly scores: high scores indicate patches far from any normal pattern, suggesting potential defects.

The modular architecture separating feature extraction, memory management, and orchestration creates clean interfaces for integration with federated learning, privacy protection, robustness mechanisms, and fairness evaluation. This design enables the complete system to address the practical requirements of deploying anomaly detection in privacy-sensitive, distributed industrial settings.

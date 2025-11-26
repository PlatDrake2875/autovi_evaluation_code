# Comprehensive Code Architecture and Technical Guide
## A Deep Dive into Federated Anomaly Detection for Visual Inspection

## Executive Summary

This document provides an exhaustive technical explanation of the AutoVI Federated Anomaly Detection codebase, a sophisticated system designed to detect manufacturing defects in visual inspection images while preserving data privacy through federated learning. The project implements PatchCore, a state-of-the-art visual anomaly detection method, augmented with a federated learning framework that enables multiple factory inspection stations to collaboratively train a global anomaly detection model without centralizing sensitive visual data.

The system operates on the fundamental principle that anomalies in manufacturing are localized visual deviations from the normal appearance of products. Rather than using traditional supervised learning that requires extensive labeled datasets, PatchCore takes a non-parametric approach: it learns what "normal" looks like by examining only defect-free training images, then during inference it identifies regions that deviate significantly from this learned normality. The federated extension ensures that each factory station can participate in training using only its local data, which is then aggregated on a central server to create a model that benefits from all available data while never transmitting raw images across the factory network.

## Part 1: Understanding the Data Pipeline

### The AutoVI Dataset Structure and Organization

The foundation of any machine learning system is the data it operates on. The AutoVI dataset represents real-world visual inspection scenarios from automotive manufacturing environments. The dataset contains images of six distinct industrial components that are inspected on different assembly line stations. These components are engine wiring harnesses, pipe clips, pipe staples, tank screws, and underbody piping systems. Each of these components requires specialized inspection because defects manifest differently depending on the component type and its physical characteristics.

The dataset is organized hierarchically to reflect how inspection systems work in real manufacturing. At the top level, there is a directory for each component type. Within each component directory, the structure mirrors the stages of quality inspection. The training directory contains images exclusively from normal, defect-free parts. This represents the "ground truth" of what a good part looks like in various lighting conditions and orientations. The training set typically contains 300 to 500 images per component, providing sufficient diversity for the model to understand the natural variation in appearance of normal parts without overfitting to specific examples.

The test directory contains both normal parts and defective parts. The normal test images serve as negative examples during evaluation—we need to verify that the model doesn't falsely flag good parts as defective. The defective parts are further subdivided into two categories: structural anomalies and logical anomalies. Structural anomalies are physical defects like cracks, deformations, or material defects that are immediately obvious from visual inspection. Logical anomalies are more subtle—they might involve incorrect assembly, missing components, or misaligned parts where the physical structure is sound but the assembly is incorrect. This distinction is important because different detection approaches might excel at different anomaly types.

Each test image with an anomaly has an accompanying ground truth mask that precisely delineates the defective region. These masks are stored as multi-channel PNG images where each channel represents the mask for a specific type of defect or defect component. The code includes functionality to combine these channel masks using logical operations to create a unified ground truth that identifies any area containing any type of defect. This ground truth is essential for computing evaluation metrics like AUC-sPRO, which measures how accurately the model localizes defects in space.

Additionally, each component directory contains a `defects_config.json` file that provides metadata about the defects specific to that component. This configuration file can specify which defect types are critical, which are merely cosmetic, and any other domain-specific information relevant to quality control. The evaluation framework can use this configuration to weight defects differently or to generate component-specific reports.

### The AutoVIDataset Class and Its Core Functionality

The `AutoVIDataset` class, implemented in `src/data/autovi_dataset.py`, serves as the primary interface for loading and accessing data from the AutoVI dataset. Understanding this class is crucial because all training, evaluation, and analysis code works through this interface. When an `AutoVIDataset` is instantiated with a root directory path, it performs a directory scan to identify all available samples and build an internal index.

The initialization process is quite thorough. The class takes several important parameters. The root_dir parameter specifies where the dataset is located on disk. The categories parameter allows selective loading—you might choose to load only engine wiring components if you want to focus on a specific assembly line. The split parameter determines whether you want training data or test data. The transform parameter allows you to specify image preprocessing operations like resizing and normalization. Finally, the include_good_only parameter is particularly important for the test split—if set to True, it means the dataset will return only normal test images, which is useful for certain types of analysis.

During initialization, the dataset walks the directory structure and for each category, it locates the training or test directory depending on the requested split. For training splits, it only loads images from the "good" subdirectory. For test splits, it loads images from both the "good" directory and all defect-specific subdirectories. For each image file discovered, the dataset records the complete file path, whether it's a normal image (label 0) or defective (label 1), which category it belongs to, and for defective images, which specific defect type it represents. This metadata is stored as a list of tuples, where each tuple contains the image path, label, category, and defect type.

The dataset also loads the defects configuration from each component's defects_config.json file. This configuration provides domain-specific information about which defects are present in that component and any special handling rules. By loading this during initialization, the dataset can provide this information to downstream analysis code that might need it.

When you access a sample from the dataset using the indexing operator (e.g., dataset[42]), several operations occur. The dataset looks up the 43rd sample in its internal list (remember, indexing is zero-based), retrieves the path to the image file, and uses PIL (Python Imaging Library) to load the PNG file and convert it to RGB format. If any transforms were specified during initialization, they are applied in sequence. These transforms typically include resizing the image to the appropriate dimensions for the model, converting the PIL Image to a PyTorch tensor, and normalizing the pixel values using ImageNet statistics.

The dataset returns all this information as a dictionary containing the image tensor, the label, the category name, the defect type, and the original file path. By returning a dictionary rather than a tuple, the code is more readable and less error-prone—you can access dataset[0]["image"] rather than needing to remember that the image is the first element of a tuple. This design decision also makes it easier to add additional fields in the future without breaking existing code.

An important feature of the `AutoVIDataset` class is that it provides ground truth mask access through a separate method. When you call `dataset.get_ground_truth(index)`, it performs additional work to load the ground truth mask for a given test sample. The method constructs the expected path to the ground truth directory based on the image's category and defect type, then searches for all PNG files in that directory. Each PNG file represents a mask for a specific defect component, so the method loads all of them and combines them using logical OR operations—if any mask marks a pixel as defective, that pixel is marked as defective in the combined mask. This unified mask is what gets used for evaluation metric computation.

The class also provides convenience methods for statistical analysis. The `get_statistics()` method iterates through all loaded samples and computes counts organized by category, defect type, and label. Similarly, `get_category_indices()` returns the indices of all samples belonging to a specific category, which is useful for category-specific processing. These methods are essential for the data partitioning logic used in federated learning, allowing the system to understand data distribution and ensure balanced partitioning across clients.

### Image Preprocessing and the Importance of Proper Resizing

Image preprocessing is often overlooked in deep learning discussions, but it's critical for good results. The images in the AutoVI dataset come from real camera systems, and these cameras have different configurations depending on the inspection station. Small components like engine wiring harnesses are photographed with the camera positioned close to the part, resulting in high-resolution images of 400x400 pixels. Larger components like underbody pipes are photographed from a slightly greater distance, resulting in 1000x750 pixel images. This difference exists because the camera system is designed to capture consistent visual detail regardless of physical object size—the pixel-per-millimeter ratio is approximately constant.

The preprocessing pipeline must account for these different source resolutions. The helper functions, primarily implemented in `src/data/autovi_dataset.py` and used throughout the training scripts, define the appropriate target resolution for each category using the `get_resize_shape()` function. When an image is loaded, it's resized to its category-specific dimensions using PIL's resize function, which performs bilinear interpolation to smoothly downsample or upsample the image as needed.

After resizing, the image is converted to a PyTorch tensor and normalized. The normalization uses ImageNet statistics: each channel is centered by subtracting the mean ([0.485, 0.456, 0.406]) and scaled by dividing by the standard deviation ([0.229, 0.224, 0.225]). These specific values come from the ImageNet dataset used to pretrain the WideResNet-50-2 backbone network. This normalization is not arbitrary—it's essential because the pretrained backbone was trained on ImageNet images that were normalized using these same values. If you don't apply matching normalization, you're essentially asking the model to interpret pixel values it's never seen during training, which severely degradates performance.

The preprocessing pipeline in the training scripts is encapsulated in helper functions like `get_transforms()`. These functions return a PyTorch `transforms.Compose` object that chains together multiple preprocessing operations in sequence. The composition ensures that operations are applied in the correct order: resize first, then convert to tensor, then normalize. The ability to parameterize transforms and pass them to the dataset constructor makes it easy to experiment with different preprocessing strategies without modifying the core dataset code.

It's worth noting that during training, we typically use fixed preprocessing without data augmentation. In traditional supervised learning, augmentation (random rotations, crops, flips) helps prevent overfitting by showing the model diverse variations of the training data. In PatchCore, the feature extractor is frozen (not trained), so augmentation would only complicate training without providing regularization benefits. During evaluation, consistent preprocessing is crucial—you want to measure model performance on the actual data, not on data that's been augmented to be easier to classify.

### Data Partitioning Strategies for Federated Learning

The transition from centralized to federated learning fundamentally changes how we think about data. In centralized training, all data is pooled in one location, and the model learns from the aggregate. In federated learning, data is distributed across multiple clients, and we need to decide how to partition the available data. The codebase implements two distinct partitioning strategies, each reflecting different real-world scenarios.

#### IID Partitioning Strategy

IID (Independent and Identically Distributed) partitioning is implemented in the `IIDPartitioner` class within `src/data/partitioner.py`. This strategy starts by gathering indices of all samples in the dataset. It then shuffles these indices using a seeded random number generator to ensure reproducibility. After shuffling, the indices are divided into equal-sized chunks using NumPy's array_split function, with each chunk assigned to a different client.

The key characteristic of IID partitioning is statistical homogeneity. If the overall dataset has 60% engine wiring and 40% tank screw images, then with IID partitioning, each client will receive approximately the same percentage of engine wiring and tank screw images. Similarly, if the overall dataset has a 70-30 split between normal and defective images, each client receives roughly this same split. This means every client has a representative sample of the overall data distribution.

IID partitioning represents an idealized scenario that doesn't typically occur in real manufacturing. However, it's valuable scientifically because it allows researchers to measure federated learning's overhead compared to centralized training. If federated learning performs worse on IID data, it suggests fundamental limitations of the federated approach. If it performs similarly to centralized learning on IID data, then differences on non-IID data reflect the challenge of handling heterogeneous distributions rather than limitations of federation itself.

The practical implementation is straightforward. The `partition()` method returns a dictionary mapping each client ID to a list of sample indices assigned to that client. The `create_subsets()` method goes further, creating `AutoVISubset` objects that wrap the original dataset and provide access only to the assigned indices. An `AutoVISubset` is a lightweight wrapper that doesn't duplicate data—it just stores indices and delegates access to the underlying dataset.

#### Category-Based Non-IID Partitioning

The `CategoryPartitioner` class implements a more realistic scenario. In a real manufacturing facility with multiple inspection stations, different stations specialize in inspecting different component types. Station A inspects engine wiring, Station B inspects tank screws, and so on. This means different stations naturally accumulate different data distributions.

The `CategoryPartitioner` class takes a parameter called `client_assignments`, which is a dictionary mapping client IDs to lists of component categories. For example, client 0 might be assigned ["engine_wiring"], meaning it gets all engine wiring images. Client 1 might be assigned ["tank_screw", "pipe_staple"], meaning it gets both tank screw and pipe staple images. This simulates a manufacturing line where some stations specialize narrowly while others handle multiple component types.

One particularly important feature is the quality control client, which receives a small sample from all categories. In realistic manufacturing, quality control stations serve as a check on line-specific stations. By including a QC client that gets a representative sample from all categories, we simulate this real-world pattern. The `qc_sample_ratio` parameter controls how much data the QC client receives from each category—typically 10%, meaning the QC client gets 10% of engine wiring images, 10% of tank screw images, and so on.

The partitioning process for category-based partitioning is more complex than IID partitioning. First, the code builds a mapping from each category to all sample indices belonging to that category. It iterates through all samples and groups indices by category. Then, it performs a two-pass assignment. In the first pass, it identifies any clients assigned to "all" categories (the QC clients) and samples from each category to provide to these clients. It's crucial to do this first because the QC client must receive its data before other clients claim it. In the second pass, it assigns remaining data to specialized clients. For example, the client assigned to ["engine_wiring"] gets all unclaimed engine wiring images.

This partitioning strategy is intentionally non-IID. Different clients have fundamentally different data distributions, which creates interesting challenges for federated learning. A model trained on only engine wiring images might develop features specific to that component, while a model trained on tank screws develops different features. The aggregation phase must synthesize these into a coherent global model that works for all components, which is more challenging than averaging models trained on homogeneous data.

#### Computing and Analyzing Partition Statistics

After partitioning, the system computes detailed statistics to understand the resulting data distribution. The `compute_partition_stats()` function analyzes a partition and produces a comprehensive report. For each client, it counts the total number of samples assigned, breaks down the count by category, and tracks the label distribution (how many normal vs. defective samples). These statistics are crucial for understanding whether the partitioning is balanced and for interpreting experimental results.

For example, with category-based partitioning on the full AutoVI dataset, statistics might show that Client 0 has 400 engine wiring images (all normal in training), Client 1 has 550 total images from underbody components (all normal in training), Client 2 has 350 images from fastener components, and so on. These imbalances are realistic and important—they mean some clients have more data to contribute to training while others have less.

The partition statistics are saved to JSON files for reproducibility. If you save the exact partition and statistics used in an experiment, others can reproduce your exact data split, ensuring that results are comparable and verifiable. This reproducibility is crucial in research because it allows others to attribute performance differences to algorithmic choices rather than random variation in data partitioning.

## Part 2: The PatchCore Anomaly Detection Model - A Deep Technical Exploration

### Foundational Concepts of Non-Parametric Anomaly Detection

PatchCore represents a paradigm shift from traditional supervised anomaly detection. In supervised approaches, you would collect thousands of labeled defects, train a neural network to classify images as normal or defective, and deploy this classifier. This approach has several limitations. First, you need extensive labeled data with diverse defect types, which is expensive to collect and label. Second, when new types of defects emerge that weren't in your training data, the classifier often fails to detect them because it hasn't learned those patterns. Third, the model learns to explicitly recognize defects, which makes it difficult to understand what made the model classify something as defective.

PatchCore takes a fundamentally different approach: instead of learning to recognize defects, it learns to recognize normality. The model is given only images of normal, defect-free parts during training. It learns to extract features from these normal images and builds a database (memory bank) of what normal features look like. During inference, when presented with a test image, the model extracts its features and compares them against the memory bank. If the test features match features in the memory bank, the image is classified as normal. If the test features don't match anything in the memory bank, the image is classified as defective.

This approach has several advantages. First, it requires only normal training data, which is abundant in manufacturing—factories have plenty of images of good parts from their normal operations. Second, it's robust to novel defect types because any defect that creates unusual features will be detected, even if that exact defect never appeared in training data. Third, by examining similarities to specific features in the memory bank, the system can explain why it flagged something as defective—it identifies the most similar normal features and can show a visual explanation.

The key insight is that at the patch level (looking at small regions of images), defects create visual patterns that deviate from normal variations. By building a comprehensive database of what normal patch-level features look like and comparing test patches against this database, the system can reliably detect anomalies without ever seeing examples of those anomalies during training.

### Deep Dive into the Feature Extraction Backbone Network

The backbone network is the foundation of the entire system, and understanding how it works is essential to understanding PatchCore. The system uses WideResNet-50-2, a convolutional neural network architecture developed by researchers at Facebook AI Research. This network was pretrained on ImageNet, which contains millions of images spanning thousands of object categories. Pretraining on ImageNet gives the network a general understanding of visual features—edges, textures, shapes, and object parts—that transfers well to many other visual domains, including manufacturing inspection.

The network has a residual structure, meaning information flows both through sequential convolutional layers and through "shortcut" connections that skip layers. These shortcuts allow the network to be much deeper than traditional architectures without suffering from training difficulties. WideResNet specifically modifies ResNet by increasing the width (number of feature channels) rather than just the depth (number of layers), which often gives better performance on modern hardware.

The network consists of multiple stages, each progressively reducing spatial resolution while increasing semantic complexity. When an image enters the network, it first passes through a convolutional layer that reduces resolution from the input size (e.g., 400x400 or 1000x750) to about 1/4 of the original size. This first stage extracts low-level features like edges and small textures. The image then flows through layers grouped into blocks, with layer2 and layer3 being the stages particularly relevant for PatchCore.

Layer 2 of the ResNet architecture processes features at 1/4 of the original image resolution. It produces 512 feature channels, with each channel encoding some aspect of the visual content. For an input image of 400x400 pixels, layer 2 output is 100x100x512. Each of the 100x100=10,000 spatial locations has 512 values representing different aspects of what's visible in that region of the image. These features encode relatively early-level information about local patterns.

Layer 3 operates on features at 1/8 of the original image resolution (50x50 for a 400x400 input) and produces 1024 feature channels. This deeper layer has seen more context from the image, so its features encode higher-level semantic information. Layer 3 features are more abstract but also more informative about what type of object or pattern is present. By combining layer 2 and layer 3, we get a complementary set of features—layer 2 provides detailed local information while layer 3 provides broader semantic context.

The feature extraction process is implemented in the `FeatureExtractor` class in `src/models/backbone.py`. When initialized, the class loads the pretrained WideResNet-50-2 network from PyTorch's model zoo. The class uses PyTorch's hook mechanism to intercept the output of specific layers during the forward pass. A hook is a function that PyTorch calls whenever data flows through a particular layer, allowing you to capture intermediate activations. The `FeatureExtractor` registers hooks on layer2 and layer3, which saves their outputs into a dictionary as the image propagates through the network.

When you call the forward method with an image, the backbone network processes the image completely, and the hooks capture layer 2 and layer 3 outputs. The `FeatureExtractor` then retrieves these captured outputs and upsamples layer 2 to match layer 3's spatial dimensions using bilinear interpolation. Layer 2 is upsampled from 1/4 resolution to 1/8 resolution to match layer 3. Finally, the two feature maps are concatenated along the channel dimension, producing a single tensor with 512+1024=1536 channels.

Importantly, the backbone network's weights are frozen—they are not updated during training. PatchCore doesn't retrain the network; it uses the network exactly as pretrained. This is fundamentally different from traditional transfer learning where you would fine-tune the network on your specific domain. By not training, PatchCore is simpler, faster, and more consistent—the feature extraction is identical regardless of the training data distribution.

### Local Neighborhood Averaging - Spatial Context Integration

After feature extraction, the system applies a spatial processing operation called local neighborhood averaging, implemented in the `apply_local_neighborhood_averaging()` function in `src/models/backbone.py`. This operation enhances the feature maps by aggregating information from spatially neighboring patches.

The operation works by applying a 3x3 convolutional filter where all weights are equal (each weight is 1/9, so they average to a uniform value). This filter slides across the feature map, and at each position, the output is the average of the 3x3 neighborhood of inputs. For a feature map that's 50x50x1536, the operation independently processes each of the 1536 channels, producing output that's the same shape.

Why is this averaging beneficial? Several reasons. First, features in neighboring spatial regions often encode similar types of information. An anomaly in a particular region often affects multiple neighboring patches. By averaging with neighbors, we aggregate information from a small region, making the features more robust to small spatial perturbations. Second, the averaging acts as a noise filter. If one patch has noisy features due to image noise or compression artifacts, averaging with neighbors provides a cleaner signal. Third, it creates spatial consistency—neighboring patches with similar content will have more similar features after averaging, which makes anomaly detection more spatially coherent.

The implementation uses PyTorch's convolution function with grouped convolution mode, where each channel is filtered independently. This is more efficient than processing channels separately in a loop. The padding parameter ensures that the output size matches the input size by padding the boundaries of the feature map with zeros.

The neighborhood size is a hyperparameter. The default is 3, meaning a 3x3 neighborhood. Larger neighborhoods (e.g., 5x5) would aggregate information from a larger region but might blur together features from different object parts. Smaller neighborhoods (1x1, no averaging) preserve fine-grained spatial information but might be more sensitive to noise. The default of 3 is a reasonable balance between these trade-offs, but it can be adjusted based on the specific inspection task—if defects are very small and precise localization is critical, use size 1; if defects are large and spatial smoothing helps, use size 3 or larger.

### Memory Bank Construction and the Coreset Subsampling Algorithm

After features are extracted and smoothed, the system must decide which features to store in the memory bank. With 400 training images and feature maps of 50x50, there are 400×50×50=1,000,000 patch features. Storing all of these would require gigabytes of memory and make inference slow. The solution is coreset subsampling—selecting a representative subset of all patches.

The coreset subsampling is implemented in the `greedy_coreset_selection()` function in `src/models/memory_bank.py`. The function is greedy in the sense that it makes locally optimal choices that lead to a good (though not necessarily optimal) global solution. The specific algorithm is a k-center algorithm, which aims to select points that are well-distributed across the feature space.

The algorithm begins by randomly selecting one feature vector as the seed. It then enters an iterative loop that runs (target_size - 1) times, where target_size is the desired number of selected features (e.g., 100,000 if we want to keep 10% of 1 million features). In each iteration, the algorithm computes the distance from every feature vector to all previously selected feature vectors. For each feature vector, it finds the distance to its nearest selected neighbor. It then identifies the feature vector with the maximum distance to its nearest selected neighbor—this is the point that's furthest from the currently selected set. This point is added to the selected set because it maximizes coverage—it fills gaps in the feature space that aren't yet represented.

This greedy approach ensures good coverage of the feature space. Features distributed across different regions of the space are selected, ensuring that the memory bank represents the diversity of normal patches. Features in dense regions of the space (common patterns) are selected with lower probability because their region is already well-represented by previously selected neighbors.

The algorithm's computational cost is important to understand. Computing distances from one feature to all others takes O(d) time where d is the feature dimension (1536 in our case). The outer loop runs target_size times, and in iteration i, we must compute distances from all n-i remaining features to the i selected features, taking O(n×i×d) time in the inner loop. Over all iterations, this is roughly O(n²×d) time total. For n=1,000,000 and d=1536, this could be very slow. The implementation includes progress reporting every 1000 iterations and uses NumPy's vectorized operations for efficiency, but this is still the bottleneck in training for large datasets.

Once subsampling is complete, the selected patches are stored as a NumPy array where each row is a 1536-dimensional vector. This array is the "memory bank"—the collection of normal patch features that the model has learned to recognize. The memory bank is typically much smaller than the original feature set, reducing memory requirements and making inference fast.

### FAISS Indexing and Nearest Neighbor Search

With the memory bank constructed, the system needs a way to quickly find the nearest neighbor of any query feature vector. A naive approach would compute distances from the query to all memory bank features, taking O(d) time per memory bank feature. With 100,000 features and inference on images with thousands of patches, this adds up to billions of distance computations.

FAISS (Facebook AI Similarity Search) solves this problem through intelligent indexing. FAISS builds specialized data structures that allow fast approximate nearest neighbor search. In the PatchCore implementation, we use the simplest FAISS index type: `IndexFlatL2`, which creates an exact index using L2 (Euclidean) distance. Despite the name "flat," this index uses optimized linear algebra libraries to compute distances very efficiently in batch mode. When you query with multiple features at once, FAISS uses vectorized operations to compute distances from all query features to all database features simultaneously, achieving speedups of 10-100x compared to naive looping.

The FAISS implementation in `src/models/memory_bank.py` handles both CPU and GPU operation. The code ensures that features are in the correct format (float32, contiguous in memory) before building the index. If GPU acceleration is enabled, the index is moved to GPU using FAISS's GPU support. GPU operation is particularly beneficial for large memory banks and high-throughput inference.

When the memory bank is saved to disk and later loaded during inference, the FAISS index is rebuilt. This ensures consistency—if the index structure were saved and loaded, it might become corrupted or misaligned with the actual feature data if the code was updated. By rebuilding the index on load, we ensure correctness.

### Anomaly Score Computation and Upsampling

During inference, the system processes a test image through the same feature extraction pipeline as training: extract features from layer 2 and layer 3, concatenate them, apply neighborhood averaging, and reshape into patch vectors. For an image of 400x400 pixels, this produces 50x50=2500 patch vectors.

For each patch vector, the system queries the FAISS index to find the nearest neighbor in the memory bank. The distance to this nearest neighbor is the anomaly score. The rationale is intuitive: if a patch is very similar to patches in the memory bank, it looks like a normal patch (low distance). If a patch is dissimilar to all memory bank patches, it's unusual and potentially anomalous (high distance).

The anomaly scores form a 50x50 anomaly map showing which regions of the feature space contain anomalous content. However, the original image is 400x400 pixels, and users need to know where in the original image anomalies are located. The system upsamples the 50x50 anomaly map back to 400x400 pixels using bilinear interpolation, implemented in `src/models/patchcore.py` using PyTorch's `F.interpolate()` function.

Bilinear interpolation estimates values at new pixel locations based on weighted averages of nearby original values. When upsampling from 50x50 to 400x400 (an 8x expansion factor), each original pixel maps to an 8x8 region of output pixels. The interpolation ensures smooth transitions between regions rather than abrupt jumps, which matches human intuition about where defects are located. If the anomaly map shows high scores at spatial position (25,25), after upsampling, these high scores spread smoothly across the corresponding region of the original image, which is roughly the center-right region.

Additionally, the system computes an image-level anomaly score as the maximum value in the anomaly map. This gives a single number representing the maximum anomaly detected anywhere in the image. Images with high maximum scores are likely to contain defects, while images with low maximum scores are likely normal. This image-level score is useful for classification tasks where you just need to know if an image is defective, without needing to know where the defect is.

## Part 3: Federated Learning Framework - Distributed Training Architecture

### Conceptual Foundation of Federated Learning

Federated learning is a paradigm for training machine learning models across distributed data sources while preserving privacy. In traditional centralized machine learning, all data is aggregated to a single location where the model is trained. In contrast, federated learning keeps data at its source—in our case, at different factory inspection stations—and only aggregates model updates.

The motivation for federated learning in this context is multifaceted. First, there are privacy concerns. Raw images from factory inspection systems may contain sensitive information about manufacturing processes, product designs, or supplier information. Centralizing this data increases risk of data breaches. Second, there are practical concerns. Transferring thousands of high-resolution images from multiple stations to a central server requires significant bandwidth and introduces latency. Third, there are regulatory concerns. Data locality regulations in many jurisdictions require that certain data remain on-premise rather than being transmitted to central servers.

The key insight of federated learning is that you don't need to transfer raw data to train a model. You can instead train a model locally using local data, extract a summary or update from that local training, send only the summary to a central server, and aggregate summaries from all local training to create a global model. In the specific case of PatchCore, the "summary" is the local memory bank (coreset), which is just feature vectors, not original images. Feature vectors are much smaller than images (1536 numbers vs. millions of pixels) and don't reveal the original visual content, providing privacy benefits.

### Client-Server Architecture and Communication Protocol

The federated system follows a classic client-server architecture. Clients are factory stations that have local training data. The server is a central aggregation point that coordinates training and maintains the global model. The architecture is implemented across multiple Python classes that work together.

In the implementation, the server doesn't physically exist as a separate machine during training—it's another Python object in the same process that coordinates the training. In a real deployment, the server would be on a separate machine or cloud service, and clients would communicate via network protocols. However, the logical structure is the same.

Communication flows as follows: the server broadcasts parameters to all clients (initially, the parameters are backbone network weights, which are identical across all clients since we use pretrained weights). Each client then performs local training using its local data partition. During local training, the client extracts features and builds a local coreset. After local training, the client sends its local coreset to the server. The server aggregates all received coresets from all clients into a global memory bank and broadcasts this global memory bank back to all clients.

This communication protocol is explicitly implemented in the `FederatedPatchCore` class in `src/federated/federated_patchcore.py`. The `train()` method orchestrates the overall process, calling `_train_single_round()` for each training round. Within a single round, the code iterates through all clients, calling each client's `extract_and_build_coreset()` method. The clients return their local coresets, which are collected and passed to the server's `receive_client_coresets()` method. The server then aggregates using `aggregate()`.

The communication of features instead of raw data provides privacy benefits. An attacker with access to feature vectors cannot reconstruct the original images. While recent research has shown that features can leak some information about training data, the privacy protection is substantial compared to transmitting original images. The system could further enhance privacy by adding differential privacy noise to feature vectors before transmission, but the current implementation doesn't include this.

### Client Implementation and Local Training Process

Each client is represented by a `PatchCoreClient` object, defined in `src/federated/client.py`. When created, a client is given a unique ID, references to the shared backbone network, and parameters like the coreset ratio and neighborhood size.

The client's primary responsibility is implemented in the `extract_and_build_coreset()` method. This method takes a DataLoader providing access to the client's local training data. The client iterates through all batches in the DataLoader, passing images through the backbone to extract features. The client accumulates all extracted patch features in a list. After processing all batches, the client has a large collection of feature vectors representing all patches in all training images assigned to this client.

The client then applies coreset subsampling to this feature collection using the same greedy k-center algorithm used in centralized training. The subsampling ratio is a local parameter—each client independently selects some fraction (e.g., 10%) of its features to retain. This is different from centralized training, where we also apply a coreset ratio, because each client only sees its own data distribution and must make its own subsampling decision. A client with diverse data (like the QC station in category-based partitioning) might need a higher coreset ratio to maintain coverage of its diverse data. A client with homogeneous data (like the engine wiring station) might use a lower ratio since its data is more uniform.

The local coreset is then returned to the orchestrator. The coreset includes feature vectors but not the original images or any information about which image each feature came from. This preserves privacy while still providing sufficient information for aggregation.

Throughout this process, the client's feature extractor is frozen—it never updates its weights. The client is not learning to extract better features; it's using the pretrained features as-is. This is fundamentally different from federated learning approaches like Federated Averaging, where each client trains a model and sends weight updates. In PatchCore, there are no weights to update—the client is purely extracting features and summarizing them.

### Server Implementation and Global Aggregation

The server, implemented in `FederatedServer` in `src/federated/server.py`, has the responsibility of coordinating aggregation. When a round of training begins, the server receives coresets from all clients via the `receive_client_coresets()` method. This method simply stores the received coresets and any client statistics.

The aggregation happens in the `aggregate()` method. The method concatenates all received client coresets into a single large feature set. If there are 5 clients, each with 10,000 features in their local coreset, concatenation produces 50,000 features. The server then applies final coreset subsampling to reduce this back to a target global size (e.g., 10,000 features).

The aggregation strategy can vary. The default "federated_coreset" strategy respects client contributions by weighting. If the system is configured to weight by samples, clients with more training data contribute proportionally more features to the global coreset. For example, if Client A has 1000 training images and Client B has 500 training images, Client A contributes roughly twice as many features as Client B to the global coreset. This is implemented by repeating each client's coreset indices based on its weight before applying the final k-center selection. Alternative strategies exist: "simple_concatenate" treats all features equally regardless of client size, and "diversity_preserving_aggregate" uses different selection criteria to maximize feature space coverage.

After aggregation, the server builds a FAISS index on the global features to enable fast inference. The global memory bank is then ready for use. The server doesn't need to send this back to clients explicitly in the implementation—inference happens on the server-side or on a client-side copy, but the architecture is flexible.

The aggregation process is where the federated learning's key benefit emerges: by aggregating coresets from multiple clients, the global memory bank sees features from all data distributions. A global model trained on aggregated coresets from category-based partitioning sees examples from all component types, even though individual clients only saw one or two types. This should lead to a more robust model than any individual client could achieve.

### FederatedPatchCore Orchestrator - Main Coordination Logic

The `FederatedPatchCore` class in `src/federated/federated_patchcore.py` is the master orchestrator that brings everything together. When initialized, it creates a specified number of `PatchCoreClient` objects, all sharing the same `FeatureExtractor`. It also creates a `FederatedServer`. These components are then coordinated through high-level methods.

The `setup_clients()` method sets up the data partitions. It takes a dataset and a partitioning strategy, creates an appropriate partitioner, and applies it to the dataset. The resulting partition (mapping of client IDs to sample indices) is stored for later use. The partition statistics are computed and reported, showing how many samples each client has and their composition by category.

The `train()` method is the main training orchestrator. For each training round, it calls `_train_single_round()`, which performs a complete round of federated training. The method uses Python's time module to measure elapsed time and tracks statistics. If a checkpoint directory is specified, it saves snapshots of the model after each round, enabling later recovery or analysis of convergence over rounds.

Within `_train_single_round()`, the code performs three distinct phases. In Phase 1 (Local Client Processing), it iterates through all clients and calls their `extract_and_build_coreset()` method. It collects the returned local coresets and any client statistics. In Phase 2 (Server Aggregation), it sends all coresets to the server and calls the server's `aggregate()` method to produce the global memory bank. In Phase 3 (Broadcasting), it broadcasts the global memory bank back to clients, though in the current implementation this is implicit through shared state.

The `predict()` method enables inference using the trained federated model. It takes images, extracts their features using the global feature extractor, queries the global memory bank for anomaly scores, and returns both pixel-level anomaly maps and image-level anomaly scores. This is identical to inference with a centralized PatchCore model, which makes it easy to compare federated and centralized approaches.

### Multi-Round Federated Training

While the current configuration often uses only 1 federated training round, the architecture supports multiple rounds. In multi-round training, after creating the global memory bank in round 1, the system can initialize round 2 using the round 1 global model and then extract features using this updated model. However, in the current implementation, the feature extractor (backbone) is never updated—it remains pretrained throughout all rounds. What could be updated is the coreset selection or aggregation strategy.

Multi-round training could be used to implement an iterative refinement approach: in round 1, build a global model from all client coresets. In round 2, clients could use this global model to identify which of their features are most useful or novel, then resubmit refined coresets. However, this isn't currently implemented. The multi-round support is architectural, allowing easy implementation of such refinements in the future.

## Part 4: Configuration System and Experimental Setup

### YAML Configuration Files and Hierarchical Parameter Management

The project uses YAML (YAML Ain't Markup Language) files to specify all experimental parameters. YAML is human-readable structured data format that's easier to work with than XML or JSON for configuration. A YAML file uses indentation to indicate hierarchy, and colons separate keys from values.

The baseline configuration file, `experiments/configs/baseline/patchcore_config.yaml`, specifies all parameters for training a single centralized PatchCore model. The file is organized into logical sections. The `model` section specifies the neural network backbone ("wide_resnet50_2"), which layers to extract features from (["layer2", "layer3"]), the coreset percentage to keep (0.1, or 10%), the neighborhood size for averaging (3), and whether to use FAISS (true). The `preprocessing` section specifies image resizing dimensions for small and large objects, the ImageNet normalization statistics, and any other preprocessing parameters. The `training` section specifies batch size (32) and number of data loading workers (4). The `inference` section specifies inference parameters like number of neighbors to use in k-NN scoring. The `seed` section specifies the random seed for reproducibility.

The federated configuration file, `experiments/configs/federated/fedavg_category_config.yaml`, extends the baseline with federated-specific sections. The `federated` section specifies the number of clients (5), the partitioning strategy ("category"), the number of training rounds (1), client-to-category assignments mapping each client ID to a list of categories, and QC sample ratio (0.1). The `aggregation` section specifies the aggregation strategy ("federated_coreset"), the target global memory bank size (10000), whether to weight aggregation by sample count (true), and an oversample factor for coreset selection.

These configuration files are loaded by the training scripts using a `load_config()` utility function that reads the YAML file and returns a Python dictionary. The script can then access configuration values using nested dictionary lookups like `config["federated"]["num_clients"]`.

### Training Script Architecture and Parameterization

The training scripts, particularly `scripts/train_federated.py`, are designed for flexibility and reproducibility. The script's typical flow is as follows. First, it parses command-line arguments that allow overriding configuration file values. This provides flexibility—you can use a configuration file for most parameters but specify a different random seed from the command line without editing the file. The script loads the configuration file and merges in any command-line overrides.

The script then performs setup operations. It sets random seeds (both NumPy and PyTorch) to ensure reproducible results. It creates output directories with timestamps to ensure different experiment runs don't overwrite each other. It prints a detailed header showing all configuration values, making it easy to verify what parameters were actually used.

Next, the script loads the dataset from disk. It scans the directory structure and builds the in-memory sample index. It then creates data partitions according to the specified strategy, computing partition statistics to understand the resulting distribution.

The script creates PyTorch DataLoaders for each client, wrapping the dataset partition in a loader that handles batching and potentially multiprocessing data loading. The number of workers for data loading can be configured, trading off between memory usage (more workers use more memory) and I/O performance (more workers enable parallel loading).

The script initializes the FederatedPatchCore system with all the parameters specified in configuration. It then calls the training method, which executes the actual federated learning process. Throughout training, the system prints progress information showing which round is running, how many features each client extracted, the size of the aggregated global memory bank, and elapsed time.

After training completes, the script saves the resulting model and all statistics to the output directory. It prints a summary showing the achieved global memory bank size, feature dimension, and other relevant statistics. This summary makes it easy to verify that training completed successfully and to understand the resulting model.

### Reproducibility Through Seeding and Deterministic Operations

Reproducibility is a core principle throughout the codebase. All random number generation is controlled through random seeds. The scripts accept a seed parameter, and this seed is passed to multiple random number generators: NumPy's random seed, PyTorch's random seed, and Python's built-in random module. This ensures that given the same seed and the same input data, the results are identical across multiple runs.

The coreset subsampling algorithm uses its own seeded random state to ensure determinism. When you run the algorithm with the same seed on the same features, you get the same subset selected every time. This determinism extends to the FAISS index construction—given the same features, the index is identical.

For reproducibility, experiments should document and save the random seed used. This is done automatically by the training scripts, which write the seed to the configuration saved in the output directory. Someone reproducing your results can use the exact same seed and get results within numerical precision of the original.

This reproducibility is essential for research. Without it, you can't confidently claim that one algorithm is better than another—differences might just be due to random variation in partitioning or subsampling rather than algorithmic differences.

### Data Generation and Preprocessing Utilities

The project includes utility functions for common preprocessing operations. The `get_transforms()` function returns a composed PyTorch transform that resizes images to category-appropriate dimensions, converts to tensor, and normalizes using ImageNet statistics. These transforms are applied to images as they're loaded from disk, ensuring consistent preprocessing.

The `get_resize_shape()` function encodes knowledge about appropriate image sizes for each category. Small objects get resized to 400x400, large objects to 1000x750. This categorization could be extended with additional category-specific preprocessing if needed (e.g., different augmentation for different component types).

The `generate_partitions.py` script can pre-compute partitions and save them to JSON files for later loading. This enables exact reproduction—if you save the partition you used in an experiment, others can load the same partition and compare results fairly.

## Part 5: Evaluation Framework and Performance Metrics

### Understanding Anomaly Map Generation

The evaluation process begins with anomaly map generation. The `AnomalyScorer` class in `src/evaluation/anomaly_scorer.py` handles this process. Given a trained model (or a pretrained global memory bank from federated training), the AnomalyScorer generates a pixel-level anomaly score for every pixel in every test image.

The process begins by iterating through all test images in a specified category. For each image, the AnomalyScorer loads it from disk, applies the same preprocessing as training, and passes it through the feature extraction backbone. The extracted features are processed identically to training: neighborhood averaging is applied, features are reshaped into patches, and each patch is queried against the memory bank.

The result is an anomaly score for each patch. These scores are spatial maps at feature resolution (50x50 for 400x400 input images). The scores are upsampled back to original image resolution using bilinear interpolation, producing a full-resolution anomaly map. This anomaly map is then saved as a PNG image with the same directory structure as the test set.

The PNG format for saving is important. The code saves the anomaly scores as floating-point values in PNG files, which some PNG libraries can handle and some cannot. This encoding preserves the full precision of anomaly scores, enabling detailed evaluation.

The AnomalyScorer can be configured to save anomaly maps for different model variants, different layers, different neighborhood sizes, or other variations. By generating maps for multiple variants and comparing their outputs, researchers can understand how architectural choices affect anomaly detection.

### AUC-sPRO: Evaluating Spatial Localization

AUC-sPRO (Area Under the Spatial PRObability of Rank curve) is a metric specifically designed for pixel-level anomaly localization. It measures how well the model can pinpoint the exact location of defects. Unlike simpler metrics that just measure whether an image is classified correctly, AUC-sPRO penalizes predictions that identify wrong regions as anomalous.

The AUC-sPRO computation is complex. For each defective test image:
1. Get the ground truth mask showing exactly which pixels are defective
2. Get the anomaly map showing the model's predicted anomaly scores
3. Compute the Probability of Rank (PRO), which measures localization accuracy

The PRO computation involves checking, for each ground truth defect pixel, what fraction of normal pixels have lower anomaly scores. If a defect pixel has a high anomaly score relative to normal pixels, this indicates accurate localization. PRO aggregates these comparisons across all pixels in an image.

To generate the final AUC-sPRO metric, the system varies an anomaly threshold across all possible values. At each threshold, it computes the rate of false positives (normal pixels incorrectly flagged as anomalous) and the probability of ranks (how well the model localizes defects). This generates a curve showing the trade-off between false positive rate and localization accuracy.

The area under this curve is AUC-sPRO. Perfect localization (all defect pixels ranked higher than all normal pixels) gives AUC-sPRO of 1.0. Random predictions give approximately 0.5. Values less than 0.5 indicate worse than random performance.

The metric is typically computed up to a maximum false positive rate (e.g., 0.1, meaning we allow 10% of normal pixels to be falsely flagged). This focuses evaluation on practical scenarios where some false positives are tolerable but excessive false positives hurt usability.

### AUC-ROC: Evaluating Image Classification

While AUC-sPRO measures spatial localization, AUC-ROC measures image-level classification accuracy. This metric answers the question: given an image, can the model correctly determine whether it contains any anomaly?

The AUC-ROC computation is more straightforward. For each test image:
1. Get the image-level ground truth (1 if the image contains any defect, 0 if all normal)
2. Get the image-level anomaly score (the maximum anomaly score in the anomaly map)
3. Record this prediction

The code varies a threshold across possible anomaly scores. At each threshold, images with anomaly scores above the threshold are classified as defective, below the threshold as normal. This generates classification decisions that can be compared against ground truth. The system computes:
- True Positive Rate (TPR): fraction of defective images correctly classified as defective
- False Positive Rate (FPR): fraction of normal images incorrectly classified as defective

Varying the threshold generates a curve showing the trade-off between catching defects (TPR) and avoiding false alarms (FPR). The area under this curve is AUC-ROC. Perfect classification gives AUC-ROC of 1.0, random performance gives 0.5.

### Defect-Specific Analysis

The evaluation framework can provide defect-specific metrics. The dataset includes different types of defects—structural anomalies, logical anomalies, and potentially others. By filtering test samples by defect type, the system can compute metrics separately for each type. This reveals whether the model detects some defect types better than others, which is valuable information for understanding model behavior.

For example, results might show AUC-sPRO of 0.92 for structural anomalies but only 0.85 for logical anomalies. This would indicate that the model excels at detecting physical defects but struggles with assembly errors. This insight guides future improvements—perhaps more training data for logical anomalies or modifications to the feature extractor to better capture assembly-related information.

### Category-Specific Performance Analysis

Just as defect types vary in detectability, so do component categories. Some components might be easier to inspect visually (high contrast defects on homogeneous backgrounds) while others are harder (subtle defects on textured backgrounds). The evaluation framework can compute metrics per category, revealing which component types the model handles well.

This category-level analysis is particularly important for federated learning. In category-based partitioning, different clients train on different categories. The evaluation can assess whether the global federated model generalizes well to all categories, or whether it performs better on categories with more training data. Imbalances in category performance might indicate that the aggregation strategy unfairly favors clients with more data.

## Part 6: Jupyter Notebooks - Interactive Workflow Examples

### Notebook 02: Baseline PatchCore Training - Detailed Walkthrough

The second notebook, `notebooks/02_baseline_training.ipynb`, is designed to be an educational resource that demonstrates the complete baseline training workflow. The notebook begins with extensive documentation explaining the overall goal and the general approach. Code cells are interspersed with markdown cells providing explanation.

The initial code cells import necessary libraries and verify the environment. It checks PyTorch version, confirms CUDA availability, and identifies the GPU device if present. This information is valuable for understanding whether training will use GPU acceleration or fall back to CPU.

The notebook then defines configuration parameters. Users can modify these parameters (data directory, output directory, model configuration, batch size) to customize training to their specific needs. The configuration is structured as a Python dictionary, making it easy to understand what options are available.

The notebook defines helper functions for common operations. The `get_transforms()` function creates a preprocessing pipeline specific to a component category. The `visualize_anomaly_map()` function displays results, showing side-by-side the original image, the anomaly map, and an overlay of the map on the image. This visualization is crucial for understanding what the model is detecting.

The notebook then demonstrates training on a single category as a quick validation. It loads training and test data for one category, creates a PatchCore model, trains the model by building the memory bank, and displays the training statistics. This single-category example trains quickly (minutes on GPU) and provides immediate feedback, helping users verify that the setup is working before running full training.

With the single category working, the notebook scales up to training on all categories. It defines a `train_category()` function that trains a model for a specified category, saves the model, and returns statistics. It then loops through all categories, training a model for each. The notebook tracks training statistics for all categories and saves them to a JSON file for later analysis.

Finally, the notebook displays a summary showing training results across all categories. This summary includes the number of training samples per category and the size of each category's memory bank. Users can examine this summary to understand the scale of the trained models and to verify that training completed successfully for all categories.

### Notebook 03: Federated Experiments - Multi-Client Learning

The third notebook would demonstrate federated learning experiments. While not fully detailed in the codebase, its structure would mirror the baseline notebook but add federated-specific components.

The notebook would begin by explaining federated learning concepts. It would then load the dataset and create partitions using both IID and category-based strategies. It would visualize the partition statistics, showing how many samples each client receives and the category composition of each client's data.

The notebook would initialize a FederatedPatchCore system with specified number of clients and configuration. It would run federated training for one or more rounds, displaying progress information. After training, it would compute statistics comparing the federated approach to the baseline, examining metrics like global memory bank size, coverage of the feature space, and diversity of selected features.

The notebook might include visualizations comparing features extracted by different clients, showing whether different clients learn complementary features or redundant features. It could also show how the aggregated global model differs from training all data centrally, quantifying the cost of federation in terms of model degradation.

### Notebook 04: Results Analysis and Comparison - Evaluation and Visualization

The fourth notebook would focus on post-training analysis. It would load evaluation results (AUC-sPRO, AUC-ROC metrics) computed by the evaluation framework and create visualizations and comparisons.

The notebook would load results from multiple experiments: baseline training, IID federated training, and category-based federated training. It would create bar charts comparing AUC-sPRO and AUC-ROC across approaches and categories. It would include tables showing detailed metrics.

The notebook might include statistical analysis examining whether differences in performance are statistically significant or could be due to random variation. It would investigate whether the federated approach degrades performance relative to centralized training, and whether this degradation is acceptable.

The notebook would also include qualitative analysis, displaying anomaly maps from different approaches side-by-side for the same test images. This visual comparison often reveals differences not apparent from numeric metrics—one approach might localize defects more accurately despite having similar AUC-sPRO, or might have different failure modes (false positives vs. false negatives).

## Part 7: Complete Training and Evaluation Pipeline

### End-to-End Workflow Overview

A complete experiment from raw data to final results follows this pipeline:

First, the raw dataset must be obtained and organized. The AutoVI dataset is downloaded from Zenodo and extracted to a directory with the expected structure. Directory validation scripts verify that the structure is correct and counts samples to ensure no corruption during download.

Second, a baseline model is trained. The `train_centralized.py` script loads the entire dataset, trains a single PatchCore model, and generates anomaly maps on the test set. This baseline provides a reference point showing the best possible performance when all data is centralized.

Third, the baseline is evaluated. The `evaluate_experiment.py` script loads the generated anomaly maps, loads ground truth masks, and computes AUC-sPRO and AUC-ROC metrics. These metrics are saved to JSON files and displayed in formatted tables.

Fourth, federated experiments are run. Data is partitioned into IID and category-based distributions. For each distribution, federated training is executed with various numbers of clients (e.g., 3, 5, 10 clients). Each federated configuration generates anomaly maps and is evaluated to produce metrics.

Fifth, results are analyzed. The metrics from all approaches (baseline, IID federated with various client counts, category-based federated with various client counts) are loaded and compared. Visualizations are created showing how federated learning performance scales with client count and how it compares to centralized baseline.

Finally, findings are documented. The experiment results are summarized in tables and figures suitable for publication or presentation. The documentation includes explanations of findings and implications for future work.

### Key Success Metrics and Interpretation

The project aims to demonstrate several things. First, that federated learning can achieve competitive performance with centralized training. Success would mean federated AUC-sPRO within 1-2% of baseline AUC-sPRO. Larger gaps would indicate that federation introduces significant performance loss.

Second, that federated learning scales gracefully with client count. Adding more clients should improve performance (more diverse data) or at worst slightly degrade it. Dramatic performance degradation with increasing clients would suggest the aggregation strategy is flawed.

Third, that category-based partitioning is harder than IID partitioning. Since category-based partitioning creates heterogeneous data distributions, federated training with category partitioning should show larger gaps from baseline than with IID partitioning. The size of this gap quantifies how hard non-IID federated learning is for this task.

Fourth, that the federated approach is communication-efficient. Federated training should send fewer bytes of data over the network than transmitting all raw training images to a central server. Memory bank vectors (tens of thousands of 1536-dimensional vectors) are smaller than images (millions or billions of pixels). Quantifying this communication savings is valuable for demonstrating practical benefits.

### Common Variations and Extensions

The basic pipeline can be extended in several ways. Multi-round federated training, where training continues for multiple iterations, can be compared against single-round training to understand whether iteration improves results. Different aggregation strategies (simple concatenation, diversity-preserving) can be compared to understand whether the weighted approach is optimal.

Different feature extractors (different ResNet depths, different pretrained models, or even non-pretrained models) can be evaluated to understand how backbone choice affects results. Different coreset ratios (keeping 1%, 5%, 10%, or 20% of features) can be compared, trading off model size against accuracy.

Noise injection experiments can assess privacy robustness. By adding Gaussian noise to features before transmission in the federated setting, you can measure how much privacy noise can be added before performance degrades significantly. This quantifies privacy-utility trade-offs.

## Part 8: Technical Implementation Details and Advanced Topics

### GPU Acceleration and Device Handling

Modern deep learning requires hardware acceleration for practical performance. The codebase automatically detects available GPUs and uses them when present, controlled through the device parameter. When set to "auto", the code checks `torch.cuda.is_available()` to detect CUDA availability and creates a torch.device object pointing to "cuda:0" (the first GPU) if available, or "cpu" otherwise.

All tensors and models are moved to the selected device using the `.to(device)` method. This is critical for correctness—if a model is on GPU but input tensors are on CPU, PyTorch will raise an error preventing silent incorrect computation. The codebase is careful to move images, models, and temporary tensors all to the same device.

FAISS also supports GPU acceleration. When the use_gpu parameter is True, the code moves the FAISS index to GPU using `faiss.index_cpu_to_gpu()`. GPU FAISS is particularly beneficial for large memory banks and batch queries, where the parallelism of GPU significantly outpaces CPU. However, GPU FAISS requires sufficient GPU memory—a memory bank with millions of features might exceed available GPU memory, requiring CPU fallback.

The device handling is transparent to the user. Configuring device="auto" automatically adapts to available hardware. This makes the code portable—the same code runs on machines with or without GPUs, automatically adapting performance characteristics.

### Memory Management Strategies

Large-scale anomaly detection requires careful memory management. The entire dataset can't be loaded into RAM (thousands of images at high resolution would exceed available memory on most machines). The codebase addresses this through lazy loading—images are loaded on-demand as DataLoaders iterate, processed, and discarded before the next image is loaded.

The coreset subsampling aggressively reduces feature storage. Rather than storing millions of patch features, only tens of thousands are retained. This reduces RAM usage by 10-100x. The selected features are still representative of the overall distribution, maintaining accuracy while drastically reducing memory.

The FAISS index is compressed using quantization techniques when necessary. While the current implementation uses a flat index (exact nearest neighbors), FAISS supports product quantization and other compression techniques that reduce memory usage at the cost of slight accuracy loss. Transitioning to compressed indices would be straightforward if memory becomes a bottleneck.

NumPy arrays are managed carefully. The code uses astype(np.float32) to use 32-bit floats instead of 64-bit doubles where appropriate, halving memory usage. It uses contiguous arrays to ensure efficient memory layout and SIMD operations.

### Reproducibility and Numerical Determinism

Achieving exact reproducibility in modern deep learning is challenging due to non-deterministic GPU operations. However, the codebase implements several practices to maximize reproducibility.

All random seeds are controlled through a `set_random_seeds()` function that sets seeds in NumPy, PyTorch, and Python's random module. The coreset subsampling algorithm is deterministic given fixed seeds. The FAISS index construction is deterministic.

However, some GPU operations remain non-deterministic. Certain CUDA operations don't have deterministic implementations. To achieve fully deterministic results, you would need to set `torch.backends.cudnn.deterministic = True`, though this can slightly reduce performance. The codebase documents that results are reproducible within floating-point precision but might have minor differences across different hardware or CUDA versions.

The solution is to always report the environment (PyTorch version, CUDA version, GPU model) along with results, so others can understand what might cause minor numerical differences.

### Error Handling and Robustness

The training scripts include error handling to prevent crashes due to corrupted images or unexpected data. If a single image fails to load, training continues with the next image rather than crashing entirely. This robustness is important because real-world datasets inevitably contain occasional corrupted files or format issues.

The DataLoader's error handling happens transparently—if a worker process encounters an error loading or processing an image, the error is captured and handled gracefully. The main training loop continues with remaining samples.

The server-side aggregation includes validation. It checks that received coresets have the expected feature dimension. It validates that aggregation didn't produce empty results. It logs warnings if some clients fail to submit coresets, continuing aggregation with available data.

This defensive programming approach recognizes that production systems must handle unexpected conditions gracefully rather than crashing.

### Integration with PyTorch Ecosystem

The codebase leverages the PyTorch ecosystem throughout. It uses torchvision for computer vision utilities (image transforms, model definitions). It uses DataLoader from torch.utils.data for efficient batching and data loading. It uses torch.nn.functional for operations like interpolation.

This tight integration with PyTorch means the codebase automatically benefits from PyTorch improvements and optimizations. If a new PyTorch version dramatically optimizes a particular operation, the codebase benefits without modification.

The code avoids unnecessary custom implementations, instead using well-tested PyTorch and NumPy operations. This improves reliability and performance compared to reimplementing algorithms from scratch.

## Part 9: Practical Workflows and Troubleshooting Guide

### Setting Up and Running Your First Experiment

To run your first experiment, start with the baseline. Create a simple configuration file or use the provided baseline configuration. Run `python scripts/train_centralized.py --config <config_file> --data_root <data_path>`. Training should complete in 10-30 minutes depending on your hardware and dataset size.

The training script will produce a directory with the trained model, statistics, and generated anomaly maps. You can immediately see anomaly maps by looking at the generated PNG files. Do they highlight actual defects? Are there many false positives in normal images?

If training is too slow, several optimizations are possible. Running on GPU provides 5-10x speedup. Reducing batch size reduces memory usage but might slightly increase training time (more batches to process). Reducing the number of workers in the DataLoader might help if data loading is the bottleneck. Reducing the dataset size to train on a subset allows quick experimentation before committing to full training.

### Adapting Code to New Datasets

To apply this code to a different visual inspection dataset, several modifications are needed. First, create a custom dataset class inheriting from or modifying AutoVIDataset. Update the CATEGORIES list to match your components. Implement the directory scanning logic to match your dataset organization. Update get_resize_shape() to return appropriate dimensions for your images.

If your dataset has different defect types than AutoVI, update the ground truth loading logic to match your annotation format. If your annotations are in different formats (XML, JSON) rather than PNG masks, modify get_ground_truth() to parse the appropriate format.

Update the preprocessing parameters in configuration files to match your dataset. If your images use different normalization statistics (e.g., because you captured them with different cameras), collect statistics on your training data and use those instead of ImageNet statistics.

Test your dataset implementation with a small subset before scaling to the full dataset. Verify that images load correctly, that ground truth masks align with images, and that basic visualization looks correct.

### Hyperparameter Tuning and Performance Optimization

If your baseline performance is lower than expected, several hyperparameters can be tuned. The coreset_ratio determines what fraction of patches are retained. Increasing this ratio (e.g., from 0.1 to 0.2) keeps more training features, which generally improves accuracy at the cost of larger memory banks and slower inference. Decreasing it reduces memory usage but might hurt accuracy.

The layers parameter controls which backbone layers are used. Using ["layer2", "layer3"] captures both local and semantic information. Using only ["layer3"] uses purely semantic features but loses fine-grained detail. Using ["layer1", "layer2", "layer3"] captures more information but increases computational cost and memory usage.

The neighborhood_size controls spatial smoothing. Larger values (e.g., 5x5 instead of 3x3) provide stronger smoothing but might blur together distinct regions. Smaller values (1x1, no smoothing) preserve fine detail but are more sensitive to noise. The optimal value depends on your defect types—very small, precise defects benefit from small neighborhood size, while larger defects benefit from larger neighborhood size.

If inference is too slow, enable FAISS GPU acceleration if you have GPU memory available. Use smaller memory bank sizes (lower global_bank_size in federated config). Consider using approximate nearest neighbor search instead of exact search if speed is more critical than accuracy.

### Debugging and Common Issues

Common issues include "CUDA out of memory" errors. Reduce batch size in configuration. Reduce the number of data loading workers. Run feature extraction and anomaly map generation on CPU instead of GPU if you have sufficient RAM.

If images fail to load, verify the dataset directory structure matches expectations. Verify image files are valid PNG files (not corrupted). Check that image paths are correctly specified in configuration.

If anomaly maps look completely wrong (random noise instead of structured maps), verify that the feature extractor is correctly loading pretrained weights. Verify that image preprocessing matches what the backbone expects (correct normalization statistics, correct image size ranges).

If federated training produces much worse results than baseline, check that all clients received data. Print partition statistics to verify clients have meaningful data. Verify that the aggregation strategy is appropriate for your data distribution.

If results aren't reproducible across runs, ensure random seeds are set and passed through all components. Check whether GPU is being used (GPU operations might be non-deterministic). Try running on CPU to achieve exact reproducibility.

## Part 10: Scientific Insights and Design Rationales

### Why Non-Parametric Anomaly Detection?

The choice of PatchCore (non-parametric, anomaly-detection-by-normality) over supervised anomaly detection reflects practical realities in manufacturing. Supervised learning requires diverse, labeled defect examples. Collecting and labeling defect examples is expensive and might miss rare but important defect types. Once the model is deployed, new defect types that weren't in training data often go undetected.

Non-parametric methods excel with unlabeled normality data. Factories generate abundant images of normal parts from routine inspection. By learning what normal looks like rather than learning to recognize defects, the system is inherently robust to novel defects. This makes PatchCore particularly suitable for manufacturing where defect diversity is high and defect examples are expensive to obtain.

### Why Patch-Level Anomaly Detection?

Examining anomalies at the patch level rather than the image level provides spatial localization. Knowing that an image contains a defect is useful, but knowing exactly where the defect is located is more valuable for quality control. Manufacturing processes are specific—a defect in one location might be critical, while the same defect in another location might be acceptable. Patch-level scoring enables precise localization.

Patch-level processing also makes the method more robust. Small defects might affect only a few patches, but the model can still detect them. Image-level classification might not notice changes affecting only a small fraction of pixels, but patch-level methods catch such localized anomalies.

### Why Federated Learning?

Federated learning preserves privacy by not transmitting raw images. It reduces communication overhead by transmitting only feature summaries. It enables participation from multiple stations without requiring a centralized data repository.

However, federated learning comes with costs. The system is more complex. Coordination overhead exists. Non-IID data distributions create challenges. The decision to use federated learning should be based on whether privacy, communication, or regulatory benefits outweigh complexity costs.

### Why Coreset Subsampling?

Raw feature extraction produces millions of patch features. Storing all of them consumes gigabytes of memory. Coreset subsampling reduces this by 10-100x, making the system practical. The greedy k-center algorithm ensures selected features are well-distributed across the feature space, maintaining coverage despite the reduction.

Alternative subsampling strategies (random sampling, density-based sampling) would be simpler but less effective. Random sampling might miss important regions of feature space. Density-based sampling might oversample common patterns. The k-center approach balances simplicity and effectiveness.

### Why Multiple Layers?

Using both layer 2 and layer 3 features provides complementary information. Layer 2 features encode low-level patterns (textures, edges). Layer 3 features encode high-level semantic information (object parts, arrangements). Anomalies might manifest as unusual low-level patterns or unusual high-level structure. By combining both, the system can detect both types.

Using deeper layers (layer 4, 5) would add more semantic information but at the cost of loss of spatial resolution. Shallower layers (layer 1) would provide more fine detail but less semantic understanding. The choice of layer 2 and 3 represents a reasonable balance for manufacturing inspection tasks.

## Conclusion and Future Directions

This comprehensive guide explains the architectural design and implementation details of the AutoVI federated anomaly detection system. The system demonstrates that privacy-preserving visual inspection is technically feasible while maintaining competitive performance with centralized approaches. The modular architecture allows researchers to experiment with different components: alternative aggregation strategies, different neural network backbones, modified partitioning schemes, and alternative anomaly scoring methods.

Future work could explore several directions. Multi-round federated training with sophisticated aggregation strategies could improve convergence. Differential privacy mechanisms could add formal privacy guarantees. Alternative backbone networks (Vision Transformers, different CNN architectures) could be evaluated. Adaptive partitioning strategies that adjust data distribution during training could address non-IID challenges. The comprehensive framework provides a solid foundation for advancing federated learning applications in manufacturing and beyond.

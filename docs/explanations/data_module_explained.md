# Data Module Explained

## The AutoVI Dataset and Its Role in the Pipeline

The data module serves as the foundation upon which the entire anomaly detection pipeline rests. Before any feature extraction, memory bank construction, or federated aggregation can occur, raw images must be loaded, organized, preprocessed, and distributed among clients. The four files in this module handle these responsibilities systematically.

The AutoVI dataset originates from Renault's automotive manufacturing quality control operations. Unlike synthetic anomaly detection benchmarks, AutoVI captures real-world complexity: varying lighting conditions, subtle defects, multiple product categories, and the inherent class imbalance between normal and anomalous samples. Understanding this dataset's structure is essential for understanding why the code makes certain design choices.

The autovi_dataset.py file defines the core dataset class that loads and organizes image samples. The datasets.py file provides wrapper classes for applying category-specific transforms. The partitioner.py file implements strategies for dividing the dataset among federated learning clients. The preprocessing.py file defines the image transformations that prepare raw images for neural network processing.

---

## The AutoVI Dataset Structure

### Six Categories of Automotive Components

The AutoVI dataset contains images from six distinct categories of automotive parts inspected during manufacturing. These categories are defined in the CATEGORIES list as engine_wiring, pipe_clip, pipe_staple, tank_screw, underbody_pipes, and underbody_screw. Each category represents a different component type with unique visual characteristics, typical defect patterns, and imaging requirements.

The engine_wiring category contains images of electrical wiring harnesses in the engine compartment. These images capture complex arrangements of cables, connectors, and tie points where defects might include loose connections, damaged insulation, or incorrect routing.

The pipe_clip and pipe_staple categories involve small fastening components that secure fluid lines and wiring throughout the vehicle. Defects in these categories often involve missing clips, incorrect placement, or damaged fasteners.

The tank_screw category captures images of screw fasteners securing fuel tanks and other reservoirs. Quality control here focuses on proper engagement, correct torque indication, and absence of cross-threading or damage.

The underbody_pipes and underbody_screw categories involve components visible from beneath the vehicle. These present particular imaging challenges due to their location and the potential for contamination from road debris.

### Size-Based Category Grouping

The dataset divides categories into two groups based on the physical size of inspected objects and consequently the image dimensions required to capture them with adequate detail.

The SMALL_OBJECTS group includes engine_wiring, pipe_clip, and pipe_staple. Images of these smaller components are captured at 400×400 pixel resolution. This resolution provides sufficient detail for the relatively compact objects while keeping computational requirements manageable.

The LARGE_OBJECTS group includes tank_screw, underbody_pipes, and underbody_screw. These physically larger components or those requiring wider field of view are captured at 1000×750 pixel resolution. The asymmetric dimensions reflect the typical aspect ratios of these inspection areas.

The get_resize_shape function maps a category name to its appropriate dimensions, returning (400, 400) for small objects and (1000, 750) for large objects. This function is used throughout the preprocessing pipeline to ensure images are resized consistently based on their category.

---

## The AutoVIDataset Class

### Initialization and Configuration

The AutoVIDataset class provides a PyTorch-compatible interface to the AutoVI image collection. During initialization, it receives the root_dir parameter specifying the filesystem path where AutoVI data is stored. The expected directory structure follows the pattern: root_dir/category/split/class/images.png where split is either "train" or "test" and class is either "good" or a specific defect type name.

The categories parameter optionally restricts which product categories to include. If omitted, all six categories are loaded. This flexibility allows training category-specific models or analyzing subsets of the data.

The split parameter selects between training and test data. The training set contains only "good" (normal, defect-free) samples since PatchCore learns exclusively from normal examples. The test set contains both good and defective samples for evaluation.

The transform parameter accepts a callable that will be applied to each image when accessed. This allows preprocessing (resizing, normalization, augmentation) to be specified at dataset creation time and applied automatically during data loading.

The include_good_only parameter provides additional filtering for the test split. When True, even the test set returns only good samples. This is useful for certain evaluation scenarios or for extending training data.

### Sample Loading Process

The _load_samples method populates the samples list with tuples of (image_path, label, category, defect_type). It iterates through each requested category, constructing the expected directory path for the current split.

For each category, the method first loads good samples from the "good" subdirectory. Each image file found receives label 0 (indicating normal) and defect_type None. The glob pattern "*.png" matches all PNG images in the directory, and sorting ensures consistent ordering across runs.

For the test split (unless include_good_only is True), the method also loads defective samples. It iterates through all subdirectories other than "good", treating each as a defect type. Images in these directories receive label 1 (indicating anomalous) and the directory name as their defect_type. This organization allows the dataset to contain multiple defect types per category, each in its own subdirectory.

The resulting samples list contains complete information about every image in the dataset, enabling efficient random access during training and evaluation.

### Defect Configuration Loading

The _load_defect_configs method attempts to load JSON configuration files that describe the defect types for each category. These configuration files (defects_config.json) provide metadata about what kinds of defects exist and how they were generated or annotated.

This metadata is stored in the defect_configs dictionary but is primarily informational. The actual defect identification comes from the directory structure, not the configuration files. The configs provide additional context that might be useful for analysis or reporting.

### Data Access Interface

The __len__ method returns the total number of samples, enabling standard Python length queries and iteration. The __getitem__ method provides indexed access to individual samples, returning a dictionary with keys for image, label, category, defect_type, and path.

When a sample is accessed, the method opens the image file using PIL, converts it to RGB format to ensure consistent channel ordering, and optionally applies the transform. The returned dictionary provides all information needed for training (the image and label) along with metadata useful for analysis and debugging (category, defect_type, path).

The dictionary format rather than a simple tuple provides clarity about what each element represents and allows easy extension with additional fields if needed.

### Ground Truth Mask Access

The get_ground_truth method retrieves pixel-level annotation masks for defective test samples. These masks indicate exactly which pixels contain the defect, enabling evaluation of localization accuracy.

Ground truth masks are stored in a specific directory structure: root_dir/category/ground_truth/defect_type/image_name/masks.png. A single defective image may have multiple mask channels (for example, if multiple defects exist in one image), which are combined using logical OR to produce a single binary mask.

The method returns None for good samples (which have no defects to localize) or when ground truth is not available. When masks exist, they are loaded, combined, and returned as a uint8 numpy array with value 255 for defective pixels and 0 for normal pixels.

### Utility Methods

The get_category_indices method returns a list of sample indices belonging to a specific category. This enables category-specific analysis or the creation of category-balanced batches.

The get_defect_indices method returns indices for samples of a specific defect type, useful for analyzing detection performance on particular defect categories.

The get_statistics method computes summary statistics about the dataset including total sample count, counts by category, counts by defect type, and counts by label. This information is valuable for understanding data distribution and diagnosing potential issues with class imbalance.

---

## The AutoVISubset Class

### Purpose in Federated Learning

The AutoVISubset class wraps an AutoVIDataset to provide access to only a subset of samples. This is essential for federated learning where each client should see only its assigned portion of the data.

Rather than copying samples or creating separate dataset objects, AutoVISubset maintains a reference to the parent dataset and a list of indices. When a sample is requested, the subset translates the local index to the corresponding global index in the parent dataset.

### Implementation Details

The constructor receives the parent AutoVIDataset and a list of integer indices specifying which samples belong to this subset. The indices list determines both which samples are accessible and their ordering within the subset.

The __len__ method returns the length of the indices list, representing how many samples this subset contains. The __getitem__ method translates local indices to global indices: when sample i is requested, it returns self.dataset[self.indices[i]].

The get_statistics method computes statistics specifically for this subset's samples, iterating through the indices and tallying categories and labels. This allows verification that partitioning produced the expected distribution.

---

## Dataset Wrapper Classes

### CategoryTransformDataset

The CategoryTransformDataset class addresses a subtle but important issue: different categories require different preprocessing due to their different image sizes. A single transform cannot resize engine_wiring images to 400×400 and tank_screw images to 1000×750.

This wrapper receives a source dataset and a dictionary mapping category names to transform functions. When a sample is accessed, the wrapper retrieves the sample from the source dataset, looks up the appropriate transform based on the sample's category, applies that transform to the image, and returns the modified sample.

This design allows a single dataset object to handle heterogeneous image sizes transparently. The calling code requests samples by index without worrying about which transform to apply; the wrapper handles that detail automatically.

### TransformedSubset

The TransformedSubset class provides a simpler subset mechanism than AutoVISubset. It works with any dataset supporting integer indexing, not specifically AutoVIDataset. The class stores a reference to the source dataset and a list of indices, translating local to global indices on access.

This class is useful when working with already-transformed datasets where the additional metadata tracking of AutoVISubset is unnecessary. It provides the same basic subsetting functionality with a leaner interface.

---

## Data Partitioning for Federated Learning

### The Partitioning Problem

Federated learning requires dividing a dataset among multiple clients such that each client has a realistic, meaningful portion of the data. How this division occurs significantly impacts both the training dynamics and the realism of the simulation.

Two fundamental approaches exist: IID (Independent and Identically Distributed) partitioning, where samples are randomly distributed so each client sees a representative sample of the whole dataset, and non-IID partitioning, where clients receive systematically different subsets that do not represent the global distribution.

Real-world federated learning scenarios are almost always non-IID. Different hospitals see different patient populations. Different factories produce different products. Different mobile devices belong to users with different behaviors. The non-IID case is both more realistic and more challenging for learning algorithms.

### IIDPartitioner

The IIDPartitioner class implements random uniform distribution of samples across clients. It is initialized with num_clients specifying how many clients to partition among and seed for reproducibility.

The partition method accepts an AutoVIDataset and returns a dictionary mapping client IDs to lists of sample indices. The implementation generates a random permutation of all sample indices, then splits this permutation into num_clients roughly equal pieces using numpy's array_split function.

The resulting partition gives each client approximately 1/num_clients of the total samples. Because the samples are randomly permuted, each client's subset is statistically representative of the whole dataset. Categories and defect types are distributed proportionally to their occurrence in the full dataset.

The create_subsets method provides a convenience wrapper that applies the partition and creates AutoVISubset objects for each client in a single call.

### CategoryPartitioner

The CategoryPartitioner class implements non-IID partitioning based on product categories. This simulates the realistic scenario where different manufacturing stations inspect different components. A station that inspects engine wiring will have data only from engine wiring, not from underbody pipes.

The default client assignments, defined in CATEGORY_CLIENT_ASSIGNMENTS, create a realistic factory scenario. Client 0 receives engine_wiring, representing an Engine Assembly station. Client 1 receives underbody_pipes and underbody_screw, representing an Underbody Line. Client 2 receives tank_screw and pipe_staple, representing a Fastener Station. Client 3 receives only pipe_clip, representing a specialized Clip Inspection station. Client 4 receives samples from all categories, representing a Quality Control station that performs random sampling across the factory.

The qc_sample_ratio parameter (default 0.1) determines what fraction of each category the QC client receives. This client gets first pick of samples, taking 10% from each category, before the specialized clients receive the remainder from their assigned categories.

### Partitioning Implementation

The partition method implements a two-pass assignment process. First, it builds a mapping from each category to all sample indices belonging to that category. Then it processes clients in two phases.

In the first phase, clients with "all" in their category list (the QC clients) receive their samples. For each category, the QC client receives qc_sample_ratio of that category's samples, selected randomly. These indices are marked as assigned to prevent double-counting.

In the second phase, specialized clients receive remaining samples from their assigned categories. A client assigned to ["tank_screw", "pipe_staple"] receives all tank_screw samples not taken by the QC client, plus all pipe_staple samples not taken by the QC client.

This two-phase approach ensures QC clients get representative samples from all categories while specialized clients receive comprehensive coverage of their assigned categories.

### Partition Utilities

The create_partition function provides a unified interface for both partitioning strategies. It accepts a strategy parameter ("iid" or "category") and dispatches to the appropriate partitioner class.

The save_partition and load_partition functions enable persistence of partition assignments. This is important for reproducibility: by saving a partition, experiments can be exactly repeated with the same client assignments. The partition is serialized as JSON with client IDs as keys and index lists as values.

The compute_partition_stats function analyzes a partition to report how many samples each client received, broken down by category and label. This information helps verify that partitioning produced the expected distribution and diagnose potential issues.

---

## Image Preprocessing Pipeline

### The Need for Preprocessing

Raw images from the AutoVI dataset cannot be directly consumed by neural networks. Several transformations are required: resizing to consistent dimensions, converting to tensor format, normalizing pixel values to match the statistics expected by pretrained models, and optionally applying augmentation to increase training diversity.

The preprocessing module provides these transformations in a flexible, composable manner. Different components of the pipeline can be enabled or disabled based on the use case.

### ImageNet Normalization

The pretrained WideResNet-50-2 backbone was trained on ImageNet, where images were normalized using specific mean and standard deviation values. The IMAGENET_MEAN tuple (0.485, 0.456, 0.406) represents the average pixel values across all ImageNet images for the red, green, and blue channels respectively. The IMAGENET_STD tuple (0.229, 0.224, 0.225) represents the standard deviations.

Normalization transforms pixel values from the range [0, 1] to approximately [-2.5, 2.5] by subtracting the mean and dividing by the standard deviation for each channel. This ensures that input images have similar statistical properties to the images the backbone saw during pretraining, enabling the pretrained features to transfer effectively.

### The get_transforms Function

The get_transforms function constructs a composed transform for a specific category. It accepts the category name (to determine resize dimensions), and boolean flags for normalize, to_tensor, and augment.

The function builds a list of transforms starting with resizing to the category-appropriate dimensions. If augmentation is enabled, it adds random horizontal flip (50% probability), random vertical flip (50% probability), random rotation up to 15 degrees, and color jitter varying brightness and contrast by up to 10%.

If to_tensor is True, the ToTensor transform is added, converting the PIL Image to a PyTorch tensor with shape [C, H, W] and values scaled to [0, 1]. If normalize is True, the ImageNet normalization transform is added.

The resulting composed transform can be applied to any PIL Image to produce a preprocessed tensor ready for the neural network.

### The Preprocessor Class

The Preprocessor class provides an object-oriented interface to the preprocessing pipeline. During initialization, it builds transforms for all categories and caches them in a dictionary. This avoids repeatedly constructing transform objects during training.

The __call__ method accepts an image and category, looks up the appropriate transform, and applies it. This provides a clean interface for processing individual images.

The preprocess_batch method processes multiple images at once, applying the appropriate transform to each based on its category, then stacking the results into a single tensor if to_tensor was enabled. This is efficient for batch processing during inference.

### Utility Functions

The resize_for_category function provides simple resizing without the full transform pipeline, useful for visualization or quick preprocessing.

The normalize_image and denormalize_image functions provide numpy implementations of ImageNet normalization and its inverse. The denormalize function is particularly useful for visualization: after processing through the network, images must be denormalized to recover valid pixel values for display.

The get_test_transform and get_train_transform functions provide preset transform combinations optimized for evaluation and training respectively. Test transforms include only resizing, tensor conversion, and normalization. Training transforms optionally add augmentation.

---

## Parameter Reference

### AutoVIDataset Parameters

**root_dir**: The filesystem path to the AutoVI dataset root directory. The expected structure is root_dir/category/split/class/images.png.

**categories**: Optional list of category names to include. Valid values are "engine_wiring", "pipe_clip", "pipe_staple", "tank_screw", "underbody_pipes", and "underbody_screw". If None, all categories are included.

**split**: Either "train" or "test". Training split contains only normal samples. Test split contains both normal and defective samples.

**transform**: Optional callable applied to each image on access. Typically a composed torchvision transform including resize, ToTensor, and Normalize.

**include_good_only**: When True and split="test", excludes defective samples. Useful for certain evaluation scenarios.

### IIDPartitioner Parameters

**num_clients**: Number of federated learning clients to partition among. Samples are divided into roughly equal portions.

**seed**: Random seed for reproducibility. The same seed produces the same partition.

### CategoryPartitioner Parameters

**client_assignments**: Dictionary mapping client IDs to lists of category names. Clients receive all samples from their assigned categories. The special value "all" indicates a QC client that samples from all categories.

**qc_sample_ratio**: Fraction of each category assigned to QC clients (default 0.1). QC clients receive this fraction from each category before specialized clients receive remainders.

**seed**: Random seed for reproducibility.

### Preprocessing Parameters

**normalize**: Whether to apply ImageNet normalization (subtracting mean, dividing by std). Required for pretrained backbone compatibility.

**to_tensor**: Whether to convert PIL Images to PyTorch tensors. Required for neural network processing.

**augment**: Whether to apply data augmentation (flips, rotation, color jitter). Increases training diversity but should be disabled for evaluation.

### Image Size Constants

**SMALL_OBJECTS** categories (engine_wiring, pipe_clip, pipe_staple) resize to **400×400** pixels.

**LARGE_OBJECTS** categories (tank_screw, underbody_pipes, underbody_screw) resize to **1000×750** pixels.

---

## Integration with the Pipeline

The data module integrates with the federated learning pipeline at several points.

During initialization, FederatedPatchCore's setup_clients method uses a partitioner to divide an AutoVIDataset among clients. The partition dictionary is stored and used to create dataloaders for each client.

During training, each client's dataloader yields batches from its AutoVISubset. The preprocessing transforms are applied automatically through the dataset's transform parameter, producing properly sized, normalized tensors ready for the feature extractor.

During evaluation, the test split AutoVIDataset provides both normal and defective samples. The get_ground_truth method retrieves localization masks for computing pixel-level metrics. The get_statistics method provides information for fairness evaluation across categories and defect types.

The CategoryTransformDataset wrapper is particularly important when working with multiple categories simultaneously. Since different categories have different image sizes, this wrapper ensures each image is processed with its appropriate transform.

The partition persistence functions (save_partition, load_partition) support reproducible experiments. By saving the partition used in one experiment, future experiments can use exactly the same data division for fair comparison.

---

## Understanding the Data Flow

The complete data flow from raw images to model-ready tensors proceeds as follows.

An AutoVIDataset is created specifying the root directory, desired categories, and split. The dataset scans the directory structure and builds a list of all available samples with their paths, labels, categories, and defect types.

A partitioner divides the sample indices among federated clients. IIDPartitioner creates random uniform splits; CategoryPartitioner creates realistic non-IID splits based on product categories.

For each client, an AutoVISubset wraps the dataset with the client's assigned indices. This subset appears as an independent dataset to the client, containing only its assigned samples.

A CategoryTransformDataset wraps the subset with category-specific transforms. When a sample is accessed, the appropriate transform is applied based on the sample's category.

A PyTorch DataLoader wraps the transformed dataset, providing batched, shuffled iteration with optional multi-worker loading. The DataLoader yields batches of preprocessed image tensors along with their metadata.

The client's feature extractor processes these batches, extracting patch features for coreset construction. The properly preprocessed images ensure that the pretrained backbone receives inputs matching its expected statistics.

This careful chain of dataset classes, wrappers, and transforms ensures that raw filesystem images are correctly transformed into the tensors needed for PatchCore's feature extraction, while maintaining the flexibility to support multiple categories, federated partitioning, and various preprocessing options.

---

## Summary

The data module provides the essential infrastructure for loading, organizing, and distributing the AutoVI dataset. The AutoVIDataset class offers a clean interface to the complex directory structure, handling multiple categories, train/test splits, and defect type organization. The subset and wrapper classes enable federated learning by dividing data among clients and applying category-appropriate preprocessing.

The partitioning strategies support both idealized IID scenarios and realistic non-IID distributions that simulate actual factory deployments. The preprocessing pipeline ensures images are properly sized, normalized, and formatted for the pretrained backbone.

Together, these components form the foundation upon which the PatchCore anomaly detection algorithm operates. Without proper data handling, the sophisticated feature extraction, memory bank construction, and federated aggregation would have no inputs to process. The careful design of the data module ensures that raw manufacturing inspection images flow smoothly through the entire pipeline, ultimately enabling effective anomaly detection for industrial quality control.

"""Memory bank implementation for PatchCore with coreset subsampling."""

from typing import Optional, Tuple

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class MemoryBank:
    """Memory bank storing representative patch features for anomaly detection.

    The memory bank stores a subset of training patch features selected via
    greedy coreset subsampling. During inference, anomaly scores are computed
    as distances to the nearest neighbors in the memory bank.

    Attributes:
        features: The stored patch features as a numpy array.
        feature_dim: Dimension of each feature vector.
        index: FAISS index for efficient nearest neighbor search.
    """

    def __init__(
        self,
        feature_dim: int,
        use_faiss: bool = True,
        use_gpu: bool = False,
    ):
        """Initialize the memory bank.

        Args:
            feature_dim: Dimension of feature vectors.
            use_faiss: Whether to use FAISS for nearest neighbor search.
            use_gpu: Whether to use GPU acceleration for FAISS.
        """
        self.feature_dim = feature_dim
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.use_gpu = use_gpu
        self.features: Optional[np.ndarray] = None
        self.index = None

    def fit(
        self,
        features: np.ndarray,
        coreset_ratio: float = 0.1,
        max_samples: Optional[int] = None,
        seed: int = 42,
        coreset_method: str = "greedy",
        use_projection: bool = False,
    ) -> None:
        """Build the memory bank from training features.

        Args:
            features: Training patch features of shape [N, D].
            coreset_ratio: Fraction of features to keep (0.0 to 1.0).
            max_samples: Maximum number of samples to keep (overrides ratio if set).
            seed: Random seed for reproducibility.
            coreset_method: Method for coreset selection. Options:
                - "greedy": Greedy k-center selection (best quality, high memory).
                           Recommended for small images (e.g., 400x400).
                - "kmeans": MiniBatchKMeans clustering (memory efficient).
                           Recommended for large images (e.g., 1000x750).
            use_projection: If True, apply Sparse Random Projection before greedy
                           coreset selection (recommended for large datasets).
        """
        n_samples = features.shape[0]

        # Determine target size
        if max_samples is not None:
            target_size = min(max_samples, n_samples)
        else:
            target_size = max(1, int(n_samples * coreset_ratio))

        method_desc = "kmeans clustering" if coreset_method == "kmeans" else "greedy coreset"
        logger.info(f"Selecting {target_size} samples from {n_samples} via {method_desc}...")

        # Apply coreset subsampling
        if target_size < n_samples:
            if coreset_method == "kmeans":
                # K-means returns centroids directly (not indices)
                self.features = kmeans_coreset_selection(
                    features, target_size=target_size, seed=seed
                )
            else:  # "greedy" (default) - works well for small images
                selected_indices = greedy_coreset_selection(
                    features, target_size=target_size, seed=seed,
                    use_projection=use_projection
                )
                self.features = features[selected_indices].copy()
        else:
            self.features = features.copy()

        logger.info(f"Memory bank size: {self.features.shape}")

        # Build FAISS index
        self._build_index()

    def _build_index(self) -> None:
        """Build FAISS index for fast nearest neighbor search."""
        if not self.use_faiss or self.features is None:
            return

        # Ensure features are float32 and contiguous
        features = np.ascontiguousarray(self.features.astype(np.float32))

        # Create L2 index
        self.index = faiss.IndexFlatL2(self.feature_dim)

        # Optionally move to GPU (if faiss-gpu is available)
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except AttributeError:
                logger.warning("FAISS GPU not available (faiss-gpu not installed), using CPU FAISS")

        self.index.add(features)

    def query(
        self,
        query_features: np.ndarray,
        k: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find nearest neighbors in the memory bank.

        Args:
            query_features: Query features of shape [M, D].
            k: Number of nearest neighbors to return.

        Returns:
            Tuple of:
                - distances: Distance to k nearest neighbors [M, k]
                - indices: Indices of k nearest neighbors [M, k]
        """
        if self.features is None:
            raise RuntimeError("Memory bank not fitted. Call fit() first.")

        query_features = np.ascontiguousarray(query_features.astype(np.float32))

        if self.use_faiss and self.index is not None:
            distances, indices = self.index.search(query_features, k)
        else:
            # Fallback to numpy-based search
            distances, indices = self._numpy_knn(query_features, k)

        return distances, indices

    def _numpy_knn(
        self,
        query_features: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """NumPy fallback for k-nearest neighbors."""
        # Compute all pairwise distances
        # Using broadcasting: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        query_norm = np.sum(query_features ** 2, axis=1, keepdims=True)
        memory_norm = np.sum(self.features ** 2, axis=1, keepdims=True)
        distances = query_norm + memory_norm.T - 2 * query_features @ self.features.T

        # Get k smallest
        indices = np.argpartition(distances, k, axis=1)[:, :k]
        # Sort the k neighbors
        row_indices = np.arange(len(query_features))[:, None]
        sorted_order = np.argsort(distances[row_indices, indices], axis=1)
        indices = indices[row_indices, sorted_order]
        distances = distances[row_indices, indices]

        return distances, indices

    def get_anomaly_scores(
        self,
        query_features: np.ndarray,
        k: int = 1,
    ) -> np.ndarray:
        """Compute anomaly scores for query features.

        Args:
            query_features: Query features of shape [M, D].
            k: Number of nearest neighbors for scoring.

        Returns:
            Anomaly scores of shape [M].
        """
        distances, _ = self.query(query_features, k=k)
        # Use distance to nearest neighbor as anomaly score
        return distances[:, 0]

    def save(self, path: str) -> None:
        """Save memory bank to file.

        Args:
            path: Output file path.
        """
        if self.features is None:
            raise RuntimeError("Memory bank is empty.")

        data = {
            "features": self.features,
            "feature_dim": self.feature_dim,
        }
        np.savez(path, **data)
        logger.info(f"Memory bank saved to {path}")

    def load(self, path: str) -> None:
        """Load memory bank from file.

        Args:
            path: Input file path.
        """
        data = np.load(path)
        self.features = data["features"]
        self.feature_dim = int(data["feature_dim"])
        self._build_index()
        logger.info(f"Memory bank loaded from {path}, shape: {self.features.shape}")

    def set_features(self, features: np.ndarray) -> None:
        """Set memory bank features directly (for federated learning).

        Args:
            features: Feature array of shape [N, D].
        """
        self.features = np.ascontiguousarray(features.astype(np.float32))
        self._build_index()
        logger.info(f"Memory bank set with {len(self.features)} patches")

    @classmethod
    def from_array(
        cls,
        features: np.ndarray,
        use_faiss: bool = True,
        use_gpu: bool = False,
    ) -> 'MemoryBank':
        """Create a MemoryBank directly from feature array.

        Args:
            features: Feature array [N, D].
            use_faiss: Whether to use FAISS.
            use_gpu: Whether to use GPU for FAISS.

        Returns:
            Initialized MemoryBank with features indexed.
        """
        feature_dim = features.shape[1]
        bank = cls(feature_dim=feature_dim, use_faiss=use_faiss, use_gpu=use_gpu)
        bank.set_features(features)
        return bank

    def __len__(self) -> int:
        if self.features is None:
            return 0
        return len(self.features)


def greedy_coreset_selection(
    features: np.ndarray,
    target_size: int,
    seed: int = 42,
    use_projection: bool = False,
) -> np.ndarray:
    """Select a representative subset using greedy k-center coreset algorithm.

    The algorithm iteratively selects the point that is furthest from the
    current set of selected points, maximizing coverage of the feature space.

    Uses PyTorch GPU acceleration when available for fast distance computation.

    When use_projection=True, applies Sparse Random Projection (as per original
    PatchCore paper) to reduce feature dimensionality before computing distances,
    significantly speeding up the selection process for high-dimensional features.

    Args:
        features: Feature array of shape [N, D].
        target_size: Number of samples to select.
        seed: Random seed for initial selection.
        use_projection: If True, apply Sparse Random Projection before distance
                       computation (recommended for large datasets).

    Returns:
        Array of selected indices.
    """
    np.random.seed(seed)
    n_samples = features.shape[0]

    if target_size >= n_samples:
        return np.arange(n_samples)

    # Convert to float32 and ensure contiguous
    features = np.ascontiguousarray(features.astype(np.float32))

    # Apply dimensionality reduction if requested (as per original PatchCore paper)
    if use_projection:
        features_for_selection = apply_sparse_random_projection(features, eps=0.9, seed=seed)
    else:
        features_for_selection = features

    # Try to use PyTorch GPU for acceleration
    if torch.cuda.is_available():
        # Check if we have enough GPU memory (features_for_selection + overhead)
        feature_memory_gb = features_for_selection.nbytes / (1024 ** 3)
        required_memory_gb = feature_memory_gb * 1.5  # 1.5x for safety margin (cdist overhead)

        torch.cuda.empty_cache()  # Clear any cached memory first
        free_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        allocated_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
        available_gb = free_memory_gb - allocated_gb - 1.0  # Keep 1GB buffer

        if required_memory_gb > available_gb:
            logger.info(f"GPU memory insufficient for coreset ({required_memory_gb:.1f}GB needed, {available_gb:.1f}GB available). Using CPU.")
        else:
            try:
                return _greedy_coreset_pytorch_gpu(features_for_selection, target_size, seed)
            except Exception as e:
                logger.warning(f"PyTorch GPU coreset failed, falling back to CPU: {e}")
                torch.cuda.empty_cache()  # Clean up after failure

    # CPU fallback
    return _greedy_coreset_cpu(features_for_selection, target_size, seed)


def _greedy_coreset_pytorch_gpu(
    features: np.ndarray,
    target_size: int,
    seed: int,
) -> np.ndarray:
    """GPU-accelerated greedy coreset selection using PyTorch."""
    n_samples = features.shape[0]

    # Move features to GPU
    device = torch.device("cuda")
    features_gpu = torch.from_numpy(features).to(device)

    # Initialize with random sample
    np.random.seed(seed)
    selected = [np.random.randint(n_samples)]
    min_distances = torch.full((n_samples,), float('inf'), device=device)

    for i in tqdm(range(target_size - 1), desc="Coreset selection"):
        # Compute distances from last selected point to all points
        last_selected = features_gpu[selected[-1]].unsqueeze(0)
        distances = torch.cdist(features_gpu, last_selected).squeeze(1)

        # Update minimum distances
        min_distances = torch.minimum(min_distances, distances)

        # Mask already selected points
        min_distances[selected[-1]] = -1

        # Select point with maximum minimum distance
        next_idx = torch.argmax(min_distances).item()
        selected.append(next_idx)

    return np.array(selected)


def _greedy_coreset_cpu(
    features: np.ndarray,
    target_size: int,
    seed: int,
) -> np.ndarray:
    """CPU fallback for greedy coreset selection."""
    n_samples = features.shape[0]

    np.random.seed(seed)
    selected = [np.random.randint(n_samples)]
    min_distances = np.full(n_samples, np.inf, dtype=np.float32)

    logger.info(f"  Coreset selection (CPU): 0/{target_size}")

    for i in range(target_size - 1):
        # Update minimum distances based on last selected point
        last_selected_feature = features[selected[-1]]
        distances = np.linalg.norm(features - last_selected_feature, axis=1)
        min_distances = np.minimum(min_distances, distances)

        # Exclude already selected points
        min_distances[selected] = -1

        # Select point with maximum minimum distance
        next_idx = np.argmax(min_distances)
        selected.append(next_idx)

        if (i + 1) % 1000 == 0:
            logger.info(f"  Coreset selection (CPU): {i + 1}/{target_size}")

    return np.array(selected)


def apply_sparse_random_projection(
    features: np.ndarray,
    eps: float = 0.9,
    seed: int = 42,
) -> np.ndarray:
    """Reduce dimensionality using Sparse Random Projection (Johnson-Lindenstrauss).

    This is used to speed up greedy k-center coreset selection by reducing the
    feature dimension before computing distances. Based on the original PatchCore
    paper and Anomalib implementation.

    Args:
        features: Input features array of shape (n_samples, feature_dim).
        eps: Distortion tolerance for Johnson-Lindenstrauss lemma (default 0.9).
             Higher values = more compression, lower values = better preservation.
        seed: Random seed for reproducibility.

    Returns:
        Projected features with reduced dimensionality.
    """
    from sklearn.random_projection import SparseRandomProjection

    n_samples, feature_dim = features.shape

    projector = SparseRandomProjection(
        n_components='auto',  # Auto-calculate via J-L lemma
        eps=eps,
        random_state=seed,
        density='auto'
    )

    projected = projector.fit_transform(features)
    logger.info(
        f"Sparse Random Projection: {feature_dim} -> {projected.shape[1]} dimensions "
        f"(eps={eps}, n_samples={n_samples})"
    )

    return projected.astype(np.float32)


def kmeans_coreset_selection(
    features: np.ndarray,
    target_size: int,
    seed: int = 42,
    batch_size: int = 1024,
) -> np.ndarray:
    """K-means based coreset selection - memory efficient alternative to greedy.

    Uses MiniBatchKMeans which processes data in chunks, avoiding OOM.
    Returns cluster centroids as representative features.

    This is recommended for large images (e.g., 1000x750) where greedy coreset
    selection would cause OOM due to the large number of patches.

    Args:
        features: Input features array of shape (n_samples, feature_dim).
        target_size: Number of coreset samples (clusters) to select.
        seed: Random seed for reproducibility.
        batch_size: Mini-batch size for K-means (controls memory usage).

    Returns:
        Selected coreset features of shape (target_size, feature_dim).
        Note: Returns features directly, not indices.
    """
    from sklearn.cluster import MiniBatchKMeans

    n_samples = features.shape[0]

    if target_size >= n_samples:
        return features.copy()

    logger.info(f"Running MiniBatchKMeans with {target_size} clusters on {n_samples} samples...")

    kmeans = MiniBatchKMeans(
        n_clusters=target_size,
        random_state=seed,
        batch_size=batch_size,
        n_init=3,
        max_iter=100,
        verbose=0,
    )

    kmeans.fit(features)

    logger.info(f"K-means clustering complete. Returning {target_size} centroids.")
    return kmeans.cluster_centers_.astype(np.float32)

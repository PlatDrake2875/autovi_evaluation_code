"""Memory bank implementation for PatchCore with coreset subsampling."""

from typing import Optional, Tuple

import numpy as np
import torch
from loguru import logger

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
    ) -> None:
        """Build the memory bank from training features.

        Args:
            features: Training patch features of shape [N, D].
            coreset_ratio: Fraction of features to keep (0.0 to 1.0).
            max_samples: Maximum number of samples to keep (overrides ratio if set).
            seed: Random seed for reproducibility.
        """
        n_samples = features.shape[0]

        # Determine target size
        if max_samples is not None:
            target_size = min(max_samples, n_samples)
        else:
            target_size = max(1, int(n_samples * coreset_ratio))

        logger.info(f"Selecting {target_size} samples from {n_samples} via coreset subsampling...")

        # Apply coreset subsampling
        if target_size < n_samples:
            selected_indices = greedy_coreset_selection(
                features, target_size=target_size, seed=seed
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

        # Optionally move to GPU
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

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
) -> np.ndarray:
    """Select a representative subset using greedy k-center coreset algorithm.

    The algorithm iteratively selects the point that is furthest from the
    current set of selected points, maximizing coverage of the feature space.

    Args:
        features: Feature array of shape [N, D].
        target_size: Number of samples to select.
        seed: Random seed for initial selection.

    Returns:
        Array of selected indices.
    """
    np.random.seed(seed)
    n_samples = features.shape[0]

    if target_size >= n_samples:
        return np.arange(n_samples)

    # Initialize with random sample
    selected = [np.random.randint(n_samples)]
    min_distances = np.full(n_samples, np.inf)

    # Convert to float32 for efficiency
    features = features.astype(np.float32)

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
            logger.debug(f"  Coreset selection: {i + 1}/{target_size - 1}")

    return np.array(selected)


def random_subsampling(
    features: np.ndarray,
    target_size: int,
    seed: int = 42,
) -> np.ndarray:
    """Simple random subsampling (baseline alternative to coreset).

    Args:
        features: Feature array of shape [N, D].
        target_size: Number of samples to select.
        seed: Random seed.

    Returns:
        Array of selected indices.
    """
    np.random.seed(seed)
    n_samples = features.shape[0]

    if target_size >= n_samples:
        return np.arange(n_samples)

    return np.random.choice(n_samples, size=target_size, replace=False)

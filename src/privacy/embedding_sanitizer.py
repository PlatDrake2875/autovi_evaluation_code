"""Embedding sanitizer for differential privacy on feature embeddings."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from loguru import logger

from .gaussian_mechanism import GaussianMechanism


@dataclass
class DPConfig:
    """Configuration for differential privacy.

    Attributes:
        enabled: Whether DP is enabled.
        epsilon: Privacy parameter (lower = more private).
        delta: Failure probability.
        clipping_norm: Maximum L2 norm for embeddings.
    """

    enabled: bool = False
    epsilon: float = 1.0
    delta: float = 1e-5
    clipping_norm: float = 1.0

    def __post_init__(self):
        """Validate configuration values."""
        if self.enabled:
            if not 0.1 <= self.epsilon <= 10.0:
                raise ValueError(f"Epsilon must be in [0.1, 10.0], got {self.epsilon}")
            if not 0 < self.delta < 1:
                raise ValueError(f"Delta must be in (0, 1), got {self.delta}")
            if self.clipping_norm <= 0:
                raise ValueError(f"Clipping norm must be positive, got {self.clipping_norm}")


class EmbeddingSanitizer:
    """Sanitizes embeddings using differential privacy.

    Applies L2 norm clipping followed by Gaussian noise addition
    to achieve (epsilon, delta)-differential privacy.

    Attributes:
        config: DP configuration.
        mechanism: Gaussian mechanism for noise addition.
        stats: Statistics from sanitization operations.
    """

    def __init__(self, config: DPConfig):
        """Initialize the embedding sanitizer.

        Args:
            config: Differential privacy configuration.
        """
        self.config = config
        self.mechanism: Optional[GaussianMechanism] = None
        self._sanitization_count = 0
        self._stats: Dict = {
            "num_sanitizations": 0,
            "total_embeddings_processed": 0,
            "embeddings_clipped": 0,
            "avg_norm_before_clip": 0.0,
            "avg_norm_after_clip": 0.0,
        }

        if config.enabled:
            # Initialize Gaussian mechanism with clipping norm as sensitivity
            self.mechanism = GaussianMechanism(
                epsilon=config.epsilon,
                delta=config.delta,
                sensitivity=config.clipping_norm,
            )
            logger.info(
                f"EmbeddingSanitizer initialized: epsilon={config.epsilon}, "
                f"delta={config.delta}, clipping_norm={config.clipping_norm}"
            )
        else:
            logger.info("EmbeddingSanitizer initialized with DP disabled")

    def clip_embeddings(
        self, embeddings: np.ndarray
    ) -> Tuple[np.ndarray, int, float, float]:
        """Clip embeddings to maximum L2 norm.

        Projects embeddings with L2 norm > clipping_norm back to the
        ball of radius clipping_norm.

        Args:
            embeddings: Input embeddings of shape [N, D].

        Returns:
            Tuple of (clipped embeddings, num_clipped, avg_norm_before, avg_norm_after).
        """
        # Compute L2 norms
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Compute scaling factors (1.0 for norms <= clipping_norm)
        scale = np.minimum(1.0, self.config.clipping_norm / (norms + 1e-8))

        # Apply clipping
        clipped = embeddings * scale

        # Track statistics
        num_clipped = int(np.sum(norms.flatten() > self.config.clipping_norm))
        avg_norm_before = float(np.mean(norms))
        avg_norm_after = float(np.mean(np.linalg.norm(clipped, axis=1)))

        logger.debug(
            f"Clipped embeddings: {num_clipped}/{len(embeddings)} exceeded norm, "
            f"avg norm {avg_norm_before:.4f} -> {avg_norm_after:.4f}"
        )

        return clipped.astype(embeddings.dtype), num_clipped, avg_norm_before, avg_norm_after

    def sanitize(
        self,
        embeddings: np.ndarray,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Sanitize embeddings with differential privacy.

        Applies L2 clipping followed by Gaussian noise addition.

        Args:
            embeddings: Input embeddings of shape [N, D].
            seed: Optional random seed for reproducibility.

        Returns:
            Sanitized embeddings with same shape.
        """
        if not self.config.enabled or self.mechanism is None:
            logger.debug("DP disabled, returning original embeddings")
            return embeddings

        logger.info(f"Sanitizing {len(embeddings)} embeddings with DP...")

        # Step 1: Clip embeddings to bound sensitivity
        clipped, num_clipped, avg_before, avg_after = self.clip_embeddings(embeddings)

        # Step 2: Add calibrated Gaussian noise
        sanitized = self.mechanism.add_noise(clipped, seed=seed)

        # Update statistics
        self._sanitization_count += 1
        self._stats["num_sanitizations"] = self._sanitization_count
        self._stats["total_embeddings_processed"] += len(embeddings)
        self._stats["embeddings_clipped"] += num_clipped
        self._stats["avg_norm_before_clip"] = avg_before
        self._stats["avg_norm_after_clip"] = avg_after
        self._stats["dp_epsilon"] = self.config.epsilon
        self._stats["dp_delta"] = self.config.delta
        self._stats["sigma"] = self.mechanism.get_sigma()

        logger.info(
            f"Sanitization complete: clipped={num_clipped}/{len(embeddings)}, "
            f"sigma={self.mechanism.get_sigma():.4f}"
        )

        return sanitized

    def get_privacy_spent(self) -> Tuple[float, float]:
        """Get the privacy budget spent so far.

        Returns:
            Tuple of (epsilon, delta) for single sanitization.
        """
        if not self.config.enabled:
            return (0.0, 0.0)
        return (self.config.epsilon, self.config.delta)

    def get_stats(self) -> Dict:
        """Get sanitization statistics.

        Returns:
            Dictionary with sanitization statistics.
        """
        return self._stats.copy()

    def __repr__(self) -> str:
        status = "enabled" if self.config.enabled else "disabled"
        return (
            f"EmbeddingSanitizer({status}, epsilon={self.config.epsilon}, "
            f"clipping_norm={self.config.clipping_norm})"
        )

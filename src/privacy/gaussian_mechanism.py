"""Gaussian mechanism for differential privacy."""

import math
from typing import Optional

import numpy as np
from loguru import logger


class GaussianMechanism:
    """Gaussian mechanism for (epsilon, delta)-differential privacy.

    Adds calibrated Gaussian noise to achieve differential privacy guarantees.
    Uses the standard Gaussian mechanism formula from Dwork & Roth.

    Attributes:
        epsilon: Privacy parameter (lower = more private).
        delta: Failure probability.
        sensitivity: L2 sensitivity of the function (bounded by clipping).
        sigma: Computed noise scale.
    """

    def __init__(
        self,
        epsilon: float,
        delta: float,
        sensitivity: float,
    ):
        """Initialize the Gaussian mechanism.

        Args:
            epsilon: Privacy parameter. Must be in [0.1, 10.0].
            delta: Failure probability. Must be in (0, 1).
            sensitivity: L2 sensitivity (typically the clipping norm).

        Raises:
            ValueError: If parameters are out of valid range.
        """
        # Validate parameters
        if not 0.1 <= epsilon <= 10.0:
            raise ValueError(f"Epsilon must be in [0.1, 10.0], got {epsilon}")
        if not 0 < delta < 1:
            raise ValueError(f"Delta must be in (0, 1), got {delta}")
        if sensitivity <= 0:
            raise ValueError(f"Sensitivity must be positive, got {sensitivity}")

        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.sigma = self._compute_sigma()

        logger.debug(
            f"GaussianMechanism initialized: epsilon={epsilon}, "
            f"delta={delta}, sensitivity={sensitivity}, sigma={self.sigma:.4f}"
        )

    def _compute_sigma(self) -> float:
        """Compute the noise scale sigma.

        Uses the standard Gaussian mechanism formula:
        sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon

        Returns:
            Noise scale sigma.
        """
        return self.sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon

    def add_noise(
        self,
        data: np.ndarray,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Add calibrated Gaussian noise to data.

        Args:
            data: Input data array of any shape.
            seed: Optional random seed for reproducibility.

        Returns:
            Noised data with same shape and dtype as input.
        """
        # Create random generator with optional seed
        rng = np.random.default_rng(seed)

        # Generate Gaussian noise
        noise = rng.normal(loc=0.0, scale=self.sigma, size=data.shape)

        # Add noise and preserve dtype
        noised_data = data + noise.astype(data.dtype)

        logger.debug(
            f"Added Gaussian noise: shape={data.shape}, "
            f"sigma={self.sigma:.4f}, seed={seed}"
        )

        return noised_data

    def get_sigma(self) -> float:
        """Get the computed noise scale.

        Returns:
            Noise scale sigma.
        """
        return self.sigma

    def __repr__(self) -> str:
        return (
            f"GaussianMechanism(epsilon={self.epsilon}, delta={self.delta}, "
            f"sensitivity={self.sensitivity}, sigma={self.sigma:.4f})"
        )

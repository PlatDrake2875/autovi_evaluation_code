"""Attack simulations for robustness evaluation.

This module provides attack implementations for testing the robustness
of federated learning systems against Byzantine/malicious clients.
These are for evaluation purposes only.
"""

from typing import List, Optional

import numpy as np
from loguru import logger


class ModelPoisoningAttack:
    """Simulate model poisoning attacks for robustness evaluation.

    This class implements various attack strategies that malicious clients
    might use to poison the federated learning process. It is intended
    for testing and evaluating robustness mechanisms only.

    Supported attack types:
    - "scaling": Multiply updates by a large factor to dominate aggregation
    - "noise": Add random noise to corrupt the update
    - "sign_flip": Flip the sign of updates to reverse learning direction

    Attributes:
        attack_type: Type of attack to simulate.
        scale_factor: Multiplication factor for scaling attack.
        noise_std: Standard deviation of noise for noise attack.
        seed: Random seed for reproducibility.
    """

    VALID_ATTACK_TYPES = ["scaling", "noise", "sign_flip"]

    def __init__(
        self,
        attack_type: str = "scaling",
        scale_factor: float = 100.0,
        noise_std: float = 10.0,
        seed: Optional[int] = None,
    ):
        """Initialize the attack simulator.

        Args:
            attack_type: Type of attack ("scaling", "noise", or "sign_flip").
            scale_factor: Multiplication factor for scaling attack.
            noise_std: Standard deviation of noise for noise attack.
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If attack_type is not valid.
        """
        if attack_type not in self.VALID_ATTACK_TYPES:
            raise ValueError(
                f"attack_type must be one of {self.VALID_ATTACK_TYPES}, "
                f"got {attack_type}"
            )

        self.attack_type = attack_type
        self.scale_factor = scale_factor
        self.noise_std = noise_std
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def apply(
        self,
        client_updates: List[np.ndarray],
        malicious_indices: List[int],
    ) -> List[np.ndarray]:
        """Apply attack to specified client updates.

        Args:
            client_updates: List of client feature arrays.
            malicious_indices: Indices of clients that are malicious.

        Returns:
            Modified client updates with attacks applied to malicious clients.

        Raises:
            ValueError: If malicious_indices are out of bounds.
        """
        if not client_updates:
            return []

        num_clients = len(client_updates)
        for idx in malicious_indices:
            if idx < 0 or idx >= num_clients:
                raise ValueError(
                    f"Malicious index {idx} out of bounds for {num_clients} clients"
                )

        # Copy all updates to avoid modifying originals
        result = [u.copy() for u in client_updates]

        # Apply attack to malicious clients
        for idx in malicious_indices:
            if self.attack_type == "scaling":
                result[idx] = self._scaling_attack(result[idx])
            elif self.attack_type == "noise":
                result[idx] = self._noise_attack(result[idx])
            elif self.attack_type == "sign_flip":
                result[idx] = self._sign_flip_attack(result[idx])

        logger.debug(
            f"ModelPoisoningAttack: applied {self.attack_type} attack to "
            f"{len(malicious_indices)} of {num_clients} clients"
        )

        return result

    def _scaling_attack(self, update: np.ndarray) -> np.ndarray:
        """Scale update by a large factor."""
        return update * self.scale_factor

    def _noise_attack(self, update: np.ndarray) -> np.ndarray:
        """Add random noise to update."""
        noise = self._rng.normal(0, self.noise_std, update.shape)
        return update + noise

    def _sign_flip_attack(self, update: np.ndarray) -> np.ndarray:
        """Flip the sign of the update."""
        return -update

    def get_attack_stats(self, num_clients: int, malicious_indices: List[int]) -> dict:
        """Get statistics about the attack configuration.

        Args:
            num_clients: Total number of clients.
            malicious_indices: Indices of malicious clients.

        Returns:
            Dictionary with attack statistics.
        """
        num_malicious = len(malicious_indices)
        return {
            "attack_type": self.attack_type,
            "num_clients": num_clients,
            "num_malicious": num_malicious,
            "malicious_fraction": num_malicious / num_clients if num_clients > 0 else 0,
            "malicious_indices": malicious_indices,
            "scale_factor": self.scale_factor if self.attack_type == "scaling" else None,
            "noise_std": self.noise_std if self.attack_type == "noise" else None,
        }

    def __repr__(self) -> str:
        if self.attack_type == "scaling":
            return f"ModelPoisoningAttack(type={self.attack_type}, scale={self.scale_factor})"
        elif self.attack_type == "noise":
            return f"ModelPoisoningAttack(type={self.attack_type}, std={self.noise_std})"
        else:
            return f"ModelPoisoningAttack(type={self.attack_type})"

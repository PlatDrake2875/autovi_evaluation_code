"""Federated memory bank aggregation strategies for PatchCore."""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from src.models.memory_bank import greedy_coreset_selection


def weighted_sampling(
    client_coresets: List[np.ndarray],
    target_samples: int,
    weights: Optional[List[float]] = None,
    oversample_factor: float = 2.0,
    seed: int = 42,
) -> np.ndarray:
    """Sample features from client coresets with optional weighting.

    Args:
        client_coresets: List of coreset arrays from each client.
        target_samples: Target number of samples to collect.
        weights: Optional weights for each client. If None, weights by data size.
        oversample_factor: Factor to oversample before final selection.
        seed: Random seed for reproducibility.

    Returns:
        Sampled features as numpy array.
    """
    np.random.seed(seed)

    # Compute weights if not provided
    if weights is None:
        total_samples = sum(len(c) for c in client_coresets)
        weights = [len(c) / total_samples for c in client_coresets]

    # Sample from each client proportionally
    sampled_features = []
    for coreset, weight in zip(client_coresets, weights):
        n_samples = int(target_samples * weight * oversample_factor)
        n_samples = max(1, min(n_samples, len(coreset)))

        if len(coreset) > n_samples:
            indices = np.random.choice(len(coreset), n_samples, replace=False)
            sampled_features.append(coreset[indices])
        else:
            sampled_features.append(coreset)

    return np.concatenate(sampled_features, axis=0)


def federated_aggregate(
    client_coresets: List[np.ndarray],
    global_bank_size: int = 10000,
    weighted_by_samples: bool = True,
    oversample_factor: float = 2.0,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict]:
    """Aggregate local memory banks from all clients into a global memory bank.

    This implements the Federated Coreset strategy:
    1. Weight client contributions by data size (for fairness)
    2. Oversample from each client proportionally
    3. Concatenate all sampled features
    4. Apply global coreset selection for diversity

    Args:
        client_coresets: List of local coreset arrays from each client.
        global_bank_size: Target size for the global memory bank.
        weighted_by_samples: If True, weight clients by their data size.
        oversample_factor: Factor to oversample before final coreset selection.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of:
            - global_bank: Global memory bank as numpy array [global_bank_size, D]
            - stats: Dictionary with aggregation statistics
    """
    np.random.seed(seed)

    # Filter out empty coresets
    valid_coresets = [c for c in client_coresets if len(c) > 0]
    if len(valid_coresets) == 0:
        raise ValueError("All client coresets are empty")

    # Compute statistics
    client_sizes = [len(c) for c in valid_coresets]
    total_samples = sum(client_sizes)

    logger.info(f"Aggregating {len(valid_coresets)} client coresets...")
    logger.debug(f"Client sizes: {client_sizes}")
    logger.debug(f"Total samples: {total_samples}")

    # Compute weights
    if weighted_by_samples:
        weights = [size / total_samples for size in client_sizes]
    else:
        # Equal weights
        weights = [1.0 / len(valid_coresets)] * len(valid_coresets)

    # Sample from each client proportionally (with oversampling)
    sampled_features = []
    for i, (coreset, weight) in enumerate(zip(valid_coresets, weights)):
        # Calculate number of samples for this client
        n_samples = int(global_bank_size * weight * oversample_factor)
        n_samples = max(1, min(n_samples, len(coreset)))

        if len(coreset) > n_samples:
            indices = np.random.choice(len(coreset), n_samples, replace=False)
            sampled = coreset[indices]
        else:
            sampled = coreset

        sampled_features.append(sampled)
        logger.debug(f"  Client {i}: sampled {len(sampled)} patches (weight={weight:.3f})")

    # Concatenate all sampled features
    all_features = np.concatenate(sampled_features, axis=0)
    logger.debug(f"Concatenated features: {all_features.shape}")

    # Apply global coreset selection for diversity
    if len(all_features) > global_bank_size:
        logger.info(f"Applying global coreset selection ({global_bank_size} from {len(all_features)})...")
        selected_indices = greedy_coreset_selection(
            all_features, target_size=global_bank_size, seed=seed
        )
        global_bank = all_features[selected_indices].copy()
    else:
        global_bank = all_features.copy()

    logger.info(f"Global memory bank size: {global_bank.shape}")

    # Compile statistics
    stats = {
        "num_clients": len(valid_coresets),
        "client_sizes": client_sizes,
        "client_weights": weights,
        "total_input_samples": total_samples,
        "concatenated_size": len(all_features),
        "global_bank_size": len(global_bank),
        "feature_dim": global_bank.shape[1] if len(global_bank) > 0 else 0,
    }

    return global_bank, stats


def simple_concatenate(
    client_coresets: List[np.ndarray],
    global_bank_size: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict]:
    """Simple concatenation strategy without weighted sampling.

    Concatenates all client coresets and optionally applies coreset selection.

    Args:
        client_coresets: List of local coreset arrays from each client.
        global_bank_size: Optional target size for global memory bank.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of:
            - global_bank: Global memory bank as numpy array
            - stats: Dictionary with aggregation statistics
    """
    # Filter out empty coresets
    valid_coresets = [c for c in client_coresets if len(c) > 0]
    if len(valid_coresets) == 0:
        raise ValueError("All client coresets are empty")

    # Simple concatenation
    all_features = np.concatenate(valid_coresets, axis=0)

    # Apply coreset if needed
    if global_bank_size is not None and len(all_features) > global_bank_size:
        selected_indices = greedy_coreset_selection(
            all_features, target_size=global_bank_size, seed=seed
        )
        global_bank = all_features[selected_indices].copy()
    else:
        global_bank = all_features.copy()

    stats = {
        "num_clients": len(valid_coresets),
        "client_sizes": [len(c) for c in valid_coresets],
        "concatenated_size": len(all_features),
        "global_bank_size": len(global_bank),
    }

    return global_bank, stats


def diversity_preserving_aggregate(
    client_coresets: List[np.ndarray],
    global_bank_size: int = 10000,
    min_samples_per_client: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict]:
    """Aggregation strategy that ensures minimum representation from each client.

    This strategy ensures that each client contributes at least a minimum
    number of samples to the global bank, preserving diversity.

    Args:
        client_coresets: List of local coreset arrays from each client.
        global_bank_size: Target size for the global memory bank.
        min_samples_per_client: Minimum samples to take from each client.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of:
            - global_bank: Global memory bank as numpy array
            - stats: Dictionary with aggregation statistics
    """
    np.random.seed(seed)

    valid_coresets = [c for c in client_coresets if len(c) > 0]
    if len(valid_coresets) == 0:
        raise ValueError("All client coresets are empty")

    num_clients = len(valid_coresets)

    # Calculate samples per client
    # First, ensure minimum per client
    guaranteed_per_client = min(min_samples_per_client, global_bank_size // num_clients)
    remaining_budget = global_bank_size - (guaranteed_per_client * num_clients)

    # Distribute remaining budget proportionally
    client_sizes = [len(c) for c in valid_coresets]
    total_samples = sum(client_sizes)
    weights = [size / total_samples for size in client_sizes]

    sampled_features = []
    actual_samples = []

    for i, (coreset, weight) in enumerate(zip(valid_coresets, weights)):
        # Guaranteed samples + proportional share of remaining
        n_samples = guaranteed_per_client + int(remaining_budget * weight)
        n_samples = max(1, min(n_samples, len(coreset)))

        if len(coreset) > n_samples:
            indices = np.random.choice(len(coreset), n_samples, replace=False)
            sampled = coreset[indices]
        else:
            sampled = coreset

        sampled_features.append(sampled)
        actual_samples.append(len(sampled))

    # Concatenate
    all_features = np.concatenate(sampled_features, axis=0)

    # Final coreset if over budget
    if len(all_features) > global_bank_size:
        selected_indices = greedy_coreset_selection(
            all_features, target_size=global_bank_size, seed=seed
        )
        global_bank = all_features[selected_indices].copy()
    else:
        global_bank = all_features.copy()

    stats = {
        "num_clients": num_clients,
        "client_sizes": client_sizes,
        "samples_per_client": actual_samples,
        "global_bank_size": len(global_bank),
    }

    return global_bank, stats


# Strategy registry for the Open/Closed Principle
# Maps strategy names to their implementation functions
STRATEGY_REGISTRY: Dict[str, Callable[..., Tuple[np.ndarray, Dict]]] = {
    "federated_coreset": federated_aggregate,
    "simple_concatenate": simple_concatenate,
    "diversity_preserving": diversity_preserving_aggregate,
}

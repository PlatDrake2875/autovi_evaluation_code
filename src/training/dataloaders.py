"""Dataloader creation utilities for AutoVI training."""

from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import DataLoader


def autovi_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for AutoVI dataset batches.

    Stacks images into a tensor and collects labels and categories.

    Args:
        batch: List of sample dictionaries with 'image', 'label', 'category' keys.

    Returns:
        Dictionary with:
            - 'image': Stacked tensor [B, C, H, W]
            - 'label': Tensor of labels [B]
            - 'category': List of category strings
    """
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    categories = [item["category"] for item in batch]
    return {"image": images, "label": labels, "category": categories}


def create_federated_dataloaders(
    dataset: Any,
    partition: Dict[int, List[int]],
    batch_size: int = 32,
    num_workers: int = 4,
    collate_fn: Optional[Callable] = None,
    pin_memory: bool = True,
    shuffle: bool = True,
) -> Dict[int, DataLoader]:
    """Create dataloaders for federated client partitions.

    Args:
        dataset: Source dataset that supports indexing.
        partition: Dictionary mapping client_id -> list of sample indices.
        batch_size: Batch size for each dataloader.
        num_workers: Number of worker processes for data loading.
        collate_fn: Optional custom collate function. Defaults to autovi_collate_fn.
        pin_memory: Whether to pin memory for faster GPU transfer.
        shuffle: Whether to shuffle data in each dataloader.

    Returns:
        Dictionary mapping client_id -> DataLoader.
    """
    from src.data.datasets import TransformedSubset

    if collate_fn is None:
        collate_fn = autovi_collate_fn

    dataloaders = {}
    for client_id, indices in partition.items():
        subset = TransformedSubset(dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory and torch.cuda.is_available(),
        )
        dataloaders[client_id] = loader

    return dataloaders

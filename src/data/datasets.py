"""Dataset wrapper classes for AutoVI training."""

from typing import Any, Callable, Dict, List


class CategoryTransformDataset:
    """Wrapper dataset that applies category-specific transforms.

    This class wraps an existing dataset and applies different transforms
    based on each sample's category. Useful for datasets where different
    object categories require different preprocessing (e.g., different sizes).

    Args:
        dataset: Source dataset that returns dicts with 'image' and 'category' keys.
        transforms_dict: Dictionary mapping category names to transform functions.

    Example:
        >>> transforms = {
        ...     "pipe_clip": get_transforms("pipe_clip"),
        ...     "tank_screw": get_transforms("tank_screw"),
        ... }
        >>> wrapped = CategoryTransformDataset(dataset, transforms)
        >>> sample = wrapped[0]  # Transform applied based on sample's category
    """

    def __init__(
        self,
        dataset: Any,
        transforms_dict: Dict[str, Callable],
    ):
        self.dataset = dataset
        self.transforms = transforms_dict

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        category = item["category"]
        if category in self.transforms:
            item["image"] = self.transforms[category](item["image"])
        return item


class TransformedSubset:
    """Subset wrapper for indexed access to transformed datasets.

    Creates a subset of a dataset using specified indices. Works with
    any dataset that supports integer indexing.

    Args:
        dataset: Source dataset supporting __getitem__ with integer indices.
        indices: List of indices to include in the subset.

    Example:
        >>> subset = TransformedSubset(dataset, [0, 5, 10, 15])
        >>> len(subset)  # Returns 4
        >>> subset[0]    # Returns dataset[0]
    """

    def __init__(self, dataset: Any, indices: List[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Any:
        return self.dataset[self.indices[idx]]

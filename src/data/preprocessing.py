"""Preprocessing pipeline for AutoVI dataset."""

from typing import Callable, List, Optional

import numpy as np
from PIL import Image

try:
    import torch
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .autovi_dataset import SMALL_OBJECTS, LARGE_OBJECTS, get_resize_shape

# ImageNet normalization statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(
    category: str,
    normalize: bool = True,
    to_tensor: bool = True,
    augment: bool = False,
) -> Callable:
    """Get preprocessing transforms for a category.

    Args:
        category: Object category name.
        normalize: Whether to apply ImageNet normalization.
        to_tensor: Whether to convert to PyTorch tensor.
        augment: Whether to apply data augmentation.

    Returns:
        Composed transform callable.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for transforms. Install with: pip install torch torchvision")

    resize_shape = get_resize_shape(category)

    transforms_list = [
        T.Resize(resize_shape),
    ]

    if augment:
        transforms_list.extend([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.1, contrast=0.1),
        ])

    if to_tensor:
        transforms_list.append(T.ToTensor())

    if normalize:
        transforms_list.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    return T.Compose(transforms_list)


class Preprocessor:
    """Preprocessing pipeline for AutoVI images.

    Handles resizing, normalization, and optional augmentation.
    """

    def __init__(
        self,
        normalize: bool = True,
        to_tensor: bool = True,
        augment: bool = False,
        cache_dir: Optional[str] = None,
    ):
        """Initialize the preprocessor.

        Args:
            normalize: Whether to apply ImageNet normalization.
            to_tensor: Whether to convert to PyTorch tensor.
            augment: Whether to apply data augmentation.
            cache_dir: Optional directory for caching preprocessed images.
        """
        self.normalize = normalize
        self.to_tensor = to_tensor
        self.augment = augment
        self.cache_dir = cache_dir

        # Build category-specific transforms
        self._transforms = {}
        for category in SMALL_OBJECTS + LARGE_OBJECTS:
            self._transforms[category] = get_transforms(
                category=category,
                normalize=normalize,
                to_tensor=to_tensor,
                augment=augment,
            )

    def __call__(self, image: Image.Image, category: str):
        """Apply preprocessing to an image.

        Args:
            image: PIL Image to preprocess.
            category: Object category for size determination.

        Returns:
            Preprocessed image (tensor if to_tensor=True).
        """
        transform = self._transforms.get(category)
        if transform is None:
            raise ValueError(f"Unknown category: {category}")
        return transform(image)

    def preprocess_batch(
        self,
        images: List[Image.Image],
        categories: List[str],
    ):
        """Preprocess a batch of images.

        Args:
            images: List of PIL Images.
            categories: List of categories corresponding to each image.

        Returns:
            Stacked tensor of preprocessed images if to_tensor=True.
        """
        if len(images) != len(categories):
            raise ValueError("Images and categories must have same length")

        processed = [self(img, cat) for img, cat in zip(images, categories)]

        if self.to_tensor and TORCH_AVAILABLE:
            return torch.stack(processed)
        return processed


def resize_for_category(image: Image.Image, category: str) -> Image.Image:
    """Resize an image according to its category.

    Args:
        image: PIL Image to resize.
        category: Object category name.

    Returns:
        Resized PIL Image.
    """
    resize_shape = get_resize_shape(category)
    return image.resize(resize_shape, Image.BILINEAR)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image using ImageNet statistics.

    Args:
        image: Image array with values in [0, 255].

    Returns:
        Normalized image array.
    """
    image = image.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD).reshape(1, 1, 3)
    return (image - mean) / std


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """Reverse ImageNet normalization.

    Args:
        image: Normalized image array.

    Returns:
        Image array with values in [0, 255].
    """
    mean = np.array(IMAGENET_MEAN).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD).reshape(1, 1, 3)
    image = image * std + mean
    return (image * 255).clip(0, 255).astype(np.uint8)


def get_test_transform(
    resize_shape: tuple,
    normalize: bool = True,
) -> Callable:
    """Get test/evaluation transforms.

    Args:
        resize_shape: Target size (width, height).
        normalize: Whether to apply ImageNet normalization.

    Returns:
        Composed transform callable.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for transforms.")

    transforms_list = [
        T.Resize(resize_shape),
        T.ToTensor(),
    ]

    if normalize:
        transforms_list.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    return T.Compose(transforms_list)


def get_train_transform(
    resize_shape: tuple,
    normalize: bool = True,
    augment: bool = False,
) -> Callable:
    """Get training transforms.

    Args:
        resize_shape: Target size (width, height).
        normalize: Whether to apply ImageNet normalization.
        augment: Whether to apply data augmentation.

    Returns:
        Composed transform callable.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for transforms.")

    transforms_list = [
        T.Resize(resize_shape),
    ]

    if augment:
        transforms_list.extend([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ])

    transforms_list.append(T.ToTensor())

    if normalize:
        transforms_list.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    return T.Compose(transforms_list)

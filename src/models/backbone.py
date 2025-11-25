"""Feature extraction backbone for PatchCore using WideResNet-50-2."""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class FeatureExtractor(nn.Module):
    """WideResNet-50-2 backbone for extracting multi-scale features.

    Extracts features from layer2 and layer3 of WideResNet-50-2, concatenates them
    after upsampling layer2 to match layer3 resolution, and optionally applies
    local neighborhood averaging.

    Attributes:
        backbone: The WideResNet-50-2 model.
        layers: List of layer names to extract features from.
        features: Dictionary storing hooked features during forward pass.
    """

    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        layers: List[str] = None,
        pretrained: bool = True,
    ):
        """Initialize the feature extractor.

        Args:
            backbone_name: Name of the backbone model.
            layers: List of layer names to extract features from.
            pretrained: Whether to use ImageNet pretrained weights.
        """
        super().__init__()

        self.layers = layers if layers else ["layer2", "layer3"]
        self.features: Dict[str, torch.Tensor] = {}

        # Load backbone
        if backbone_name == "wide_resnet50_2":
            weights = models.Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.wide_resnet50_2(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Freeze backbone weights
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Register forward hooks
        self._register_hooks()

        # Set to eval mode
        self.backbone.eval()

    def _register_hooks(self) -> None:
        """Register forward hooks on specified layers."""
        for layer_name in self.layers:
            layer = getattr(self.backbone, layer_name)
            layer.register_forward_hook(self._get_hook(layer_name))

    def _get_hook(self, layer_name: str):
        """Create a hook function for a specific layer."""

        def hook(module, input, output):
            self.features[layer_name] = output

        return hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and concatenate multi-scale features.

        Args:
            x: Input tensor of shape [B, 3, H, W].

        Returns:
            Concatenated features of shape [B, C, H', W'] where C is the sum
            of channel dimensions from all extracted layers.
        """
        self.features.clear()

        # Forward pass through backbone
        with torch.no_grad():
            _ = self.backbone(x)

        # Get features from hooked layers
        layer2_features = self.features["layer2"]  # [B, 512, H/4, W/4]
        layer3_features = self.features["layer3"]  # [B, 1024, H/8, W/8]

        # Upsample layer2 to match layer3 spatial dimensions
        target_size = layer3_features.shape[-2:]
        layer2_upsampled = F.interpolate(
            layer2_features,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        # Concatenate features
        concatenated = torch.cat([layer2_upsampled, layer3_features], dim=1)

        return concatenated

    def get_feature_dim(self) -> int:
        """Get the output feature dimension.

        Returns:
            Total feature dimension (512 + 1024 = 1536 for default layers).
        """
        dim = 0
        if "layer2" in self.layers:
            dim += 512
        if "layer3" in self.layers:
            dim += 1024
        return dim


def apply_local_neighborhood_averaging(
    features: torch.Tensor,
    neighborhood_size: int = 3,
) -> torch.Tensor:
    """Apply local neighborhood averaging to patch features.

    This aggregates information from neighboring patches to make features
    more robust and locally-aware.

    Args:
        features: Feature tensor of shape [B, C, H, W].
        neighborhood_size: Size of the averaging kernel (must be odd).

    Returns:
        Averaged features of the same shape.
    """
    if neighborhood_size <= 1:
        return features

    padding = neighborhood_size // 2
    kernel = torch.ones(1, 1, neighborhood_size, neighborhood_size, device=features.device)
    kernel = kernel / (neighborhood_size * neighborhood_size)

    # Apply averaging to each channel independently
    B, C, H, W = features.shape
    features_reshaped = features.view(B * C, 1, H, W)
    averaged = F.conv2d(features_reshaped, kernel, padding=padding)
    averaged = averaged.view(B, C, H, W)

    return averaged


def reshape_features_to_patches(features: torch.Tensor) -> torch.Tensor:
    """Reshape spatial feature map to patch vectors.

    Args:
        features: Feature tensor of shape [B, C, H, W].

    Returns:
        Patch features of shape [B * H * W, C].
    """
    B, C, H, W = features.shape
    # Permute to [B, H, W, C] then reshape to [B*H*W, C]
    patches = features.permute(0, 2, 3, 1).reshape(-1, C)
    return patches


def get_patch_locations(
    feature_shape: Tuple[int, int, int, int],
) -> torch.Tensor:
    """Get the spatial locations of each patch.

    Args:
        feature_shape: Shape tuple (B, C, H, W) of the feature map.

    Returns:
        Location tensor of shape [B * H * W, 2] containing (y, x) coordinates.
    """
    B, C, H, W = feature_shape
    y_coords = torch.arange(H).view(-1, 1).expand(H, W).reshape(-1)
    x_coords = torch.arange(W).view(1, -1).expand(H, W).reshape(-1)

    # Repeat for batch dimension
    locations = torch.stack([y_coords, x_coords], dim=1)
    locations = locations.unsqueeze(0).expand(B, -1, -1).reshape(-1, 2)

    return locations

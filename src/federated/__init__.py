"""Federated learning module for PatchCore anomaly detection."""

from .client import PatchCoreClient
from .server import FederatedServer
from .federated_patchcore import FederatedPatchCore
from .strategies.federated_memory import federated_aggregate

__all__ = [
    "PatchCoreClient",
    "FederatedServer",
    "FederatedPatchCore",
    "federated_aggregate",
]

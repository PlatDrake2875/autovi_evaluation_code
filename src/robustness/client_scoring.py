"""Client scoring and anomaly detection for federated learning."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class ClientScore:
    """Score and metadata for a single client.

    Attributes:
        client_id: Identifier for the client.
        score: Anomaly score (higher = more anomalous).
        is_outlier: Whether the client is flagged as an outlier.
        details: Additional details about the scoring.
    """

    client_id: int
    score: float
    is_outlier: bool
    details: Dict[str, Any] = field(default_factory=dict)

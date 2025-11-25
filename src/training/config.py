"""Configuration loading and management utilities."""

from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config file doesn't exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def merge_config_with_args(
    config: Dict[str, Any],
    args: Namespace,
    arg_mappings: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Merge CLI arguments with config file values.

    CLI arguments take precedence over config file values when not None.

    Args:
        config: Base configuration dictionary.
        args: Parsed CLI arguments.
        arg_mappings: Optional mapping of arg names to config paths.
            Format: {"arg_name": "section.key"} or {"arg_name": "key"}

    Returns:
        Merged configuration dictionary.

    Example:
        >>> config = {"federated": {"num_rounds": 1}, "seed": 42}
        >>> args = Namespace(num_rounds=5, seed=None)
        >>> mappings = {"num_rounds": "federated.num_rounds"}
        >>> merge_config_with_args(config, args, mappings)
        {"federated": {"num_rounds": 5}, "seed": 42}
    """
    if arg_mappings is None:
        arg_mappings = {}

    merged = config.copy()

    for arg_name, config_path in arg_mappings.items():
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            # Handle nested paths like "federated.num_rounds"
            parts = config_path.split(".")
            if len(parts) == 1:
                merged[parts[0]] = arg_value
            elif len(parts) == 2:
                if parts[0] not in merged:
                    merged[parts[0]] = {}
                merged[parts[0]][parts[1]] = arg_value

    return merged


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary to save.
        output_path: Path to save the YAML file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

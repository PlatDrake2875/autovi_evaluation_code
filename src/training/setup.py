"""Training setup utilities."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for numpy, torch CPU, and torch CUDA (if available).

    Args:
        seed: Random seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_output_directory(
    output_dir: str,
    config: Optional[Dict[str, Any]] = None,
    config_filename: str = "config.yaml",
) -> Path:
    """Setup output directory and optionally save config.

    Creates the output directory and any parent directories if they don't exist.
    Optionally saves the configuration to a YAML file in the output directory.

    Args:
        output_dir: Path to the output directory.
        config: Optional configuration dictionary to save.
        config_filename: Name of the config file to save (default: "config.yaml").

    Returns:
        Path object for the output directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if config is not None:
        config_save_path = output_path / config_filename
        with open(config_save_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    return output_path


def setup_logging(
    output_dir: Path,
    log_subdir: str = "logs",
    log_prefix: str = "training",
    level: int = logging.INFO,
) -> logging.Logger:
    """Setup logging to file and console.

    Creates a logger that outputs to both a timestamped log file and the console.

    Args:
        output_dir: Base output directory.
        log_subdir: Subdirectory for log files (default: "logs").
        log_prefix: Prefix for log filename (default: "training").
        level: Logging level (default: logging.INFO).

    Returns:
        Configured logger instance.
    """
    log_dir = output_dir / log_subdir
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{log_prefix}_{timestamp}.log"

    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    return logging.getLogger(__name__)


def print_training_header(
    title: str,
    params: Dict[str, Any],
    width: int = 60,
) -> None:
    """Print a formatted training header with parameters.

    Args:
        title: Title for the training run.
        params: Dictionary of parameters to display.
        width: Width of the separator lines.
    """
    print(f"\n{'=' * width}")
    print(title)
    print(f"{'=' * width}")
    for key, value in params.items():
        print(f"{key}: {value}")

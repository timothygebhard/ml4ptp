"""
Methods for configuration files.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Any

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:  # pragma: no cover
    from yaml import Loader  # type: ignore


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def load_yaml(file_path: Path) -> Any:
    """
    Load the (YAML) file at the given `file_path`. The result is
    usually a ``dict`` or a ``list`` of ``dict``s.
    """

    # Open the YAML file and parse its contents
    with open(file_path, 'r') as yaml_file:
        return yaml.load(yaml_file, Loader=Loader)


def load_experiment_config(file_path: Path) -> dict:
    """
    Load the experiment configuration file at the given ``file_path``.
    The configuration file is a YAML file, whose contents are parsed
    and returned as a ``dict``.

    Args:
        file_path: Path to the experiment configuration file.

    Returns:
        A ``dict`` with the experiment configuration.
    """

    return dict(load_yaml(file_path))

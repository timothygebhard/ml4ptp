"""
Methods for configuration files.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:  # pragma: no cover
    from yaml import Loader  # type: ignore


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def load_config(file_path: Path) -> dict:
    """
    Load the (YAML) configuration file at the given ``file_path``.

    Args:
        file_path: Path to a YAML file to be loaded.

    Returns:
        A ``dict`` with the contents of the target YAML file.
    """

    # Open the YAML file and parse its contents
    with open(file_path, 'r') as yaml_file:
        config = dict(yaml.load(yaml_file, Loader=Loader))
    return config

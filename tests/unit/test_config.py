"""
Unit tests for config.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import pytest
import yaml

from ml4ptp.config import load_config


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

@pytest.fixture()
def config_file(tmp_path: Path) -> Path:

    data = dict(a=1, b=2.0, c=False, d=None)

    file_path = tmp_path / 'config.yaml'
    with open(file_path, 'w') as yaml_file:
        yaml.dump(data, yaml_file)

    return file_path


def test__load_config(config_file: Path) -> None:

    # Case 1
    config = load_config(config_file)
    assert isinstance(config, dict)
    assert config['a'] == 1
    assert config['b'] == 2.0
    assert config['c'] is False
    assert config['d'] is None

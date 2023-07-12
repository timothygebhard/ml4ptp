"""
Unit tests for config.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import pytest
import yaml

from ml4ptp.config import load_yaml, load_experiment_config


# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------

@pytest.fixture()
def yaml_file(tmp_path: Path) -> Path:

    data = [
        dict(a=1, b=2.0),
        dict(c=False, d=None),
    ]

    file_path = tmp_path / 'yaml_file.yaml'
    with open(file_path, 'w') as yaml_file:
        yaml.dump(data, yaml_file)

    return file_path


@pytest.fixture()
def experiment_config_file(tmp_path: Path) -> Path:

    data = dict(a=1, b=2.0, c=False, d=None)

    file_path = tmp_path / 'config.yaml'
    with open(file_path, 'w') as yaml_file:
        yaml.dump(data, yaml_file)

    return file_path


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

def test__load_yaml(yaml_file: Path) -> None:

    # Case 1
    data = load_yaml(yaml_file)
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]['a'] == 1
    assert data[0]['b'] == 2.0
    assert data[1]['c'] is False
    assert data[1]['d'] is None


def test__load_experiment_config(experiment_config_file: Path) -> None:

    # Case 1
    config = load_experiment_config(experiment_config_file)
    assert isinstance(config, dict)
    assert config['a'] == 1
    assert config['b'] == 2.0
    assert config['c'] is False
    assert config['d'] is None

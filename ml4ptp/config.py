"""
Functions and utilities related to the general configuration.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import ml4ptp


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_ml4ptp_dir() -> Path:
    return Path(ml4ptp.__file__).parent.parent.resolve()


def get_datasets_dir() -> Path:
    return get_ml4ptp_dir() / 'datasets'


def get_dataset_path(name: str, stage: str) -> Path:
    """
    Get the path to the HDF file that contains the given `dataset`.

    Args:
        name: Name of the dataset (e.g., "pyatmos").
        stage: Either "train" or "test".

    Returns:
        The path to the HDF file for the given `dataset`.
    """

    # Define path to output directory
    path = get_datasets_dir() / name / 'output' / f'{stage}.hdf'

    # Double-check that the target file exists
    if not path.exists():
        raise RuntimeError(f'Target file ({path}) does not exist!')

    return path


def get_experiments_dir() -> Path:
    return get_ml4ptp_dir() / 'experiments'

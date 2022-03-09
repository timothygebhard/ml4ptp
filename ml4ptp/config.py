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


def get_dataset_path(dataset: str) -> Path:
    """
    Get the path to the HDF file that contains the given `dataset`.

    Args:
        dataset: Name of the dataset (e.g., "pyatmos").

    Returns:
        The path to the HDF file for the given `dataset`.
    """

    # Define path to output directory
    path = get_datasets_dir() / dataset / 'output'

    # Depending on the dataset, append the name of the HDF file
    if dataset == 'pyatmos':
        path /= 't_for_fixed_p.hdf'
    elif dataset == 'ms-100k':
        path /= 'ms-100k.hdf'
    else:
        raise ValueError(f'Illegal value for dataset: {dataset}')

    # Double-check that the target file exists
    if not path.exists():
        raise RuntimeError(f'Target file ({path}) does not exist!')

    return path


def get_experiments_dir() -> Path:
    return get_ml4ptp_dir() / 'experiments'

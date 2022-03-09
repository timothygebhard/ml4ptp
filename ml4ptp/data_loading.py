"""
Utility functions for loading data.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Optional, Tuple

import h5py
import numpy as np

from ml4ptp.config import get_dataset_path


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def load_pressure_grid(name: str) -> np.ndarray:
    """
    Load the pressure grid from the dataset of the given `name`.

    (The pressure grid is the same for training and testing, so we do
    not need the `stage` parameter here.)

    Args:
        name: The name of the data set (e.g., "pyatmos").

    Returns:
        A 1D numpy array containing the pressure grid.
    """

    # Get path to (train) dataset, open HDF file, and select pressure grid
    file_path = get_dataset_path(name=name, stage='train')
    with h5py.File(file_path, 'r') as hdf_file:
        pressure_grid = np.array(hdf_file['pressure_grid'])

    return pressure_grid


def load_temperatures(
    name: str,
    stage: str,
    size: Optional[int] = None,
) -> np.ndarray:
    """
    Load the temperatures from the data set of the given `name`.

    Args:
        name: The name of the data set (e.g., "pyatmos").
        stage: Either "train" or "test".
        size: Number of temperature vectors to load.

    Returns:
        A 2D numpy array of shape `(n_samples, grid_size)` containing
        the temperature vectors.
    """

    # Get path to dataset, open HDF file, and select temperatures
    file_path = get_dataset_path(name=name, stage=stage)
    with h5py.File(file_path, 'r') as hdf_file:
        temperatures = np.array(hdf_file['temperatures'][:size])

    # Raise an error if we requested more profiles than we had available
    if length := len(temperatures) < size:
        raise RuntimeError(
            f'Requested {size} profiles, but only {length} could be found!'
        )

    return temperatures


def load_normalization(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load normalization values (mean and standard deviation of the train
    data) for the data set of the given `name`.

    The normalization is always based on the training data set, so we
    do not need the `stage` parameter here.

    Args:
        name: The name of the data set (e.g., "pyatmos").

    Returns:
        A tuple `(train_mean, train_std)` of 1D numpy arrays of shape
        `(grid_size,)` that contain the mean and standard deviation of
        the training data.
    """

    # Get path to dataset, open HDF file, and select temperatures
    file_path = get_dataset_path(name=name, stage='train')
    with h5py.File(file_path, 'r') as hdf_file:
        train_mean = np.array(hdf_file['normalization']['train_mean'])
        train_std = np.array(hdf_file['normalization']['train_std'])

    return train_mean, train_std

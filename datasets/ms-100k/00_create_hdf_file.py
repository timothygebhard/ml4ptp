"""
Combine the different *.txt files of the MS-100k data set into a
single HDF file.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import time

import h5py
import numpy as np

from ml4ptp.config import get_datasets_dir


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print("\nCREATE HDF FILE FOR MS-100k dataset\n", flush=True)

    # -------------------------------------------------------------------------
    # Read in data from text files
    # -------------------------------------------------------------------------

    # Define input and output directory
    input_directory = get_datasets_dir() / 'ms-100k' / 'input'
    output_directory = get_datasets_dir() / 'ms-100k' / 'output'
    output_directory.mkdir(exist_ok=True)

    # Read in the parameters, the pressure grid, and the temperatures
    print('Loading data from *.txt files...', end=' ', flush=True)
    parameters = np.loadtxt(input_directory / 'parameters.txt')
    pressure_grid = np.loadtxt(input_directory / 'pressure_grid.txt')
    temperatures = np.loadtxt(input_directory / 'temperatures.txt')
    print('Done!', flush=True)

    # Print information about shapes
    print('\nShapes:', flush=True)
    print('  parameters:   ', parameters.shape, flush=True)
    print('  pressure_grid:', pressure_grid.shape, flush=True)
    print('  temperatures: ', temperatures.shape, flush=True)
    print('', flush=True)

    # -------------------------------------------------------------------------
    # Randomly shuffle the entire data set
    # -------------------------------------------------------------------------

    # If we want to split the data set into training and test, we need to be
    # sure that we get a representative sample. The easiest solution is to
    # randomly shuffle everything to destroy any particular ordering that is
    # due to the way that they were created.

    print('Randomly shuffling data...', end=' ', flush=True)

    # Create random indices and shuffle them (reproducibly)
    np.random.seed(42)
    random_idx = np.arange(len(parameters))
    np.random.shuffle(random_idx)

    # Use random indices to shuffle the data in a consistent way
    parameters = parameters[random_idx, :]
    temperatures = temperatures[random_idx, :]

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Write everything to the output HDF file
    # -------------------------------------------------------------------------

    print('Writing everything to HDF file...', end=' ', flush=True)

    # Define path and open HDF file
    file_path = output_directory / 'ms-100k.hdf'
    with h5py.File(file_path, 'w') as hdf_file:

        # Store pressure grid and temperatures
        hdf_file.create_dataset(
            name='pressure_grid', data=pressure_grid, dtype=float,
        )
        hdf_file.create_dataset(
            name='temperatures', data=temperatures, dtype=float,
        )

        # Create a group for the parameters and store them there
        group = hdf_file.create_group(name='parameters')
        group.create_dataset(
            name='T0', data=parameters[:, 0], dtype=float,
        )
        group.create_dataset(
            name='a1', data=parameters[:, 1], dtype=float,
        )
        group.create_dataset(
            name='a2', data=parameters[:, 2], dtype=float,
        )
        group.create_dataset(
            name='P1', data=parameters[:, 3], dtype=float,
        )
        group.create_dataset(
            name='P2', data=parameters[:, 4], dtype=float,
        )
        group.create_dataset(
            name='P3', data=parameters[:, 5], dtype=float,
        )

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f"\nThis took {time.time() - script_start:.1f} seconds!\n")

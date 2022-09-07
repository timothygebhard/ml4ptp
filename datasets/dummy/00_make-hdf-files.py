"""
Create HDF files for a dummy data set.

The dummy data set consists of low-order Taylor series with coefficients
are drawn from a Gaussian distribution, meaning that the model should be
able to achieve close to perfect performance.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from argparse import ArgumentParser
from pathlib import Path

import time

import h5py
import numpy as np


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print("\nCREATE HDF FILES FOR DUMMY DATA SET\n", flush=True)

    # -------------------------------------------------------------------------
    # Set up argument parser to get path to input directory
    # -------------------------------------------------------------------------

    # Set up parser, add arguments, and parse them
    parser = ArgumentParser()
    parser.add_argument('--grid-size', type=int, default=101)
    parser.add_argument('--max-degree', type=int, default=3)
    parser.add_argument('--n-profiles', type=int, default=120_000)
    parser.add_argument('--test-size', type=int, default=20_000)
    parser.add_argument('--output-dir', type=str, default='./output')
    args = parser.parse_args()

    # Print arguments
    print('Received the following arguments:\n')
    for key, value in vars(args).items():
        print(f'  {key} = {value}')
    print()

    # Define output directory and sure that it exists
    print('Creating output directory...', end=' ', flush=True)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(exist_ok=True, parents=True)
    print('Done!\n', flush=True)

    # -------------------------------------------------------------------------
    # Create random data set
    # -------------------------------------------------------------------------

    print('Creating dummy data...', end=' ', flush=True)

    # Set up a new random number generator
    rng = np.random.RandomState(seed=42)

    # Create random "pressure" grids
    log_P = np.array(
        [
            np.linspace(
                -6 + rng.uniform(-1, 1),
                0 + rng.uniform(-1, 1),
                args.grid_size,
            )
            for i in range(args.n_profiles)
        ]
    )
    P = 10**log_P

    # Create corresponding "temperatures"
    coefficients = rng.normal(0, 1, (args.n_profiles, args.max_degree))
    T = np.zeros_like(log_P)
    for i in range(args.max_degree):
        T += coefficients[:, i].reshape(-1, 1) * np.sin(i * log_P)
        T += coefficients[:, i].reshape(-1, 1) * np.cos(i * log_P)
    T -= np.min(T)  # Important because of hard constraint that T > 0
    T /= np.max(T)  # Optional

    print('Done!', flush=True)

    # Print minimum and maximum (for plotting)
    print('Generated data cover the following value range:\n')
    print(f'  log_P: ({np.min(log_P):.2f}, {np.max(log_P):.2f})')
    print(f'  T:     ({np.min(T):.2f}, {np.max(T):.2f})')
    print()

    # -------------------------------------------------------------------------
    # Create indices for training, validation and test
    # -------------------------------------------------------------------------

    print('Creating random indices for split...', end=' ', flush=True)

    # Create indices and shuffle them randomly
    all_idx = np.arange(0, args.n_profiles)
    rng.shuffle(all_idx)

    # Define indices for training, validation and test
    train_idx = all_idx[: args.n_profiles - args.test_size]
    test_idx = all_idx[args.n_profiles - args.test_size :]

    print('Done!\n', flush=True)

    # -------------------------------------------------------------------------
    # Create HDF files for the training, validation and test data sets
    # -------------------------------------------------------------------------

    output_dir.mkdir(exist_ok=True)

    print("Saving everything to HDF files:", flush=True)

    for file_name, idx in [
        ('train.hdf', train_idx),
        ('test.hdf', test_idx),
    ]:

        print(f'  Creating {file_name}...', end=' ', flush=True)

        # Create new HDF file
        file_path = output_dir / file_name
        with h5py.File(file_path, "w") as hdf_file:
            hdf_file.create_dataset(
                name='coefficients', data=coefficients[idx]
            )
            hdf_file.create_dataset(name='P', data=P[idx])
            hdf_file.create_dataset(name='T', data=T[idx])

        print("Done!", flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f"\nThis took {time.time() - script_start:.1f} seconds!\n")

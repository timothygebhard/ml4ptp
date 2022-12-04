"""
Compute baseline, that is, fit the PT profiles in the test set with an
n-th order polynomial and store the results in an HDF file.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import argparse
import time

from rich.progress import track

import h5py
import numpy as np

from ml4ptp.config import load_config
from ml4ptp.paths import expandvars


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

def get_cli_args() -> argparse.Namespace:

    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment-dir',
        required=True,
        default='$ML4PTP_EXPERIMENTS_DIR/pyatmos/polynomial-baseline',
        help='Path to the experiment directory with the config.yaml.',
    )
    args = parser.parse_args()

    return args


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nCREATE POLYNOMIAL BASELINE\n', flush=True)

    # -------------------------------------------------------------------------
    # Get experiment dir and load configuration file
    # -------------------------------------------------------------------------

    # Get CLI arguments
    args = get_cli_args()

    # Load experiment configuration from YAML
    print('Loading experiment configuration...', end=' ', flush=True)
    experiment_dir = expandvars(Path(args.experiment_dir)).resolve()
    config = load_config(experiment_dir / 'config.yaml')
    print('Done!', flush=True)

    # Define shortcuts
    min_n_parameters = int(config['min_n_parameters'])
    max_n_parameters = int(config['max_n_parameters'])

    # -------------------------------------------------------------------------
    # Load the test set from HDF; reset output HDF file
    # -------------------------------------------------------------------------

    # Load test set from HDF
    print('Loading test set...', end=' ', flush=True)
    file_path = expandvars(Path(config['test_file_path']))
    with h5py.File(file_path, 'r') as hdf_file:
        log_P = np.log10(np.array(hdf_file[config['key_P']]))
        T_true = np.array(hdf_file[config['key_T']])
    print('Done!\n', flush=True)

    # Make sure output HDF file exists and is empty
    print('Resetting output HDF file...', end=' ', flush=True)
    output_file_path = experiment_dir / 'results_on_test_set.hdf'
    with h5py.File(output_file_path, 'w') as hdf_file:
        pass
    print('Done!\n', flush=True)

    # -------------------------------------------------------------------------
    # For each degree, fit the test set with a polynomial and store the results
    # -------------------------------------------------------------------------

    # Loop over degrees
    for degree in range(min_n_parameters - 1, max_n_parameters):

        # Fit the test set with a polynomial
        print(f'Fitting test set with a polynomial of degree {degree}:')

        # Store fit results
        coefs = np.full((len(log_P), degree + 1), np.nan)
        T_pred = np.full_like(T_true, np.nan)

        # Loop over all profiles in test set and fit them with a polynomial
        for i, (x, y) in track(
            sequence=enumerate(zip(log_P, T_true)),
            description='',
            total=len(log_P),
        ):

            # Fit the profile with a polynomial
            p = np.polyfit(x, y, deg=degree)
            t = np.polyval(p=p, x=x)

            # Store fit results
            coefs[i] = p
            T_pred[i] = t

        # Compute the mean squared error
        mse = np.mean((T_true - T_pred) ** 2, axis=1)

        # Store fit results in HDF file
        print('Storing fit results in HDF file...', end=' ', flush=True)
        with h5py.File(output_file_path, 'a') as hdf_file:
            group = hdf_file.create_group(f'{degree + 1}-fit-parameters')
            group.create_dataset('coefs', data=coefs)
            group.create_dataset('log_P', data=log_P)
            group.create_dataset('T_true', data=T_true)
            group.create_dataset('T_pred', data=T_pred)
            group.create_dataset('mse', data=mse)
        print('Done!\n', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nDone! This took {time.time() - script_start:.1f} seconds.\n')

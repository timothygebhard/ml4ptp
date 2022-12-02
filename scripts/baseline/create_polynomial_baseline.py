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
        # required=True,
        default='$ML4PTP_EXPERIMENTS_DIR/pyatmos/baseline',
        help='Path to the experiment directory with the config.yaml.',
    )
    parser.add_argument(
        '--n-parameters',
        type=int,
        default=2,
        help='Number of free parameters to use for fitting (= degree + 1).',
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
    print('\nCOMPUTE POLYNOMIAL BASELINE\n', flush=True)

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

    # -------------------------------------------------------------------------
    # Load the test set from HDF
    # -------------------------------------------------------------------------

    print('Loading test set...', end=' ', flush=True)
    file_path = expandvars(Path(config['test_file_path']))
    with h5py.File(file_path, 'r') as hdf_file:
        log_P = np.log10(np.array(hdf_file[config['key_P']]))
        T = np.array(hdf_file[config['key_T']])
    print('Done!\n', flush=True)

    # -------------------------------------------------------------------------
    # Load the test set from HDF
    # -------------------------------------------------------------------------

    # Store fit results
    coefs = []
    T_pred = []

    # Loop over all profiles in test set and fit them with a polynomial
    for x, y in track(
        sequence=zip(log_P, T),
        description='Fitting profiles:',
        total=len(log_P),
    ):
        p = np.polyfit(x, y, deg=args.n_parameters - 1)
        t = np.polyval(p=p, x=x)
        coefs.append(p)
        T_pred.append(t)
    print()

    # -------------------------------------------------------------------------
    # Save results to an HDF file
    # -------------------------------------------------------------------------

    print('Saving results to HDF file...', end=' ', flush=True)
    file_path = experiment_dir / f'n-parameters_{args.n_parameters}.hdf'
    with h5py.File(file_path, 'w') as hdf_file:
        hdf_file.create_dataset(name='log_P', data=log_P)
        hdf_file.create_dataset(name='T_true', data=T)
        hdf_file.create_dataset(name='T_pred', data=np.array(T_pred))
        hdf_file.create_dataset(name='coefficients', data=np.array(coefs))
    print('Done!\n', flush=True)

    # -------------------------------------------------------------------------
    # Compute average error
    # -------------------------------------------------------------------------

    absolute_error = np.sqrt(np.mean(np.square(np.array(T_pred) - T)))
    relative_error = np.sqrt(np.mean(np.square((np.array(T_pred) - T) / T)))
    print(f'Mean absolute error: {absolute_error:.2f}')
    print(f'Mean relative error: {relative_error:.2f}')

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nDone! This took {time.time() - script_start:.1f} seconds.\n')

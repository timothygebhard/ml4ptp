"""
Compute PCA baseline, that is, fit the PT profiles in the test set with
a PCA learned on the training set and store the results in an HDF file.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import argparse
import time

from sklearn.decomposition import PCA

import h5py
import numpy as np

from ml4ptp.config import load_experiment_config
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
        default='$ML4PTP_EXPERIMENTS_DIR/pyatmos/pca-baseline',
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
    print('\nCREATE PCA BASELINE\n', flush=True)

    # -------------------------------------------------------------------------
    # Get experiment dir and load configuration file
    # -------------------------------------------------------------------------

    # Get CLI arguments
    args = get_cli_args()

    # Load experiment configuration from YAML
    print('Loading experiment configuration...', end=' ', flush=True)
    experiment_dir = expandvars(Path(args.experiment_dir)).resolve()
    config = load_experiment_config(experiment_dir / 'config.yaml')
    print('Done!', flush=True)

    # Define shortcuts
    min_n_parameters = int(config['min_n_parameters'])
    max_n_parameters = int(config['max_n_parameters'])
    random_seeds = [int(_) for _ in config['random_seeds']]

    # -------------------------------------------------------------------------
    # Load the test set from HDF; reset output HDF file
    # -------------------------------------------------------------------------

    # Store the data (both training and test)
    log_P = dict()
    T_true = dict()

    # Load training set from HDF
    print('Loading training set from HDF...', end=' ', flush=True)
    file_path = expandvars(Path(config['train_file_path']))
    with h5py.File(file_path, 'r') as hdf_file:
        log_P['train'] = np.log10(np.array(hdf_file[config['key_P']]))
        T_true['train'] = np.array(hdf_file[config['key_T']])
    print('Done!', flush=True)

    # Load test set from HDF
    print('Loading test set...', end=' ', flush=True)
    file_path = expandvars(Path(config['test_file_path']))
    with h5py.File(file_path, 'r') as hdf_file:
        log_P['test'] = np.log10(np.array(hdf_file[config['key_P']]))
        T_true['test'] = np.array(hdf_file[config['key_T']])
    print('Done!\n', flush=True)

    # Make sure output HDF file exists and is empty
    print('Resetting output HDF file...', end=' ', flush=True)
    output_file_path = experiment_dir / 'results_on_test_set.hdf'
    with h5py.File(output_file_path, 'w') as hdf_file:
        pass
    print('Done!\n', flush=True)

    # -------------------------------------------------------------------------
    # Find a common pressure grid for all profiles
    # -------------------------------------------------------------------------

    # Get minimum and maximum pressure for the grid
    # Note: We need to find the interval that is covered by *all* profiles,
    # therefore the minimum is given by the maximum of the minimum pressures
    # and the maximum is given by the minimum of the maximum pressures.
    min_log_P = max(np.max(_[:, 0]) for _ in log_P.values())
    max_log_P = min(np.min(_[:, -1]) for _ in log_P.values())

    # Define pressure grid
    log_P_grid = np.linspace(min_log_P, max_log_P, log_P['train'].shape[1])

    # -------------------------------------------------------------------------
    # Project all profiles onto the common pressure grid
    # -------------------------------------------------------------------------

    print('Projecting all profiles onto common grid...', end=' ', flush=True)
    T_projected = dict()
    for key in log_P.keys():
        T_projected[key] = np.empty_like(T_true[key])
        for i, (log_P_, T_true_) in enumerate(zip(log_P[key], T_true[key])):
            T_projected[key][i] = np.interp(log_P_grid, log_P_, T_true_)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # For each degree, fit the test set with a polynomial and store the results
    # -------------------------------------------------------------------------

    # Loop over random seeds
    for random_seed in random_seeds:

        print(f'\nRunning for random seed: {random_seed}', flush=True)

        # Set random seed
        np.random.seed(random_seed)

        # Create indices to select a random subset of the training set
        n_profiles = len(T_projected['train'])
        idx = np.random.choice(
            a=np.arange(n_profiles),
            size=int(0.9 * n_profiles),
            replace=False,
        )

        # Loop over n_parameters
        for n in range(min_n_parameters, max_n_parameters + 1):

            print(f'\n  Fitting PCA with {n} params...', end=' ', flush=True)

            # Fit PCA
            pca = PCA(n_components=n)
            pca.fit(T_projected['train'][idx])

            # Find best fit with PCA
            T_pred = pca.inverse_transform(pca.transform(T_projected['test']))

            # Compute mean squared error and mean relative error
            mse = np.mean((T_pred - T_projected['test']) ** 2, axis=1)
            mre = np.mean(
                np.abs(T_pred - T_projected['test']) / T_projected['test'],
                axis=1
            )

            print('Done!', flush=True)

            # Store fit results in HDF file
            print('  Storing fit results in HDF file...', end=' ', flush=True)
            with h5py.File(output_file_path, 'a') as hdf_file:
                name = f'{n}-principal-components/run-{random_seed}'
                group = hdf_file.create_group(name)
                group.create_dataset('log_P', data=log_P['test'])
                group.create_dataset('T_true', data=T_projected['test'])
                group.create_dataset('T_pred', data=T_pred)
                group.create_dataset('mse', data=mse)
                group.create_dataset('mre', data=mre)
            print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nDone! This took {time.time() - script_start:.1f} seconds.\n')

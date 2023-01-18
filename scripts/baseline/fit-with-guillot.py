"""
This script fits a given subset of the Goyal-2020 test set with a
Guillot parameterization for the PT profile.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from dataclasses import asdict
from typing import List

import argparse
import socket
import time

import h5py
import numpy as np
import pandas as pd

from ml4ptp.guillot import fit_profile_with_guillot, FitResult
from ml4ptp.paths import get_datasets_dir, get_experiments_dir


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

def get_cli_args() -> argparse.Namespace:

    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment-dir',
        default='$ML4PTP_EXPERIMENTS_DIR/goyal-2020/fit-with-guillot',
        required=True,
        help='Path to the output directory for the experiment.',
    )
    parser.add_argument(
        '--n-splits',
        default=1,
        type=int,
        help='When running this script in parallel: How many splits to use.',
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for PyTorch, numpy, ... . Default: 42.',
    )
    parser.add_argument(
        '--split-idx',
        default=0,
        type=int,
        help=(
            'When running this script in parallel: '
            'Index of the split to evaluate; must be in [0, n_splits).'
        ),
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
    print('\nFIT PT PROFILE WITH GUILLOT PARAMETERIZATION\n', flush=True)

    # Print hostname for debugging purposes
    print(f'Running on machine: {socket.gethostname()}\n', flush=True)

    # -------------------------------------------------------------------------
    # Get experiment dir and load configuration file
    # -------------------------------------------------------------------------

    # Get CLI arguments
    args = get_cli_args()
    print('Received the following arguments:\n', flush=True)
    for key, value in vars(args).items():
        print(f'  {key + ":":<16} {value}', flush=True)
    print('\n', flush=True)

    # Define shortcuts
    split_idx = args.split_idx
    n_splits = args.n_splits

    # Set random seed
    np.random.seed(args.random_seed)

    # -------------------------------------------------------------------------
    # Load the Goyal-2020 test set and select the subset to fit
    # -------------------------------------------------------------------------

    print('Loading data from HDF...', end=' ', flush=True)

    file_path = get_datasets_dir() / 'goyal-2020' / 'output' / 'test.hdf'
    with h5py.File(file_path, 'r') as hdf_file:
        n = len(hdf_file['pt_profiles/pressure'])
        P = np.array(hdf_file['pt_profiles/pressure'])[split_idx::n_splits]
        T = np.array(hdf_file['pt_profiles/temperature'])[split_idx::n_splits]
        idx = np.arange(n)[split_idx::n_splits]

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Run fitting
    # -------------------------------------------------------------------------

    # Run fitting
    print('Fitting profiles:\n', flush=True)
    results: List[FitResult] = []
    for i, (P_true, T_true, idx_i) in enumerate(zip(P, T, idx)):

        print(f'  Profile {i + 1:>3d}/{len(P)} ...', end=' ', flush=True)
        result = fit_profile_with_guillot(
            idx_i,
            P_true,
            T_true,
            n_runs=100,
            random_seed=args.random_seed,
        )
        results.append(result)
        print(
            f'Done! (runtime = {result.runtime:.1f} seconds, '
            f'idx = {result.idx}), ',
            f'mre = {result.mre:.3f})',
            flush=True,
        )

    # Sort results by idx
    results = sorted(results, key=lambda x: x.idx)

    # Convert results to a dataframe for easier handling
    results_df = pd.DataFrame([asdict(_) for _ in results])

    # -------------------------------------------------------------------------
    # Save the (partial) results to an HDF file
    # -------------------------------------------------------------------------

    print('\nSaving results...', end=' ', flush=True)

    # Create output directory if it does not exist
    file_dir = get_experiments_dir() / 'goyal-2020' / 'fit-with-guillot'
    file_dir.mkdir(parents=True, exist_ok=True)

    # Define name and path for the (partial) output HDF file
    suffix = f'__{split_idx + 1:03d}-{n_splits:03d}' if n_splits > 1 else ''
    file_name = f'results_on_test_set{suffix}.hdf'
    file_path = file_dir / file_name

    # Define keys to save: if we only save a single file, we can drop the idx
    keys = sorted(list(results_df.keys()))
    if n_splits == 1:
        keys.remove('idx')

    # Save results to HDF file
    with h5py.File(file_path, 'w') as hdf_file:
        for key in keys:
            hdf_file.create_dataset(
                name=key,
                data=np.row_stack(results_df[key].values),
            )

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nDone! This took {time.time() - script_start:.1f} seconds.\n')

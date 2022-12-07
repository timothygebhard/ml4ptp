"""
Merge partial HDF files from running `evaluate-with-nested-sampling.py`
in parallel.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Dict, List

import argparse
import re
import socket
import time

import h5py
import numpy as np

from ml4ptp.paths import expandvars


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

def get_cli_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--keep-partial-hdf-files',
        action='store_true',
        default=False,
        help='If given, do not delete the partial HDF files.',
    )
    parser.add_argument(
        '--run-dir',
        required=True,
        default=(
            '$ML4PTP_EXPERIMENTS_DIR/pyatmos/default/latent-size-2/runs/run_0'
        ),
        help='Path to the directory containing the partial HDF files.',
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
    print('\nMERGE PARTIAL HDF FILES\n', flush=True)

    # Print hostname for debugging purposes
    print(f'Running on machine: {socket.gethostname()}\n', flush=True)

    # -------------------------------------------------------------------------
    # Get list of partial HDF files; ensure that we're not missing any
    # -------------------------------------------------------------------------

    # Get command line arguments
    args = get_cli_args()
    run_dir = expandvars(Path(args.run_dir)).resolve()

    # Get list of partial HDF files
    print('Getting list of partial HDF files...', end=' ', flush=True)
    partial_hdf_files = list(run_dir.glob('results_on_test_set__*.hdf'))
    print('Done!', flush=True)

    # Get the number of expected partial HDF files
    matches = re.match(
        '^.*__(?P<split_idx>\d+)-(?P<n_splits>\d+).hdf$',
        partial_hdf_files[0].name
    )
    if matches is not None:
        n_files_expected = int(matches.group('n_splits'))
    else:
        raise RuntimeError('Could not parse number of splits from file name!')

    # Check that the number of partial HDF files is as expected
    if len(partial_hdf_files) != n_files_expected:
        raise RuntimeError(
            f'Expected {n_files_expected} partial HDF files, but found '
            f'{len(partial_hdf_files)}!'
        )

    # -------------------------------------------------------------------------
    # Loop over partial files and read them in
    # -------------------------------------------------------------------------

    print('Reading in partial HDF files...', end=' ', flush=True)

    data_as_lists: Dict[str, List[np.ndarray]] = dict(
        idx=[],
        log_P=[],
        T_true=[],
        z_initial=[],
        z_refined=[],
        T_pred_initial=[],
        T_pred_refined=[],
        mse_initial=[],
        mse_refined=[],
        niter=[],
        ncall=[],
        converged=[],
    )

    # Loop over partial HDF files and read them in
    for file_name in partial_hdf_files:
        with h5py.File(run_dir / file_name, 'r') as hdf_file:
            for key in data_as_lists.keys():
                data_as_lists[key].append(np.array(hdf_file[key]))

    print('Done!', flush=True)

    # Concatenate arrays
    print('Concatenating arrays...', end=' ', flush=True)
    data_as_arrays: Dict[str, np.ndarray] = dict()
    for key in data_as_lists.keys():
        data_as_arrays[key] = np.concatenate(data_as_lists[key]).squeeze()
        if data_as_arrays[key].ndim == 1:
            data_as_arrays[key] = data_as_arrays[key][:, np.newaxis]
    print('Done!', flush=True)

    # Sort arrays by `idx`
    print('Sorting by idx...', end=' ', flush=True)
    sort_idx = np.argsort(data_as_arrays['idx'].squeeze())
    for key in data_as_arrays.keys():
        data_as_arrays[key] = data_as_arrays[key][sort_idx]
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Save the merged HDF file
    # -------------------------------------------------------------------------

    print('Saving merged HDF file...', end=' ', flush=True)

    with h5py.File(run_dir / 'results_on_test_set.hdf', 'w') as hdf_file:

        for key in data_as_arrays.keys():

            # We do not need to save the `idx` in the merged HDF file
            if key == 'idx':
                continue

            # Create dataset
            hdf_file.create_dataset(
                name=key,
                data=data_as_arrays[key],
                shape=data_as_arrays[key].shape,
            )

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Delete the partial HDF files
    # -------------------------------------------------------------------------

    if args.keep_partial_hdf_files:
        print('Keeping partial HDF files!', flush=True)

    else:
        print('Deleting partial HDF files...', end=' ', flush=True)
        for file_name in partial_hdf_files:
            file_path = run_dir / file_name
            file_path.unlink()
        print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nDone! This took {time.time() - script_start:.1f} seconds.\n')

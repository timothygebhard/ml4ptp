"""
Evaluate model on test set using `ultranest`.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from dataclasses import asdict
from pathlib import Path
from typing import List

import argparse
import socket
import time

import h5py
import numpy as np
import onnx
import pandas as pd

from ml4ptp.config import load_experiment_config
from ml4ptp.data_modules import DataModule
from ml4ptp.evaluation import find_optimal_z_with_ultranest, EvaluationResult
from ml4ptp.paths import expandvars


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

def get_cli_args() -> argparse.Namespace:

    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment-dir',
        default='$ML4PTP_EXPERIMENTS_DIR/pyatmos/default/latent-size-2',
        required=True,
        help='Path to the experiment directory with the config.yaml',
    )
    parser.add_argument(
        '--limit',
        default=4.0,
        type=float,
        help='Limit parameter for the prior. Default: 4.0.',
    )
    parser.add_argument(
        '--n-live-points',
        default=400,
        type=int,
        help='Number of live points to use for the nested sampling.',
    )
    parser.add_argument(
        '--n-splits',
        default=1,
        type=int,
        help='When running this script in parallel: How many splits to use.',
    )
    parser.add_argument(
        '--prior',
        default='gaussian',
        choices=['uniform', 'gaussian'],
        help='Prior to use for the latent variables. Default: "gaussian".',
    )
    parser.add_argument(
        '--run-dir',
        default=(
            '$ML4PTP_EXPERIMENTS_DIR/pyatmos/default/latent-size-2/runs/run_0'
        ),
        type=str,
        required=True,
        help=(
            'Path to the directory of the run to evaluate. Should contain the '
            'saved encoder and decoder models: encoder.onnx and decoder.onnx.'
        ),
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
    parser.add_argument(
        '--timeout',
        type=int,
        default=600,
        help='Timeout in seconds for the evaluation. Default: 600.',
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
    print('\nEVALUATE A MODEL\n', flush=True)

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

    # Load experiment configuration from YAML
    print('Loading config file...', end=' ', flush=True)
    experiment_dir = expandvars(Path(args.experiment_dir)).resolve()
    config = load_experiment_config(experiment_dir / 'config.yaml')
    print('Done!', flush=True)

    # Define shortcuts
    split_idx = args.split_idx
    n_splits = args.n_splits
    n_live_points = args.n_live_points
    run_dir = expandvars(Path(args.run_dir)).resolve()

    # Set random seed
    np.random.seed(args.random_seed)

    # -------------------------------------------------------------------------
    # Prepare the dataset and load the trained encoder and decoder models
    # -------------------------------------------------------------------------

    # Instantiate the DataModule
    print('Creating DataModule...', end=' ', flush=True)
    datamodule = DataModule(**config['datamodule'])
    datamodule.prepare_data()
    print('Done!', flush=True)

    # Load the trained encoder (as a byte string that can be serialized)
    print('Loading trained encoder...', end=' ', flush=True)
    file_path = run_dir / 'encoder.onnx'
    encoder_bytes = onnx.load(file_path.as_posix()).SerializeToString()
    print('Done!', flush=True)

    # Load the trained decoder (as a byte string that can be serialized)
    print('Loading trained decoder...', end=' ', flush=True)
    file_path = run_dir / 'decoder.onnx'
    decoder_bytes = onnx.load(file_path.as_posix()).SerializeToString()
    print('Done!\n\n', flush=True)

    # -------------------------------------------------------------------------
    # Run fitting
    # -------------------------------------------------------------------------

    # Prepare inputs for p_map
    log_P, T_true, _ = datamodule.get_test_data()
    idx = np.arange(len(log_P))
    log_P = log_P[split_idx::n_splits]
    T_true = T_true[split_idx::n_splits]
    idx = idx[split_idx::n_splits]

    # Run fitting
    # We do this sequentially, because combining multiprocessing with limiting
    # the runtime of nested sampling is just asking for trouble ...
    print('Fitting profiles:\n', flush=True)
    results: List[EvaluationResult] = []
    for i, (log_P_i, T_true_i, idx_i) in enumerate(zip(log_P, T_true, idx)):

        print(f'  Profile {i + 1:>3d}/{len(log_P)} ...', end=' ', flush=True)
        result = find_optimal_z_with_ultranest(
            log_P=log_P_i,
            T_true=T_true_i,
            idx=idx_i,
            encoder_bytes=encoder_bytes,
            decoder_bytes=decoder_bytes,
            random_seed=args.random_seed,
            n_live_points=args.n_live_points,
            n_max_calls=500_000,
            timeout=args.timeout,
            prior=args.prior,
            limit=args.limit,
        )
        print(
            f'Done! (runtime = {result.runtime:.1f} seconds, '
            f'success = {bool(result.success)}, '
            f'mre = {result.mre_refined:.3f})'
            ,
            flush=True
        )

    # Sort results by idx
    results = sorted(results, key=lambda x: x.idx)

    # Convert results to a dataframe for easier handling
    results_df = pd.DataFrame([asdict(_) for _ in results])

    # -------------------------------------------------------------------------
    # Save the (partial) results to an HDF file
    # -------------------------------------------------------------------------

    print('\nSaving results...', end=' ', flush=True)

    # Define name and path for the (partial) output HDF file
    suffix = f'__{split_idx + 1:03d}-{n_splits:03d}' if n_splits > 1 else ''
    file_name = f'results_on_test_set{suffix}.hdf'
    file_path = run_dir / file_name

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

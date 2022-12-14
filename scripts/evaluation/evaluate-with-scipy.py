"""
Evaluate model on test set using `scipy.optimize.minimize()`.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from functools import partial
from pathlib import Path
from typing import Dict, List, Union

import argparse
import socket
import time

from p_tqdm import p_umap
from scipy.optimize import minimize

import h5py
import numpy as np
import onnx
import pandas as pd

from ml4ptp.config import load_experiment_config
from ml4ptp.data_modules import DataModule
from ml4ptp.onnx import ONNXEncoder, ONNXDecoder
from ml4ptp.paths import expandvars
from ml4ptp.utils import get_number_of_available_cores


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
        '--split-idx',
        default=0,
        type=int,
        help=(
            'When running this script in parallel: '
            'Index of the split to evaluate; must be in [0, n_splits).'
        ),
    )
    parser.add_argument(
        '--n-splits',
        default=1,
        type=int,
        help='When running this script in parallel: How many splits to use.',
    )
    parser.add_argument(
        '--run-dir',
        default=(
            '$ML4PTP_EXPERIMENTS_DIR/pyatmos/default/latent-size-2/runs/run_1'
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
        help='Random seed for PyTorch, numpy, ....',
    )
    args = parser.parse_args()

    return args


def find_optimal_z(
    log_P: np.ndarray,
    T_true: np.ndarray,
    idx: int,
    encoder_bytes: bytes,
    decoder_bytes: bytes,
    random_seed: int,
) -> Dict[str, Union[int, float, np.ndarray]]:
    """
    Use `scipy.optimize.minimize()` to find the optimal latent variable
    `z` for a given PT profile (based on initial guess from encoder).
    """

    # Set random seed
    np.random.seed(random_seed)

    # Fix the shapes of the inputs
    log_P = log_P.reshape(1, -1)
    T_true = T_true.reshape(1, -1)

    # Load encoder and decoder from byte strings
    encoder = ONNXEncoder(encoder_bytes)
    decoder = ONNXDecoder(decoder_bytes)

    # Prepare dict with results
    results: Dict[str, Union[int, float, np.ndarray]] = dict(
        idx=idx,
        log_P=log_P,
        T_true=T_true,
    )

    # -------------------------------------------------------------------------
    # Get initial guess for z from encoder and compute error
    # -------------------------------------------------------------------------

    z_initial = encoder(log_P=log_P, T=T_true)
    T_pred_initial = decoder(log_P=log_P, z=z_initial)
    mse_initial = float(np.mean((T_pred_initial - T_true) ** 2))

    results['z_initial'] = z_initial.squeeze()
    results['T_pred_initial'] = T_pred_initial.squeeze()
    results['mse_initial'] = mse_initial

    # -------------------------------------------------------------------------
    # Try a basic simplex optimization of the MSE
    # -------------------------------------------------------------------------

    # Define function to minimize
    def mse(z: np.ndarray) -> float:
        T_pred = decoder(log_P=log_P, z=z)
        mse = float(np.mean((T_true - T_pred) ** 2, axis=1))
        return 1e300 if np.isnan(mse) else mse

    # Compute optimal z (within given bounds)
    optimum = minimize(
        fun=lambda z: mse(z.reshape(1, -1)),
        x0=z_initial.squeeze(),
        bounds=[(-5, 5)] * z_initial.shape[1],
    )

    # Store results
    results['z_refined'] = optimum.x.squeeze()
    results['ncall'] = int(optimum.nfev)
    results['niter'] = int(optimum.nit)
    results['converged'] = int(optimum.success)

    # Compute final predicted temperatures and MSE
    T_pred_refined = decoder(log_P=log_P, z=np.atleast_2d(optimum.x))
    mse_refined = float(np.mean((T_pred_refined - T_true) ** 2))
    results['T_pred_refined'] = T_pred_refined.squeeze()
    results['mse_refined'] = mse_refined

    return results


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

    # Load experiment configuration from YAML
    print('Loading experiment configuration...', end=' ', flush=True)
    experiment_dir = expandvars(Path(args.experiment_dir)).resolve()
    config = load_experiment_config(experiment_dir / 'config.yaml')
    print('Done!\n', flush=True)

    # Define shortcuts
    split_idx = args.split_idx
    n_splits = args.n_splits
    random_seed = args.random_seed
    run_dir = expandvars(Path(args.run_dir)).resolve()
    latent_size = config['model']['decoder']['parameters']['latent_size']

    # Set random seed
    np.random.seed(random_seed)

    # -------------------------------------------------------------------------
    # Prepare the dataset and load the trained encoder and decoder models
    # -------------------------------------------------------------------------

    # Instantiate the DataModule
    print('Instantiating DataModule...', end=' ', flush=True)
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
    print('Done!\n', flush=True)

    # -------------------------------------------------------------------------
    # Run fitting
    # -------------------------------------------------------------------------

    # Prepare inputs for p_map
    log_P, T_true = datamodule.get_test_data()
    idx = np.arange(len(log_P))
    log_P = log_P[split_idx::n_splits]
    T_true = T_true[split_idx::n_splits]
    idx = idx[split_idx::n_splits]

    # Get number of cores to use
    n_jobs = get_number_of_available_cores()
    print(f'Number of available cores: {n_jobs}\n', flush=True)
    print('Fitting:', flush=True)

    # Run fitting (in parallel)
    # We use umap here, because it should (theoretically) be faster than map,
    # and because we can sort by the `idx` of the input data afterwards.
    results: List[Dict[str, Union[int, float, np.ndarray]]] = p_umap(
        partial(
            find_optimal_z,
            encoder_bytes=encoder_bytes,
            decoder_bytes=decoder_bytes,
            random_seed=random_seed,
        ),
        log_P,
        T_true,
        idx,
        num_cpus=n_jobs,
        ncols=80,  # for tqdm
    )

    # Sort results by idx
    results = sorted(results, key=lambda x: x['idx'])  # type: ignore

    # Convert results (i.e., list of dicts) to a dataframe for easier handling
    results_df = pd.DataFrame(results)

    # -------------------------------------------------------------------------
    # Save the (partial) results to an HDF file
    # -------------------------------------------------------------------------

    print('\nSaving results...', end=' ', flush=True)

    # Define name and path for the (partial) output HDF file
    suffix = f'__{split_idx + 1:03d}-{n_splits:03d}' if n_splits > 1 else ''
    file_name = f'results_on_test_set{suffix}.hdf'
    file_path = run_dir / file_name

    # Define keys to save
    keys = [
        'log_P',
        'T_true',
        'z_initial',
        'z_refined',
        'T_pred_initial',
        'T_pred_refined',
        'mse_initial',
        'mse_refined',
        'ncall',
        'niter',
        'converged',
    ]
    if n_splits > 1:
        keys += ['idx']

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
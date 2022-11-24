"""
Evaluate model on test set.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import argparse
import io
import logging
import time

from joblib import delayed, Parallel
from tqdm.auto import tqdm

import h5py
import scipy.stats
import torch
import numpy as np
import pandas as pd
import ultranest

from ml4ptp.config import load_config
from ml4ptp.data_modules import DataModule
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
        default=(
            '/Users/timothy/Desktop/projects/ml4ptp/experiments/playground'
        ),
        # required=True,
        help='Path to the experiment directory with the config.yaml',
    )
    parser.add_argument(
        '--run-dir',
        default=0,
        type=str,
        # required=True,
        help=(
            'Path to the directory of the run to evaluate. Should contain the '
            'saved encoder and decoder models: encoder.pt and decoder.pt.'
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


def find_optimal_z_with_nested_sampling(
    log_P: torch.Tensor,
    T_true: torch.Tensor,
    encoder_buffer: io.BytesIO,
    decoder_buffer: io.BytesIO,
    latent_size: int,
) -> dict:

    # Load decoder from buffer
    loc = torch.device('cpu')
    encoder = torch.jit.load(encoder_buffer, map_location=loc)  # type: ignore
    decoder = torch.jit.load(decoder_buffer, map_location=loc)  # type: ignore

    # Prepare dict with results
    results = dict()

    # -------------------------------------------------------------------------
    # Get initial guess for z from encoder and compute error
    # -------------------------------------------------------------------------

    with torch.no_grad():

        z_initial = encoder(log_P=log_P, T=T_true)
        T_pred_initial = decoder(log_P=log_P, z=z_initial)
        mse_initial = float(torch.mean((T_pred_initial - T_true) ** 2))

        results['z_initial'] = z_initial.numpy()
        results['T_pred_initial'] = T_pred_initial.numpy()
        results['mse_initial'] = mse_initial
        results['T_true'] = T_true.numpy()

    # -------------------------------------------------------------------------
    # Define (Gaussian) prior and likelihood
    # -------------------------------------------------------------------------

    def gaussian_prior(cube: np.ndarray) -> np.ndarray:
        """
        Gaussian prior for z.
        """

        params = cube.copy()
        # noinspection PyUnresolvedReferences
        gaussdistribution = scipy.stats.norm(0, 1)

        for i in range(latent_size):
            params[:, i] = gaussdistribution.ppf(cube[:, i])

        return params

    def likelihood(params: np.ndarray) -> np.ndarray:
        """
        Likelihood for comparing PT profiles (= negative MSE).
        """

        z = torch.from_numpy(params).float()
        log_P_ = torch.from_numpy(np.tile(log_P, (z.shape[0], 1))).float()

        with torch.no_grad():
            T_pred = (
                decoder(
                    log_P=log_P_,
                    z=z,
                )
                .detach()
                .numpy()
                .squeeze()
            )

        T_true_ = np.tile(T_true, (z.shape[0], 1))

        mse = np.asarray(np.mean(np.square(T_true_ - T_pred), axis=1))

        return -mse

    # -------------------------------------------------------------------------
    # Set up nested sampling and run
    # -------------------------------------------------------------------------

    logger = logging.getLogger("ultranest")
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.WARNING)

    sampler = ultranest.ReactiveNestedSampler(
        param_names=[f'z{i}' for i in range(latent_size)],
        loglike=likelihood,
        transform=gaussian_prior,
        vectorized=True,
    )

    # noinspection PyTypeChecker
    result = sampler.run(
        min_num_live_points=400,
        show_status=False,
        viz_callback=False,
    )

    # -------------------------------------------------------------------------
    # Decode refined z and compute error
    # -------------------------------------------------------------------------

    z_refined = torch.FloatTensor(result['maximum_likelihood']['point'])
    results['z_refined'] = z_refined.numpy()
    results['niter'] = int(result['niter'])

    with torch.no_grad():
        T_pred_refined = decoder(log_P=log_P, z=z_refined)
        mse_refined = float(torch.mean((T_pred_refined - T_true) ** 2))
        results['T_pred_refined'] = T_pred_refined.numpy()
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

    # -------------------------------------------------------------------------
    # Get experiment dir and load configuration file
    # -------------------------------------------------------------------------

    # Get CLI arguments
    args = get_cli_args()

    # Load experiment configuration from YAML
    print('Loading experiment configuration...', end=' ', flush=True)
    experiment_dir = expandvars(Path(args.experiment_dir)).resolve()
    config = load_config(experiment_dir / 'config.yaml')
    print('Done!\n', flush=True)

    # Define shortcuts
    random_seed = args.random_seed
    run_dir = expandvars(Path(args.run_dir)).resolve()
    latent_size = config['model']['decoder']['parameters']['latent_size']

    # -------------------------------------------------------------------------
    # Prepare the dataset and load the trained encoder and decoder models
    # -------------------------------------------------------------------------

    # Instantiate the DataModule
    print('Instantiating DataModule...', end=' ', flush=True)
    datamodule = DataModule(
        **config['datamodule'],
        test_batch_size=1,
        random_state=random_seed,
    )
    datamodule.prepare_data()
    print('Done!', flush=True)

    # Load the trained encoder (as a buffer that can be pickled)
    print('Loading trained encoder...', end=' ', flush=True)
    file_path = run_dir / 'encoder.pt'
    with open(file_path, 'rb') as f:
        encoder_buffer = io.BytesIO(f.read())
    print('Done!', flush=True)

    # Load the trained decoder
    print('Loading trained decoder...', end=' ', flush=True)
    file_path = run_dir / 'decoder.pt'
    with open(file_path, 'rb') as f:
        decoder_buffer = io.BytesIO(f.read())
    print('Done!\n', flush=True)

    # -------------------------------------------------------------------------
    # Run fitting
    # -------------------------------------------------------------------------

    n_jobs = get_number_of_available_cores()
    print(f'Number of available cores: {n_jobs}\n', flush=True)

    print('Fitting:', flush=True)
    results = Parallel(n_jobs=n_jobs)(
        delayed(find_optimal_z_with_nested_sampling)(
            log_P=log_P,
            T_true=T_true,
            encoder_buffer=encoder_buffer,
            decoder_buffer=decoder_buffer,
            latent_size=latent_size,
        )
        for log_P, T_true in tqdm(list(datamodule.test_dataloader()))
    )

    # Convert results (i.e., list of dicts) to a dataframe for easier handling
    results_df = pd.DataFrame(results)

    # -------------------------------------------------------------------------
    # Save the results to an HDF file
    # -------------------------------------------------------------------------

    print('\nSaving results...', end=' ', flush=True)
    file_path = run_dir / 'results_on_test_set.hdf'

    with h5py.File(file_path, 'w') as hdf_file:

        for key in (
            'z_initial',
            'z_refined',
            'T_pred_initial',
            'T_pred_refined',
            'mse_initial',
            'mse_refined',
            'niter',
        ):

            hdf_file.create_dataset(
                name=key,
                data=np.row_stack(results_df[key].values),
            )

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nDone! This took {time.time() - script_start:.1f} seconds.\n')

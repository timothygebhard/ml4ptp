"""
Evaluate model on test set.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import argparse
import time

import h5py
import torch
import numpy as np

from ml4ptp.config import load_experiment_config
from ml4ptp.data_modules import DataModule
from ml4ptp.evaluation import get_initial_predictions, get_refined_predictions
from ml4ptp.models import Model
from ml4ptp.paths import expandvars
from ml4ptp.utils import get_device_from_model


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

def get_cli_args() -> argparse.Namespace:

    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment-dir',
        required=True,
        help='Path to the experiment directory with the config.yaml',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=512,
        help='Batch size for refinement / optimization.',
    )
    parser.add_argument(
        '--n-epochs',
        type=int,
        default=50,
        help='Number of epochs for the refinement / optimization.',
    )
    parser.add_argument(
        '--run',
        type=int,
        required=True,
        help='Which run to evaluate.',
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for PyTorch, numpy, ....',
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
    random_seed = args.random_seed
    run_dir = experiment_dir / 'runs' / f'run_{args.run}'

    # -------------------------------------------------------------------------
    # Prepare the dataset
    # -------------------------------------------------------------------------

    # Instantiate the DataModule
    print('Instantiating DataModule...', end=' ', flush=True)
    datamodule = DataModule(**config['datamodule'], random_state=random_seed)
    datamodule.prepare_data()
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Set up the model
    # -------------------------------------------------------------------------

    # Collect the parameters for the normalization from the data module,
    # unless we explicitly specify values in the configuration file
    if 'normalization' not in config['model'].keys():
        config['normalization'] = dict(
            T_offset=datamodule.T_offset,
            T_factor=datamodule.T_factor,
        )

    # Instantiate the model as prescribed by the configuration file
    print('Setting up model...', end=' ', flush=True)
    model = Model(
        encoder_config=config['model']['encoder'],
        decoder_config=config['model']['decoder'],
        optimizer_config=config['optimizer'],
        loss_config=config['loss'],
        normalization_config=config['normalization'],
        lr_scheduler_config=config['lr_scheduler'],
        plotting_config=config['plotting'],
    )
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Load the model and move it to the GPU
    # -------------------------------------------------------------------------

    # Load "best" checkpoint
    print('Loading best checkpoint...', end=' ', flush=True)
    file_path = run_dir / 'checkpoints' / 'best.ckpt'
    model = model.load_from_checkpoint(
        checkpoint_path=file_path.as_posix(),
        map_location=get_device_from_model(model)
    )
    print('Done!\n', flush=True)

    # Move model to GPU, if possible
    if torch.cuda.is_available():
        print('Running on device: CUDA\n')
        device = torch.device('cuda')
        model.to(device)
    else:
        print('Running on device: CPU\n')
        device = torch.device('cpu')

    # -------------------------------------------------------------------------
    # Get initial predictions (without refinement)
    # -------------------------------------------------------------------------

    # Run test set through model and get initial predictions for everything
    print('Getting initial predictions on test set:', flush=True)
    z_initial, T_true, log_P, T_pred_initial = get_initial_predictions(
        model=model,
        test_dataloader=datamodule.test_dataloader(),
        device=device,
    )
    print()

    # Compute initial error
    mae_initial = np.mean(np.abs(T_true - T_pred_initial), axis=1)
    q_5 = np.quantile(mae_initial, 0.05)
    q_50 = np.quantile(mae_initial, 0.50)
    q_95 = np.quantile(mae_initial, 0.95)
    print(f'Initial MAE: {q_50:.2f} [{q_5:.2f}-{q_95:.2f}]', flush=True)

    # Save results to HDF file
    print('Saving initial results to HDF file...', end=' ', flush=True)
    file_path = run_dir / 'results_on_test_set.hdf'
    with h5py.File(file_path, 'w') as hdf_file:
        hdf_file.create_dataset(name='z_initial', data=z_initial)
        hdf_file.create_dataset(name='T_true', data=T_true)
        hdf_file.create_dataset(name='log_P', data=log_P)
        hdf_file.create_dataset(name='T_pred_initial', data=T_pred_initial)
    print('Done!\n', flush=True)

    # -------------------------------------------------------------------------
    # Get refined predictions
    # -------------------------------------------------------------------------

    print('Getting refined predictions on test set:', flush=True)
    z_refined, T_pred_refined = get_refined_predictions(
        model=model,
        z_initial=z_initial,
        T_true=T_true,
        log_P=log_P,
        device=device,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
    )
    print()

    # Compute refined error
    mae_refined = np.mean(np.abs(T_true - T_pred_refined), axis=1)
    q_5 = np.quantile(mae_refined, 0.05)
    q_50 = np.quantile(mae_refined, 0.50)
    q_95 = np.quantile(mae_refined, 0.95)
    print(f'Refined MAE: {q_50:.2f} [{q_5:.2f}-{q_95:.2f}]', flush=True)

    # Save results to HDF file
    print('Saving refined results to HDF file...', end=' ', flush=True)
    file_path = run_dir / 'results_on_test_set.hdf'
    with h5py.File(file_path, 'a') as hdf_file:
        hdf_file.create_dataset(name='z_refined', data=z_refined)
        hdf_file.create_dataset(name='T_pred_refined', data=T_pred_refined)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nDone! This took {time.time() - script_start:.1f} seconds.\n')

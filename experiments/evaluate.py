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

from ml4ptp.config import load_config
from ml4ptp.data_modules import DataModule
from ml4ptp.evaluation import evaluate_on_test_set
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
    config = load_config(experiment_dir / 'config.yaml')
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

    # Instantiate the model as prescribed by the configuration file
    print('Setting up model...', end=' ', flush=True)
    model = Model(
        encoder_config=config['model']['encoder'],
        decoder_config=config['model']['decoder'],
        optimizer_config=config['optimizer'],
        loss_config=config['loss'],
        normalization_config=dict(T_mean=float('nan'), T_std=float('nan')),
        lr_scheduler_config=config['lr_scheduler'],
        plotting_config=config['plotting'],
    )
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Evaluate the model on the test set
    # -------------------------------------------------------------------------

    # Load "best" checkpoint
    print('Loading best checkpoint...', end=' ', flush=True)
    file_path = run_dir / 'checkpoints' / 'best.ckpt'
    model.load_from_checkpoint(
        checkpoint_path=file_path.as_posix(),
        map_location=get_device_from_model(model)
    )
    print('Done!', flush=True)

    # Run test set through model and apply LBFGS optimization to latent z
    print('Evaluating trained model on test set:', flush=True)
    (
        z_initial,
        z_optimal,
        T_true,
        log_P,
        T_pred_initial,
        T_pred_optimal,
    ) = evaluate_on_test_set(
        model=model,
        test_dataloader=datamodule.test_dataloader(),
        device=get_device_from_model(model),
    )

    # Save results to HDF file
    print('Saving evaluation results to HDF file...', end=' ', flush=True)
    file_path = run_dir / 'results_on_test_set.hdf'
    with h5py.File(file_path, 'w') as hdf_file:
        hdf_file.create_dataset(name='z_initial', data=z_initial)
        hdf_file.create_dataset(name='z_optimal', data=z_optimal)
        hdf_file.create_dataset(name='T_true', data=T_true)
        hdf_file.create_dataset(name='log_P', data=log_P)
        hdf_file.create_dataset(name='T_pred_initial', data=T_pred_initial)
        hdf_file.create_dataset(name='T_pred_optimal', data=T_pred_optimal)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nDone! This took {time.time() - script_start:.1f} seconds.\n')

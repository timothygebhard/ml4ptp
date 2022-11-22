"""
Train a model for a PT profile parameterization.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from shutil import copy

import argparse
import time

import yaml

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

from ml4ptp.config import load_config
from ml4ptp.data_modules import DataModule
from ml4ptp.exporting import export_model_with_torchscript
from ml4ptp.git_utils import document_git_status
from ml4ptp.models import Model
from ml4ptp.paths import expandvars
from ml4ptp.utils import get_run_dir


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

def get_cli_args() -> argparse.Namespace:

    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment-dir',
        required=True,
        default='$ML4PTP_EXPERIMENTS_DIR/pyatmos/default/latent-size-2',
        help='Path to the experiment directory with the config.yaml',
    )
    parser.add_argument(
        '--run-dir',
        default=None,
        help=(
            'Path to the run directory (should be inside the experiment_dir). '
            'If given, do not generate a new run dir, but use the given one.'
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


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nTRAIN A MODEL\n', flush=True)

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

    # -------------------------------------------------------------------------
    # Construct run directory; save config and git status
    # -------------------------------------------------------------------------

    # Get the directory where everything for this run is stored
    if args.run_dir is not None:
        run_dir = Path(args.run_dir).resolve()
    else:
        run_dir = get_run_dir(experiment_dir=experiment_dir)

    # Print run directory
    print('All data for this run will be stored in:')
    print(' ', run_dir)
    print()

    # Save a copy of the original configuration file for this run
    print('Create a copy of the config file...', end=' ', flush=True)
    copy(experiment_dir / 'config.yaml', run_dir)
    print('Done!', flush=True)

    # Save a copy of the command line arguments passed to this script
    print('Saving command line arguments...', end=' ', flush=True)
    file_path = run_dir / 'arguments.yaml'
    with open(file_path, 'w') as json_file:
        json_file.write(yaml.safe_dump(vars(args), indent=2))
    print('Done!', flush=True)

    # Document the state of the git repository
    document_git_status(target_dir=run_dir, verbose=True)
    print()

    # -------------------------------------------------------------------------
    # Setup the Trainer
    # -------------------------------------------------------------------------

    # Set random seed for PyTorch, numpy, ...
    random_seed = args.random_seed
    seed_everything(seed=random_seed, workers=True)

    # Manually create a logger to get control over the run directory
    logger = TensorBoardLogger(
        save_dir=run_dir.parent.as_posix(), name="", version=run_dir.name
    )

    # Create a callback for creating (best) checkpoints
    checkpoint_callback = ModelCheckpoint(
        filename='best',
        save_top_k=1,
        save_last=True,
        monitor="val/total_loss",
    )

    # Create a callback for early stopping
    early_stopping_callback = EarlyStopping(
        monitor="val/total_loss", **config['callbacks']['early_stopping']
    )

    # Finally, create the trainer
    print('\nPreparing Trainer:', flush=True)
    trainer = Trainer(
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            LearningRateMonitor(logging_interval='step'),
            RichProgressBar(),
        ],
        default_root_dir=expandvars(experiment_dir).as_posix(),
        logger=logger,
        **config['trainer'],
    )
    print()

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
    # Train the model
    # -------------------------------------------------------------------------

    print('\n\nStarting training:\n', flush=True)
    trainer.fit(model=model, datamodule=datamodule)
    print()

    # -------------------------------------------------------------------------
    # Export models using torchscript
    # -------------------------------------------------------------------------

    print('Exporting trained models...', end=' ', flush=True)

    # Export encoder
    file_path = run_dir / 'encoder.pt'
    export_model_with_torchscript(model=model.encoder, file_path=file_path)

    # Export decoder
    file_path = run_dir / 'decoder.pt'
    export_model_with_torchscript(model=model.decoder, file_path=file_path)

    print('Done!\n', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nDone! This took {time.time() - script_start:.1f} seconds.\n')

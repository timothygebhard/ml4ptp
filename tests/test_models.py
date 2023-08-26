"""
Unit tests for models.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

from lightning.pytorch import Trainer, seed_everything

import h5py
import numpy as np
import pytest

from ml4ptp.data_modules import DataModule
from ml4ptp.models import Model


# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------

@pytest.fixture()
def hdf_file(tmp_path: Path) -> Path:
    """
    Fixture to create dummy HDF file.
    """

    np.random.seed(42)

    file_path = tmp_path / 'hdf_file.hdf'
    with h5py.File(file_path, 'w') as hdf_file:
        hdf_file.create_dataset(
            name='P',
            data=10 ** np.random.normal(0, 1, (128, 13)),
        )
        hdf_file.create_dataset(
            name='T',
            data=10 ** np.random.normal(0, 1, (128, 13)),
        )
    return file_path


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

def test__model(hdf_file: Path, tmp_path: Path) -> None:

    seed_everything(42)

    # -------------------------------------------------------------------------
    # Setup data module
    # -------------------------------------------------------------------------

    datamodule = DataModule(
        train_file_path=hdf_file,
        test_file_path=hdf_file,
        key_P='P',
        key_T='T',
        train_size=128,
        val_size=64,
        train_batch_size=16,
        val_batch_size=16,
        test_batch_size=16,
    )
    datamodule.prepare_data()

    # -------------------------------------------------------------------------
    # Setup model
    # -------------------------------------------------------------------------

    encoder_config = dict(
        name='MLPEncoder',
        parameters=dict(
            input_size=13,
            layer_size=32,
            latent_size=3,
            n_layers=2,
        ),
    )
    decoder_config = dict(
        name='Decoder',
        parameters=dict(
            layer_size=32,
            latent_size=3,
            n_layers=2,
            activation='leaky_relu',
        ),
    )
    optimizer_config = dict(
        name='AdamW',
        parameters=dict(lr=3e-4),
    )
    loss_config = dict(
        beta=100,
    )
    lr_scheduler_config = dict(
        name='StepLR',
        parameters=dict(step_size=30, gamma=0.1),
    )
    plotting_config = dict(
        enable_plotting=True,
        pt_profile=dict(min_T=0, max_T=350, min_log_P=0.5, max_log_P=-6.5),
    )

    model = Model(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        optimizer_config=optimizer_config,
        loss_config=loss_config,
        normalization=datamodule.get_normalization(),
        lr_scheduler_config=lr_scheduler_config,
        plotting_config=plotting_config,
    )

    # -------------------------------------------------------------------------
    # Setup Trainer and run tests
    # -------------------------------------------------------------------------

    trainer = Trainer(
        accelerator='auto',
        default_root_dir=tmp_path.as_posix(),
        max_epochs=2,
        log_every_n_steps=1,
        detect_anomaly=True,
        fast_dev_run=True,
    )

    trainer.fit(model=model, datamodule=datamodule)
    assert np.isclose(
        trainer.logged_metrics['val/total_loss_epoch'],
        508.5398,
    )
    assert np.isclose(
        trainer.logged_metrics['train/total_loss_epoch'],
        72.4137,
    )

    trainer.test(model=model, datamodule=datamodule, verbose=False)
    assert np.isclose(
        trainer.logged_metrics['test/total_loss_epoch'],
        49.0438,
    )

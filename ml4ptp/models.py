"""
Define models.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import List, Optional, Tuple

from pytorch_lightning.utilities.types import (
    EVAL_DATALOADERS,
    TRAIN_DATALOADERS,
)

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch

from ml4ptp.decoders import Decoder
from ml4ptp.encoders import Encoder
from ml4ptp.importing import get_member_by_name
from ml4ptp.kernels import compute_mmd
from ml4ptp.plotting import plot_profile_to_tensorboard, plot_z_to_tensorboard


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

class Model(pl.LightningModule):
    """
    A wrapper to combine the encoder and decoder for training.
    """

    def __init__(
        self,
        encoder_config: dict,
        decoder_config: dict,
        optimizer_config: dict,
        loss_config: dict,
        normalization_config: dict,
        plotting_config: dict,
        lr_scheduler_config: Optional[dict] = None,
    ) -> None:
        """
        Set up a new model.

        Args:
            encoder_config: Configuration for the encoder.
            decoder_config: Configuration for the decoder.
            optimizer_config: Configuration for the optimizer.
            loss_config: Configuration  for the loss.
            normalization_config: Configuration for the normalization.
            plotting_config: Configuration for TensorBoard plots.
            lr_scheduler_config: Configuration for the LR scheduler.
        """

        super().__init__()

        # Store hyperparameters used to instantiate this network
        self.save_hyperparameters()

        # Store the constructor arguments
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.optimizer_config = optimizer_config
        self.loss_config = loss_config
        self.normalization_config = normalization_config
        self.plotting_config = plotting_config
        self.lr_scheduler_config = lr_scheduler_config

        # Set up the encoder and decoder networks
        self.encoder = Encoder(**encoder_config, **normalization_config)
        self.decoder = Decoder(**decoder_config, **normalization_config)

    def configure_optimizers(self) -> dict:
        """
        Set up the optimizer (and, optionally, the LR scheduler).
        """

        # Set up the optimizer
        # Note: Something like `getattr(torch.optim, 'AdamW')` works, but is
        # actually not allowed by Python, and mypy will complain about it.
        optimizer = get_member_by_name(
            module_name='torch.optim',
            member_name=self.optimizer_config['name'],
        )(
            params=self.parameters(),
            **self.optimizer_config['parameters'],
        )
        result = {'optimizer': optimizer}

        # Set up the learning rate scheduler (if desired)
        if self.lr_scheduler_config is not None:
            lr_scheduler = get_member_by_name(
                module_name='torch.optim.lr_scheduler',
                member_name=self.lr_scheduler_config['name'],
            )(
                optimizer=optimizer,
                **self.lr_scheduler_config['parameters'],
            )
            result['lr_scheduler'] = lr_scheduler

        return result

    def forward(  # type: ignore
        self,
        log_P: torch.Tensor,
        T: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Run through encoder to get latent code
        z = self.encoder.forward(log_P=log_P, T=T)

        # Run through decoder to get predicted temperatures
        T_pred = self.decoder.forward(z=z, log_P=log_P)

        return z, T_pred

    def loss(
        self,
        T_true: torch.Tensor,
        T_pred: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the loss.
        """

        # Compute the reconstruction loss
        reconstruction_loss = (T_true - T_pred).pow(2).mean()

        # Compute the MMD between z and a sample from a standard Gaussian
        true_samples = torch.randn(
            self.loss_config['n_samples'], self.encoder.latent_size
        ).type_as(z)
        mmd_loss = self.loss_config['beta'] * compute_mmd(true_samples, z)

        # Compute the total loss
        total_loss = reconstruction_loss + mmd_loss

        return total_loss, reconstruction_loss, mmd_loss

    def training_step(  # type: ignore
        self,
        batch: List[torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._common_step(batch, batch_idx, "train")

    def validation_step(  # type: ignore
        self,
        batch: List[torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._common_step(batch, batch_idx, "val")

    def test_step(  # type: ignore
        self,
        batch: List[torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._common_step(batch, batch_idx, "test")

    def _common_step(
        self,
        batch: List[torch.Tensor],
        batch_idx: int,
        stage: str,
    ) -> torch.Tensor:

        # Set model either to training or evaluation mode
        if stage == 'train':
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()

        # Unpack the batch and compute pass through encoder and decoder
        log_P, T_true = batch
        z, T_pred = self.forward(log_P=log_P, T=T_true)

        # Compute the loss terms
        total_loss, reconstruction_loss, latent_loss = self.loss(
            T_true=T_true, T_pred=T_pred, z=z
        )

        # Log the loss terms to TensorBoard
        self.log_dict(
            dictionary={
                f"{stage}/total_loss": total_loss,
                f"{stage}/reconstruction_loss": reconstruction_loss,
                f"{stage}/latent_loss": latent_loss,
            },
            on_step=True,
            on_epoch=True,
        )

        # At the beginning of every epoch, save some plots to TensorBoard
        if batch_idx == 0:
            self.plot_z_to_tensorboard(z=z, label=stage)
            self.plot_profile_to_tensorboard(
                z=z, log_P=log_P, T_true=T_true, T_pred=T_pred, label=stage
            )

        # Only return the total loss for backpropagation
        return total_loss

    def plot_z_to_tensorboard(self, z: torch.Tensor, label: str) -> None:

        # Create the figure
        figure = plot_z_to_tensorboard(z)

        # Log the figure to TensorBoard
        self.logger.experiment.add_figure(
            tag=f'Latent distribution ({label})',
            figure=figure,
            global_step=self.current_epoch,
        )

        # Close the figure
        plt.close(figure)

    def plot_profile_to_tensorboard(
        self,
        z: torch.Tensor,
        log_P: torch.Tensor,
        T_true: torch.Tensor,
        T_pred: torch.Tensor,
        label: str,
    ) -> None:

        # Create the figure
        figure = plot_profile_to_tensorboard(
            z,
            log_P,
            T_true,
            T_pred,
            self.plotting_config,
        )

        # Add figure to TensorBoard
        self.logger.experiment.add_figure(
            tag=f'Example profiles ({label})',
            figure=figure,
            global_step=self.current_epoch,
        )

        # Close the figure
        plt.close(figure)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

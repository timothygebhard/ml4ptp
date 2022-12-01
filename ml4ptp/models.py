"""
Define models.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from math import exp
from typing import List, Optional, Tuple

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import (
    EVAL_DATALOADERS,
    TRAIN_DATALOADERS,
)

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch

from ml4ptp.importing import get_member_by_name
from ml4ptp.mixins import NormalizerMixin
from ml4ptp.mmd import compute_mmd
from ml4ptp.plotting import plot_profile_to_tensorboard, plot_z_to_tensorboard


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

class Model(pl.LightningModule, NormalizerMixin):
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
        self.plotting_config = plotting_config
        self.lr_scheduler_config = lr_scheduler_config

        # Store normalizer
        self.T_offset = normalization_config['T_offset']
        self.T_factor = normalization_config['T_factor']

        # Define some shortcuts
        self.beta = self.loss_config['beta']
        self.use_weighted_loss = self.loss_config.get('weighted_loss', False)

        # Define other attributes
        self.rl_weight = 0.0

        # Set up the encoder and decoder networks
        self.encoder = get_member_by_name(
            module_name='ml4ptp.encoders', member_name=encoder_config['name']
        )(**encoder_config['parameters'], **normalization_config)
        self.decoder = get_member_by_name(
            module_name='ml4ptp.decoders', member_name=decoder_config['name']
        )(**decoder_config['parameters'], **normalization_config)

    def get_loss_weights_like(self, x: torch.Tensor) -> torch.Tensor:

        # If we are not using a weighted loss, every sample has the same weight
        if not self.use_weighted_loss:
            return torch.ones_like(x, device=x.device) / x.numel()

        # Otherwise, we need to compute the weights
        # The linear scheme used here is somewhat arbitrary
        weights = torch.tile(
            input=torch.linspace(0.001, 1.0, x.shape[1], device=x.device),
            dims=(x.shape[0], 1),
        )
        weights = weights / weights.sum()
        return weights

    @property
    def tensorboard_logger(self) -> TensorBoardLogger:
        if not isinstance(self.logger, TensorBoardLogger):  # pragma: no cover
            raise RuntimeError('No TensorBoard logger found!')
        # noinspection PyTypeChecker
        return self.logger

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
            result['lr_scheduler'] = {
                'scheduler': lr_scheduler,
                'interval': self.lr_scheduler_config.get('interval', 'epoch'),
                'monitor': 'val/total_loss',
                'frequency': 1,
            }

        return result

    def forward(
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

        # Compute the reconstruction loss on the normalized temperatures.
        # This is a manual version of an MSE loss which can also handle
        # weighted samples (e.g., give more weight to higher pressure).
        rl = (self.normalize(T_true) - self.normalize(T_pred)).pow(2)
        rl = rl * self.get_loss_weights_like(rl)
        reconstruction_loss = rl.sum()

        # Compute the MMD between z and a sample from a standard Gaussian.
        # We do this multiple times to get a better estimate of the MMD.
        mmd_loss = torch.zeros(z.shape[0], device=self.device)  # type: ignore
        for i in range(10):
            true_samples = torch.randn(  # type: ignore
                z.shape[0],
                self.encoder.latent_size,
                device=self.device,
            )
            mmd_loss = compute_mmd(true_samples, z)
        mmd_loss = mmd_loss / 10.0

        # Compute the total loss
        # Note: the `use_rl` parameter is used to turn off the reconstruction
        # loss during the first few epochs for encoder pre-training. This is
        # a hack to stop the encoder from collapsing to a single point.
        total_loss = (
            self.rl_weight * reconstruction_loss
            + self.beta * mmd_loss
        )

        return total_loss, reconstruction_loss, mmd_loss

    def on_train_epoch_start(self) -> None:

        # Slowly turn on the reconstruction loss during the first few epochs
        # to prevent the encoder from collapsing to a single point.
        pretrain_encoder = self.loss_config.get('pretrain_encoder', 0)
        if pretrain_encoder == 0:
            self.rl_weight = 1.0
        else:
            self.rl_weight = 1 / (1 + exp(-self.rl_weight + pretrain_encoder))

    def training_step(
        self,
        batch: List[torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._common_step(batch, batch_idx, "train")

    def validation_step(
        self,
        batch: List[torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._common_step(batch, batch_idx, "val")

    def test_step(
        self,
        batch: List[torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._common_step(batch, batch_idx, "test")  # pragma: no cover

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
        if batch_idx == 0 and self.plotting_config['enable_plotting']:
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
        self.tensorboard_logger.experiment.add_figure(
            tag=f'{label}/latent-distribution',
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
            self.plotting_config['pt_profile'],
        )

        # Add figure to TensorBoard
        self.tensorboard_logger.experiment.add_figure(
            tag=f'{label}/example-profiles',
            figure=figure,
            global_step=self.current_epoch,
        )

        # Close the figure
        plt.close(figure)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        raise NotImplementedError  # pragma: no cover

    def test_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError  # pragma: no cover

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError  # pragma: no cover

    def val_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError  # pragma: no cover

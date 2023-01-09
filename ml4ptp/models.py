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
from torch.nn.functional import softplus

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch

from ml4ptp.importing import get_member_by_name
from ml4ptp.mixins import NormalizerMixin
from ml4ptp.mmd import compute_mmd
from ml4ptp.plotting import plot_profile_to_tensorboard, plot_z_to_tensorboard
from ml4ptp.utils import fix_weak_reference, weighted_mse_loss


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
        normalization: dict,
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
            normalization: A dictionary containing the normalization
                parameters for the pressure and temperature.
            plotting_config: Configuration for TensorBoard plots.
            lr_scheduler_config: Configuration for the LR scheduler.
        """

        super().__init__()

        # Save hyperparameters.
        # This is needed so that at the end of training, we can load the best
        # model checkpoint with `model.load_from_checkpoint()` without having
        # to pass the constructor arguments again.
        self.save_hyperparameters()

        # Store the constructor arguments
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.optimizer_config = optimizer_config
        self.normalization = normalization
        self.loss_config = loss_config
        self.plotting_config = plotting_config
        self.lr_scheduler_config = lr_scheduler_config

        # Define some shortcuts
        self.beta = self.loss_config['beta']
        self.n_mmd_loops = self.loss_config.get('n_mmd_loops', 10)
        self.plot_interval = self.plotting_config.get('plot_interval', 10)

        # Keep track of the number of times we had to re-initialize the encoder
        self.n_failures = 0

        # Set up the encoder and decoder networks
        self.encoder = get_member_by_name(
            module_name='ml4ptp.encoders',
            member_name=encoder_config['name']
        )(
            **encoder_config['parameters'],
            normalization=normalization,
        )
        self.decoder = get_member_by_name(
            module_name='ml4ptp.decoders',
            member_name=decoder_config['name'],
        )(
            **decoder_config['parameters'],
            normalization=normalization,
        )

    def configure_optimizers(self) -> dict:
        """
        Set up the optimizer and, optionally, the LR scheduler.
        """

        # ---------------------------------------------------------------------
        # Set up the optimizer
        # ---------------------------------------------------------------------

        optimizer = get_member_by_name(
            module_name=self.optimizer_config.get('module', 'torch.optim'),
            member_name=self.optimizer_config.get('name', None),
        )(
            params=self.parameters(),
            **self.optimizer_config['parameters'],
        )
        result = {'optimizer': optimizer}

        # ---------------------------------------------------------------------
        # Set up the learning rate scheduler (if desired)
        # ---------------------------------------------------------------------

        if (config := self.lr_scheduler_config) is not None:

            # Create the scheduler
            lr_scheduler = get_member_by_name(
                module_name=config.get('module', 'torch.optim.lr_scheduler'),
                member_name=config.get('name', None)
            )(
                optimizer=optimizer,
                **config['parameters'],
            )

            # For some LR schedulers, we need to fix an issue with a weak
            # reference that prevents them from being checkpointed
            fix_weak_reference(lr_scheduler)

            # Add the scheduler to the result
            result['lr_scheduler'] = {
                'scheduler': lr_scheduler,
                'interval': config.get('interval', 'epoch'),
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
        z = self.encoder(log_P=log_P, T=T)

        # Check if we got unlucky with the weight initialization (i.e., the
        # norm of our latents is too small) and we need to re-initialize the
        # encoder to prevent the model from collapsing. We only do this at
        # the beginning of training, though.
        # This is obviously a hack, but it kinda seems to work?
        while self.current_epoch < 10:  # pragma: no cover

            # Check the norm of the latent codes
            # The threshold here is somewhat arbitrary, but it seems to work?
            mean_norm = torch.norm(z, dim=1).mean()  # type: ignore
            if 0.05 < mean_norm:
                break

            # If we got unlucky, re-initialize the encoder weights
            self.n_failures += 1
            print(
                f'\nWARNING: mean(norm(z)) = {mean_norm}! '
                f'[epoch={self.current_epoch}, n_failures={self.n_failures}]\n'
                f'Re-initializing encoder network!\n',
                flush=True
            )
            self.encoder.initialize_weights()
            z = self.encoder(log_P=log_P, T=T)

            # Abort training if we have already failed too many times
            if self.n_failures >= 100:  # pragma: no cover
                raise RuntimeError('Too many initialization failures!')

        # Run through decoder to get predicted temperatures
        T_pred = self.decoder(z=z, log_P=log_P)

        return z, T_pred

    def loss(
        self,
        z: torch.Tensor,
        T_true: torch.Tensor,
        T_pred: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Compute the (total) loss.
        """

        # Compute the reconstruction loss on the *normalized* temperatures,
        # so that the scale of the loss is independent of the temperature.
        rec_loss__normalized = weighted_mse_loss(
            x_pred=self.normalize_T(T_pred),
            x_true=self.normalize_T(T_true),
            weights=weights,
        )

        # For logging purposes, we also compute the unnormalized loss
        with torch.no_grad():
            rec_loss__unnormalized = weighted_mse_loss(T_pred, T_true, weights)

        # Compute the MMD between z and a sample from a standard Gaussian.
        # We do this multiple times to get a better estimate of the MMD.
        # Perhaps we could also increase the size of the `true_samples` tensor,
        # but the MMD scaled quadratically with the number of samples, so maybe
        # just doing several iterations is better? (Or maybe this not useful at
        # all, and we should just use a single sample?)
        mmd_loss = torch.tensor(0.0)
        for i in range(self.n_mmd_loops):
            sample = torch.randn(*z.shape, device=self.device)  # type: ignore
            mmd_loss = mmd_loss + compute_mmd(sample, z) / self.n_mmd_loops

        # Compute loss on the norm of the latent codes.
        # Essentially, we want to make sure that no latent codes end up too
        # far from the center (making it hard to define a prior over them).
        # In theory, the MMD should already take care of this, but in practice
        # it seems like adding this extra "soft barrier function" term to our
        # loss is helpful to catch cases that the MMD lets slip through.
        # One alternative was to place a (scaled) Tanh() layer at the end of
        # the encoder, but in practice, this makes it harder to train the model
        # because it sometimes gets stuck in a zero-gradient region.
        norms = torch.linalg.norm(z, dim=1)
        norm_loss = 10 * softplus(norms - 3.5, beta=100, threshold=10).mean()

        # Compute the total loss
        total_loss = rec_loss__normalized + self.beta * mmd_loss + norm_loss

        return (
            total_loss,
            rec_loss__normalized,
            rec_loss__unnormalized,
            mmd_loss,
            norm_loss,
        )

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
            self.train()
        else:
            self.eval()

        # Unpack the batch and compute pass through encoder and decoder
        log_P, T_true, weights = batch
        z, T_pred = self.forward(log_P=log_P, T=T_true)

        # Compute the loss terms
        (
            total_loss,
            rec_loss__normalized,
            rec_loss__unnormalized,
            latent_loss,
            norm_loss,
        ) = self.loss(z=z, T_true=T_true, T_pred=T_pred, weights=weights)

        # Log the loss terms to TensorBoard
        self.log_dict(
            dictionary={
                f"{stage}/total_loss": total_loss,
                f"{stage}/rec_loss__normalized": rec_loss__normalized,
                f"{stage}/rec_loss__unnormalized": rec_loss__unnormalized,
                f"{stage}/latent_loss": latent_loss,
                f"{stage}/norm_loss": norm_loss,
            },
            on_step=True,
            on_epoch=True,
        )

        # Every N epochs, create some plots and log them to TensorBoard.
        # We don't want to do this every epoch because plotting is quite slow
        # and it can slow down training significantly, plus it produces a lot
        # of data (hundreds of MB) that we don't really need.
        if (
            batch_idx == 0
            and self.logger is not None
            and self.plot_interval > 0
            and self.current_epoch % self.plot_interval == 0
        ):
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
        self.logger.experiment.add_figure(  # type: ignore
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
        self.logger.experiment.add_figure(  # type: ignore
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

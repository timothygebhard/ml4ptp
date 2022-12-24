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
        self.use_weighted_loss = self.loss_config.get('weighted_loss', False)

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

    def configure_optimizers(self) -> dict:
        """
        Set up the optimizer and, optionally, the LR scheduler.
        """

        # ---------------------------------------------------------------------
        # Set up the optimizer
        # ---------------------------------------------------------------------

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

            # For cyclic LR schedulers, we need the following hack to get rid
            # of a weak reference that would otherwise prevent the scheduler
            # from being pickled (which is required for checkpointing).
            # See: https://github.com/pytorch/pytorch/issues/88684
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.CyclicLR):
                lr_scheduler._scale_fn_custom = (  # type: ignore
                    lr_scheduler._scale_fn_ref()  # type: ignore
                )
                lr_scheduler._scale_fn_ref = None  # type: ignore

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
        # norm of our generated latents is too small or too big) and we need
        # to re-initialize the encoder to prevent the model from collapsing.
        # This is obviously a hack, but it kinda seems to work?
        while True:

            # Compute the norm of the latent codes
            mean_norm = torch.norm(z, dim=1).mean()  # type: ignore
            if 0.1 < mean_norm < 2.5:
                break  # pragma: no cover

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
            if self.n_failures > 100:
                raise RuntimeError('Too many initialization failures!')

        # Run through decoder to get predicted temperatures
        T_pred = self.decoder(z=z, log_P=log_P)

        return z, T_pred

    def loss(
        self,
        T_true: torch.Tensor,
        T_pred: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the loss.
        """

        # Compute the reconstruction loss on the *normalized* temperatures,
        # so that the scale of the loss is independent of the temperature.
        # This is a manual version of an MSE loss which can also handle
        # weighted samples (e.g., give more weight to higher pressure).
        rl = (self.normalize_T(T_true) - self.normalize_T(T_pred)).pow(2)
        rl = rl * self.get_loss_weights_like(rl)
        reconstruction_loss__normalized = rl.sum()

        # For logging purposes, we also compute the unnormalized loss
        with torch.no_grad():
            rl = (T_true - T_pred).pow(2)
            rl = rl * self.get_loss_weights_like(rl)
            reconstruction_loss__unnormalized = rl.sum()

        # Compute the MMD between z and a sample from a standard Gaussian.
        # We do this multiple times to get a better estimate of the MMD.
        # Perhaps we could also increase the size of the `true_samples` tensor,
        # but the MMD scaled quadratically with the number of samples, so maybe
        # just doing several iterations is better? (Or maybe this not useful at
        # all, and we should just use a single sample?)
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
        total_loss = reconstruction_loss__normalized + self.beta * mmd_loss

        return (
            total_loss,
            reconstruction_loss__normalized,
            reconstruction_loss__unnormalized,
            mmd_loss,
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
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()

        # Unpack the batch and compute pass through encoder and decoder
        log_P, T_true = batch
        z, T_pred = self.forward(log_P=log_P, T=T_true)

        # Compute the loss terms
        (
            total_loss,
            rec_loss__normalized,
            rec_loss__unnormalized,
            latent_loss,
        ) = self.loss(T_true=T_true, T_pred=T_pred, z=z)

        # Log the loss terms to TensorBoard
        self.log_dict(
            dictionary={
                f"{stage}/total_loss": total_loss,
                f"{stage}/rec_loss__normalized": rec_loss__normalized,
                f"{stage}/rec_loss__unnormalized": rec_loss__unnormalized,
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

        if self.logger:

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

        if self.logger:

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

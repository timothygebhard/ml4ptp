"""
Define decoder architectures.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Callable

import torch
import torch.nn as nn

from ml4ptp.layers import get_mlp_layers
from ml4ptp.mixins import NormalizerMixin


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

class Decoder(nn.Module, NormalizerMixin):

    def __init__(
        self,
        latent_size: int,
        layer_size: int,
        n_layers: int,
        T_mean: float,
        T_std: float,
        activation: str = 'leaky_relu',
        final_sigmoid: bool = False,
    ) -> None:

        super().__init__()

        # Store constructor arguments
        self.latent_size = latent_size
        self.layer_size = layer_size
        self.n_layers = n_layers
        self.T_mean = float(T_mean)
        self.T_std = float(T_std)

        # Define encoder architecture
        # Note: The `+ 1` on the input is for the (log) pressure at which we
        # want to evaluate the profile represented by this decoder.
        self.layers: Callable[[torch.Tensor], torch.Tensor] = get_mlp_layers(
            input_size=latent_size + 1,
            n_layers=n_layers,
            layer_size=layer_size,
            output_size=1,
            activation=activation,
            final_sigmoid=final_sigmoid,
        )

    def forward(self, z: torch.Tensor, log_P: torch.Tensor) -> torch.Tensor:

        # Reminder:
        #  * z has shape (batch_size, latent_size)
        #  * log_p has shape (batch_size, grid_size)
        #  * The decoder takes inputs of size latent_size + 1
        #  * We now need to combine z and log_p into a single tensor of
        #    shape (batch_size * grid_size, latent_size + 1) that we can
        #    give to the decoder
        #  * After passing through the decoder, we want to reshape the output
        #    to (batch_size, grid_size) again: one T for each log_P

        # Get batch size and grid size
        batch_size, grid_size = log_P.shape

        # Repeat z so that we can concatenate it with every pressure value.
        # This changes the shape of z:
        #   (batch_size, latent_size) -> (batch_size, grid_size, latent_size)
        z = z.unsqueeze(1).repeat(1, grid_size, 1)

        # Reshape the pressure grid and z into the batch dimension. They now
        # show have shapes:
        #   p_flat: (batch_size * grid_size, 1)
        #   z_flat: (batch_size * grid_size, latent_size)
        p_flat = log_P.reshape(batch_size * grid_size, 1)
        z_flat = z.reshape(batch_size * grid_size, self.latent_size)

        # Get the decoder input by concatenating x and z. The shape should be:
        #   (batch_size * n_points, 1 + z_dim)
        decoder_input = torch.cat((p_flat, z_flat), dim=1)

        # Send through decoder. The decoder output will have shape:
        #   (batch_size * grid_size, 1)
        decoded = self.layers(decoder_input)

        # Reshape the decoder output to (batch_size, grid_size), that is,
        # the original shape of `log_p`
        T_pred = decoded.reshape(batch_size, grid_size)

        # Undo normalization (i.e., transform back to Kelvin)
        T_pred = self.normalize(T_pred, undo=True)

        return T_pred

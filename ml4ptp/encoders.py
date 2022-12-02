"""
Define encoder architectures.
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

class MLPEncoder(nn.Module, NormalizerMixin):

    def __init__(
        self,
        input_size: int,
        latent_size: int,
        layer_size: int,
        n_layers: int,
        T_offset: float,
        T_factor: float,
    ) -> None:

        super().__init__()

        # Store constructor arguments
        self.input_size = input_size
        self.latent_size = latent_size
        self.layer_size = layer_size
        self.n_layers = n_layers
        self.T_offset = T_offset
        self.T_factor = T_factor

        # Define encoder architecture
        self.layers: Callable[[torch.Tensor], torch.Tensor] = get_mlp_layers(
            input_size=2 * self.input_size,
            layer_size=self.layer_size,
            n_layers=self.n_layers,
            output_size=self.latent_size,
            activation='leaky_relu',
            final_tanh=True,
        )

    def forward(self, log_P: torch.Tensor, T: torch.Tensor) -> torch.Tensor:

        # Normalize temperatures and construct encoder input
        normalized_T = self.normalize(T)
        encoder_input = torch.concat(tensors=(log_P, normalized_T), dim=1)

        # Compute forward pass through encoder to get latent variable z
        z = self.layers(encoder_input)

        return z


class ModifiedMLPEncoder(nn.Module, NormalizerMixin):
    """
    A modified version of the MLP encoder that first maps each tuple
    `(log_P, T)` to a single value (via a small MLP), effectively
    reducing the input to a one-dimesional vector. This is then used
    as the input to the second MLP encoder.
    """

    def __init__(
        self,
        input_size: int,
        latent_size: int,
        layer_size: int,
        n_layers: int,
        T_offset: float,
        T_factor: float,
    ) -> None:

        super().__init__()

        # Store constructor arguments
        self.input_size = input_size
        self.latent_size = latent_size
        self.layer_size = layer_size
        self.n_layers = n_layers
        self.T_offset = T_offset
        self.T_factor = T_factor

        # Define encoder architecture
        self.layers_1: Callable[[torch.Tensor], torch.Tensor] = get_mlp_layers(
            input_size=2,
            layer_size=64,
            n_layers=2,
            output_size=1,
            activation='leaky_relu',
            final_tanh=False,
        )
        self.layers_2: Callable[[torch.Tensor], torch.Tensor] = get_mlp_layers(
            input_size=self.input_size,
            layer_size=self.layer_size,
            n_layers=self.n_layers,
            output_size=self.latent_size,
            activation='leaky_relu',
            final_tanh=True,
        )

    def forward(self, log_P: torch.Tensor, T: torch.Tensor) -> torch.Tensor:

        # Normalize temperatures and construct encoder input
        normalized_T = self.normalize(T)

        # Construct encoder input: Reshape grid into batch dimension
        batch_size, grid_size = log_P.shape
        log_P_flat = log_P.reshape(batch_size * grid_size, 1)
        normalized_T_flat = normalized_T.reshape(batch_size * grid_size, 1)
        encoder_input = torch.concat((log_P_flat, normalized_T_flat), dim=1)

        # Compute forward pass through first encoder part
        output = torch.nn.functional.leaky_relu(self.layers_1(encoder_input))

        # Reshape output back to grid
        output = output.reshape(batch_size, grid_size)

        # Compute forward pass through second encoder part
        z = self.layers_2(output)

        return z


class CNPEncoder(nn.Module, NormalizerMixin):

    def __init__(
        self,
        latent_size: int,
        layer_size: int,
        n_layers: int,
        T_offset: float,
        T_factor: float,
    ) -> None:

        super().__init__()

        # Store constructor arguments
        self.latent_size = latent_size
        self.layer_size = layer_size
        self.n_layers = n_layers
        self.T_offset = T_offset
        self.T_factor = T_factor

        # Define encoder architecture
        self.layers: Callable[[torch.Tensor], torch.Tensor] = get_mlp_layers(
            input_size=2,
            layer_size=self.layer_size,
            n_layers=self.n_layers,
            output_size=self.latent_size,
            activation='leaky_relu',
            final_tanh=True,
        )

    def forward(self, log_P: torch.Tensor, T: torch.Tensor) -> torch.Tensor:

        # Normalize temperatures
        normalized_T = self.normalize(T)

        # Construct encoder input: Reshape grid into batch dimension
        batch_size, grid_size = log_P.shape
        log_P_flat = log_P.reshape(batch_size * grid_size, 1)
        normalized_T_flat = normalized_T.reshape(batch_size * grid_size, 1)
        encoder_input = torch.concat((log_P_flat, normalized_T_flat), dim=1)

        # Compute forward pass through encoder to get latent variable z
        z_values = self.layers(encoder_input)

        # Reshape to get grid dimension back
        z_values = z_values.reshape(batch_size, grid_size, self.latent_size)

        # Aggregate along grid dimension to get final z
        z = torch.mean(z_values, dim=1)

        return z

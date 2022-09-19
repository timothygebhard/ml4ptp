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
        T_mean: float,
        T_std: float,
    ) -> None:

        super().__init__()

        # Store constructor arguments
        self.input_size = input_size
        self.latent_size = latent_size
        self.layer_size = layer_size
        self.n_layers = n_layers
        self.T_mean = T_mean
        self.T_std = T_std

        # Define encoder architecture
        self.layers: Callable[[torch.Tensor], torch.Tensor] = get_mlp_layers(
            input_size=2 * self.input_size,
            layer_size=self.layer_size,
            n_layers=self.n_layers,
            output_size=self.latent_size,
        )

    def forward(self, log_P: torch.Tensor, T: torch.Tensor) -> torch.Tensor:

        # Normalize temperatures and construct encoder input
        normalized_T = self.normalize(T)
        encoder_input = torch.column_stack((log_P, normalized_T))

        # Compute forward pass through encoder to get latent variable z
        z = self.layers(encoder_input)

        return z


class CNPEncoder(nn.Module, NormalizerMixin):

    def __init__(
        self,
        latent_size: int,
        layer_size: int,
        n_layers: int,
        T_mean: float,
        T_std: float,
    ) -> None:

        super().__init__()

        # Store constructor arguments
        self.latent_size = latent_size
        self.layer_size = layer_size
        self.n_layers = n_layers
        self.T_mean = T_mean
        self.T_std = T_std

        # Define encoder architecture
        self.layers: Callable[[torch.Tensor], torch.Tensor] = get_mlp_layers(
            input_size=2,
            layer_size=self.layer_size,
            n_layers=self.n_layers,
            output_size=self.latent_size,
        )

    def forward(self, log_P: torch.Tensor, T: torch.Tensor) -> torch.Tensor:

        # Normalize temperatures
        normalized_T = self.normalize(T)

        # Construct encoder input: Reshape grid into batch dimension
        batch_size, grid_size = log_P.shape
        log_P = log_P.reshape(batch_size * grid_size, 1)
        normalized_T = normalized_T.reshape(batch_size * grid_size, 1)
        encoder_input = torch.column_stack((log_P, normalized_T))

        # Compute forward pass through encoder to get latent variable z
        z_values = self.layers(encoder_input)

        # Reshape to get grid dimension back
        z_values = z_values.reshape(batch_size, grid_size, self.latent_size)

        # Aggregate along grid dimension to get final z
        z = torch.mean(z_values, dim=1)

        return z

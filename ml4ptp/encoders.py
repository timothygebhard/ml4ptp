"""
Define encoder architectures.
"""


# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Any, Dict

import torch
import torch.nn as nn

from ml4ptp.layers import get_mlp_layers, Squeeze
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
        normalization: Dict[str, Any],
        batch_norm: bool = False,
    ) -> None:

        super().__init__()

        # Store constructor arguments
        self.input_size = input_size
        self.latent_size = latent_size
        self.layer_size = layer_size
        self.n_layers = n_layers
        self.normalization = normalization
        self.batch_norm = batch_norm

        # Define encoder architecture
        self.layers: torch.nn.Sequential = get_mlp_layers(
            input_size=2 * self.input_size,
            layer_size=self.layer_size,
            n_layers=self.n_layers,
            output_size=self.latent_size,
            activation='leaky_relu',
            final_tanh=True,
            batch_norm=batch_norm,
        )

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """
        Initialize weights of encoder layers. Preliminary experiments
        show that this can help to prevent the encoder from predicting
        all zeros (which leads to the model not learning anything).
        """

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.01)

    def forward(self, log_P: torch.Tensor, T: torch.Tensor) -> torch.Tensor:

        # Normalize and concatenate log_P and T to form encoder input
        encoder_input = torch.cat(
            tensors=(
                self.normalize_log_P(log_P),
                self.normalize_T(T),
            ),
            dim=1,
        )

        # Compute forward pass through encoder to get latent variable z
        z: torch.Tensor = self.layers(encoder_input)

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
        normalization: Dict[str, Any],
        batch_norm: bool = False,
    ) -> None:

        super().__init__()

        # Store constructor arguments
        self.input_size = input_size
        self.latent_size = latent_size
        self.layer_size = layer_size
        self.n_layers = n_layers
        self.normalization = normalization
        self.batch_norm = batch_norm

        # Define encoder architecture
        self.layers_1: torch.nn.Sequential = get_mlp_layers(
            input_size=2,
            layer_size=64,
            n_layers=2,
            output_size=1,
            activation='leaky_relu',
            final_tanh=False,
            batch_norm=batch_norm,
        )
        self.layers_2: torch.nn.Sequential = get_mlp_layers(
            input_size=self.input_size,
            layer_size=self.layer_size,
            n_layers=self.n_layers,
            output_size=self.latent_size,
            activation='leaky_relu',
            final_tanh=True,
            batch_norm=batch_norm,
        )

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """
        Initialize weights of encoder layers. Preliminary experiments
        show that this can help to prevent the encoder from predicting
        all zeros (which leads to the model not learning anything).
        """

        # Initialize weights
        for layer in self.layers_1:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.01)
        for layer in self.layers_2:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.01)

    def forward(self, log_P: torch.Tensor, T: torch.Tensor) -> torch.Tensor:

        # Normalize inputs
        normalized_log_P = self.normalize_log_P(log_P)
        normalized_T = self.normalize_T(T)

        # Construct encoder input: Reshape grid into batch dimension
        batch_size, grid_size = log_P.shape
        encoder_input = torch.cat(
            tensors=(
                normalized_log_P.reshape(batch_size * grid_size, 1),
                normalized_T.reshape(batch_size * grid_size, 1),
            ),
            dim=1,
        )

        # Compute forward pass through first encoder part
        output = torch.nn.functional.leaky_relu(self.layers_1(encoder_input))

        # Reshape output back to grid
        output = output.reshape(batch_size, grid_size)

        # Compute forward pass through second encoder part
        z: torch.Tensor = self.layers_2(output)

        return z


class ConvolutionalEncoder(nn.Module, NormalizerMixin):

    def __init__(
        self,
        input_size: int,
        latent_size: int,
        layer_size: int,
        n_layers: int,
        normalization: Dict[str, Any],
        n_channels: int = 64,
        kernel_size: int = 3,
        batch_norm: bool = False,
    ) -> None:

        super().__init__()

        # Store constructor arguments
        self.input_size = input_size
        self.latent_size = latent_size
        self.layer_size = layer_size
        self.n_layers = n_layers
        self.normalization = normalization
        self.batch_norm = batch_norm

        # Define encoder architecture
        self.convnet = nn.Sequential(
            torch.nn.Conv1d(
                in_channels=2,
                out_channels=n_channels,
                kernel_size=kernel_size,
            ),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(n_channels) if batch_norm else nn.Identity(),
            torch.nn.Conv1d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
            ),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(n_channels) if batch_norm else nn.Identity(),
            torch.nn.Conv1d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
            ),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(n_channels) if batch_norm else nn.Identity(),
            torch.nn.Conv1d(
                in_channels=n_channels,
                out_channels=1,
                kernel_size=kernel_size,
            ),
            torch.nn.LeakyReLU(),
            Squeeze(),
        )
        self.mlp: torch.nn.Sequential = get_mlp_layers(
            input_size=self.input_size - 4 * (kernel_size - 1),  # no padding
            layer_size=self.layer_size,
            n_layers=self.n_layers,
            output_size=self.latent_size,
            activation='leaky_relu',
            final_tanh=True,
            batch_norm=batch_norm,
        )

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """
        Initialize weights of encoder layers. Preliminary experiments
        show that this can help to prevent the encoder from predicting
        all zeros (which leads to the model not learning anything).
        """

        for layer in self.convnet:
            if isinstance(layer, nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.01)  # type: ignore

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.01)

    def forward(self, log_P: torch.Tensor, T: torch.Tensor) -> torch.Tensor:

        # Stack log_P and T together into a tensor with 2 channels, that is,
        # the shape is `(batch_size, 2, grid_size)`
        encoder_input = torch.stack(
            tensors=(
                self.normalize_log_P(log_P),
                self.normalize_T(T),
            ),
            dim=1,
        )

        # Compute forward pass through encoder to get latent variable z
        x: torch.Tensor = self.convnet(encoder_input)
        z: torch.Tensor = self.mlp(x)

        return z


class CNPEncoder(nn.Module, NormalizerMixin):

    def __init__(
        self,
        latent_size: int,
        layer_size: int,
        n_layers: int,
        normalization: Dict[str, Any],
        batch_norm: bool = False,
    ) -> None:

        super().__init__()

        # Store constructor arguments
        self.latent_size = latent_size
        self.layer_size = layer_size
        self.n_layers = n_layers
        self.normalization = normalization
        self.batch_norm = batch_norm

        # Define encoder architecture
        self.layers: torch.nn.Sequential = get_mlp_layers(
            input_size=2,
            layer_size=self.layer_size,
            n_layers=self.n_layers,
            output_size=self.latent_size,
            activation='leaky_relu',
            final_tanh=True,
            batch_norm=batch_norm,
        )

    def forward(self, log_P: torch.Tensor, T: torch.Tensor) -> torch.Tensor:

        # Normalize inputs
        normalized_log_P = self.normalize_log_P(log_P)
        normalized_T = self.normalize_T(T)

        # Construct encoder input: Reshape grid into batch dimension
        batch_size, grid_size = log_P.shape
        encoder_input = torch.cat(
            tensors=(
                normalized_log_P.reshape(batch_size * grid_size, 1),
                normalized_T.reshape(batch_size * grid_size, 1),
            ),
            dim=1,
        )

        # Compute forward pass through encoder to get latent variable z
        z_values = self.layers(encoder_input)

        # Reshape to get grid dimension back
        z_values = z_values.reshape(batch_size, grid_size, self.latent_size)

        # Aggregate along grid dimension to get final z
        z: torch.Tensor = torch.mean(z_values, dim=1)

        return z

"""
Various unused methods.

This is meant as a backup for future work: Ultimately, we did use the
things in here for the paper, but maybe they will come in handy in the
future, so let's keep the code in the repository?
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from ml4ptp.layers import get_activation, Sine, Identity
from ml4ptp.mixins import NormalizerMixin


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

class FourierEncoding(torch.nn.Module):
    """
    Fourier encoding layer.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        kind: How to generate the frequencies. Options are:
            - 'gaussian': Sample frequencies from a Gaussian with
                the given `sigma`. All channels are independent.
            - 'linear': Sample frequencies linearly from `min_freq` to
                `max_freq`. All channels use the same frequencies.
            - 'log': Sample frequencies log-spaced from `min_freq` to
                `max_freq`. All channels use the same frequencies.
        sigma: Standard deviation of the Gaussian from which the
            frequencies are sampled. Only used if `kind` is set to
            'gaussian'.
        min_freq: Minimum frequency. Only used if `kind` is set to
            'linear' or 'log'.
        max_freq: Maximum frequency. Only used if `kind` is set to
            'linear' or 'log'.
    """

    def __init__(
        self,
        in_features: int = 2,
        out_features: int = 256,
        kind: str = 'gaussian',
        sigma: float = 10.0,
        min_freq: float = 1.0,
        max_freq: float = 10.0,
    ):

        super().__init__()

        # Ensure that `out_features` is even
        assert out_features % 2 == 0, 'Number of output features must be even.'

        # Store constructor arguments
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.kind = kind
        self.sigma = float(sigma)
        self.min_freq = float(min_freq)
        self.max_freq = float(max_freq)

        # Create frequency matrix
        if kind == 'gaussian':
            frequencies = (
                sigma * torch.randn(out_features // 2, in_features)
            )
        elif kind == 'linear':
            frequencies = torch.linspace(
                start=min_freq,
                end=max_freq,
                steps=out_features // 2,
            ).tile((in_features, 1)).T
        elif kind == 'log':
            frequencies = torch.logspace(
                start=np.log10(min_freq),
                end=np.log10(max_freq),
                steps=out_features // 2,
                base=10.0,
            ).tile((in_features, 1)).T
        else:
            raise ValueError(f'Unknown encoding kind: {kind}.')

        # Register frequencies as a buffer. This makes sure that they are
        # saved and loaded with the model, and that they are moved to the
        # correct device when the model is moved. The typehint is needed so
        # that mypy does not complain in the `__call__()` method.
        self.frequencies: torch.Tensor
        self.register_buffer('frequencies', frequencies)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:

        # Shapes (reminder):
        #   frequencies: (out_features // 2, in_features)
        #   x: (batch_size, in_features, grid_size)
        # The multiplication below broadcasts over the batch dimension.
        # The result is of shape (batch_size, out_features // 2, grid_size).
        out = 2 * 3.14159265359 * self.frequencies @ x

        # Concatenate sine and cosine parts along the channel dimension
        out = torch.cat((torch.cos(out), torch.sin(out)), dim=1)

        return out

    def __repr__(self) -> str:

        return (
            f'FourierEncoding(\n'
            f'  in_features={self.in_features}, \n'
            f'  out_features={self.out_features}, \n'
            f'  kind="{self.kind}", \n'
            f'  sigma={self.sigma}, \n'
            f'  min_freq={self.min_freq}, \n'
            f'  max_freq={self.max_freq}, \n'
            f')'
        )


class ModulateWithZ(nn.Module):
    """
    Wrapper around `ModulatedLinear` that takes care of the `z` input.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        latent_size: int,
    ) -> None:

        super().__init__()

        # Store constructor arguments
        self.in_features = in_features
        self.out_features = out_features
        self.latent_size = latent_size

        # Inialize dummy `z` tensor
        self.z = torch.full((1, latent_size), torch.nan)

        # Create modulated linear layer
        self.layer = ModulatedLinear(
            in_features=in_features,
            out_features=out_features,
            latent_size=latent_size,
        )

    def update_z(self, z: torch.Tensor) -> None:
        assert z.shape[1] == self.layer.latent_size, 'z has wrong latent size!'
        self.z = z

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        assert not torch.isnan(self.z).any(), 'z has not been updated yet!'
        return self.layer.forward(x=tensor, z=self.z)

    def __repr__(self) -> str:
        return (
            f'ModulateWithZ('
            f'in_features={self.in_features}, '
            f'out_features={self.out_features}, '
            f'latent_size={self.latent_size}'
            f')'
        )


class ModulatedLinear(nn.Module):
    """
    Linear layer whose weights are modulated using the latent vector.

    Supposedly, this is a better than just concatenating the latent
    vector to the input and using a regular linear layer, as it is
    multiplicative instead of additive.

    References:
    - https://arxiv.org/abs/2011.13775 (original idea)
    - https://arxiv.org/abs/2110.09788 (more detailed description of
        the implementation that we have used here)

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        latent_size: Size of the latent vector.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        latent_size: int,
    ) -> None:

        super().__init__()

        # Store constructor arguments
        self.in_features = in_features
        self.out_features = out_features
        self.latent_size = latent_size

        # Weight and bias of the modulated linear layer
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        # Initialize weight and bias
        nn.init.xavier_normal_(self.weight)
        nn.init.constant_(self.bias, 0.01)

        # Affine transformation for the latent vector
        self.latent_transform = nn.Linear(latent_size, in_features)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:

        # Apply affine transformation to the latent vector
        # Shape after transformation: (batch_size, in_features)
        z = self.latent_transform(z)

        # Some reshaping to make the following operations easier
        W = self.weight.reshape(1, self.in_features, self.out_features)
        S = z.reshape(-1, self.in_features, 1)
        x = x.reshape(-1, 1, self.in_features)

        # Modulate weights (broadcast over batch dimension)
        # Shape after modulation: (batch_size, in_features, out_features)
        W_ = W * S

        # Compute normalization factor
        factor = torch.rsqrt(W_.pow(2).sum(dim=1) + 1e-8)
        factor = factor.reshape(-1, 1, self.out_features)

        # Apply normalization ("demodulation")
        # Shape after normalization: (batch_size, in_features, out_features)
        W__ = W_ * factor

        # Apply modulated weights to input
        out = torch.bmm(x, W__) + self.bias
        out = out.reshape(-1, self.out_features)

        return out

    def __repr__(self) -> str:
        return (
            f'ModulatedLinear('
            f'in_features={self.in_features}, '
            f'out_features={self.out_features}, '
            f'latent_size={self.latent_size}'
            f')'
        )


class ModulatedDecoder(nn.Module, NormalizerMixin):
    """
    Similar to the standard decoder, but with skip connections: The
    latent variable is concatenated to the output of each layer.

    Arguments:
        latent_size: The size of the latent space.
        layer_size: The size of the hidden layers of the decoder.
        n_layers: The number of hidden layers of the decoder.
        normalization: A dictionary containing the normalization
            parameters for the pressure and temperature.
        activation: The activation function to use in the decoder.
    """

    def __init__(
        self,
        latent_size: int,
        layer_size: int,
        n_layers: int,
        normalization: Dict[str, Any],
        activation: str = 'leaky_relu',
        dropout: float = 0.0,
    ) -> None:

        super().__init__()

        # Store constructor arguments
        self.latent_size = latent_size
        self.layer_size = layer_size
        self.n_layers = n_layers
        self.normalization = normalization
        self.activation = activation
        self.dropout = dropout

        # Mapping network for latent vector
        self.mapping = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
        )

        # ---------------------------------------------------------------------
        # Collect list of all layers
        # ---------------------------------------------------------------------

        # Start with the first layer (special case for SIRENs)
        layers = nn.ModuleList(
            [
                ModulateWithZ(
                    in_features=1,  # pressure value
                    out_features=layer_size,
                    latent_size=128,
                ),
                (
                    Sine(w0=3.0)
                    if activation == 'siren'
                    else get_activation(self.activation)
                ),
                (
                    nn.Dropout(self.dropout)
                    if self.dropout > 0.0
                    else Identity()
                )
            ]
        )

        # Add hidden layers
        for _ in range(n_layers):
            layers += [
                ModulateWithZ(
                    in_features=layer_size,
                    out_features=layer_size,
                    latent_size=128,
                ),
                get_activation(self.activation),
                (
                    nn.Dropout(self.dropout)
                    if self.dropout > 0.0
                    else Identity()
                )
            ]

        # Add final layer (no activation function)
        layers += [
            ModulateWithZ(
                in_features=layer_size,
                out_features=1,
                latent_size=128,
            ),
        ]

        # Drop Identity layers (which were only a stand-in for "no batch norm")
        layers = nn.ModuleList(
            [_ for _ in layers if not isinstance(_, Identity)]
        )

        # Combine all layers into a single sequential module
        self.layers = torch.nn.Sequential(*layers)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """
        Initialize the weights of the decoder. Only needed for SIRENs.
        """

        if self.activation == 'siren':
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    n = layer.weight.shape[-1]
                    nn.init.uniform_(
                        layer.weight, -np.sqrt(6 / n), np.sqrt(6 / n)
                    )

    def forward(self, z: torch.Tensor, log_P: torch.Tensor) -> torch.Tensor:

        # Most of this is analogous to the standard `Decoder` class, so see
        # there for a more detailed explanation.

        # Get batch size and grid size
        batch_size, grid_size = log_P.shape

        # Repeat z so that we can concatenate it with every pressure value.
        # This changes the shape of z:
        #   (batch_size, latent_size) -> (batch_size, grid_size, latent_size)
        z = z.unsqueeze(1).repeat(1, grid_size, 1)

        # Reshape the pressure grid and z into the batch dimension.
        # They now show have shapes:
        #   p_flat: (batch_size * grid_size, 1)
        #   z_flat: (batch_size * grid_size, latent_size)
        log_P_flat = log_P.reshape(batch_size * grid_size, 1)
        z_flat = z.reshape(batch_size * grid_size, self.latent_size)

        # Pass z through mapping network
        z_mapped = self.mapping(z_flat)

        # Update z in each layer
        # NOTE: This is the "condition D on z" step!
        for layer in self.layers:
            if isinstance(layer, ModulateWithZ):
                layer.update_z(z=z_mapped)

        # Normalize pressure and send through decoder.
        # The decoder output will have shape: (batch_size * grid_size, 1)
        decoded: torch.Tensor = self.layers(self.normalize_log_P(log_P_flat))

        # Reshape the decoder output back to to the original shape of log_P
        T_pred = decoded.reshape(batch_size, grid_size)

        # Undo normalization (i.e., transform back to Kelvin)
        T_pred = self.normalize_T(T_pred, undo=True)

        return T_pred

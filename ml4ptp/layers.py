"""
Useful custom layers for PyTorch.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_activation(name: str) -> nn.Module:
    """
    Get an activation function based on its `name`.
    """

    if name == 'elu':
        return nn.ELU()
    if name == 'gelu':
        return nn.GELU()
    if name == 'leaky_relu':
        return nn.LeakyReLU()
    if name == 'mish':
        return nn.Mish()
    if name == 'prelu':
        return nn.PReLU()
    if name == 'relu':
        return nn.ReLU()
    if name == 'sine' or name == 'siren':
        return Sine()
    if name == 'swish':
        return nn.SiLU()
    if name == 'tanh':
        return nn.Tanh()

    raise ValueError(f'Could not resolve name "{name}"!')


def get_mlp_layers(
    input_size: int,
    n_layers: int,
    layer_size: int,
    output_size: int = 1,
    activation: str = 'leaky_relu',
    batch_norm: bool = False,
    dropout: float = 0.0,
) -> nn.Sequential:
    """
    Create a multi-layer perceptron with the layer sizes.

    Args:
        input_size: Number of input neurons.
        n_layers: Number of *hidden* layers: If this is set to 0, the
            resulting network still have 2 layers for input and output.
        layer_size: Number of neurons in the hidden layers.
        output_size: Number of output neurons.
        activation: Which kind of activation function to use.
            If "siren" is used, the MLP will use sine as the activation
            function and apply the special initialization scheme from
            Sitzmann et al. (2020).
        batch_norm: Whether to use batch normalization.
        dropout: Dropout probability. If 0, no dropout is used.

    Returns:
        A `nn.Sequential` container with the desired MLP.
    """

    # Define layers: Start with input layer + activation + batch norm
    layers = [
        nn.Linear(input_size, layer_size),
        (
            Sine(w0=3.0)
            if activation == 'siren'
            else get_activation(activation)
        ),
        (
            nn.BatchNorm1d(layer_size)
            if batch_norm
            else Identity()
        ),
        (
            nn.Dropout(dropout)
            if dropout > 0.0
            else Identity()
        )
    ]

    # Add hidden layers
    for _ in range(n_layers):
        layers += [
            nn.Linear(layer_size, layer_size),
            get_activation(name=activation),
            (
                nn.BatchNorm1d(layer_size)
                if batch_norm
                else Identity()
            ),
            (
                nn.Dropout(dropout)
                if dropout > 0.0
                else Identity()
            )
        ]

    # Add output layer (no activation, no batch norm, no dropout)
    layers += [nn.Linear(layer_size, output_size)]

    # Drop any `Identity` layers
    layers = [_ for _ in layers if not isinstance(_, Identity)]

    # Apply special initialization scheme for SIREN networks
    if activation == 'siren':
        for layer in layers:
            if isinstance(layer, nn.Linear):
                n = layer.weight.shape[-1]
                nn.init.uniform_(layer.weight, -np.sqrt(6 / n), np.sqrt(6 / n))

    return nn.Sequential(*layers)


def get_cnn_layers(
    n_layers: int,
    n_channels: int,
    kernel_size: int,
    activation: str = 'leaky_relu',
    batch_norm: bool = False,
    in_channels: int = 2,
    out_channels: int = 1,
) -> nn.Sequential:
    """
    Create a 1D convolutional neural network with 2 input and 1 output
    channels, and the given number of "hidden" layers.

    Args:
        n_layers: Number of *hidden* convolutional layers: If this is
            set to 0, the resulting network still have 2 layers for
            input and output.
        n_channels: Number of channels in the hidden layers.
        kernel_size: Size of the 1D convolutional kernels.
        activation: Which kind of activation function to use.
        batch_norm: Whether to use batch normalization.
        in_channels: Number of input channels. This should be 2 for
            most cases, except, for example, if we use Fourier features
            from a `FourierEncoding` layer.
        out_channels: Number of output channels. (Usually 1.)

    Returns:
        A `nn.Sequential` container with the desired CNN.
    """

    # Define layers: Start with input layer
    layers = [
        nn.Conv1d(
            in_channels=in_channels,
            out_channels=n_channels,
            kernel_size=kernel_size,
        ),
        get_activation(name=activation),
        (
            nn.BatchNorm1d(num_features=n_channels)
            if batch_norm
            else Identity()
        ),
    ]

    # Add hidden layers
    for _ in range(n_layers):
        layers += [
            nn.Conv1d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
            ),
            get_activation(name=activation),
            (
                nn.BatchNorm1d(num_features=n_channels)
                if batch_norm
                else Identity()
            ),
        ]

    # Add output layer (and squeeze out the channel dimension)
    layers += [
        nn.Conv1d(
            in_channels=n_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
        ),
        Squeeze(),
    ]

    # Drop any `Identity` layers (which were stand-ins for batch norm)
    layers = [_ for _ in layers if not isinstance(_, Identity)]

    return nn.Sequential(*layers)


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class ConcatenateWithZ(nn.Module):
    """
    Concatenate the input with a given `z`.

    Note: This always needs to be updated in the `forward()` method of
        a decoder network to achieve the "condition D on z" behavior.

    Args:
        z: The latent code `z` to concatenate with the input.
        layer_size: The size of the extra layer through which `z` is
            sent before concatenation. This layer has no bias, so it
            is really only a matrix multiplication. If `layer_size` is
            set to 0, no layer is used and `z` is concatenated directly.
    """

    def __init__(self, z: torch.Tensor, layer_size: int = 64) -> None:

        super().__init__()

        self.z = z
        self.layer_size = layer_size

        # If `layer_size` is set to 0, we don't need an extra layer
        if self.layer_size > 0:
            self.layer = nn.Linear(z.shape[1], self.layer_size, bias=False)

    def update_z(self, z: torch.Tensor) -> None:
        self.z = z

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:

        # If we have a layer, use it to transform z
        z = self.layer(self.z) if self.layer_size > 0 else self.z

        return torch.cat(tensors=(tensor, z), dim=1)

    def __repr__(self) -> str:

        # Shortcut for output dimensionality
        out_size = self.z.shape[1] if self.layer_size == 0 else self.layer_size

        return (
            f'ConcatenateWithZ('
            f'latent_size={self.z.shape[1]}, '
            f'layer_size={self.layer_size}, '
            f'out_size={out_size}+in_size'
            f')'
        )


class Mean(nn.Module):
    """
    Wrap the `.mean()` method into a `nn.Module`.
    """

    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.mean(dim=self.dim)

    def __repr__(self) -> str:
        return f'Mean(dim={self.dim})'


class PrintShape(torch.nn.Module):
    """
    An identity mapping that prints the shape of the tensor that is
    passed through it; optionally with a label (for the layer name).
    """

    def __init__(self, label: str = '') -> None:
        super().__init__()
        self.label = label

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        print(self.label + ': ', tensor.shape)
        return tensor


class PrintValue(torch.nn.Module):
    """
    An identity mapping that prints the value of the tensor that is
    passed through it; optionally with a label (for the layer name).
    """

    def __init__(self, label: str = '') -> None:
        super().__init__()
        self.label = label

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        print(self.label + ':\n', tensor)
        return tensor


class Sine(torch.nn.Module):
    """
    A wrapper around 'torch.sin()` to use it as an activation function.
    Optionally with a frequency parameter `w0`.
    """

    def __init__(self, w0: float = 1.0) -> None:
        super().__init__()
        self.w0 = w0

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * tensor)

    def __repr__(self) -> str:
        return f'Sine(w0={self.w0})'


class Identity(torch.nn.Module):
    """
    A dummy layer that simply returns its input without doing anything.
    """

    def __init__(self) -> None:
        super().__init__()

    # Note: This cannot be a @staticmethod because otherwise the export
    # using torchscript seems to break.
    # noinspection PyMethodMayBeStatic
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


class ScaledTanh(nn.Module):
    """
    Scaled version of a Tanh activation function: `a * tanh(x / b)`.
    """

    def __init__(self, a: float = 3.0, b: float = 3.0):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.a * torch.tanh(tensor / self.b)

    def __repr__(self) -> str:
        return f'ScaledTanh(a={self.a}, b={self.b})'


class Squeeze(nn.Module):
    """
    Wrap the `.squeeze()` method into a `nn.Module`.
    """

    def __init__(self) -> None:
        super().__init__()

    # noinspection PyMethodMayBeStatic
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.squeeze()


class View(nn.Module):
    """
    Wrap the `.view()` method into a `nn.Module`.
    """

    def __init__(self, size: Union[int, Tuple[int, ...]]) -> None:
        super().__init__()
        self.size = size

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(self.size)

    def __repr__(self) -> str:
        return f'View(size={self.size})'

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
    final_tanh: bool = True,
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
        final_tanh: If True, add a scaled Tanh() activation after the
            last layer to limit outputs to [-5, 5]. (For encoders.)
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
    for i in range(n_layers):
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

    # Add final tanh activation, if desired
    # This function is approximately linear from -1.5 to 1.5, but makes sure
    # all values are inside [-3, 3], which is want we want for sampling z.
    if final_tanh:
        layers += [ScaledTanh(3.0, 3.0)]

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

    Returns:
        A `nn.Sequential` container with the desired CNN.
    """

    # Define layers: Start with input layer
    layers = [
        nn.Conv1d(
            in_channels=2,
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
    for i in range(n_layers):
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
            out_channels=1,
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
            sent before concatenation. (This seems to be useful for
            SIREN-style `SkipConnectionDecoder`s, which otherwise often
            struggle to learn a good latent space.) If this is set to
            0, `z` is concatenated directly.
    """

    def __init__(self, z: torch.Tensor, layer_size: int = 64) -> None:

        super().__init__()

        self.z = z
        self.layer_size = layer_size

        # If `layer_size` is set to 0, we don't need an extra layer
        if self.layer_size > 0:
            self.layers = nn.Sequential(
                nn.Linear(z.shape[1], self.layer_size),
                nn.LeakyReLU(),
                nn.Linear(self.layer_size, self.layer_size),
            )

    def update_z(self, z: torch.Tensor) -> None:
        self.z = z

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:

        # If we have a layer, use it to transform z
        z = self.layers(self.z) if self.layer_size > 0 else self.z

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

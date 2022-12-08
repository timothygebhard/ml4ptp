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

    Returns:
        A `nn.Sequential` container with the desired MLP.
    """

    # Set up "normal" activation function
    nonlinearity = get_activation(name=activation)

    # Define layers: Start with input layer
    layers = [nn.Linear(input_size, layer_size), nonlinearity]
    if batch_norm:
        layers.append(nn.BatchNorm1d(layer_size))

    # Add hidden layers
    for i in range(n_layers):
        layers += [nn.Linear(layer_size, layer_size), nonlinearity]
        if batch_norm:
            layers.append(nn.BatchNorm1d(layer_size))

    # Add output layer
    layers += [nn.Linear(layer_size, output_size)]

    # Add final tanh activation, if desired
    # This function is approximately linear from -3 to 3, but makes sure that
    # all values are inside [-5, 5], which is want we want for sampling z.
    if final_tanh:
        layers += [ScaledTanh(5.0, 5.0)]

    # Apply special initialization scheme for SIREN networks
    if activation == 'siren':
        for layer in layers:
            if isinstance(layer, nn.Linear):
                n = layer.weight.shape[-1]
                nn.init.uniform_(layer.weight, -np.sqrt(6 / n), np.sqrt(6 / n))

    return nn.Sequential(*layers)


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class Mean(nn.Module):
    """
    Wrap the `.mean()` method into a `nn.Module`.
    """

    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.mean(dim=self.dim)


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

    def __init__(self, w0: float = 1) -> None:
        super().__init__()
        self.w0 = w0

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * tensor)


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

    def __init__(self, a: float = 5.0, b: float = 5.0):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.a * torch.tanh(tensor / self.b)


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

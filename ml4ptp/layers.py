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

    if name == 'leaky_relu':
        return nn.LeakyReLU()
    if name == 'relu':
        return nn.ReLU()
    if name == 'sine' or name == 'siren':
        return Sine()
    if name == 'tanh':
        return nn.Tanh()

    raise ValueError(f'Could not resolve name "{name}"!')


def get_mlp_layers(
    input_size: int,
    n_layers: int,
    layer_size: int,
    output_size: int = 1,
    activation: str = 'leaky_relu',
    final_sigmoid: bool = True,
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
        final_sigmoid: If True, add a sigmoid activation function after
            the last layer to ensure all outputs are in [0, 1].

    Returns:
        A `nn.Sequential` container with the desired MLP.
    """

    # Set up "normal" activation function
    nonlinearity = get_activation(name=activation)

    # Set up the final activation function
    final_nonlinearity = nn.Sigmoid() if final_sigmoid else Identity()

    # Define layers
    layers = [nn.Linear(input_size, layer_size), nonlinearity]
    for i in range(n_layers):
        layers += [nn.Linear(layer_size, layer_size), nonlinearity]
    layers += [nn.Linear(layer_size, output_size), final_nonlinearity]

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

    @staticmethod
    def forward(tensor: torch.Tensor) -> torch.Tensor:
        return tensor


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

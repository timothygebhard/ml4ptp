"""
Useful custom layers for PyTorch.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Tuple, Union

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_mlp_layers(
    input_size: int,
    n_layers: int,
    layer_size: int,
    output_size: int = 1,
) -> nn.Sequential:

    layers = [nn.Linear(input_size, layer_size), nn.LeakyReLU()]
    for i in range(n_layers):
        layers += [nn.Linear(layer_size, layer_size), nn.LeakyReLU()]
    layers += [nn.Linear(layer_size, output_size)]

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

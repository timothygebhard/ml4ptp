"""
Mixins for other classes; e.g., a normalizer for encoders and decoders.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Callable

import torch


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

class NormalizerMixin:
    """
    Mixin for normalizing temperatures.
    """

    T_offset: float
    T_factor: float

    def normalize(self, T: torch.Tensor, undo: bool = False) -> torch.Tensor:
        if not undo:
            return (T - self.T_offset) / self.T_factor
        return T * self.T_factor + self.T_offset


class InitializeEncoderWeights:
    """
    Initialize the weights of the encoder.

    The standard initialization of PyTorch does not seem to work well
    for our purposes (it often produces encoders that map everything
    into a small environment around 0), so we use this mixin do define
    our own custom initialization.
    """

    children: Callable

    def initialize_weights(self) -> None:
        for child in self.children():
            for layer in child.children():
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.kaiming_uniform_(layer.weight)
                    torch.nn.init.constant_(layer.bias, 0.01)

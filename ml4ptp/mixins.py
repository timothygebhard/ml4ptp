"""
Mixins for other classes; e.g., a normalizer for encoders and decoders.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

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

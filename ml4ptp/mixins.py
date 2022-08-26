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

    T_mean: float
    T_std: float

    def normalize(self, T: torch.Tensor, undo: bool = False) -> torch.Tensor:
        if not undo:
            return (T - self.T_mean) / self.T_std
        return T * self.T_std + self.T_mean

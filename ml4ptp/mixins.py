"""
Mixins for other classes; e.g., a normalizer for encoders and decoders.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Any, Dict

import torch


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

class NormalizerMixin:
    """
    Mixin for normalizing (log)-pressures and temperatures.
    """

    normalization: Dict[str, Any]

    def normalize_T(
        self,
        T: torch.Tensor,
        undo: bool = False,
    ) -> torch.Tensor:

        offset = float(self.normalization['T_offset'])
        factor = float(self.normalization['T_factor'])
        assert factor != 0.0, 'factor must not be zero!'

        return T * factor + offset if undo else (T - offset) / factor

    def normalize_log_P(
        self,
        log_P: torch.Tensor,
        undo: bool = False,
    ) -> torch.Tensor:

        offset = float(self.normalization['log_P_offset'])
        factor = float(self.normalization['log_P_factor'])
        assert factor != 0.0, 'factor must not be zero!'

        return log_P * factor + offset if undo else (log_P - offset) / factor

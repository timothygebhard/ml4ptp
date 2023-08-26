"""
Customized TensorBoard callback for PyTorch.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Mapping, Optional

from lightning.pytorch import loggers
from lightning.pytorch.utilities import rank_zero_only


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

class CustomTensorBoardLogger(loggers.TensorBoardLogger):
    """
    Customized TensorBoard callback for PyTorch.

    This is a thin wrapper around the original TensorBoardLogger
    callback, except it does not log the `epoch` to TensorBoard.
    """

    @rank_zero_only
    def log_metrics(
        self,
        metrics: Mapping[str, float],
        step: Optional[int] = None,
    ) -> None:

        # Drop the "epoch" from the metrics
        metrics = dict(metrics)
        metrics.pop('epoch', None)

        # Call the original method
        super().log_metrics(metrics, step)

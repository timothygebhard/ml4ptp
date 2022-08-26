"""
Utilities for exporting models.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Union

import numpy as np
import torch


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

class PTProfile:
    """
    Wrapper class for exporting trained (decoder) models which represent
    a (parameterized) pressure-temperature profile.
    """

    def __init__(self, file_path: Path):

        # Load the (decoder) model from the given file path
        self.model = torch.jit.load(file_path)  # type: ignore

        # Make some other properties available
        self.latent_size = self.model.latent_size
        self.T_mean = self.model.T_mean
        self.T_std = self.model.T_std

    def __call__(
        self,
        z: np.ndarray,
        log_P: Union[np.ndarray, float],
    ) -> np.ndarray:

        # Convert log_P to array, if needed
        if isinstance(log_P, float):
            log_P = np.array([log_P])

        # Make sure inputs have the right shape
        if not z.shape == (self.model.latent_size,):
            raise ValueError(f'z must be {self.model.latent_size}D!')
        if log_P.ndim != 1:
            raise ValueError('log_P must be 1D!')

        # Construct inputs to model
        log_P_in = torch.from_numpy(log_P.reshape(-1, 1)).float()
        z_in = torch.from_numpy(np.tile(A=z, reps=(log_P.shape[0], 1))).float()

        # Send through the model
        with torch.no_grad():  # type: ignore
            T = self.model.forward(z=z_in, log_P=log_P_in).numpy()

        return np.asarray(np.atleast_1d(T.squeeze()))


def export_model_with_torchscript(
    model: torch.nn.Module,
    file_path: Path,
) -> None:
    """
    Basic auxiliary function for exporting a model with torchscript.
    """

    script = torch.jit.script(model)
    torch.jit.save(m=script, f=file_path)  # type: ignore

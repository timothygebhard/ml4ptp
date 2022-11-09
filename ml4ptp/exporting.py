"""
Utilities for exporting models.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Optional, Union

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

    def __init__(
        self,
        decoder_file_path: Path,
        flow_file_path: Optional[Path] = None,
    ) -> None:

        # Load the (decoder) model from the given file path
        self.model = torch.jit.load(decoder_file_path)  # type: ignore

        # If a flow file path is given, load the flow
        self.flow = None
        if flow_file_path is not None:
            self.flow = torch.load(flow_file_path)  # type: ignore

        # Make some other properties available
        self.latent_size = self.model.latent_size
        self.T_offset = self.model.T_offset
        self.T_factor = self.model.T_factor

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

        # Apply flow, if needed
        z_in = torch.from_numpy(z).float().unsqueeze(0)
        if self.flow is not None:
            with torch.no_grad():
                for layer in self.flow.flows:
                    z_in, _ = layer(z_in)

        # Construct inputs to model
        log_P_in = torch.from_numpy(log_P.reshape(-1, 1)).float()
        z_in = torch.tile(z_in, (log_P.shape[0], 1))

        # Send through the model
        with torch.no_grad():
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

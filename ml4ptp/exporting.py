"""
Utilities for exporting models.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import torch

from ml4ptp.onnx import ONNXDecoder


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

class PTProfile:
    """
    Wrapper class for loading a decoder model from a (ONNX) file that
    provides an intuitive interface for running the model.
    """

    def __init__(self, path_or_bytes: Union[str, bytes]) -> None:

        # Load the (decoder) model from the given file path
        self.model = ONNXDecoder(path_or_bytes)

        # Get the latent size from the model
        self.latent_size = self.model.session.get_inputs()[0].shape[1]

    def __call__(
        self,
        z: np.ndarray,
        log_P: Union[np.ndarray, float],
    ) -> np.ndarray:

        # Ensure that the input arrays have the correct shape
        z = np.atleast_2d(z)
        log_P = np.atleast_2d(log_P)

        # Run some sanity checks on the shapes
        if not z.shape[1] == self.latent_size:
            raise ValueError(f'z must be {self.latent_size}-dimensional!')
        if not z.shape[0] == log_P.shape[0]:
            raise ValueError('Batch size of z and log_P must match!')

        # Send through the model
        T = self.model(z=z, log_P=log_P)

        return np.asarray(np.atleast_1d(T.squeeze()))


def export_encoder_with_onnx(
    model: torch.nn.Module,
    example_inputs: Dict[str, torch.Tensor],
    file_path: Path,
) -> None:
    """
    Basic auxiliary function for exporting an encoder with ONNX.

    Args:
        model: The (trained) encoder model to export.
        example_inputs: A dictionary with example inputs to the model.
            Keys should be "log_P" and "T", values should be tensors
            of shape `(batch_size, grid_size)`.
        file_path: The file path to which to export the model.
    """

    torch.onnx.export(
        model=model,
        args=(example_inputs, ),
        f=file_path.as_posix(),
        dynamic_axes={
            "log_P": {0: "batch_size"},
            "T": {0: "batch_size"},
            "z": {0: "batch_size"},
        },
        input_names=['log_P', 'T'],
        output_names=['z'],
    )


def export_decoder_with_onnx(
    model: torch.nn.Module,
    example_inputs: Dict[str, torch.Tensor],
    file_path: Path,
) -> None:

    torch.onnx.export(
        model=model,
        args=(example_inputs, ),
        f=file_path.as_posix(),
        dynamic_axes={
            "z": {0: "batch_size"},
            "log_P": {0: "batch_size", 1: "grid_size"},
            "T_pred": {0: "batch_size", 1: "grid_size"},
        },
        input_names=['z', 'log_P'],
        output_names=['T_pred'],
    )


def export_model_with_torchscript(
    model: torch.nn.Module,
    example_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    file_path: Path,
) -> None:
    """
    Basic auxiliary function for exporting a model with torchscript.
    """

    script = torch.jit.trace(  # type: ignore
        func=model,
        example_inputs=example_inputs,
    )
    torch.jit.save(m=script, f=file_path)  # type: ignore

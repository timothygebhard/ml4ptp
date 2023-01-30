"""
Utilities for exporting models.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

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

    Args:
        path_or_bytes: The path to the ONNX file (or the file contents
            loaded using ``onnx.load(file_path).SerializeToString()``)
            that contains the trained decoder model.
        log_P_min: The minimum value of the log of the pressure. If this
            value is set, `T(log_p)` will be set to `T(log_P_min)` for
            all `log_p` < `log_P_min`. If this value is set to None, no
            such clipping will be performed.
        log_P_max: The maximum value of the log of the pressure. If this
            value is set, `T(log_p)` will be set to `T(log_P_max)` for
            all `log_p` > `log_P_max`. If this value is set to None, no
            such clipping will be performed.
        T_min: The minimum value of the predicted temperature. If this
            value is set, all `T` < `T_min` will be set to `T_min`.
            Default: 0 (negative temperature could break the simulator).
        T_max: The maximum value of the predicted temperature. If this
            value is set, all `T` > `T_max` will be set to `T_max`.
            Default: None.
    """

    def __init__(
        self,
        path_or_bytes: Union[str, bytes],
        log_P_min: Optional[float] = None,
        log_P_max: Optional[float] = None,
        T_min: Optional[float] = 0,
        T_max: Optional[float] = None,
    ) -> None:

        # Store constructor arguments
        self.log_P_min = log_P_min
        self.log_P_max = log_P_max
        self.T_min = T_min
        self.T_max = T_max

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

        # Clip the log_P values if necessary
        if self.log_P_min is not None:
            log_P = np.clip(log_P, self.log_P_min, None)
        if self.log_P_max is not None:
            log_P = np.clip(log_P, None, self.log_P_max)

        # Run some sanity checks on the shapes
        if not z.shape[1] == self.latent_size:
            raise ValueError(f'z must be {self.latent_size}-dimensional!')
        if not z.shape[0] == log_P.shape[0]:
            raise ValueError('Batch size of z and log_P must match!')

        # Send through the model
        T = self.model(z=z, log_P=log_P)

        # Clip the temperature values if necessary
        if self.T_min is not None:
            T = np.clip(T, self.T_min, None)
        if self.T_max is not None:
            T = np.clip(T, None, self.T_max)

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

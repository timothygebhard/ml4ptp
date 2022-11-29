"""
Methods for dealing with ONNX models.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Union

import numpy as np
import onnxruntime


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

class ONNXEncoder:

    def __init__(
        self,
        path_or_bytes: Union[str, bytes],
        n_threads: int = 1,
    ) -> None:

        # Define the session options
        sess_options = onnxruntime.SessionOptions()
        sess_options.intra_op_num_threads = n_threads

        # Create a session for the ONNX model
        self.session = onnxruntime.InferenceSession(
            path_or_bytes=path_or_bytes,
            sess_options=sess_options,
        )

    def __call__(self, log_P: np.ndarray, T: np.ndarray) -> np.ndarray:

        # Run the model
        inputs = {
            'log_P': log_P.astype(np.float32),
            'T': T.astype(np.float32),
        }
        outputs = self.session.run(None, inputs)

        return np.asarray(outputs[0])


class ONNXDecoder:
    """
    A thin wrapper around ``onnxruntime.InferenceSession`` that can
    load an ONNX model from a file path or a byte string and provides
    a simple ``__call__`` method that can be used to run the model.
    """

    def __init__(
        self,
        path_or_bytes: Union[str, bytes],
        n_threads: int = 1,
    ) -> None:

        # Define the session options
        sess_options = onnxruntime.SessionOptions()
        sess_options.intra_op_num_threads = n_threads

        # Create a session for the ONNX model
        self.session = onnxruntime.InferenceSession(
            path_or_bytes=path_or_bytes,
            sess_options=sess_options,
        )

    def __call__(self, z: np.ndarray, log_P: np.ndarray) -> np.ndarray:

        # Run the model
        inputs = {
            'z': z.astype(np.float32),
            'log_P': log_P.astype(np.float32),
        }
        outputs = self.session.run(None, inputs)

        return np.asarray(outputs[0])

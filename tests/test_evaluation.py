"""
Unit tests for evaluation.py.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

from ml4ptp.evaluation import find_optimal_z_with_ultranest
from ml4ptp.exporting import (
    export_encoder_with_onnx,
    export_decoder_with_onnx,
)
from ml4ptp.onnx import ONNXEncoder, ONNXDecoder


# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------

@pytest.fixture
def dummy_encoder_bytes(tmp_path: Path) -> bytes:

    class DummyEncoder(torch.nn.Module):

        # noinspection PyMethodMayBeStatic
        def forward(
            self, log_P: torch.Tensor, T: torch.Tensor
        ) -> torch.Tensor:

            return (
                2.5 * torch.ones(log_P.shape[0])
                + torch.nn.functional.relu(-log_P).sum()  # Dummy computation
                + torch.nn.functional.relu(-T).sum()  # Dummy computation
            ).reshape(-1, 1)

    encoder = DummyEncoder()

    file_path = tmp_path / 'encoder.onnx'
    export_encoder_with_onnx(
        model=encoder,
        file_path=file_path,
        example_inputs=dict(log_P=torch.rand(3, 7), T=torch.rand(3, 7)),
    )

    encoder_bytes = onnx.load(file_path.as_posix()).SerializeToString()

    return bytes(encoder_bytes)


@pytest.fixture
def dummy_decoder_bytes(tmp_path: Path) -> bytes:

    class DummyDecoder(torch.nn.Module):

        # noinspection PyMethodMayBeStatic
        def forward(
            self, z: torch.Tensor, log_P: torch.Tensor
        ) -> torch.Tensor:
            return log_P + z.mean(dim=1, keepdim=True)

    decoder = DummyDecoder()

    file_path = tmp_path / 'decoder.onnx'
    export_decoder_with_onnx(
        model=decoder,
        file_path=file_path,
        example_inputs=dict(z=torch.rand(3, 1), log_P=torch.rand(3, 7)),
    )

    decoder_bytes = onnx.load(file_path.as_posix()).SerializeToString()

    return bytes(decoder_bytes)


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

def test__find_optimal_z_with_ultranest(
    dummy_encoder_bytes: bytes,
    dummy_decoder_bytes: bytes,
) -> None:

    # Load encoder from byte string
    encoder = ONNXEncoder(dummy_encoder_bytes)
    z = encoder(log_P=np.ones((3, 7)), T=np.ones((3, 7)))
    assert z.shape == (3, 1)
    assert np.allclose(z, 2.5 * np.ones((3, )))

    # Load decoder from byte string
    decoder = ONNXDecoder(dummy_decoder_bytes)
    T = decoder(log_P=np.ones((3, 7)), z=np.zeros((3, 1)))
    assert T.shape == (3, 7)
    assert np.allclose(T, np.ones((3, 7)))

    # -------------------------------------------------------------------------
    # Case 1: 1D latent variable, uniform prior
    # -------------------------------------------------------------------------

    log_P = np.ones((1, 7))
    T_true = log_P + 3.14159 * np.ones((1, 7))

    result = find_optimal_z_with_ultranest(
        log_P=log_P,
        T_true=T_true,
        idx=0,
        encoder_bytes=dummy_encoder_bytes,
        decoder_bytes=dummy_decoder_bytes,
        random_seed=0,
        n_live_points=100,
        n_max_calls=1000,
        prior='uniform',
        limit=5.0,
    )

    assert result.success
    assert np.allclose(result.z_initial, 2.5 * np.ones((1, 1)))
    assert np.allclose(result.z_refined, 3.14 * np.ones((1, 1)), atol=1e-2)
    assert np.allclose(result.T_pred_refined, T_true, atol=1e-2)
    assert np.isclose(result.mse_initial, 0.41163772809999977)
    assert np.isclose(result.mse_refined, 9.92954956941911e-07)

    # -------------------------------------------------------------------------
    # Case 2: 1D latent variable, truncated Gaussian prior
    # -------------------------------------------------------------------------

    log_P = np.ones((1, 7))
    T_true = log_P + 1.234 * np.ones((1, 7))

    result = find_optimal_z_with_ultranest(
        log_P=log_P,
        T_true=T_true,
        idx=0,
        encoder_bytes=dummy_encoder_bytes,
        decoder_bytes=dummy_decoder_bytes,
        random_seed=0,
        n_live_points=100,
        n_max_calls=1000,
        prior='gaussian',
        limit=4.0,
    )

    assert result.success
    assert np.allclose(result.z_initial, 2.5 * np.ones((1, 1)))
    assert np.allclose(result.z_refined, 1.234 * np.ones((1, 1)), atol=1e-2)
    assert np.allclose(result.T_pred_refined, T_true, atol=1e-2)
    assert np.isclose(result.mse_initial, 1.6027559999999998)
    assert np.isclose(result.mse_refined, 2.319126553229717e-07)

    # -------------------------------------------------------------------------
    # Case 3: illegal prior
    # -------------------------------------------------------------------------

    with pytest.raises(ValueError) as value_error:
        find_optimal_z_with_ultranest(
            log_P=log_P,
            T_true=T_true,
            idx=0,
            encoder_bytes=dummy_encoder_bytes,
            decoder_bytes=dummy_decoder_bytes,
            prior='illegal',
            random_seed=0,
        )
    assert 'Invalid prior' in str(value_error)

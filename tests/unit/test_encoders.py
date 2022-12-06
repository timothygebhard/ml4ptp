"""
Unit tests for encoders.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Any, Dict

import numpy as np
import pytest
import torch

from ml4ptp.encoders import CNPEncoder, MLPEncoder, ModifiedMLPEncoder


# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------

@pytest.fixture()
def normalization() -> Dict[str, Any]:
    return dict(
        normalization='whiten',
        T_offset=0,
        T_factor=1,
        log_P_offset=0,
        log_P_factor=1,
    )


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

def test__mlp_encoder(normalization: Dict[str, Any]) -> None:

    torch.manual_seed(42)

    # Case 1: batch_norm = False
    encoder = MLPEncoder(
        input_size=29,
        latent_size=5,
        layer_size=16,
        n_layers=2,
        normalization=normalization,
        batch_norm=False,
    )
    log_P = torch.randn(17, 29)
    T = torch.randn(17, 29)
    z = encoder(log_P=log_P, T=T)

    assert z.shape == (17, 5)
    assert np.isclose(z.mean().item(), 0.04472377523779869)

    # Case 2: batch_norm = True
    encoder = MLPEncoder(
        input_size=29,
        latent_size=5,
        layer_size=16,
        n_layers=2,
        normalization=normalization,
        batch_norm=True,
    )
    log_P = torch.randn(17, 29)
    T = torch.randn(17, 29)
    z = encoder(log_P=log_P, T=T)

    assert z.shape == (17, 5)
    assert np.isclose(z.mean().item(), 0.022653179243206978)


def test__modified_mlp_encoder(normalization: Dict[str, Any]) -> None:

    torch.manual_seed(42)

    # Case 1: batch_norm = False
    encoder = ModifiedMLPEncoder(
        input_size=29,
        latent_size=5,
        layer_size=16,
        n_layers=2,
        normalization=normalization,
        batch_norm=False,
    )
    log_P = torch.randn(17, 29)
    T = torch.randn(17, 29)
    z = encoder(log_P=log_P, T=T)

    assert z.shape == (17, 5)
    assert np.isclose(z.mean().item(), -0.021120455116033554)

    # Case 2: batch_norm = True
    encoder = ModifiedMLPEncoder(
        input_size=29,
        latent_size=5,
        layer_size=16,
        n_layers=2,
        normalization=normalization,
        batch_norm=False,
    )
    log_P = torch.randn(17, 29)
    T = torch.randn(17, 29)
    z = encoder(log_P=log_P, T=T)

    assert z.shape == (17, 5)
    assert np.isclose(z.mean().item(), 4.957984492648393e-05)


def test__cnp_encoder(normalization: Dict[str, Any]) -> None:

    torch.manual_seed(42)

    # Case 1: batch_norm = False
    encoder = CNPEncoder(
        latent_size=5,
        layer_size=16,
        n_layers=2,
        normalization=normalization,
    )
    log_P = torch.randn(17, 29)
    T = torch.randn(17, 29)
    z = encoder(log_P=log_P, T=T)

    assert z.shape == (17, 5)
    assert np.isclose(z.mean().item(), 0.00246062851510942)

    # Case 2: batch_norm = True
    encoder = CNPEncoder(
        latent_size=5,
        layer_size=16,
        n_layers=2,
        normalization=normalization,
    )
    log_P = torch.randn(17, 29)
    T = torch.randn(17, 29)
    z = encoder(log_P=log_P, T=T)

    assert z.shape == (17, 5)
    assert np.isclose(z.mean().item(), 0.09216536581516266)

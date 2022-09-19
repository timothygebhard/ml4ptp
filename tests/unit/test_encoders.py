"""
Unit tests for encoders.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Tuple

import numpy as np
import pytest
import torch

from ml4ptp.encoders import MLPEncoder, CNPEncoder


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

@pytest.fixture()
def data() -> Tuple[torch.Tensor, torch.Tensor]:

    torch.manual_seed(42)
    log_P = torch.randn(17, 29)
    T = torch.randn(17, 29)
    return log_P, T


def test__mlp_encoder(data: Tuple[torch.Tensor, torch.Tensor]) -> None:

    torch.manual_seed(42)
    encoder = MLPEncoder(
        input_size=29,
        latent_size=5,
        layer_size=16,
        n_layers=2,
        T_offset=1,
        T_factor=2,
    )

    # Case 1
    log_P, T = data
    output = encoder.forward(log_P=log_P, T=T)
    assert output.shape == (17, 5)
    assert np.isclose(output.mean().item(), 0.0004023064102511853)


def test__cnp_encoder(data: Tuple[torch.Tensor, torch.Tensor]) -> None:

    torch.manual_seed(42)

    # Case 1
    encoder = CNPEncoder(
        latent_size=5,
        layer_size=16,
        n_layers=2,
        T_offset=1,
        T_factor=2,
    )
    log_P, T = data
    output = encoder.forward(log_P=log_P, T=T)
    assert output.shape == (17, 5)
    assert np.isclose(output.mean().item(), 0.0027798849623650312)

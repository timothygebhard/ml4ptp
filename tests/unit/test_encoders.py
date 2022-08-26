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

from ml4ptp.encoders import Encoder


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

@pytest.fixture()
def data() -> Tuple[torch.Tensor, torch.Tensor]:

    torch.manual_seed(42)
    log_P = torch.randn(17, 29)
    T = torch.randn(17, 29)
    return log_P, T


def test__encoder(data: Tuple[torch.Tensor, torch.Tensor]) -> None:

    torch.manual_seed(42)
    encoder = Encoder(latent_size=5, layer_size=16, T_mean=1, T_std=2)

    # Case 1
    log_P, T = data
    output = encoder.forward(log_P=log_P, T=T)
    assert output.shape == (17, 5)
    assert np.isclose(output.mean().item(), 0.013009665533900261)

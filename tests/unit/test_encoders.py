"""
Unit tests for encoders.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import torch

from ml4ptp.encoders import Encoder


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

def test__encoder() -> None:

    torch.manual_seed(42)

    encoder = Encoder(latent_size=5, layer_size=16, T_mean=0, T_std=1)

    # Case 1
    log_P = torch.randn(17, 29)
    T = torch.randn(17, 29)
    output = encoder.forward(log_P=log_P, T=T)
    assert output.shape == (17, 5)
    assert np.isclose(output.mean().item(), 0.016820348799228668)

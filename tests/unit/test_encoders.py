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

    encoder = Encoder(latent_size=5, layer_size=16, T_mean=1, T_std=2)

    # Case 1
    log_P = torch.randn(17, 29)
    T = torch.randn(17, 29)
    output = encoder.forward(log_P=log_P, T=T)
    assert output.shape == (17, 5)
    assert np.isclose(output.mean().item(), 0.013212296180427074)

    # Case 2
    assert torch.equal(
        encoder.normalize(T=torch.Tensor([3., 5., 7.]), undo=False),
        torch.Tensor([1., 2., 3.]),
    )
    assert torch.equal(
        encoder.normalize(T=torch.Tensor([1., 2., 3.]), undo=True),
        torch.Tensor([3., 5., 7.]),
    )

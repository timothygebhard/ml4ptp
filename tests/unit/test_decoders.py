"""
Unit tests for decoders.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import torch

from ml4ptp.decoders import Decoder


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

def test__decoder() -> None:

    torch.manual_seed(42)

    decoder = Decoder(latent_size=5, layer_size=16, T_mean=1, T_std=2)

    # Case 1
    z = torch.randn(17, 5)
    log_P = torch.randn(17, 19)
    output = decoder.forward(z=z, log_P=log_P)
    assert output.shape == (17, 19)
    assert np.isclose(output.mean().item(), 1.7334630489349365)

    # Case 2
    assert torch.equal(
        decoder.normalize(T=torch.Tensor([3., 5., 7.]), undo=False),
        torch.Tensor([1., 2., 3.]),
    )
    assert torch.equal(
        decoder.normalize(T=torch.Tensor([1., 2., 3.]), undo=True),
        torch.Tensor([3., 5., 7.]),
    )

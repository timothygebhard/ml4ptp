"""
Unit tests for decoders.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import torch

from ml4ptp.decoders import Decoder, HyperSIREN


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

def test__decoder() -> None:

    torch.manual_seed(42)

    # Case 1
    decoder = Decoder(
        latent_size=5,
        layer_size=16,
        n_layers=2,
        T_offset=1,
        T_factor=2,
        activation='leaky_relu',
    )
    z = torch.randn(17, 5)
    log_P = torch.randn(17, 19)
    output = decoder.forward(z=z, log_P=log_P)
    assert len(decoder.layers) == 2 + 2 + 2 + 1  # type: ignore
    assert output.shape == (17, 19)
    assert np.isclose(output.mean().item(), 1.7334630489349365)

    # Case 2
    decoder = Decoder(
        latent_size=5,
        layer_size=16,
        n_layers=2,
        T_offset=1,
        T_factor=2,
        activation='siren',
    )
    z = torch.randn(17, 5)
    log_P = torch.randn(17, 19)
    output = decoder.forward(z=z, log_P=log_P)
    assert output.shape == (17, 19)
    assert np.isclose(output.mean().item(), 0.29760393500328064)


def test__hypersiren() -> None:
   
    # Case 1
    model = HyperSIREN(
        latent_size=2,
        layer_size=16,
        n_layers=2,
        T_offset=0,
        T_factor=1,
        activation='siren',
    )

    z = torch.randn(17, 2)
    log_P = torch.randn(17, 39)
    T_pred = model(z=z, log_P=log_P)
    assert T_pred.shape == log_P.shape

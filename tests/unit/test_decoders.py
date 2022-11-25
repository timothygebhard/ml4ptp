"""
Unit tests for decoders.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import torch

from ml4ptp.decoders import Decoder, SkipConnectionsDecoder, HypernetDecoder


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


def test__skip_connections_decoder() -> None:

    # Case 1
    decoder = SkipConnectionsDecoder(
        latent_size=5,
        layer_size=16,
        n_layers=2,
        T_offset=1,
        T_factor=2,
        activation='leaky_relu',
    )
    z = torch.randn(17, 5)
    log_P = torch.randn(17, 19)
    T_pred = decoder(z=z, log_P=log_P)
    assert T_pred.shape == log_P.shape


def test__hypernet_decoder() -> None:

    # Case 1
    model = HypernetDecoder(
        T_offset=0,
        T_factor=1,
        latent_size=2,
        hypernet_layer_size=16,
        decoder_layer_size=16,
        hypernet_n_layers=3,
        decoder_n_layers=3,
        hypernet_activation='leaky_relu',
        decoder_activation='siren',
    )

    z = torch.randn(17, 2)
    log_P = torch.randn(17, 39)
    T_pred = model(z=z, log_P=log_P)
    assert T_pred.shape == log_P.shape

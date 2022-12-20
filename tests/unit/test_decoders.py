"""
Unit tests for decoders.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Any, Dict

import numpy as np
import pytest
import torch

from ml4ptp.decoders import Decoder, SkipConnectionsDecoder, HypernetDecoder


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

def test__decoder(normalization: Dict[str, Any]) -> None:

    torch.manual_seed(42)

    # Case 1: Leaky ReLU activation
    decoder = Decoder(
        latent_size=5,
        layer_size=16,
        n_layers=2,
        normalization=normalization,
        activation='leaky_relu',
        batch_norm=False,
    )
    z = torch.randn(17, 5)
    log_P = torch.randn(17, 19)
    T_pred = decoder(z=z, log_P=log_P)

    assert len(decoder.layers) == 2 + 2 + 2 + 1
    assert T_pred.shape == (17, 19)
    assert np.isclose(T_pred.mean().item(), 0.3667314648628235)

    # Case 2: Siren activation
    decoder = Decoder(
        latent_size=5,
        layer_size=16,
        n_layers=2,
        normalization=normalization,
        activation='siren',
        batch_norm=False,
    )
    z = torch.randn(17, 5)
    log_P = torch.randn(17, 19)
    T_pred = decoder(z=z, log_P=log_P)

    assert T_pred.shape == (17, 19)
    assert np.isclose(T_pred.mean().item(), -0.3511980473995209)

    # Case 3: batch_norm = True
    decoder = Decoder(
        latent_size=5,
        layer_size=16,
        n_layers=2,
        normalization=normalization,
        activation='leaky_relu',
        batch_norm=True,
    )
    z = torch.randn(17, 5)
    log_P = torch.randn(17, 19)
    T_pred = decoder(z=z, log_P=log_P)

    assert len(decoder.layers) == 3 + 3 + 3 + 1
    assert T_pred.shape == (17, 19)
    assert np.isclose(T_pred.mean().item(), -0.17229308187961578)


def test__skip_connections_decoder(normalization: Dict[str, Any]) -> None:

    torch.manual_seed(42)

    # Case 1: batch_norm = False
    decoder = SkipConnectionsDecoder(
        latent_size=5,
        layer_size=16,
        n_layers=2,
        normalization=normalization,
        activation='leaky_relu',
        batch_norm=False,
    )
    z = torch.randn(17, 5)
    log_P = torch.randn(17, 19)
    T_pred = decoder(z=z, log_P=log_P)

    assert T_pred.shape == log_P.shape
    assert np.isclose(T_pred.mean().item(), -0.27519676089286804)

    # Case 2: batch_norm = True
    decoder = SkipConnectionsDecoder(
        latent_size=5,
        layer_size=16,
        n_layers=2,
        normalization=normalization,
        activation='leaky_relu',
        batch_norm=True,
    )
    z = torch.randn(17, 5)
    log_P = torch.randn(17, 19)
    T_pred = decoder(z=z, log_P=log_P)

    assert T_pred.shape == log_P.shape
    assert np.isclose(T_pred.mean().item(), 0.06192253530025482)

    # Case 3: SIREN activation
    decoder = SkipConnectionsDecoder(
        latent_size=5,
        layer_size=16,
        n_layers=2,
        normalization=normalization,
        activation='siren',
        batch_norm=False,
    )
    z = torch.randn(17, 5)
    log_P = torch.randn(17, 19)
    T_pred = decoder(z=z, log_P=log_P)

    assert T_pred.shape == log_P.shape
    assert np.isclose(T_pred.mean().item(), 0.3681025207042694)


def test__hypernet_decoder(normalization: Dict[str, Any]) -> None:

    torch.manual_seed(42)

    # Case 1: batch_norm = False
    model = HypernetDecoder(
        normalization=normalization,
        latent_size=2,
        hypernet_layer_size=16,
        decoder_layer_size=16,
        hypernet_n_layers=3,
        decoder_n_layers=3,
        hypernet_activation='leaky_relu',
        decoder_activation='siren',
        batch_norm=False,
    )

    z = torch.randn(17, 2)
    log_P = torch.randn(17, 39)
    T_pred = model(z=z, log_P=log_P)

    assert T_pred.shape == log_P.shape
    assert np.isclose(T_pred.mean().item(), -0.21794116497039795)

    # Case 2: batch_norm = True
    model = HypernetDecoder(
        normalization=normalization,
        latent_size=2,
        hypernet_layer_size=16,
        decoder_layer_size=16,
        hypernet_n_layers=3,
        decoder_n_layers=3,
        hypernet_activation='leaky_relu',
        decoder_activation='siren',
        batch_norm=True,
    )

    z = torch.randn(17, 2)
    log_P = torch.randn(17, 39)
    T_pred = model(z=z, log_P=log_P)

    assert T_pred.shape == log_P.shape
    assert np.isclose(T_pred.mean().item(), 0.29131171107292175)

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

from ml4ptp.encoders import (
    CNPEncoder,
    ConvolutionalEncoder,
    MLPEncoder,
    ModifiedMLPEncoder,
)


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
        dropout=0.0,
    )
    log_P = torch.randn(17, 29)
    T = torch.randn(17, 29)
    z = encoder(log_P=log_P, T=T)

    assert z.shape == (17, 5)
    assert np.isclose(z.mean().item(), 0.054241057485342026)

    # Case 2: batch_norm = True
    encoder = MLPEncoder(
        input_size=29,
        latent_size=5,
        layer_size=16,
        n_layers=2,
        normalization=normalization,
        batch_norm=True,
        dropout=0.0,
    )
    log_P = torch.randn(17, 29)
    T = torch.randn(17, 29)
    z = encoder(log_P=log_P, T=T)

    assert z.shape == (17, 5)
    assert np.isclose(z.mean().item(), 0.009999962523579597)


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
        dropout=0.0,
    )
    log_P = torch.randn(17, 29)
    T = torch.randn(17, 29)
    z = encoder(log_P=log_P, T=T)

    assert sum(_.numel() for _ in encoder.layers_1.parameters()) == 8577
    assert z.shape == (17, 5)
    assert np.isclose(z.mean().item(), -0.02112540230154991)

    # Case 2: batch_norm = True
    encoder = ModifiedMLPEncoder(
        input_size=29,
        latent_size=5,
        layer_size=16,
        n_layers=2,
        normalization=normalization,
        batch_norm=False,
        dropout=0.0,
    )
    log_P = torch.randn(17, 29)
    T = torch.randn(17, 29)
    z = encoder(log_P=log_P, T=T)

    assert z.shape == (17, 5)
    assert np.isclose(z.mean().item(), -0.07541168481111526)


def test__convolutional_encoder(normalization: Dict[str, Any]) -> None:

    torch.manual_seed(42)

    # Case 1: batch_norm = False
    encoder = ConvolutionalEncoder(
        input_size=29,
        latent_size=5,
        cnn_n_layers=2,
        cnn_n_channels=64,
        cnn_kernel_size=1,
        mlp_n_layers=3,
        mlp_layer_size=256,
        normalization=normalization,
        batch_norm=False,
        dropout=0.0,
    )
    log_P = torch.randn(17, 29)
    T = torch.randn(17, 29)
    z = encoder(log_P=log_P, T=T)

    assert sum(_.numel() for _ in encoder.convnet.parameters()) == 8577
    assert z.shape == (17, 5)
    assert np.isclose(z.mean().item(), -1.084885835647583)

    # Case 2: batch_norm = True
    encoder = ConvolutionalEncoder(
        input_size=29,
        latent_size=5,
        cnn_n_layers=4,
        cnn_n_channels=16,
        cnn_kernel_size=1,
        mlp_n_layers=3,
        mlp_layer_size=256,
        normalization=normalization,
        batch_norm=True,
        dropout=0.0,
    )
    log_P = torch.randn(17, 29)
    T = torch.randn(17, 29)
    z = encoder(log_P=log_P, T=T)

    assert z.shape == (17, 5)
    assert np.isclose(z.mean().item(), 0.01000002957880497)

    # Case 3: bigger convolutional network
    encoder = ConvolutionalEncoder(
        input_size=29,
        latent_size=5,
        cnn_n_layers=4,
        cnn_n_channels=512,
        cnn_kernel_size=1,
        mlp_n_layers=1,
        mlp_layer_size=256,
        normalization=normalization,
        batch_norm=False,
        dropout=0.0,
    )

    assert sum(_.numel() for _ in encoder.convnet.parameters()) == 1_052_673


def test__cnp_encoder(normalization: Dict[str, Any]) -> None:

    torch.manual_seed(42)

    # Case 1: batch_norm = False
    encoder = CNPEncoder(
        latent_size=5,
        layer_size=16,
        n_layers=2,
        normalization=normalization,
        batch_norm=False,
        dropout=0.0,
    )
    log_P = torch.randn(17, 29)
    T = torch.randn(17, 29)
    z = encoder(log_P=log_P, T=T)

    assert z.shape == (17, 5)
    assert np.isclose(z.mean().item(), 0.002398358192294836)

    # Case 2: batch_norm = True
    encoder = CNPEncoder(
        latent_size=5,
        layer_size=16,
        n_layers=2,
        normalization=normalization,
        batch_norm=True,
        dropout=0.0,
    )
    log_P = torch.randn(17, 29)
    T = torch.randn(17, 29)
    z = encoder(log_P=log_P, T=T)

    assert z.shape == (17, 5)
    assert np.isclose(z.mean().item(), 0.10353609919548035)

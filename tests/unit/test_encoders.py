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

from ml4ptp.encoders import ConvolutionalEncoder, MLPEncoder, CNPEncoder


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

@pytest.fixture()
def data() -> Tuple[torch.Tensor, torch.Tensor]:

    torch.manual_seed(42)
    log_P = torch.randn(17, 29)
    T = torch.randn(17, 29)
    return log_P, T


def test__convolutional_encoder(
    data: Tuple[torch.Tensor, torch.Tensor]
) -> None:

    torch.manual_seed(42)
    encoder = ConvolutionalEncoder(
        latent_size=5, layer_size=16, T_mean=1, T_std=2
    )

    # Case 1
    log_P, T = data
    output = encoder.forward(log_P=log_P, T=T)
    assert output.shape == (17, 5)
    assert np.isclose(output.mean().item(), 0.013009665533900261)


def test__mlp_encoder(data: Tuple[torch.Tensor, torch.Tensor]) -> None:

    torch.manual_seed(42)
    encoder = MLPEncoder(
        input_size=29,
        latent_size=5,
        layer_size=16,
        n_layers=2,
        T_mean=1,
        T_std=2,
    )

    # Case 1
    log_P, T = data
    output = encoder.forward(log_P=log_P, T=T)
    assert output.shape == (17, 5)
    assert np.isclose(output.mean().item(), 0.0008047096198424697)


def test__cnp_encoder(data: Tuple[torch.Tensor, torch.Tensor]) -> None:

    torch.manual_seed(42)

    # Case 1
    encoder = CNPEncoder(
        latent_size=5,
        layer_size=16,
        n_layers=2,
        T_mean=1,
        T_std=2,
        deterministic=True,
    )
    log_P, T = data
    output = encoder.forward(log_P=log_P, T=T)
    assert output.shape == (17, 5)
    assert np.isclose(output.mean().item(), 0.00554615119472146)

    # Case 2
    encoder = CNPEncoder(
        latent_size=5,
        layer_size=16,
        n_layers=2,
        T_mean=1,
        T_std=2,
        deterministic=False,
    )
    encoder.train()
    log_P, T = data
    output = encoder.forward(log_P=log_P, T=T)
    assert output.shape == (17, 5)
    assert np.isclose(output.mean().item(), 0.12951301038265228)


    # Case 3
    encoder = CNPEncoder(
        latent_size=5,
        layer_size=16,
        n_layers=2,
        T_mean=1,
        T_std=2,
        deterministic=False,
    )
    encoder.eval()
    log_P, T = data
    output = encoder.forward(log_P=log_P, T=T)
    assert output.shape == (17, 5)
    assert np.isclose(output.mean().item(), 0.03222157806158066)

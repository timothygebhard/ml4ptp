"""
Unit tests for exporting.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import numpy as np
import pytest
import torch

from ml4ptp.decoders import Decoder
from ml4ptp.exporting import export_model_with_torchscript, PTProfile


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

@pytest.fixture()
def model() -> torch.nn.Sequential:
    torch.manual_seed(42)
    model = torch.nn.Sequential(
        torch.nn.Linear(in_features=3, out_features=32),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(in_features=32, out_features=1),
    )
    return model


@pytest.fixture()
def decoder() -> torch.nn.Module:
    torch.manual_seed(42)
    model = Decoder(
        latent_size=2, layer_size=32, n_layers=2, T_mean=300, T_std=50
    )
    return model


def test__load_config(model: torch.nn.Sequential, tmp_path: Path) -> None:

    # Case 1: Ensure that model gives same outputs after saving and loading
    x_in = torch.rand(17, 3)
    x_out_1: torch.Tensor = model.forward(x_in)  # type: ignore
    file_path = tmp_path / 'exported_model.pt'
    export_model_with_torchscript(model=model, file_path=file_path)
    loaded_model = torch.jit.load(file_path)  # type: ignore
    x_out_2 = loaded_model.forward(x_in)
    assert torch.equal(x_out_1, x_out_2)


def test__pt_profile(decoder: torch.nn.Module, tmp_path: Path) -> None:

    file_path = tmp_path / 'decoder.pt'
    export_model_with_torchscript(model=decoder, file_path=file_path)

    # Case 1
    pt_profile = PTProfile(file_path=file_path)
    assert pt_profile.T_mean == 300
    assert pt_profile.T_std == 50
    assert pt_profile.latent_size == 2

    z = np.zeros(pt_profile.latent_size)
    log_P = np.linspace(-6, 0, 100)
    T = pt_profile(z=z, log_P=log_P)
    assert T.shape == (100, )
    assert np.isclose(np.mean(T), 302.72156)

    with pytest.raises(ValueError) as value_error:
        pt_profile(z=np.array([0, 0, 0]), log_P=log_P)
    assert 'z must be 2D!' in str(value_error)

    with pytest.raises(ValueError) as value_error:
        pt_profile(z=z, log_P=log_P.reshape((2, 2, 5, 5)))
    assert 'log_P must be 1D!' in str(value_error)

    # Case 2
    T = pt_profile(z=z, log_P=-3.0)
    assert T.shape == (1,)
    assert np.isclose(float(T), 302.4253)

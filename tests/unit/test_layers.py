"""
Unit tests for layers.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import pytest
import torch

from ml4ptp.layers import (
    get_activation,
    get_mlp_layers,
    Identity,
    Mean,
    PrintShape,
    ScaledTanh,
    Sine,
    Squeeze,
    View,
)


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

def test__get_activation() -> None:

    assert isinstance(get_activation('leaky_relu'), torch.nn.LeakyReLU)
    assert isinstance(get_activation('relu'), torch.nn.ReLU)
    assert isinstance(get_activation('sine'), Sine)
    assert isinstance(get_activation('siren'), Sine)
    assert isinstance(get_activation('tanh'), torch.nn.Tanh)

    with pytest.raises(ValueError) as value_error:
        get_activation('illegal')
    assert 'Could not resolve name' in str(value_error)


def test__get_mlp_layers() -> None:

    # Case 1
    layers = get_mlp_layers(
        input_size=2,
        n_layers=0,
        layer_size=2,
        output_size=1,
        activation='leaky_relu',
    )
    assert len(layers) == 3
    assert isinstance(layers[0], torch.nn.Linear)
    assert isinstance(layers[1], torch.nn.LeakyReLU)


def test__identity() -> None:

    identity = Identity()

    # Case 1
    x_in = torch.rand(17, 39)
    x_out = identity(x_in)
    assert torch.equal(x_out, x_in)


def test__mean() -> None:

    x_in = torch.arange(11 * 13).reshape(11, 13).float()

    # Case 1
    mean = Mean(dim=0)
    x_out = mean(x_in)
    assert torch.equal(x_out, torch.arange(65, 78).float())

    # Case 2
    mean = Mean(dim=1)
    x_out = mean(x_in)
    assert torch.equal(x_out, torch.arange(6, 137, 13).float())


def test_print_shape(capfd: pytest.CaptureFixture) -> None:

    print_shape = PrintShape(label='Some layer')

    # Case 1
    x_in = torch.arange(5 * 7).reshape(5, 7).float()
    x_out = print_shape(x_in)
    assert torch.equal(x_out, x_in)

    # Case 2
    out, err = capfd.readouterr()
    assert 'Some layer:  torch.Size([5, 7])' in str(out)


def test_sine() -> None:

    sine = Sine()

    # Case 1
    x_in = torch.linspace(0, 2 * 3.1415926, 100)
    x_out = sine(x_in)
    assert torch.equal(x_out, torch.sin(x_in))


def test_squeeze() -> None:

    squeeze = Squeeze()

    # Case 1
    x_in = torch.arange(42).reshape(1, 1, 42).float()
    x_out = squeeze(x_in)
    assert torch.equal(x_out, torch.arange(42).float())


def test_view() -> None:

    view = View(size=(4, 3))

    # Case 1
    x_in = torch.arange(12).reshape(2, 6).float()
    x_out = view(x_in)
    assert torch.equal(x_out, torch.arange(12).reshape(4, 3).float())

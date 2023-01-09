"""
Unit tests for layers.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import pytest
import torch

from ml4ptp.layers import (
    ConcatenateWithZ,
    get_activation,
    get_cnn_layers,
    get_mlp_layers,
    Identity,
    Mean,
    PrintShape,
    PrintValue,
    ScaledTanh,
    Sine,
    Squeeze,
    View,
)


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

def test__get_activation() -> None:

    assert isinstance(get_activation('elu'), torch.nn.ELU)
    assert isinstance(get_activation('gelu'), torch.nn.GELU)
    assert isinstance(get_activation('leaky_relu'), torch.nn.LeakyReLU)
    assert isinstance(get_activation('mish'), torch.nn.Mish)
    assert isinstance(get_activation('prelu'), torch.nn.PReLU)
    assert isinstance(get_activation('relu'), torch.nn.ReLU)
    assert isinstance(get_activation('sine'), Sine)
    assert isinstance(get_activation('siren'), Sine)
    assert isinstance(get_activation('swish'), torch.nn.SiLU)
    assert isinstance(get_activation('tanh'), torch.nn.Tanh)

    with pytest.raises(ValueError) as value_error:
        get_activation('illegal')
    assert 'Could not resolve name' in str(value_error)


def test__get_mlp_layers() -> None:

    torch.manual_seed(42)

    # Case 1
    layers = get_mlp_layers(
        input_size=2,
        n_layers=0,
        layer_size=2,
        output_size=1,
        activation='leaky_relu',
        batch_norm=False,
        dropout=0.0,
    )
    assert len(layers) == 3
    assert isinstance(layers[0], torch.nn.Linear)
    assert isinstance(layers[1], torch.nn.LeakyReLU)

    # Case 2
    layers = get_mlp_layers(
        input_size=2,
        n_layers=1,
        layer_size=2,
        output_size=1,
        activation='relu',
        batch_norm=False,
        dropout=0.0,
    )
    assert len(layers) == 5
    assert isinstance(layers[0], torch.nn.Linear)
    assert isinstance(layers[1], torch.nn.ReLU)

    # Case 3
    layers = get_mlp_layers(
        input_size=2,
        n_layers=1,
        layer_size=2,
        output_size=1,
        activation='relu',
        batch_norm=False,
        dropout=0.5,
    )
    assert len(layers) == 7


def test__get_cnn_layers() -> None:

    torch.manual_seed(42)

    # Case 1
    layers = get_cnn_layers(
        n_layers=2,
        n_channels=8,
        kernel_size=3,
        activation='leaky_relu',
        batch_norm=False,
    )
    assert len(layers) == 8
    assert isinstance(layers[0], torch.nn.Conv1d)
    assert isinstance(layers[1], torch.nn.LeakyReLU)
    x = torch.randn(17, 2, 101)
    y = layers(x)
    assert y.shape == (17, 101 - (2 + 2) * (3 - 1))
    assert np.isclose(y.mean().item(), 0.1323554664850235)

    # Case 2
    layers = get_cnn_layers(
        n_layers=0,
        n_channels=8,
        kernel_size=5,
        activation='leaky_relu',
        batch_norm=True,
    )
    assert len(layers) == 5
    assert isinstance(layers[0], torch.nn.Conv1d)
    assert isinstance(layers[1], torch.nn.LeakyReLU)
    x = torch.randn(17, 2, 101)
    y = layers(x)
    assert y.shape == (17, 101 - (2 + 0) * (5 - 1))
    assert np.isclose(y.mean().item(), 0.04575807601213455)


def test__concatenate_with_z() -> None:

    # Case 1
    z = torch.randn(5 * 2).reshape(5, 2).float()
    x_in = torch.randn(5 * 7).reshape(5, 7).float()
    concatenate_with_z = ConcatenateWithZ(z=z, layer_size=0)
    x_out = concatenate_with_z(x_in)
    assert torch.equal(concatenate_with_z.z, z)
    assert x_out.shape == (5, 9)
    assert torch.equal(x_out, torch.cat((x_in, z), dim=1))

    # Case 2
    z_new = torch.randn(5 * 3).reshape(5, 3).float()
    concatenate_with_z.update_z(z_new)
    x_out = concatenate_with_z(x_in)
    assert torch.equal(concatenate_with_z.z, z_new)
    assert x_out.shape == (5, 10)
    assert torch.equal(x_out, torch.cat((x_in, z_new), dim=1))

    # Case 3
    z = torch.randn(5 * 2).reshape(5, 2).float()
    x_in = torch.randn(5 * 7).reshape(5, 7).float()
    concatenate_with_z = ConcatenateWithZ(z=z, layer_size=0)
    x_out = concatenate_with_z(x_in)
    assert torch.equal(concatenate_with_z.z, z)
    assert x_out.shape == (5, 9)
    assert torch.equal(x_out[:, :7], x_in)

    # Case 4
    z = torch.randn(5 * 2).reshape(5, 2).float()
    x_in = torch.randn(5 * 7).reshape(5, 7).float()
    concatenate_with_z = ConcatenateWithZ(z=z, layer_size=16)
    x_out = concatenate_with_z(x_in)
    assert torch.equal(concatenate_with_z.z, z)
    assert x_out.shape == (5, 7 + 16)

    # Case 5
    assert (
        repr(concatenate_with_z) ==
        'ConcatenateWithZ(latent_size=2, layer_size=16, out_size=16+in_size)'
    )


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
    assert repr(mean) == 'Mean(dim=0)'
    assert torch.equal(x_out, torch.arange(65, 78).float())

    # Case 2
    mean = Mean(dim=1)
    x_out = mean(x_in)
    assert repr(mean) == 'Mean(dim=1)'
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


def test_print_value(capfd: pytest.CaptureFixture) -> None:

    print_value = PrintValue(label='Some layer')

    # Case 1
    x_in = torch.tensor([1.0, 2.0, 3.0])
    x_out = print_value(x_in)
    assert torch.equal(x_out, x_in)

    # Case 2
    out, err = capfd.readouterr()
    assert 'Some layer:\n tensor([1., 2., 3.])' in str(out)


def test_scaled_tanh() -> None:

    # Case 1
    scaled_tanh = ScaledTanh(a=1.2, b=3.4)
    assert repr(scaled_tanh) == 'ScaledTanh(a=1.2, b=3.4)'

    # Case 2
    scaled_tanh = ScaledTanh(a=5.0, b=5.0)
    assert torch.isclose(
        scaled_tanh(torch.Tensor([0.0])), torch.Tensor([0.0])
    )
    assert torch.isclose(
        scaled_tanh(torch.Tensor([1.0])), torch.Tensor([0.9868766069412231])
    )
    assert torch.isclose(
        scaled_tanh(torch.Tensor([100.0])), torch.Tensor([5.0])
    )

    # Case 3
    scaled_tanh = ScaledTanh(a=3.0, b=1.0)
    assert torch.isclose(
        scaled_tanh(torch.Tensor([0.0])), torch.Tensor([0.0])
    )
    assert torch.isclose(
        scaled_tanh(torch.Tensor([1.0])), torch.Tensor([2.28478247])
    )
    assert torch.isclose(
        scaled_tanh(torch.Tensor([100.0])), torch.Tensor([3.0])
    )


def test_sine() -> None:

    # Case 1
    sine = Sine(w0=3.141)
    assert repr(sine) == 'Sine(w0=3.141)'

    sine = Sine()
    assert repr(sine) == 'Sine(w0=1.0)'

    # Case 2
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

    # Case 1
    view = View(size=(4, 3))
    assert repr(view) == 'View(size=(4, 3))'

    # Case 2
    x_in = torch.arange(12).reshape(2, 6).float()
    x_out = view(x_in)
    assert torch.equal(x_out, torch.arange(12).reshape(4, 3).float())

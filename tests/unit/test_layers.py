"""
Unit tests for layers.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import pytest
import torch

from ml4ptp.layers import Mean, PrintShape, Squeeze, View


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

def test__mean() -> None:

    x_in = torch.arange(11 * 13).reshape(11, 13).float()

    # Case 1
    mean = Mean(dim=0)
    x_out = mean(x_in)
    assert torch.equal(x_out, torch.range(65, 77))

    # Case 2
    mean = Mean(dim=1)
    x_out = mean(x_in)
    assert torch.equal(x_out, torch.range(6, 136, 13))


def test_print_shape(capfd: pytest.CaptureFixture) -> None:

    print_shape = PrintShape(label='Some layer')

    # Case 1
    x_in = torch.arange(5 * 7).reshape(5, 7).float()
    x_out = print_shape(x_in)
    assert torch.equal(x_out, x_in)

    # Case 2
    out, err = capfd.readouterr()
    assert 'Some layer:  torch.Size([5, 7])' in str(out)


def test_squeeze() -> None:

    squeeze = Squeeze()

    # Case 1
    x_in = torch.range(1, 42).reshape(1, 1, 42)
    x_out = squeeze(x_in)
    assert torch.equal(x_out, torch.range(1, 42))


def test_view() -> None:

    view = View(size=(4, 3))

    # Case 1
    x_in = torch.range(1, 12).reshape(2, 6)
    x_out = view(x_in)
    assert torch.equal(x_out, torch.range(1, 12).reshape(4, 3))

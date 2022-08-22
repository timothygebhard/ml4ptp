"""
Unit tests for kernels.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import torch

from ml4ptp.kernels import gaussian_kernel, compute_mmd


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

def test__gaussian_kernel() -> None:

    torch.manual_seed(42)

    # Case 1: Ensure that diagonal of kernel(x, x) is all ones
    x = torch.randn(1000, 1)
    assert torch.all(torch.diagonal(gaussian_kernel(x, x)) == 1)


def test__compute_mmd() -> None:

    torch.manual_seed(42)

    # Case 1
    x = torch.randn(1_000, 1)
    y = torch.randn(1_000, 1)
    assert np.isclose(compute_mmd(x, y).item(), 0.0010222196578979492)

    # Case 2
    x = torch.randn(1_000, 1)
    y = torch.randn(1_000, 1) + 1_000
    assert np.isclose(compute_mmd(x, y).item(), 0.9043338298797607)

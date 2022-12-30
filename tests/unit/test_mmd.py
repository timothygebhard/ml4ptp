"""
Unit tests for mmd.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import torch

from ml4ptp.mmd import compute_mmd


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

def test__compute_mmd() -> None:

    torch.manual_seed(42)

    # Case 1
    x = torch.randn(1000, 2)
    y_1 = torch.randn(1000, 2)
    y_2 = torch.randn(1000, 2) + 100
    mmd_0 = compute_mmd(x, x)
    mmd_1 = compute_mmd(x, y_1)
    mmd_2 = compute_mmd(x, y_2)
    assert torch.isclose(mmd_0, torch.tensor(0.0), atol=1e-3)
    assert mmd_1 >= 0
    assert mmd_2 >= 0
    assert mmd_1 < mmd_2

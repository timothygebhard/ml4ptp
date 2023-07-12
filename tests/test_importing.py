"""
Unit tests for importing.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import os

import torch

from ml4ptp.importing import get_member_by_name


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

def test__get_member_by_name() -> None:

    # Case 1
    assert get_member_by_name('os', 'getcwd') is os.getcwd

    # Case 2
    assert get_member_by_name('torch.optim', 'Adam') is torch.optim.Adam

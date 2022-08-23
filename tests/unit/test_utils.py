"""
Unit tests for utils.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from sys import platform

import torch

from ml4ptp.utils import (
    get_number_of_available_cores,
    setup_rich_progress_bar,
    tensor_to_str
)


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

def test__get_number_of_available_cores() -> None:

    n_cores = get_number_of_available_cores(default=42)

    # Case 1: os.sched_getaffinity() is not available
    if platform in ('darwin', 'win32'):
        assert n_cores == 42

    # Case 2: os.sched_getaffinity() is available
    else:
        assert isinstance(n_cores, int)


def test__setup_rich_progress_bar() -> None:

    progress_bar = setup_rich_progress_bar()
    assert len(progress_bar.columns) == 6


def test__tensor_to_str() -> None:

    # Case 1
    tensor = torch.Tensor([0.123456789, 1.234567890])
    string = tensor_to_str(tensor)
    assert string == '(0.123, 1.235)'

    # Case 2
    tensor = torch.Tensor([3.14159265])
    string = tensor_to_str(tensor, n_digits=1)
    assert string == '(3.1)'

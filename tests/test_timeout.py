"""
Unit tests for timeout.py.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from time import sleep

import pytest

from ml4ptp.timeout import timelimit, TimeoutException


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

def test__timeout_exception() -> None:

    with pytest.raises(TimeoutException) as timeout_exception:
        raise TimeoutException("Timed out!")
    assert "Timed out!" in str(timeout_exception)


def test__timelimit() -> None:

    # Case 1
    flag = False
    assert not flag
    with timelimit(seconds=1):
        sleep(0.5)
        flag = True
    assert flag

    # Case 2
    flag = False
    assert not flag
    with pytest.raises(TimeoutException) as timeout_exception:
        with timelimit(seconds=1):
            sleep(1.5)
            flag = True
    assert not flag
    assert "Timed out!" in str(timeout_exception)

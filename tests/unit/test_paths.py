"""
Unit tests for paths.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import pytest

from ml4ptp.paths import expandvars


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

def test__expandvars(monkeypatch: pytest.MonkeyPatch) -> None:

    # Set up an environmental variables
    monkeypatch.setenv('SOME_DIR', 'test')

    # Case 1 (expected use case)
    path = Path('/this/is/a/$SOME_DIR/path')
    assert isinstance(expandvars(path), Path)
    assert expandvars(path).as_posix() == '/this/is/a/test/path'

    # Case 2 (environmental variable does not exist)
    path = Path('/this/is/a/$DOES_NOT_EXIST/path')
    assert expandvars(path).as_posix() == '/this/is/a/$DOES_NOT_EXIST/path'

    # Case 3 (no environmental variables)
    path = Path('/this/is/a/path')
    assert expandvars(path).as_posix() == '/this/is/a/path'

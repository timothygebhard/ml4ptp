"""
Unit tests for paths.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import pytest

from ml4ptp.paths import expandvars, get_datasets_dir, get_scripts_dir


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


def test__get_datasets_dir(monkeypatch: pytest.MonkeyPatch) -> None:

    # Case 1
    monkeypatch.setenv("ML4PTP_DATASETS_DIR", ".")
    assert isinstance(get_datasets_dir(), Path)
    assert get_datasets_dir().as_posix() == "."

    # Case 2
    monkeypatch.delenv("ML4PTP_DATASETS_DIR", raising=False)
    with pytest.raises(FileNotFoundError) as file_not_found_error:
        get_datasets_dir()
    assert 'The datasets directory does not exist' in str(file_not_found_error)


def test__get_scripts_dir() -> None:

    assert isinstance(get_scripts_dir(), Path)
    assert get_scripts_dir().name == 'scripts'

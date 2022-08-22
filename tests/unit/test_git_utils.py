"""
Unit tests for git_utils.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

from git import Repo

from ml4ptp.git_utils import (
    document_git_status,
    get_diff,
    get_git_hash,
    get_repo,
    is_dirty,
)


# -----------------------------------------------------------------------------
# UNIT TESTS
# -----------------------------------------------------------------------------

def test__get_repo() -> None:

    # Case 1: Check that function runs without errors and returns a Repo
    assert isinstance(get_repo(), Repo)


def test__get_diff() -> None:

    # Case 1: Check that function runs without errors and returns a string
    assert isinstance(get_diff(), str)


def test__get_git_hash() -> None:

    # Case 1: Check that function runs without errors and returns a string
    # that has the correct length to be a git hash
    assert isinstance(get_git_hash(), str)
    assert len(get_git_hash()) == 40


def test__document_git_status(tmp_path: Path) -> None:

    # Case 1: Check that we can document the git status to a temporary
    # directory and get a file with the correct name if the repo is dirty
    document_git_status(target_dir=tmp_path, verbose=False)

    hash_file_path = tmp_path / 'git-hash.txt'
    assert hash_file_path.exists()
    assert hash_file_path.is_file()

    diff_file_path = tmp_path / 'git-diff.txt'
    if is_dirty():
        assert diff_file_path.exists()
        assert diff_file_path.is_file()
    else:
        assert not diff_file_path.exists()

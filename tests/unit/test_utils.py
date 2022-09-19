"""
Unit tests for utils.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from sys import platform

import torch

from ml4ptp.utils import (
    find_run_dirs_with_results,
    get_device_from_model,
    get_number_of_available_cores,
    get_run_dir,
    setup_rich_progress_bar,
    tensor_to_str
)


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

def test__find_run_dirs_with_results(tmp_path: Path) -> None:

    runs_dir = tmp_path / 'runs'
    runs_dir.mkdir(exist_ok=True)

    for i in range(10):
        run_dir = runs_dir / f'run_{i}'
        run_dir.mkdir(exist_ok=True)
        if i in (1, 3, 7):
            file_path = run_dir / 'results_on_test_set.hdf'
            file_path.touch()

    run_dirs = find_run_dirs_with_results(tmp_path)
    assert len(run_dirs) == 3
    assert run_dirs[0].as_posix().endswith('1')
    assert run_dirs[1].as_posix().endswith('3')
    assert run_dirs[2].as_posix().endswith('7')


def test__get_device_from_model() -> None:

    # Case 1: Create model on CPU
    model = torch.nn.Sequential(torch.nn.Linear(1, 1))
    assert get_device_from_model(model) == torch.device('cpu')


def test__get_number_of_available_cores() -> None:

    n_cores = get_number_of_available_cores(default=42)

    # Case 1: os.sched_getaffinity() is not available
    if platform in ('darwin', 'win32'):
        assert n_cores == 42

    # Case 2: os.sched_getaffinity() is available
    else:
        assert isinstance(n_cores, int)


def test__get_run_dir(tmp_path: Path) -> None:

    # Case 1: runs_dir does not (yet) exist
    run_dir = get_run_dir(experiment_dir=tmp_path)
    assert run_dir.parent == tmp_path / 'runs'
    assert run_dir == tmp_path / 'runs' / 'run_0'

    # Case 2: runs_dir does already exist
    run_dir = get_run_dir(experiment_dir=tmp_path)
    assert run_dir == tmp_path / 'runs' / 'run_1'


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

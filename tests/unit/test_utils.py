"""
Unit tests for utils.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from sys import platform

import numpy as np
import torch

from ml4ptp.utils import (
    find_run_dirs_with_results,
    get_batch_idx,
    get_device_from_model,
    get_number_of_available_cores,
    get_run_dir,
    setup_rich_progress_bar,
    tensor_to_str,
    weighted_mse_loss,
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


def test__get_batch_idx() -> None:

    # Case 1
    array = np.array([0, 1, 2, 3, 4, 5])
    idx = get_batch_idx(array, 2, shuffle=False)
    assert isinstance(idx, list)
    assert len(idx) == 3
    assert np.array_equal(idx[0], array[0:2])
    assert np.array_equal(idx[1], array[2:4])
    assert np.array_equal(idx[2], array[4:6])

    # Case 2
    array = np.array([0, 1, 2, 3, 4])
    idx = get_batch_idx(array, 3, shuffle=False)
    assert len(idx) == 2
    assert np.array_equal(idx[0], array[0:3])
    assert np.array_equal(idx[1], array[3:])

    # Case 3
    array = np.array([0, 1, 2, 3, 4])
    idx = get_batch_idx(array, 3, shuffle=True)
    assert len(idx) == 2
    assert np.array_equal(idx[0], np.array([4, 2, 3]))
    assert np.array_equal(idx[1], np.array([1, 0]))


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


def test__weighted_mse_loss() -> None:

    # Case 1
    y_true = torch.randn(17, 43)
    y_pred = torch.randn(17, 43)
    weights = torch.ones(17, 43) / 43
    assert torch.allclose(
        weighted_mse_loss(y_true, y_pred, weights),
        torch.nn.functional.mse_loss(y_true, y_pred),
    )

    # Case 2
    y_true = torch.Tensor([[1, 2, 3, 4]])
    y_pred = torch.Tensor([[1, 2, 3, 4]])
    weights = torch.Tensor([[0.25, 0.25, 0.25, 0.25]])
    loss = weighted_mse_loss(y_true, y_pred, weights)
    assert loss == 0

    # Case 3
    y_true = torch.Tensor([[0, 0, 0, 0]])
    y_pred = torch.Tensor([[1, 2, 3, 4]])
    weights = torch.Tensor([[0.1, 0.2, 0.3, 0.4]])
    loss = weighted_mse_loss(y_true, y_pred, weights)
    assert loss == 10

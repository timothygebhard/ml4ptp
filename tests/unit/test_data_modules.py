"""
Unit tests for data_modules.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from ml4ptp.data_modules import DataModule


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

@pytest.fixture()
def hdf_file(tmp_path: Path) -> Path:
    """
    Fixture to create dummy HDF file.
    """

    file_path = tmp_path / 'hdf_file.hdf'
    with h5py.File(file_path, 'w') as hdf_file:
        hdf_file.create_dataset(
            name='P',
            data=np.arange(13 * 17).reshape(13, 17),
        )
        hdf_file.create_dataset(
            name='T',
            data=np.arange(13 * 17).reshape(13, 17) + 1,
        )
        hdf_file.create_dataset(
            name='weights',
            data=2 * np.eye(13, 17),
        )
    return file_path


def test__data_module(hdf_file: Path) -> None:
    """
    Test DataModule.
    """

    # Type hints
    log_P: torch.Tensor
    T: torch.Tensor
    weights: torch.Tensor

    # -------------------------------------------------------------------------
    # Case 1: Check error messages
    # -------------------------------------------------------------------------

    dm = DataModule(
        key_P='P',
        key_T='T',
        train_file_path=None,
        test_file_path=None,
        train_size=11,
        val_size=3,
        train_batch_size=1,
        val_batch_size=3,
        normalization='whiten',
        random_state=42,
    )

    with pytest.raises(RuntimeError) as runtime_error:
        dm.train_dataloader()
    assert "No train_dataset defined!" in str(runtime_error)

    with pytest.raises(RuntimeError) as runtime_error:
        dm.val_dataloader()
    assert "No val_dataset defined!" in str(runtime_error)

    with pytest.raises(RuntimeError) as runtime_error:
        dm.test_dataloader()
    assert "No test_dataset defined!" in str(runtime_error)

    with pytest.raises(NotImplementedError):
        dm.predict_dataloader()

    with pytest.raises(RuntimeError) as runtime_error:
        dm.get_normalization()
    assert 'Normalization constants have not yet' in str(runtime_error)

    # -------------------------------------------------------------------------
    # Case 2: whiten normalization + equal weights
    # -------------------------------------------------------------------------

    dm = DataModule(
        key_P='P',
        key_T='T',
        key_weights='weights',
        weighting_scheme='equal',
        train_file_path=hdf_file,
        test_file_path=hdf_file,
        train_size=11,
        val_size=3,
        train_batch_size=1,
        val_batch_size=3,
        test_batch_size=1,
        normalization='whiten',
        random_state=42,
    )
    dm.prepare_data()

    assert np.isclose(dm.get_normalization()['T_offset'], 96.125)
    assert np.isclose(dm.get_normalization()['T_factor'], 50.306236267089844)
    assert len(dm.train_dataloader()) == 8
    assert len(dm.val_dataloader()) == 1

    log_P, T, weights = next(iter(dm.train_dataloader()))  # type: ignore
    assert log_P.shape == (1, 17)
    assert T.shape == (1, 17)
    assert weights.shape == (1, 17)

    log_P, T, weights = next(iter(dm.val_dataloader()))
    assert log_P.shape == (3, 17)
    assert T.shape == (3, 17)
    assert weights.shape == (3, 17)

    log_P, T, weights = next(iter(dm.test_dataloader()))
    assert log_P.shape == (1, 17)
    assert T.shape == (1, 17)
    assert weights.shape == (1, 17)

    assert len(torch.unique(weights)) == 1

    # -------------------------------------------------------------------------
    # Case 3: minmax normalization + linear weights
    # -------------------------------------------------------------------------

    dm = DataModule(
        key_P='P',
        key_T='T',
        key_weights=None,
        weighting_scheme='linear',
        train_file_path=hdf_file,
        test_file_path=hdf_file,
        train_size=10,
        val_size=2,
        train_batch_size=2,
        val_batch_size=2,
        test_batch_size=2,
        normalization='minmax',
        random_state=42,
    )
    dm.prepare_data()

    assert np.isclose(dm.get_normalization()['T_offset'], 1.0)
    assert np.isclose(dm.get_normalization()['T_factor'], 169.0)

    assert len(dm.train_dataloader()) == 4
    assert len(dm.val_dataloader()) == 1
    assert len(dm.test_dataloader()) == 7

    log_P, T, weights = next(iter(dm.train_dataloader()))  # type: ignore
    assert log_P.shape == (2, 17)
    assert T.shape == (2, 17)
    assert weights.shape == (2, 17)

    assert torch.equal(weights[0], weights[1])

    # -------------------------------------------------------------------------
    # Case 4: contribution function
    # -------------------------------------------------------------------------

    dm = DataModule(
        key_P='P',
        key_T='T',
        key_weights='weights',
        weighting_scheme='contribution_function',
        train_file_path=hdf_file,
        test_file_path=None,
        train_size=10,
        val_size=2,
        train_batch_size=8,
        val_batch_size=2,
        test_batch_size=13,
    )
    dm.prepare_data()

    log_P, T, weights = next(iter(dm.train_dataloader()))  # type: ignore
    assert torch.allclose(weights.sum(dim=1), torch.ones(8))

    # -------------------------------------------------------------------------
    # Case 5: get_test_data()
    # -------------------------------------------------------------------------

    with pytest.raises(RuntimeError) as runtime_error:
        dm.get_test_data()
    assert 'No test_dataset defined!' in str(runtime_error)

    dm.test_file_path = hdf_file
    dm.prepare_data()

    test_log_P, test_T, test_weights = dm.get_test_data()
    assert isinstance(test_log_P, np.ndarray)
    assert isinstance(test_T, np.ndarray)
    assert isinstance(test_weights, np.ndarray)
    assert test_log_P.shape == (13, 17)
    assert test_T.shape == (13, 17)
    assert test_weights.shape == (13, 17)

    # -------------------------------------------------------------------------
    # Case 6: get_weights()
    # -------------------------------------------------------------------------

    raw_weights = torch.randn(13, 17)
    dm.weighting_scheme = 'illegal'
    with pytest.raises(ValueError) as value_error:
        dm.get_weights(raw_weights=raw_weights)
    assert 'Invalid weighting scheme!' in str(value_error)

    dm.weighting_scheme = 'equal'
    weights = dm.get_weights(raw_weights=raw_weights)
    assert torch.equal(weights, torch.ones_like(raw_weights) / 17)

    dm.weighting_scheme = 'linear'
    weights = dm.get_weights(raw_weights=raw_weights)
    assert torch.allclose(weights.sum(dim=1), torch.ones(13))

    dm.weighting_scheme = 'contribution_function'
    raw_weights = 3.141 * torch.eye(13, 17)
    weights = dm.get_weights(raw_weights=raw_weights)
    assert torch.allclose(weights.sum(dim=1), torch.ones(13))
    assert torch.allclose(weights, raw_weights / 3.141)

    # -------------------------------------------------------------------------
    # Case 7: compute_normalization()
    # -------------------------------------------------------------------------

    log_P = torch.arange(1, 8).float()
    T = torch.arange(1, 8).float()

    dm.normalization = 'illegal'
    with pytest.raises(ValueError) as value_error:
        dm.compute_normalization(log_P=log_P, T=T)
    assert 'Invalid normalization!' in str(value_error)

    dm.normalization = 'minmax'
    dm.compute_normalization(log_P=log_P, T=T)
    normalization = dm.get_normalization()
    assert np.isclose(normalization['log_P_offset'], 1.0)
    assert np.isclose(normalization['log_P_factor'], 6.0)
    assert np.isclose(normalization['T_offset'], 1.0)
    assert np.isclose(normalization['T_factor'], 6.0)

    dm.normalization = 'whiten'
    dm.compute_normalization(log_P=log_P, T=T)
    normalization = dm.get_normalization()
    assert np.isclose(normalization['log_P_offset'], 4.0)
    assert np.isclose(normalization['log_P_factor'], 2.1602468490600586)
    assert np.isclose(normalization['T_offset'], 4.0)
    assert np.isclose(normalization['T_factor'], 2.1602468490600586)

    # -------------------------------------------------------------------------
    # Case 8: __repr__()
    # -------------------------------------------------------------------------

    table = dm.__repr__()
    assert len(table.split('\n')) == 9

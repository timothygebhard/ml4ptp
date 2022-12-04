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
            data=np.arange(17 * 13).reshape(17, 13),
        )
        hdf_file.create_dataset(
            name='T',
            data=np.arange(17 * 13).reshape(17, 13) + 1,
        )
    return file_path


def test__data_module(hdf_file: Path) -> None:
    """
    Test DataModule.
    """

    # Case 1
    dm = DataModule(
        key_P='P',
        key_T='T',
        train_file_path=hdf_file,
        test_file_path=None,
        train_size=11,
        val_size=3,
        train_batch_size=1,
        val_batch_size=3,
        normalization='whiten'
    )
    dm.prepare_data()

    assert np.isclose(dm.T_offset, 72)
    assert np.isclose(dm.T_factor, 41.42462921142578)
    assert len(dm.train_dataloader()) == 8
    assert len(dm.val_dataloader()) == 1

    with pytest.raises(RuntimeError) as runtime_error:
        dm.test_dataloader()
    assert "No test_dataset defined!" in str(runtime_error)

    # Case 2
    dm = DataModule(
        key_P='P',
        key_T='T',
        train_file_path=None,
        test_file_path=hdf_file,
        test_batch_size=2,
        normalization='whiten'
    )
    dm.prepare_data()

    assert len(dm.test_dataloader()) == 9

    with pytest.raises(RuntimeError) as runtime_error:
        dm.train_dataloader()
    assert "No train_dataset defined!" in str(runtime_error)

    with pytest.raises(RuntimeError) as runtime_error:
        dm.val_dataloader()
    assert "No val_dataset defined!" in str(runtime_error)

    with pytest.raises(NotImplementedError):
        dm.predict_dataloader()

    # Case 3
    dm = DataModule(
        key_P='P',
        key_T='T',
        train_file_path=hdf_file,
        test_file_path=None,
        train_size=11,
        val_size=3,
        train_batch_size=1,
        normalization='minmax'
    )
    dm.prepare_data()

    assert np.isclose(dm.T_offset, 1)
    assert np.isclose(dm.T_factor, 11 * 13 - 1)

    # Case 4
    with pytest.raises(ValueError) as value_error:
        dm = DataModule(
            key_P='P',
            key_T='T',
            train_file_path=hdf_file,
            test_file_path=None,
            train_size=11,
            val_size=3,
            train_batch_size=1,
            normalization='illegal'
        )
        dm.prepare_data()
    assert 'Invalid normalization!' in str(value_error)

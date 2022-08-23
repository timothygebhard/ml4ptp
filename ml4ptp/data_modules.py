"""
Data module(s) that encapsulate the data handling for PyTorch Lightning.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Optional, Union

from pytorch_lightning.utilities.types import (
    EVAL_DATALOADERS,
    TRAIN_DATALOADERS,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import h5py
import numpy as np
import pytorch_lightning as pl
import torch

from ml4ptp.paths import expandvars


# -----------------------------------------------------------------------------
# DATA MODULES
# -----------------------------------------------------------------------------

class DataModule(pl.LightningDataModule):
    """
    Wrap data loading in a PL-compatible way.

    This class handles reading the data sets from HDF files, casting
    them to tensors, splitting into training / validation, and creating
    the required `Dataset` and `DataLoader` instances.

    The DataLoaders return a 2-tuple consisting of:

        - log_P: A tensor with the log10 of the pressure values in bar.
        - T: A tensor with the temperature values in Kelvin.
    """

    def __init__(
        self,
        key_P: str,
        key_T: str,
        train_file_path: Optional[Path],
        test_file_path: Optional[Path],
        train_size: int = 10_000,
        val_size: Union[float, int] = 0.1,
        batch_size: int = 1_024,
        num_workers: int = 4,
        persistent_workers: bool = True,
        random_state: int = 42,
    ) -> None:

        super().__init__()  # type: ignore

        # Store constructor arguments
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.key_P = key_P
        self.key_T = key_T
        self.train_size = train_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.random_state = random_state

        # Initialize variables that will hold the normalization constants
        self.T_mean: float = np.nan
        self.T_std: float = np.nan

        # Initialize variables that will hold the data sets
        self.train_dataset: Optional[TensorDataset] = None
        self.val_dataset: Optional[TensorDataset] = None
        self.test_dataset: Optional[TensorDataset] = None

    def prepare_data(self) -> None:
        """
        Prepare data. This function is called within a single process on
        CPU, and is meant to be used, for example, for downloading data,
        or tokenizing it.

        Note: There exists also the possibility to define as `setup()`
        method which is run on every GPU indepedently. However, since
        we do not plan to train on multiple GPUs for now, everything is
        placed in the `prepare_data()` method instead.
        """

        # Load the training data
        if self.train_file_path is not None:

            # Read data from HDF file
            file_path = expandvars(self.train_file_path).resolve()
            with h5py.File(file_path, "r") as hdf_file:
                P = torch.as_tensor(hdf_file[self.key_P]).float()
                P = P[:self.train_size]
                log_P = torch.log10(P)
                T = torch.as_tensor(hdf_file[self.key_T]).float()
                T = T[:self.train_size]

            # Split the data into training and validation
            train_log_P, val_log_P, train_T, val_T = train_test_split(
                log_P,
                T,
                test_size=self.val_size,
                random_state=self.random_state,
            )

            # Compute the normalization for T from training data
            self.T_mean = float(torch.mean(T))
            self.T_std = float(torch.std(T))

            # Create data sets for training and validation
            self.train_dataset = TensorDataset(train_log_P, train_T)
            self.val_dataset = TensorDataset(val_log_P, val_T)

        # Load the test data
        if self.test_file_path is not None:

            # Read data from HDF file
            file_path = expandvars(self.test_file_path).resolve()
            with h5py.File(file_path, "r") as hdf_file:
                test_P = torch.as_tensor(hdf_file[self.key_P]).float()
                test_P = test_P[:self.train_size]
                test_log_P = torch.log10(test_P)
                test_T = torch.as_tensor(hdf_file[self.key_T]).float()
                test_T = test_T[:self.train_size]

            # Create data sets for testing
            self.test_dataset = TensorDataset(test_log_P, test_T)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """
        Returns: The train data loader.
        """

        if self.train_dataset is None:
            raise RuntimeError("No train_dataset defined!")

        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """
        Returns: The validation data loader.
        """

        if self.val_dataset is None:
            raise RuntimeError("No valid_dataset defined!")

        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """
        Returns: The test data loader.
        """

        if self.test_dataset is None:
            raise RuntimeError("No test_dataset defined!")

        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError()

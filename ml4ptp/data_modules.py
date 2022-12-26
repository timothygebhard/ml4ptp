"""
Data module(s) that encapsulate the data handling for PyTorch Lightning.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Dict, Optional, Union, Tuple

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
        train_batch_size: int = 1_024,
        val_batch_size: int = 1_024,
        test_batch_size: int = 1_024,
        num_workers: int = 4,
        persistent_workers: bool = True,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = True,
        normalization: str = 'whiten',
        random_state: int = 42,
    ) -> None:

        super().__init__()

        # Store constructor arguments
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.key_P = key_P
        self.key_T = key_T
        self.train_size = train_size
        self.val_size = val_size
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.normalization = normalization
        self.random_state = random_state

        # Initialize variables that will hold the normalization constants.
        # for the temperature. The `offset` is either the mean (for whitening)
        # or the minimum (for minmax); the `factor` is either the standard
        # deviation or the maximum minus the minimum.
        self.normalization_dict: Dict[str, float] = dict(
            T_offset=np.nan,
            T_factor=np.nan,
            log_P_offset=np.nan,
            log_P_factor=np.nan,
        )

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
            file_path = expandvars(Path(self.train_file_path)).resolve()
            with h5py.File(file_path, "r") as hdf_file:
                P = torch.from_numpy(np.array(hdf_file[self.key_P]))
                P = P[:self.train_size].float()
                log_P = torch.log10(P)
                T = torch.from_numpy(np.array(hdf_file[self.key_T]))
                T = T[:self.train_size].float()

            # Split the data into training and validation
            train_log_P, val_log_P, train_T, val_T = train_test_split(
                log_P,
                T,
                test_size=self.val_size,
                random_state=self.random_state,
            )

            # Compute normalization constants
            self.compute_normalization(log_P=train_log_P, T=train_T)

            # Create data sets for training and validation
            self.train_dataset = TensorDataset(train_log_P, train_T)
            self.val_dataset = TensorDataset(val_log_P, val_T)

        # Load the test data
        if self.test_file_path is not None:

            # Read data from HDF file
            file_path = expandvars(Path(self.test_file_path)).resolve()
            with h5py.File(file_path, "r") as hdf_file:
                test_P = torch.from_numpy(np.array(hdf_file[self.key_P]))
                test_P = test_P[:self.train_size].float()
                test_log_P = torch.log10(test_P)
                test_T = torch.from_numpy(np.array(hdf_file[self.key_T]))
                test_T = test_T[:self.train_size].float()

            # Create data sets for testing
            self.test_dataset = TensorDataset(test_log_P, test_T)

    def compute_normalization(
        self,
        log_P: torch.Tensor,
        T: torch.Tensor,
    ) -> None:
        """
        Compute normalization constants and update `normalization_dict`.
        """

        # Compute the normalization from the given training data
        if self.normalization == 'whiten':
            T_offset = float(torch.mean(T))
            T_factor = float(torch.std(T))
            log_P_offset = float(torch.mean(log_P))
            log_P_factor = float(torch.std(log_P))
        elif self.normalization == 'minmax':
            T_offset = float(torch.min(T))
            T_factor = float(torch.max(T) - torch.min(T))
            log_P_offset = float(torch.min(log_P))
            log_P_factor = float(torch.max(log_P) - torch.min(log_P))
        else:
            raise ValueError('Invalid normalization!')

        # Store the normalization constants
        self.normalization_dict['T_offset'] = T_offset
        self.normalization_dict['T_factor'] = T_factor
        self.normalization_dict['log_P_offset'] = log_P_offset
        self.normalization_dict['log_P_factor'] = log_P_factor

    def get_normalization(self) -> Dict[str, float]:
        """
        Return the normalization constants (after checking that they
        have been computed already).
        """

        # Check if normalization constants have been computed
        if any(np.isnan(_) for _ in self.normalization_dict.values()):
            raise RuntimeError(
                'Normalization constants have not yet been computed!'
            )

        return self.normalization_dict

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """
        Returns: The train data loader.
        """

        if self.train_dataset is None:
            raise RuntimeError("No train_dataset defined!")

        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """
        Returns: The validation data loader.
        """

        if self.val_dataset is None:
            raise RuntimeError("No val_dataset defined!")

        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """
        Returns: The test data loader.
        """

        if self.test_dataset is None:
            raise RuntimeError("No test_dataset defined!")

        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError()

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Auxiliary function to get the test data as numpy arrays.
        """

        if self.test_dataset is None:
            raise RuntimeError("No test_dataset defined!")

        log_P, T_true = self.test_dataset.tensors

        return log_P.numpy(), T_true.numpy()

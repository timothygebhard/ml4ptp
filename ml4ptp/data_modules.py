"""
Data module(s) that encapsulate the data handling for PyTorch Lightning.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Dict, Optional, Union, Tuple

from lightning.pytorch.utilities.types import (
    EVAL_DATALOADERS,
    TRAIN_DATALOADERS,
)
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import h5py
import numpy as np
import lightning.pytorch as pl
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

    Args:
        key_P: The key of the pressure data in the HDF file.
        key_T: The key of the temperature data in the HDF file.
        key_weights: The key of the weights data in the HDF file.
            May be `None` if no weights are available (default).
        weighting_scheme: Which scheme to use for weighting the data.
            Currently, the following options are supported:
            - 'equal': All data points have the same weight (default).
            - 'linear': Arbitrary linear weighting scheme that gives
                more weight to the higher pressure data points.
        train_file_path: The path to the HDF file containing the
            training data.
        test_file_path: The path to the HDF file containing the
            test data.
        train_size: The number of training data points to use. This is
            the sum of the number of training and validation samples.
        val_size: The number of validation data points to use. If this
            is a float, it is interpreted as the fraction of the total
            number of training data points to use for validation.
        train_batch_size: The batch size to use for training.
        val_batch_size: The batch size to use for validation.
        test_batch_size: The batch size to use for testing.
        num_workers: The number of workers to use for loading the data.
        persistent_workers: Whether to use persistent workers for the
            data loaders.
        shuffle: Whether to shuffle the data.
        pin_memory: Whether to pin the data in memory.
        drop_last: Whether to drop the last batch if it is incomplete.
        normalization: Which normalization scheme to use for the
            temperature data. Currently the following options are
            supported:
            - 'whiten': Subtract the mean and divide by the standard
                deviation ("standardization"). This is the default.
            - 'minmax': Subtract the minimum and divide by the maximum
                minus the minimum.
        random_state: The random seed to use for splitting the data.

    Returns:
        The DataLoaders return a 3-tuple consisting of:

        - log_P: A tensor with the log10 of the pressure values in bar.
        - T: A tensor with the temperature values in Kelvin.
        - weights: A tensor with the weights for each data point (to
            compute a weighted loss).
    """

    def __init__(
        self,
        key_P: str,
        key_T: str,
        key_weights: Optional[str] = None,
        weighting_scheme: str = 'equal',
        train_file_path: Optional[Path] = None,
        test_file_path: Optional[Path] = None,
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
        self.key_weights = key_weights
        self.weighting_scheme = weighting_scheme
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

                # Select pressure and compute log10
                P = torch.from_numpy(np.array(hdf_file[self.key_P]))
                P = P[:self.train_size].float()
                log_P = torch.log10(P)

                # Select temperature
                T = torch.from_numpy(np.array(hdf_file[self.key_T]))
                T = T[:self.train_size].float()

                # Select (raw) weights.
                # These are not normalized yet, and depending on the weighting
                # scheme, they may be replaced (e.g., by equal weights).
                if self.key_weights is not None:
                    raw_weights = torch.from_numpy(
                        np.array(hdf_file[self.key_weights])
                    )
                    raw_weights = raw_weights[:self.train_size].float()
                else:
                    raw_weights = torch.empty_like(T)

            # Split the data into training and validation
            (
                train_log_P,
                val_log_P,
                train_T,
                val_T,
                train_raw_weights,
                val_raw_weights,
            ) = train_test_split(
                log_P,
                T,
                raw_weights,
                test_size=self.val_size,
                random_state=self.random_state,
            )

            # Create weights for the training and validation data
            train_weights = self.get_weights(train_raw_weights)
            val_weights = self.get_weights(val_raw_weights)

            # Compute normalization constants
            self.compute_normalization(log_P=train_log_P, T=train_T)

            # Create data sets for training and validation
            self.train_dataset = TensorDataset(
                train_log_P, train_T, train_weights
            )
            self.val_dataset = TensorDataset(
                val_log_P, val_T, val_weights
            )

        # Load the test data
        if self.test_file_path is not None:

            # Read data from HDF file
            file_path = expandvars(Path(self.test_file_path)).resolve()
            with h5py.File(file_path, "r") as hdf_file:

                # Select pressure and compute log10
                test_P = torch.from_numpy(np.array(hdf_file[self.key_P]))
                test_P = test_P.float()
                test_log_P = torch.log10(test_P)

                # Select temperature
                test_T = torch.from_numpy(np.array(hdf_file[self.key_T]))
                test_T = test_T.float()

                # Select (raw) weights
                if self.key_weights is not None:
                    test_raw_weights = torch.from_numpy(
                        np.array(hdf_file[self.key_weights])
                    )
                    test_raw_weights = test_raw_weights.float()
                else:
                    test_raw_weights = torch.empty_like(test_T)

            # Create weights for the test data
            test_weights = self.get_weights(test_raw_weights)

            # Create data sets for testing
            self.test_dataset = TensorDataset(test_log_P, test_T, test_weights)

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

    def get_weights(self, raw_weights: torch.Tensor) -> torch.Tensor:
        """
        Return the weights for the given raw weights.

        Depending on the weighting scheme, the raw weights may be
        replaced by equal weights, or an arbitrary linear scheme.
        In any case, the weights are normalized to sum up to one.
        """

        # Option 1 (default): All data points have the same weight
        if self.weighting_scheme == 'equal':
            weights = torch.ones_like(raw_weights, device=raw_weights.device)

        # Option 2: Arbitrary linear weighting
        elif self.weighting_scheme == 'linear':
            weights = torch.tile(
                input=torch.linspace(
                    0.001, 1.0, raw_weights.shape[1], device=raw_weights.device
                ),
                dims=(raw_weights.shape[0], 1),
            )

        # Option 3: Weighting based on contribution function (i.e., the raw
        # weights that we loaded from the HDF file)
        elif self.weighting_scheme == 'contribution_function':
            weights = raw_weights

        # Invalid weighting scheme
        else:
            raise ValueError('Invalid weighting scheme!')

        # Make sure weights are normalized to sum up to 1 for each profile
        weights = weights / torch.sum(weights, dim=1, keepdim=True)

        return weights

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

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Auxiliary function to get the test data as numpy arrays.
        """

        if self.test_dataset is None:
            raise RuntimeError("No test_dataset defined!")

        log_P, T_true, weights = self.test_dataset.tensors

        return log_P.numpy(), T_true.numpy(), weights.numpy()

    def __repr__(self) -> str:
        """
        Return a string representation of the DataModule.
        """

        # Collect attributes of training dataset
        if self.train_dataset:
            n_train_samples = len(self.train_dataset)
            n_train_batches = len(self.train_dataloader())
        else:  # pragma: no cover
            n_train_samples = 0
            n_train_batches = 0

        # Collect attributes of validation dataset
        if self.val_dataset:
            n_val_samples = len(self.val_dataset)
            n_val_batches = len(self.val_dataloader())
        else:  # pragma: no cover
            n_val_samples = 0
            n_val_batches = 0

        # Collect attributes of test dataset
        if self.test_dataset:
            n_test_samples = len(self.test_dataset)
            n_test_batches = len(self.test_dataloader())
        else:  # pragma: no cover
            n_test_samples = 0
            n_test_batches = 0

        # Create new table object
        table = Table()
        table.add_column('')
        table.add_column('training', justify='right')
        table.add_column('validation', justify='right')
        table.add_column('test', justify='right')

        # Add rows
        table.add_row(
            "size",
            f'{self.train_size:,}',
            f'{self.val_size:,}',
            '---',
        )
        table.add_row(
            "batch_size",
            f'{self.train_batch_size:,}',
            f'{self.val_batch_size:,}',
            f'{self.test_batch_size:,}',
        )
        table.add_row(
            "n_samples",
            f'{n_train_samples:,}',
            f'{n_val_samples:,}',
            f'{n_test_samples:,}',
        )
        table.add_row(
            "n_batches",
            f'{n_train_batches:,}',
            f'{n_val_batches:,}',
            f'{n_test_batches:,}',
        )

        # Create a silent console to render the table
        console = Console(width=80, record=True)
        with console.capture() as capture:
            console.print(table)

        # Return the rendered table
        return capture.get()

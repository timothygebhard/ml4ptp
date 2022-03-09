"""
Data modules that encapsulate the data handling for PyTorch Lightning.
Note: Currently, there is only one module that handles the access for
all data sets.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Optional

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
import torch

from ml4ptp.data_loading import load_temperatures, load_normalization


# -----------------------------------------------------------------------------
# DATA MODULES
# -----------------------------------------------------------------------------

class DataModule(pl.LightningDataModule):
    """
    This class wraps the access to the training / validation dataset.

    Args:
        name: Name of the data set (e.g., "pyatmos").
        train_size: Number of profiles to use for the training and
            validation set. This is the total number, which is split
            according to the given `val_fraction`.
        test_size: Number of profiles to use for the test set.
        val_fraction: Fraction of profiles used for validation.
        batch_size: Batch size for training and validation.
        num_workers: Number of workers for DataLoader instances.
        random_state: Random seed for `train_test_split()` which is
            used for splitting the training and validation data.
    """

    def __init__(
        self,
        name: str,
        train_size: int = 10_000,
        test_size: int = 1_000,
        val_fraction: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 4,
        random_state: int = 42,
    ) -> None:

        super().__init__()

        # Store constructor arguments
        self.name = name
        self.train_size = train_size
        self.test_size = test_size
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_state = random_state

        # Initialize variables that will hold the normalization constants
        self.mean = None
        self.std = None

        # Initialize variables that will hold the data sets
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Prepare the data (e.g., splitting into training / validation /
        test and applying the normalization transforms).

        (The main motivation for putting this into its own method seems
        to be what happens when you train on multiple GPUs...)

        Args:
            stage: Either "fit", "validate", or "test".
        """

        # Set default value for stage
        if stage is None or stage in ('fit', 'validate'):
            stage = 'train'

        # Load the normalization values
        train_mean, train_std = load_normalization(name=self.name)

        # Either create train / validation data set...
        if stage == 'train':
    
            # Load temperatures and normalize
            temperatures = load_temperatures(
                name=self.name, stage=stage, size=self.train_size,
            )
            temperatures -= train_mean
            temperatures /= train_std

            # Split into training and validation
            t_train, t_val = train_test_split(
                temperatures,
                test_size=self.val_fraction,
                random_state=self.random_state,
            )

            # Create datasets
            self.dataset_train = TensorDataset(torch.Tensor(t_train))
            self.dataset_val = TensorDataset(torch.Tensor(t_val))

        # ...or test data set
        elif stage == 'test':

            # Load temperatures and normalize
            temperatures = load_temperatures(
                name=self.name, stage=stage, size=self.test_size,
            )
            temperatures -= train_mean
            temperatures /= train_std

            # Create dataset
            self.dataset_test = TensorDataset(torch.Tensor(temperatures))

    def train_dataloader(self) -> DataLoader:
        """
        Returns: The train data loader.
        """

        return DataLoader(
            dataset=self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns: The validation data loader.
        """

        return DataLoader(
            dataset=self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns: The test data loader.
        """

        return DataLoader(
            dataset=self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def predict_dataloader(self) -> DataLoader:
        """
        Returns: The predict data loader (= test data loader).
        """

        return DataLoader(
            dataset=self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

"""
Utility functions for evaluation on the test set.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Tuple

from rich.progress import track
from pytorch_lightning.utilities.types import EVAL_DATALOADERS

import numpy as np
import torch

from ml4ptp.models import Model
from ml4ptp.optimization import optimize_z_with_lbfgs


# -----------------------------------------------------------------------------
# DEFINTIONS
# -----------------------------------------------------------------------------

def evaluate_on_test_set(
    model: Model,
    test_dataloader: EVAL_DATALOADERS,
    device: torch.device = torch.device('cpu'),
    n_epochs: int = 100,
    batch_size: int = 128,
    history_size: int = 10,
) -> Tuple[np.ndarray, ...]:

    # Make sure model is in evaluation mode
    model = model.eval()

    # Keep track of batch-wise results
    z_initial_all = []
    z_optimal_all = []
    T_true_all = []
    log_P_all = []
    T_pred_initial_all = []
    T_pred_optimal_all = []

    # Loop over data in batches
    for log_P_, T in track(
        sequence=test_dataloader,
        description='Optimizing z:',
    ):

        # Compute "normal" pass through the model (encoder and decoder)
        z_initial_, T_pred_initial_ = model(
            log_P=log_P_.to(device), T=T.to(device)
        )

        # Use the value of z from the encoder as a starting point for an
        # optimizer to find the best input value to the decoder
        z_optimal_, T_pred_optimal_ = optimize_z_with_lbfgs(
            z_initial=z_initial_.detach(),
            log_P=log_P_,
            T_true=T,
            decoder=model.decoder,
            n_epochs=n_epochs,
            batch_size=batch_size,
            device=device,
            history_size=history_size,
        )

        # Store batch results (as numpy array on CPU!)
        z_initial_all.append(z_initial_.detach().cpu().numpy())
        z_optimal_all.append(z_optimal_.detach().cpu().numpy())
        T_true_all.append(T.detach().cpu().numpy())
        log_P_all.append(log_P_.detach().cpu().numpy())
        T_pred_initial_all.append(T_pred_initial_.detach().cpu().numpy())
        T_pred_optimal_all.append(T_pred_optimal_.detach().cpu().numpy())

        # Add some sanity checks to ensure we are not just producing NaNs
        assert not np.any(np.isnan(z_initial_all[-1]))
        assert not np.any(np.isnan(z_optimal_all[-1]))
        assert not np.any(np.isnan(T_true_all[-1]))
        assert not np.any(np.isnan(log_P_all[-1]))
        assert not np.any(np.isnan(T_pred_initial_all[-1]))
        assert not np.any(np.isnan(T_pred_optimal_all[-1]))

    # Merge batch results
    z_initial = np.row_stack(z_initial_all)
    z_optimal = np.row_stack(z_optimal_all)
    T_true = np.row_stack(T_true_all)
    log_P = np.row_stack(log_P_all)
    T_pred_initial = np.row_stack(T_pred_initial_all)
    T_pred_optimal = np.row_stack(T_pred_optimal_all)

    return z_initial, z_optimal, T_true, log_P, T_pred_initial, T_pred_optimal

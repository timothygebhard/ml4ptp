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
) -> Tuple[np.ndarray, ...]:

    # Make sure model is in evaluation mode
    model = model.eval()

    # Keep track of batch-wise results
    z_initial_all = []
    z_optimal_all = []
    T_true_all = []
    T_pred_initial_all = []
    T_pred_optimal_all = []

    # Loop over data in batches
    for log_P, T in track(
        sequence=test_dataloader,
        description='Optimizing z:',
    ):

        # Compute "normal" pass through the model (encoder and decoder)
        z_initial_, T_pred_initial_ = model(log_P=log_P, T=T)

        # Use the value of z from the encoder as a starting point for an
        # optimizer to find the best input value to the decoder
        z_optimal_, T_pred_optimal_ = optimize_z_with_lbfgs(
            z_initial=z_initial_.detach(),
            log_P=log_P,
            T_true=T,
            decoder=model.decoder,
            n_epochs=100,
            batch_size=512,
            device=device,
        )

        # Store batch results
        z_initial_all.append(z_initial_)
        z_optimal_all.append(z_optimal_)
        T_true_all.append(T)
        T_pred_initial_all.append(T_pred_initial_)
        T_pred_optimal_all.append(T_pred_optimal_)

    # Merge batch results and cast to numpy
    z_initial = np.asarray(
        torch.row_stack(z_initial_all).detach().cpu().numpy()
    )
    z_optimal = np.asarray(
        torch.row_stack(z_optimal_all).detach().cpu().numpy()
    )
    T_true = np.asarray(
        torch.row_stack(T_true_all).detach().cpu().numpy()
    )
    T_pred_initial = np.asarray(
        torch.row_stack(T_pred_initial_all).detach().cpu().numpy()
    )
    T_pred_optimal = np.asarray(
        torch.row_stack(T_pred_optimal_all).detach().cpu().numpy()
    )

    return z_initial, z_optimal, T_true, T_pred_initial, T_pred_optimal

"""
Utility functions for evaluation on the test set.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Tuple

from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from tqdm.auto import tqdm

import numpy as np
import torch

from ml4ptp.models import Model
from ml4ptp.utils import get_batch_idx


# -----------------------------------------------------------------------------
# DEFINTIONS
# -----------------------------------------------------------------------------

def get_initial_predictions(
    model: Model,
    test_dataloader: EVAL_DATALOADERS,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # Make sure model is in evaluation mode
    model = model.eval()

    # Keep track of batch-wise results
    z_initial_all = []
    T_true_all = []
    log_P_all = []
    T_pred_initial_all = []

    # Loop over the test set in batches to get z and T_pred
    for _log_P, _T_true in tqdm(test_dataloader, ncols=80):

        # Pass through encoder / decoder to get initial z and T_pred
        _z_initial, _T_pred_initial = model(
            log_P=_log_P.to(device), T=_T_true.to(device)
        )

        # Store batch results (as numpy array on CPU!)
        z_initial_all.append(_z_initial.detach().cpu().numpy())
        T_true_all.append(_T_true.detach().cpu().numpy())
        log_P_all.append(_log_P.detach().cpu().numpy())
        T_pred_initial_all.append(_T_pred_initial.detach().cpu().numpy())

        # Add some sanity checks to ensure we are not just producing NaNs
        assert not np.any(np.isnan(z_initial_all[-1]))
        assert not np.any(np.isnan(T_true_all[-1]))
        assert not np.any(np.isnan(log_P_all[-1]))
        assert not np.any(np.isnan(T_pred_initial_all[-1]))

    # Merge batch results
    z_initial = np.row_stack(z_initial_all)
    T_true = np.row_stack(T_true_all)
    log_P = np.row_stack(log_P_all)
    T_pred_initial = np.row_stack(T_pred_initial_all)

    return z_initial, T_true, log_P, T_pred_initial


def get_refined_predictions(
    model: Model,
    z_initial: np.ndarray,
    T_true: np.ndarray,
    log_P: np.ndarray,
    device: torch.device,
    n_epochs: int = 50,
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:

    # Make sure model is in evaluation mode
    model = model.eval()
    decoder = model.decoder

    # Get batch indices for looping over the inputs
    batch_idx = get_batch_idx(a=z_initial, batch_size=batch_size)

    # Store the refined results that we obtain through optimization
    z_refined_all = []
    T_pred_refined_all = []

    # Loop over data in batches
    for idx in tqdm(batch_idx, ncols=80):

        # Select batch and create tensors on the target device
        _z_refined = (
            torch.from_numpy(z_initial[idx]).to(device).requires_grad_(True)
        )
        _T_true = (
            torch.from_numpy(T_true[idx]).to(device).requires_grad_(False)
        )
        _log_P = torch.from_numpy(log_P[idx]).to(device).requires_grad_(False)

        # Create a new optimizer that optimizes _z_refined plus LR scheduler
        optimizer = torch.optim.AdamW(params=[_z_refined], lr=3e-2)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5
        )

        # Compute errors before refinement
        _T_pred = decoder(z=_z_refined, log_P=_log_P)
        errors_initial = torch.nn.functional.mse_loss(_T_pred, _T_true)

        # Run for the desired number of optimization steps
        for epoch in range(n_epochs):

            # Reset the gradients
            optimizer.zero_grad()

            # Compute T_pred, compute loss, and take step with optimizer
            _T_pred = decoder(z=_z_refined, log_P=_log_P)
            loss = torch.nn.functional.mse_loss(_T_pred, _T_true)
            loss.backward()  # type: ignore
            optimizer.step()
            scheduler.step()

        # Compute T_pred for final value of z, plus errors after refinement
        _T_pred = decoder(z=_z_refined, log_P=_log_P)
        errors_refined = torch.nn.functional.mse_loss(_T_pred, _T_true)

        # Find profiles where the refinement did not decrease the error
        not_improved = errors_refined.squeeze() > errors_initial.squeeze()
        _z_refined = _z_refined.detach().cpu().numpy()
        _z_refined[not_improved] = z_initial[idx][not_improved]

        # Store batch results (as numpy array on CPU!)
        z_refined_all.append(_z_refined)
        T_pred_refined_all.append(_T_pred.detach().cpu().numpy())

        # Add some sanity checks to ensure we are not just producing NaNs
        assert not np.any(np.isnan(z_refined_all[-1]))
        assert not np.any(np.isnan(T_pred_refined_all[-1]))

    # Merge batch results
    z_refined = np.row_stack(z_refined_all)
    T_pred_refined = np.row_stack(T_pred_refined_all)

    return z_refined, T_pred_refined

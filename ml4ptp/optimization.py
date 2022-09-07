"""
Utility methods for dealing with optimization.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Tuple

import numpy as np
import torch


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

def optimize_z_with_lbfgs(
    z_initial: torch.Tensor,
    log_P: torch.Tensor,
    T_true: torch.Tensor,
    decoder: torch.nn.Module,
    batch_size: int,
    device: torch.device,
    n_epochs: int = 10,
    lr: float = 1,
    max_iter: int = 10,
    history_size: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given a `decoder` and an initial guess for `z`, use the L-BFGS
    optimizer to find the "optimal" z to make the decoder output the
    PT profile given by `log_P` and `T_true`.

    This function can be used to "fine-tune" the prediction of the
    encoder to "disentangle" the usefulness of the PT profile family
    represented by the decoder from the encoder performance.

    Args:
        z_initial: Initial guess for `z`; usually the encoder output.
        log_P: Grid with logarithm of the pressure in bar.
        T_true: Grid with temperatures corresponding to `log_P`.
        decoder: A model that takes `z` and `log_P` and predicts `T`.
        batch_size: Number of profiles to optimize simultaneously.
        device: Device (CUDA or CPU) on which to run optimization.
        n_epochs: Number of optimization epochs.
        lr: The learning rate for the LBFGS optimizer.
        max_iter: The maximum number of iterations.
        history_size: The amount of history to keep. Note that LBFGS is
            a very memory-intensive optimizer: As per the PyTorch docs,
            it requires `param_bytes * (history_size + 1)` bytes of
            memory.

    Returns:
        A tuple `(z_optimal, T_pred)` with the optimized values for `z`
        and the corresponding temperatures (for `log_P`).
    """

    # Ensure the decoder model is in evaluation mode; move things to device
    decoder.eval().to(device)
    log_P = log_P.to(device).detach()
    T_true = T_true.to(device).detach()

    # Instantiate list to store results from batch-wise processing
    z_optimal_all = []
    T_pred_all = []

    # Get indices for batches
    all_idx = np.arange(len(z_initial))
    n_splits = int(np.ceil(len(z_initial) / batch_size))

    # Loop over batches
    for idx in np.array_split(all_idx, n_splits):

        # Create a new optimization target
        z_optimal = z_initial[idx].clone().to(device).detach()
        z_optimal.requires_grad = True

        # Set up a new LBFGS optimizer
        optim = torch.optim.LBFGS(
            params=[z_optimal],
            lr=lr,
            max_iter=max_iter,
            history_size=history_size,
        )

        # Run several multiple optimization rounds
        for i in range(n_epochs):

            # Zero out all gradients
            optim.zero_grad()

            # Compute loss associated with z_optimal
            T_pred = decoder.forward(z=z_optimal, log_P=log_P[idx])
            loss = (T_pred - T_true[idx]).pow(2).mean()

            # Backpropagate and use optimizer to update z_optimal
            loss.backward()
            optim.step(closure=lambda: float(loss))

        # Compute the optimized PT profiles from z_optimal for this batch
        T_pred = decoder.forward(z=z_optimal, log_P=log_P[idx])

        # Store z_optimal and PT profiles for this batch
        z_optimal_all.append(z_optimal.detach().cpu())
        T_pred_all.append(T_pred.detach().cpu())

    # Combine batches and return results
    return torch.row_stack(z_optimal_all), torch.row_stack(T_pred_all)

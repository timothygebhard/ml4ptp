"""
Unit tests for optimization.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import pytest
import torch

from ml4ptp.optimization import optimize_z_with_lbfgs


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

@pytest.fixture()
def decoder() -> torch.nn.Module:

    class Decoder(torch.nn.Module):

        @staticmethod
        def forward(
            z: torch.Tensor,
            log_P: torch.Tensor,
        ) -> torch.Tensor:
            result = torch.ones_like(log_P)
            for i in range(z.shape[1]):
                result = (
                    result
                    + z[:, i].reshape(-1, 1) * torch.sin(i * log_P)
                    + z[:, i].reshape(-1, 1) * torch.cos(i * log_P)
                )
            return result

    return Decoder()


def test__optimize_z_with_lbfgs(decoder: torch.nn.Module) -> None:

    torch.manual_seed(423)

    batch_size = 7
    grid_size = 11
    latent_size = 5

    log_P = torch.tile(torch.linspace(1, 10, grid_size), (batch_size, 1))
    z_true = torch.randn(batch_size, latent_size)
    T_true = decoder.forward(z=z_true, log_P=log_P)
    z_initial = torch.randn(batch_size, latent_size)

    z_optimal, T_pred = optimize_z_with_lbfgs(
        z_initial=z_initial,
        log_P=log_P,
        T_true=T_true,
        decoder=decoder,
        n_epochs=10,
        device=torch.device('cpu'),
        batch_size=2,
    )

    max_delta_T = torch.max(torch.abs(T_pred - T_true)).detach()
    assert np.isclose(max_delta_T, 0, atol=1e-4)

    max_delta_z = torch.max(torch.abs(z_optimal - z_true)).detach()
    assert np.isclose(max_delta_z, 0, atol=1e-4)

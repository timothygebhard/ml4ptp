"""
Unit tests for plotting.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import torch

from ml4ptp.plotting import plot_profile_to_tensorboard, plot_z_to_tensorboard


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

def test__plot_profile_to_tensorboard() -> None:

    figure = plot_profile_to_tensorboard(
        z=torch.randn(17, 5),
        log_P=torch.randn(17, 19),
        T_pred=torch.randn(17, 19),
        T_true=torch.randn(17, 19),
        plotting_config=dict(
            min_T=0,
            max_T=100,
            min_log_P=-5,
            max_log_P=5,
        ),
    )
    assert isinstance(figure, plt.Figure)
    plt.close(figure)


def test__plot_z_to_tensorboard() -> None:

    figure = plot_z_to_tensorboard(z=torch.randn(17, 3))
    assert isinstance(figure, plt.Figure)
    plt.close(figure)

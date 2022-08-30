"""
Methods for plotting.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from corner import corner

import matplotlib.pyplot as plt
import torch

from ml4ptp.utils import tensor_to_str


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

def plot_z_to_tensorboard(z: torch.Tensor) -> plt.Figure:

    z_numpy = z.detach().cpu().numpy()

    # Set up a new figure for a corner plot
    figure = corner(
        data=z_numpy,
        bins=25,
        range=z_numpy.shape[1] * [(-5, 5)],
        plot_density=False,
        plot_contours=False,
    )
    figure.tight_layout()

    return figure


def plot_profile_to_tensorboard(
    z: torch.Tensor,
    log_P: torch.Tensor,
    T_true: torch.Tensor,
    T_pred: torch.Tensor,
    plotting_config: dict,
) -> plt.Figure:

    # Create a 3x3 figure to plot some examples
    figure, axes = plt.subplots(ncols=3, nrows=3, figsize=(8, 8))
    axes = axes.flatten()

    # Create plots for the first 9 elements
    for i, ax in enumerate(axes):

        # Plot true and reconstructed PT profile
        ax.plot(
            T_true[i].detach().cpu().numpy(),
            log_P[i].detach().cpu().numpy(),
            '.',
            color='C0',
            label='True',
        )
        ax.plot(
            T_pred[i].detach().cpu().numpy(),
            log_P[i].detach().cpu().numpy(),
            '-',
            lw=0.5,
            color='C1',
            label='Pred.',
        )

        # Set plot options
        ax.set_title(f'z = {tensor_to_str(z[i])}', fontsize=10)
        ax.set_xlim(plotting_config['min_T'], plotting_config['max_T'])
        ax.set_xlabel('Temperature (K)')
        ax.set_ylim(plotting_config['min_log_P'], plotting_config['max_log_P'])
        ax.set_ylabel('log10(Pressure / bar)')
        ax.grid(ls='--', alpha=0.5)
        ax.legend(loc='lower left', fontsize=8)

    figure.tight_layout()

    return figure

"""
Methods for plotting.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import List, Tuple

from corner import corner

import matplotlib.pyplot as plt
import numpy as np
import torch

from ml4ptp.utils import tensor_to_str


# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS
# -----------------------------------------------------------------------------


def set_fontsize(ax: plt.Axes, fontsize: int) -> None:
    """
    Set the ``fontsize`` for all labels (title, x- and y-label, and tick
    labels) of a target ``ax`` at once.

    Args:
        ax: The ax which contains the plot.
        fontsize: The target font size for the labels.
    """

    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(fontsize)


# -----------------------------------------------------------------------------
# TENSORBOARD
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
            lw=1.5,
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


# -----------------------------------------------------------------------------
# CREATING FIGURES
# -----------------------------------------------------------------------------


def plot_pt_profile(
    pt_profiles: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    latent_size: int,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> Tuple[plt.Figure, plt.Axes]:

    # Create a new figure
    fig, ax = plt.subplots(figsize=(3.2 / 2.54, 3.0 / 2.54))

    # Loop over different file paths (usually those are different runs)
    for i, (log_P, T_true, T_pred) in enumerate(pt_profiles):

        # Plot the true PT profile and the best polynomial fit (only once)
        if i == 0:

            # Draw the true PT profile
            ax.plot(T_true, log_P, 'o', ms=1, mec='none', mfc='k', zorder=99)

            # Fit the true PT profile with a polynomial and plot the result
            p = np.polyfit(log_P, T_true, deg=latent_size - 1)
            T_pred_poly = np.polyval(p=p, x=log_P)
            ax.plot(T_pred_poly, log_P, lw=1, color='C1')

        # Plot the best fit obtained with the decoder model
        ax.plot(T_pred, log_P, lw=1, color='C0')

    # Set up ax labels, limits, etc.
    ax.set_xticks(np.linspace(int(xlim[0]), int(xlim[1]), 3, endpoint=True))
    ax.set_yticks(range(int(ylim[1]), int(ylim[0]) + 1, 1))
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('log(Pressure / bar)')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    set_fontsize(ax, 6)

    # Adjust spacing around the plot
    plt.subplots_adjust(
        left=0.30,
        bottom=0.275,
        right=0.90,
        top=0.975,
    )

    return fig, ax

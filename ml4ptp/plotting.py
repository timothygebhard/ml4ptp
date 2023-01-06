"""
Methods for plotting.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Any

from corner import corner
from matplotlib.colorbar import Colorbar
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import torch

from ml4ptp.utils import tensor_to_str


# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS
# -----------------------------------------------------------------------------

def add_colorbar_to_ax(
    img: AxesImage,
    fig: plt.Figure,
    ax: plt.Axes,
    where: str = 'right',
    **kwargs: Any,
) -> Colorbar:
    """
    Add a "nice" colorbar to a plot.

    Args:
        img: The return of a plotting command.
        fig: The figure that the plot is part of.
        ax: The ax which contains the plot.
        where: Where to place the colorbar (`"left"`, `"right"`,
            `"top"` or `"bottom"`).
        **kwargs: Additional keyword arguments which will be passed
            to `fig.colorbar()`.

    Returns:
        The colorbar that was added to the axis.
    """

    if where in ('left', 'right'):
        orientation = 'vertical'
    elif where in ('top', 'bottom'):
        orientation = 'horizontal'
    else:
        raise ValueError(
            f'Illegal value for `where`: "{where}". Must be one '
            'of ["left", "right", "top", "bottom"].'
        )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(where, size='5%', pad=0.05)
    cbar = fig.colorbar(
        img, cax=cax, orientation=orientation, ticklocation=where, **kwargs,
    )
    cbar.ax.tick_params(labelsize=8)

    return cbar


def set_fontsize(ax: plt.Axes, fontsize: float) -> None:
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


def disable_ticks(ax: plt.Axes) -> None:
    """
    Disable the ticks and labels on the given matplotlib ``ax``.

    This is similar to calling ``ax.axis('off')``, except that the frame
    around the plot is preserved.

    Args:
        ax: A matplotlib axis.
    """

    ax.tick_params(
        axis='both',
        which='both',
        top=False,
        bottom=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )


# -----------------------------------------------------------------------------
# TENSORBOARD
# -----------------------------------------------------------------------------

def plot_z_to_tensorboard(z: torch.Tensor) -> plt.Figure:

    z_numpy = z.detach().cpu().numpy()

    # Set up a new figure for a corner plot
    figure = corner(
        data=z_numpy,
        bins=25,
        range=z_numpy.shape[1] * [(-4, 4)],
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

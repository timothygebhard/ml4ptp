"""
Create a plot showing the error distributions for a given set of
experiments (or baselines), which is given via a configuration file.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Any, Dict, List, Tuple

import argparse
import time
import warnings

from KDEpy import TreeKDE
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ml4ptp.config import load_yaml
from ml4ptp.paths import expandvars
from ml4ptp.plotting import set_fontsize, CBF_COLORS


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

def get_cli_arguments() -> argparse.Namespace:
    """
    Get CLI arguments.
    """

    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-files',
        required=True,
        nargs='+',
        default=[
            './configs/pyatmos__polynomial.yaml',
            './configs/pyatmos__pca.yaml',
            './configs/pyatmos__our-method.yaml',
        ],
        help=(
            'Path to the configuration file(s) which specifies which error '
            'distributions we are plotting.'
        ),
    )
    parser.add_argument(
        '--output-file-name',
        default='pyatmos.pdf',
        help='Name of the output file (including file extension).',
    )
    args = parser.parse_args()

    return args


def add_plot_group_to_figure(
    dataset: str,
    title: str,
    plot_group: Dict[str, Any],
    fig: plt.Figure,
    ax: plt.Axes,
    kde_grid: np.ndarray,
) -> Tuple[plt.Figure, plt.Axes, float]:
    """
    Add a plot group to the figure.
    A plot group consists of several runs with the same settings, for
    which we plot the mean and standard deviation.
    """

    # -------------------------------------------------------------------------
    # Load data from HDF file
    # -------------------------------------------------------------------------

    # Collect all data from all runs
    rmses = []
    for run in plot_group['runs']:

        # Check if the file exists
        file_path = expandvars(Path(run['file_path'])).resolve()
        if not file_path.exists():
            warnings.warn(f'File {file_path} does not exist!', stacklevel=2)
            continue

        # Open HDF file and get the RSME
        with h5py.File(file_path, 'r') as hdf_file:

            # Get all (R)MSEs
            rsme = np.sqrt(np.array(hdf_file[run['key']]))

            # Only keep the RMSEs where the nested sampling did not fail
            if 'success' in hdf_file.keys():
                success = np.array(hdf_file['success']).astype(bool)
                rsme = rsme[success]

            rmses.append(rsme)

    # If we have no data, return the figure as is
    if not rmses:
        warnings.warn(
            f'No runs found for plot group {plot_group["label"]}',
            stacklevel=2,
        )
        return fig, ax, np.nan

    # -------------------------------------------------------------------------
    # Compute median and KDE
    # -------------------------------------------------------------------------

    # Compute the median (or rather: the mean of the median of each run)
    median = float(np.mean([np.median(rmse) for rmse in rmses]))

    # Compute a histogram for each run
    kdes = []
    for rmse in rmses:
        kde = TreeKDE(kernel='gaussian', bw='ISJ').fit(rmse)
        kdes.append(kde.evaluate(kde_grid))

    # Compute the mean and standard deviation of the KDE
    kde_mean = np.mean(kdes, axis=0)
    kde_std = np.std(kdes, axis=0)

    # -------------------------------------------------------------------------
    # Plot median and KDE
    # -------------------------------------------------------------------------

    # Determine color
    color = CBF_COLORS[plot_group['n'] - 1]

    # Plot the mean KDE
    ax.plot(
        kde_grid,
        kde_mean,
        label=plot_group['label'],
        color=color,
        lw=1,
    )

    # Plot the standard deviation of the KDE
    ax.fill_between(
        kde_grid,
        kde_mean - kde_std,
        kde_mean + kde_std,
        facecolor=color,
        edgecolor='none',
        alpha=0.25,
    )

    # Add dashed line with the median to the plot
    for i, (c, lw, ls) in enumerate(
        [
            ('white', 1.25, '-'),
            (color, 0.75, '--'),
        ]
    ):
        ax.axvline(
            x=median,
            color=c,
            ls=ls,
            lw=lw,
            zorder=97 + i,
        )

    # Define the horizontal alignment and background color of the median label.
    # It's not pretty, but otherwise we get overlapping labels for PyATMOS.
    ha = 'center'
    x_offset = 0.00
    if dataset == 'pyatmos' and title == 'Our method':
        if plot_group['n'] == 4:
            x_offset = -0.007
        elif plot_group['n'] == 3:
            x_offset = 0.007

    # Define the vertical position of the median label
    if dataset == 'pyatmos' and title == 'Our method':
        y = 0.04
        va = 'bottom'
        dy0 = 0.00
        dy1 = 0.02
    else:
        y = 0.96
        va = 'top'
        dy0 = 0.02
        dy1 = 0.00

    # Add text box with median
    label = ax.text(
        x=(
            (median - np.min(kde_grid))
            / (np.max(kde_grid) - np.min(kde_grid))
            + x_offset
            + 0.0014  # adjust improper alignment due to lack of descenders
        ),
        y=y,
        s=f'{median:.2f}',
        rotation=90,
        va=va,
        ha=ha,
        transform=ax.transAxes,
        fontsize=7,
        color=color,
        bbox=dict(alpha=0, boxstyle='square,pad=0'),
        zorder=100,
    )

    # Manually add a white background to the median label
    # We can't simply use the `bbox` kwarg because it's alignment is not ideal
    # with text that does not have descenders (e.g., numbers only).
    # See: https://stackoverflow.com/a/69254920/4100721
    fig.canvas.draw()
    extent = label.get_window_extent(renderer=fig.canvas.get_renderer())
    extent = extent.transformed(ax.transAxes.inverted())
    x0 = extent.x0 + 0.00125
    x1 = extent.x1 - 0.00400
    y0 = extent.y0 + dy0
    y1 = extent.y1 - dy1
    dx = x1 - x0
    dy = y1 - y0
    ax.add_patch(
        Rectangle(
            xy=(x0, y0),
            width=dx,
            height=dy,
            transform=ax.transAxes,
            fc='white',
            ec='none',
            alpha=0.9,
            zorder=99,
        )
    )

    return fig, ax, median


def get_plot_options(dataset: str) -> dict:
    """
    Get plot options for a given dataset ("pyatmos" or "goyal-2020").
    """

    plot_options: Dict[str, Any] = {}

    if dataset == 'pyatmos':
        plot_options['kde_grid'] = np.linspace(-0.25, 16.25, 1000)
        plot_options['ylim'] = (0.001, 10)
        plot_options['yformatter'] = FormatStrFormatter('%.3f')
    elif dataset == 'goyal-2020':
        plot_options['kde_grid'] = np.linspace(-2.5, 202.5, 1000)
        plot_options['ylim'] = (0.0001, 1.0)
        plot_options['yformatter'] = FormatStrFormatter('%.3f')
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    return plot_options


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nPLOT ERROR DISTRIBUTIONS\n', flush=True)

    # Get CLI arguments
    args = get_cli_arguments()

    # -------------------------------------------------------------------------
    # Create new figure and loop over configuration files
    # -------------------------------------------------------------------------

    # Set default font
    mpl.rcParams['font.sans-serif'] = "Arial"
    mpl.rcParams['font.family'] = "sans-serif"

    # Create a new figure
    pad_inches = 0.025
    fig, axes = plt.subplots(
        nrows=len(args.config_files),
        figsize=(
            18.4 / 2.54 - 2 * pad_inches,
            len(args.config_files) * 3.4 / 2.54 - 2 * pad_inches,
        ),
        sharex='all',
    )
    axes = np.atleast_1d(axes)
    plt.subplots_adjust(hspace=0, wspace=0.05)

    # Loop over configuration files
    for i, config_file in enumerate(args.config_files):

        # Load configuration file
        print('\nLoading configuration file...', end=' ', flush=True)
        file_path = expandvars(Path(config_file)).resolve()
        config = load_yaml(file_path)
        print('Done!', flush=True)

        # Define shortcuts
        ax = axes[i]
        dataset = file_path.name.split('__')[0]
        title = config['title']
        plot_groups: List[Dict[str, Any]] = config['plot_groups']

        # Get plot options
        plot_options = get_plot_options(dataset)

        # Loop over the plot groups
        print('Creating plots...', end=' ', flush=True)
        medians = {}
        for plot_group in plot_groups:
            fig, ax, median = add_plot_group_to_figure(
                dataset=dataset,
                title=title,
                plot_group=plot_group,
                fig=fig,
                ax=ax,
                kde_grid=plot_options['kde_grid'],
            )
            medians[plot_group['label']] = median
        print('Done!\n', flush=True)

        # Add legend
        legend = ax.legend(loc='center right', fontsize=7)
        legend.set_zorder(1000)
        legend.set_title(
            title=title,
            prop={'size': 7, 'weight': 'bold'},
        )
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_linewidth(0)

        # Set width and z-order of the frame
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_zorder(1000)
            ax.spines[axis].set_linewidth(0.25)
            ax.xaxis.set_tick_params(width=0.25)
            ax.yaxis.set_tick_params(width=0.25)

        # Set x- and y-limits (and make them log-scale); format ticks
        ax.set_xlim(plot_options['kde_grid'][0], plot_options['kde_grid'][-1])
        ax.set_ylim(*plot_options['ylim'])
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(plot_options['yformatter'])
        ax.yaxis.get_major_locator().set_params(numticks=99)
        ax.yaxis.get_minor_locator().set_params(
            numticks=99,
            subs=np.arange(0.1, 1, 0.1),
        )

        # Set font sizes
        set_fontsize(ax, 7)
        ax.xaxis.label.set_fontsize(8)
        ax.yaxis.label.set_fontsize(8)

        # Set x- and y-labels
        ax.set_ylabel('Density')
        if i == len(args.config_files) - 1:
            ax.set_xlabel('Root Mean Squared Error (K)')

        # Set width and length of the tick marks
        ax.tick_params('x', length=0, width=0.25, which='major')
        ax.tick_params('y', length=2, width=0.25, which='major')
        ax.tick_params('x', length=0, width=0.25, which='minor')
        ax.tick_params('y', length=1, width=0.25, which='minor')
        if i == len(args.config_files) - 1:
            ax.tick_params('x', length=2, width=0.25, which='major')
            ax.tick_params('x', length=1, width=0.25, which='minor')

        # Print the medians of the different plot groups
        print('Medians of the different plot groups:\n')
        for label, median in medians.items():
            print(f'  {label}: {median:.2f}')
        print()

    # -------------------------------------------------------------------------
    # Save the figure
    # -------------------------------------------------------------------------

    print('\nSaving figure to PDF...', end=' ', flush=True)

    fig.tight_layout(pad=0)

    plots_dir = Path(__file__).resolve().parent / 'plots'
    plots_dir.mkdir(exist_ok=True)
    file_path = plots_dir / args.output_file_name
    fig.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=pad_inches)

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.2f} seconds.\n')

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
from matplotlib.ticker import FormatStrFormatter

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ml4ptp.config import load_yaml
from ml4ptp.paths import expandvars
from ml4ptp.plotting import set_fontsize


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
        # required=True,
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
        default='pyatmos__all.pdf',
        help='Name of the output file (including file extension).',
    )
    args = parser.parse_args()

    return args


def add_plot_group_to_figure(
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
            warnings.warn(f'File {file_path} does not exist!')
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
        warnings.warn(f'No runs found for plot group {plot_group["label"]}')
        return fig, ax, np.nan

    # -------------------------------------------------------------------------
    # Compute median and KDE
    # -------------------------------------------------------------------------

    # Compute the median (or rather: the mean of the median of each run)
    median = float(np.mean([np.median(_) for _ in rmses]))

    # Compute a histogram for each run
    kdes = []
    for i, rmse in enumerate(rmses):
        kde = TreeKDE(kernel='gaussian', bw='ISJ').fit(rmse)
        kdes.append(kde.evaluate(kde_grid))

    # Compute the mean and standard deviation of the KDE
    kde_mean = np.mean(kdes, axis=0)
    kde_std = np.std(kdes, axis=0)

    # -------------------------------------------------------------------------
    # Plot median and KDE
    # -------------------------------------------------------------------------

    # Determine color
    color = f"C{plot_group['n'] - 1}"

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
    for i, (c, lw, ls) in enumerate([('white', 1, '-'), (color, 0.5, '--')]):
        ax.axvline(
            x=median,
            color=c,
            ls=ls,
            lw=lw,
            zorder=98 + i,
        )

    # Add text box with median
    for c, alpha in [('white', 1), (color, 0)]:
        ax.text(
            x=(
                (np.log10(median) - np.log10(kde_grid[0]))
                / (np.log10(np.max(kde_grid)) - np.log10(kde_grid[0]))
            ),
            y=0.9,
            s=f' {median:.2f} ',
            rotation=90,
            va='top',
            ha='center',
            transform=ax.transAxes,
            fontsize=3.5,
            color=c,
            bbox=dict(
                fc='white', ec='none', alpha=alpha, boxstyle='square,pad=0'
            ),
            zorder=100 - alpha,
        )
 
    return fig, ax, median


def get_plot_options(dataset: str) -> dict:
    """
    Get plot options for a given dataset ("pyatmos" or "goyal-2020").
    """

    plot_options: Dict[str, Any] = {}

    if dataset == 'pyatmos':
        plot_options['kde_grid'] = np.geomspace(0.01, 50, 1000)
        plot_options['ylim'] = (0.01, 20)
        plot_options['yformatter'] = FormatStrFormatter('%.2f')
    elif dataset == 'goyal-2020':
        plot_options['kde_grid'] = np.geomspace(1, 300, 1000)
        plot_options['ylim'] = (0.001, 2)
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
            17 / 2.54 - 2 * pad_inches,
            len(args.config_files) * 2.5 / 2.54 - 2 * pad_inches,
        ),
        sharex='all',
    )
    axes = np.atleast_1d(axes)
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

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
                plot_group=plot_group,
                fig=fig,
                ax=ax,
                kde_grid=plot_options['kde_grid'],
            )
            medians[plot_group['label']] = median
        print('Done!\n', flush=True)

        # Add legend
        legend = ax.legend(loc='center left', fontsize=5.5)
        legend.set_title(
            title=title,
            prop={'size': 5.5, 'weight': 'bold'},
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
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(plot_options['yformatter'])

        # Set font sizes
        set_fontsize(ax, 5.5)
        ax.xaxis.label.set_fontsize(6.5)
        ax.yaxis.label.set_fontsize(6.5)

        # Set x- and y-labels
        ax.set_ylabel('Density')
        if i == len(args.config_files) - 1:
            ax.set_xlabel('Root Mean Squared Error (in Kelvin)')

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

    plots_dir = Path(__file__).resolve().parent / 'plots'
    plots_dir.mkdir(exist_ok=True)

    file_path = plots_dir / args.output_file_name
    fig.tight_layout(pad=0)
    fig.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=pad_inches)

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.2f} seconds.\n')

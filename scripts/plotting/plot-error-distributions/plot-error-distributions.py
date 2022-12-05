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

from matplotlib.ticker import FormatStrFormatter
from scipy.stats import gaussian_kde

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

    # Read in the RSME (root mean squared error) for different runs
    # Note: We assume that the error stored in the HDF file is the MSE.
    rmses = []
    for run in plot_group['runs']:

        file_path = expandvars(Path(run['file_path'])).resolve()
        if not file_path.exists():
            warnings.warn(f'File {file_path} does not exist!')
            continue

        with h5py.File(file_path, 'r') as hdf_file:
            rsme = np.sqrt(np.array(hdf_file[run['key']])).squeeze()
            rmses.append(rsme)

    if not rmses:
        warnings.warn(f'No runs found for plot group {plot_group["label"]}')
        return fig, ax, np.nan

    # Compute the median (or rather: the mean of the median of each run)
    median = float(np.mean([np.median(_) for _ in rmses]))

    # Compute a KDE for each run and evaluate it on the given grid
    kdes = [gaussian_kde(_)(kde_grid) for _ in rmses]

    # Compute the mean and standard deviation of the KDE
    kde_mean = np.mean(kdes, axis=0)
    kde_std = np.std(kdes, axis=0)

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

    # Add median to the plot
    ax.axvline(
        x=median,
        color=color,
        linestyle='--',
        linewidth=0.5,
        zorder=99,
    )
    ax.text(
        x=median / np.max(kde_grid),
        y=0.5,
        s=f'{median:.1f}',
        rotation=90,
        va='center',
        ha='center',
        transform=ax.transAxes,
        fontsize=4,
        color=color,
        bbox=dict(
            facecolor='white',
            alpha=0.9,
            edgecolor='none',
            boxstyle='round,pad=0.1',
        ),
        zorder=100,
    )

    return fig, ax, median


def get_plot_options(dataset: str) -> dict:

    plot_options = {}

    if dataset == 'pyatmos':
        plot_options['kde_grid'] = np.linspace(0, 16, 500)
    elif dataset == 'goyal-2020':
        plot_options['kde_grid'] = np.linspace(0, 160, 500)
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
        legend = ax.legend(loc='center right', fontsize=5.5)
        legend.set_title(
            title=title,
            prop={'size': 5.5, 'weight': 'bold'},
        )
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_linewidth(0)

        # Set general plot options
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(0.25)
            ax.xaxis.set_tick_params(width=0.25)
            ax.yaxis.set_tick_params(width=0.25)

        ax.set_ylabel('Density')
        ax.set_xlim(
            plot_options['kde_grid'][0],
            plot_options['kde_grid'][-1],
        )
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        set_fontsize(ax, 5.5)
        ax.xaxis.label.set_fontsize(6.5)
        ax.yaxis.label.set_fontsize(6.5)
        ax.tick_params('y', length=2, width=0.25, which='major')
        ax.tick_params('x', length=0, width=0.25, which='major')

        if i == len(args.config_files) - 1:
            ax.set_xlabel('Root Mean Squared Error (in Kelvin)')
            ax.tick_params('x', length=2, width=0.25, which='major')

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

"""
Create a plot showing the error distributions for a given set of
experiments (or baselines), which is given via a configuration file.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List, Tuple
try:
    from yaml import CLoader as Loader
except ImportError:  # pragma: no cover
    from yaml import Loader  # type: ignore

import argparse
import time
import warnings
import yaml

from matplotlib.ticker import FormatStrFormatter
from scipy.stats import gaussian_kde

import h5py
import matplotlib.pyplot as plt
import numpy as np

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
        '--config-file',
        required=True,
        default='./configs/pyatmos__polynomial.yaml',
        help=(
            'Path to the configuration file which specifies which error '
            'distributions we are plotting.'
        ),
    )
    args = parser.parse_args()

    return args


def add_plot_group_to_figure(
    idx: int,
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

    # Plot the mean KDE
    ax.plot(
        kde_grid,
        kde_mean,
        label=plot_group['label'],
        color=f'C{idx}',
    )

    # Plot the standard deviation of the KDE
    ax.fill_between(
        kde_grid,
        kde_mean - kde_std,
        kde_mean + kde_std,
        facecolor=f'C{idx}',
        edgecolor='none',
        alpha=0.25,
    )

    # Add median to the plot
    ax.axvline(
        x=median,
        color=f'C{idx}',
        linestyle='--',
        linewidth=1,
        zorder=99,
    )

    return fig, ax, median


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nPLOT ERROR DISTRIBUTIONS\n', flush=True)

    # -------------------------------------------------------------------------
    # Get CLI arguments and load configuration file
    # -------------------------------------------------------------------------

    # Get CLI arguments
    args = get_cli_arguments()

    # Load configuration file
    print('Loading configuration file...', end=' ', flush=True)
    file_path = expandvars(Path(args.config_file)).resolve()
    with open(file_path, 'r') as yaml_file:
        plot_groups: List[Dict[str, Any]] = yaml.load(yaml_file, Loader=Loader)
    print('Done!', flush=True)

    print('Loaded the following from the configuration file:\n')
    pprint(plot_groups)
    print()

    # -------------------------------------------------------------------------
    # Determine plot options based on dataset
    # -------------------------------------------------------------------------

    # Get dataset
    dataset = file_path.name.split('__')[0]

    # Set plot options based on dataset
    if dataset == 'pyatmos':
        kde_grid = np.linspace(0, 15, 500)
    elif dataset == 'goyal-2020':
        kde_grid = np.linspace(0, 120, 500)
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    # -------------------------------------------------------------------------
    # Loop over the plot groups and create the plots
    # -------------------------------------------------------------------------

    # Create a new figure
    pad_inches = 0.025
    fig, ax = plt.subplots(
        figsize=(17 / 2.54 - 2 * pad_inches,  3.5 / 2.54 - 2 * pad_inches),
    )

    # Loop over the plot groups
    print('Creating plots...', end=' ', flush=True)
    medians = {}
    for idx, plot_group in enumerate(plot_groups):
        fig, ax, median = add_plot_group_to_figure(
            idx=idx,
            plot_group=plot_group,
            fig=fig,
            ax=ax,
            kde_grid=kde_grid,
        )
        medians[plot_group['label']] = median
    print('Done!\n', flush=True)

    # Set general plot options
    ax.legend(loc='upper right', fontsize=6)
    ax.set_xlabel('Root Mean Squared Error (in Kelvin)')
    ax.set_ylabel('Density')
    ax.set_xlim(kde_grid[0], kde_grid[-1])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    set_fontsize(ax, 6)
    fig.tight_layout(pad=0)

    # Print the medians of the different plot groups
    print('Medians of the different plot groups:\n')
    for label, median in medians.items():
        print(f'  {label}: {median:.2f}')

    # Save the figure
    print('\nSaving figure to PDF...', end=' ', flush=True)
    plots_dir = Path(__file__).resolve().parent / 'plots'
    plots_dir.mkdir(exist_ok=True)
    file_name = Path(args.config_file).name.replace('.yaml', '.pdf')
    file_path = plots_dir / file_name
    fig.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=pad_inches)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.2f} seconds.\n')

"""
Create plot of the error distribution.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import argparse
import time

from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

import h5py
import matplotlib.pyplot as plt
import numpy as np

from ml4ptp.paths import expandvars
from ml4ptp.plotting import set_fontsize
from ml4ptp.utils import find_run_dirs_with_results


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

def get_cli_args() -> argparse.Namespace:

    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--baseline-dir',
        required=True,
        help='Path to the experiment directory with baseline results.',
    )
    parser.add_argument(
        '--dataset',
        required=True,
        choices=('pyatmos', 'goyal-2020'),
        help='Dataset on which the experiments where performed.',
    )
    parser.add_argument(
        '--latent-sizes',
        nargs='+',
        type=int,
        default=(2, 3, 4, 5),
        help='Latent sizes to be included in the plot.',
    )
    parser.add_argument(
        '--parent-dir',
        required=True,
        help=(
            'Path to the parent directory of the experiment directories. '
            'This directory needs to contain directories <latent-size-N>.'
        ),
    )
    args = parser.parse_args()

    return args


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nPLOT ERROR DISTRIBUTION\n', flush=True)

    # -------------------------------------------------------------------------
    # Get experiment dir and load configuration file
    # -------------------------------------------------------------------------

    # Get CLI arguments, define shortcuts
    args = get_cli_args()
    dataset = args.dataset
    latent_sizes = list(args.latent_sizes)

    # Load experiment configuration from YAML
    baseline_dir = expandvars(Path(args.baseline_dir)).resolve()
    parent_dir = expandvars(Path(args.parent_dir)).resolve()
    print('Looking for results in the following locations:')
    print('  Baseline:   ', baseline_dir)
    print('  Experiments:', parent_dir)
    print()

    # Print other parameters for debugging
    print('Other parameters:')
    print('  dataset:     ', dataset)
    print('  latent_sizes:', latent_sizes)
    print()

    # Define grid (depending on data set)
    if dataset == 'goyal-2020':
        grid = np.linspace(0, 100, 2_000)
    elif dataset == 'pyatmos':
        grid = np.linspace(0, 12.5, 2_000)
    else:
        raise ValueError('Invalid data set!')

    # -------------------------------------------------------------------------
    # Create a new figure, add a (custom) legend
    # -------------------------------------------------------------------------

    # Create a new figure
    fig, axes = plt.subplots(
        figsize=(6.8 / 2.54, 3.6 / 2.54),
        nrows=3,
        sharex='all',
        sharey='all',
        gridspec_kw={'height_ratios': [0.5, 2, 2]},
    )

    # Create custom legend and add it to axes[0]
    handles = [
        Line2D([0], [0], color=f'C{i}', lw=2) for i in range(len(latent_sizes))
    ]
    labels = [f'dim(z) = {z}' for z in latent_sizes]
    leg = axes[0].legend(
        handles=handles,
        labels=labels,
        columnspacing=1.0,
        fontsize=6,
        frameon=False,
        handlelength=0.75,
        handletextpad=0.5,
        loc='center',
        ncol=4,
    )
    axes[0].axis('off')

    # -------------------------------------------------------------------------
    # Plot error distribution for polynomial baseline
    # -------------------------------------------------------------------------

    print('Plotting results from baseline...', end=' ', flush=True)

    for i, latent_size in enumerate(latent_sizes):

        # Load the true and predicted temperatures from HDF file
        file_path = baseline_dir / f'n-parameters_{latent_size}.hdf'
        with h5py.File(file_path, 'r') as hdf_file:
            T_true = np.array(hdf_file['T_true'])
            T_pred = np.array(hdf_file['T_pred'])

        # Compute the mean absolute error for each profile; compute KDE
        error = np.mean(np.abs(T_true - T_pred), axis=1)
        kde = gaussian_kde(error)

        # Plot the error distribution, add vertical line for median
        axes[1].plot(grid, kde(grid), lw=1, color=f'C{i}')
        axes[1].axvline(x=np.median(error), lw=0.5, ls='--', color=f'C{i}')

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Plot error distribution for our method
    # -------------------------------------------------------------------------

    print('Plotting result from our method...', end=' ', flush=True)

    for i, latent_size in enumerate(latent_sizes):

        # Define expected experiment_dir and collect available runs
        experiment_dir = parent_dir / f'latent-size-{latent_size}'
        run_dirs = find_run_dirs_with_results(experiment_dir)

        # Collect results for each run
        run_results = []
        for run_dir in run_dirs:

            # Load the true and predicted temperatures from HDF file
            file_path = run_dir / 'results_on_test_set.hdf'
            with h5py.File(file_path, 'r') as hdf_file:
                T_true = np.array(hdf_file['T_true'])
                T_pred = np.array(hdf_file['T_pred_refined'])

            # Compute the mean absolute error for each profile; compute KDE
            error = np.mean(np.abs(T_true - T_pred), axis=1)
            kde = gaussian_kde(error)
            run_results.append(kde(grid))

        # Compute mean and standard deviation of all runs
        all_kde = np.row_stack(run_results)
        kde_mean = np.mean(all_kde, axis=0)
        kde_std = np.std(all_kde, axis=0)

        # Plot the mean distribution with an error band
        axes[2].plot(grid, kde_mean, lw=1, color=f'C{i}')
        axes[2].fill_between(
            grid,
            kde_mean - kde_std,
            kde_mean + kde_std,
            ec='none',
            fc=f'C{i}',
            alpha=0.3,
        )

        # Compute the median of the mean distribution and add vertical line
        cdf = np.cumsum(kde_mean)
        cdf /= np.max(cdf)
        median = grid[np.abs(cdf - 0.5).argmin()]
        axes[2].axvline(x=median, lw=0.5, ls='--', color=f'C{i}')

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Set up ax labels, limits, ...
    # -------------------------------------------------------------------------

    # Setup ticks
    axes[1].tick_params(
        axis='both',
        which='both',
        bottom=False,
        left=False,
        labelleft=False,
    )
    axes[2].tick_params(
        axis='both',
        which='both',
        left=False,
        labelleft=False,
    )
    axes[2].set_xticks(np.linspace(min(grid), max(grid), 6))

    # Adjust whitespace around figure
    plt.subplots_adjust(
        left=0.055,
        bottom=0.22,
        right=0.97,
        top=1.01,
        wspace=0.1,
        hspace=0.1,
    )

    # Set labels and limits
    axes[1].set_ylabel('Freq.')
    axes[2].set_ylabel('Freq.')
    axes[2].set_xlabel('Mean absolute error (K)')
    axes[2].set_xlim(min(grid) - max(grid) / 50, max(grid) + max(grid) / 50)

    # Set font sizes
    set_fontsize(axes[0], 6)
    set_fontsize(axes[1], 6)
    set_fontsize(axes[2], 6)

    # Add boxes with labels for "Baseline" and "Our method"
    for i, label in enumerate(['Baseline', 'Our method']):
        leg = axes[i + 1].legend(
            [Line2D([0], [0], alpha=0)],
            [label],
            fontsize=6,
            frameon=True,
            handlelength=0,
            handletextpad=0,
            loc='upper right',
        )
        leg.get_frame().set_facecolor('white')
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_linewidth(0.8)
        leg.get_frame().set_alpha(1.0)

    # -------------------------------------------------------------------------
    # Save the plot
    # -------------------------------------------------------------------------

    print('Saving plot to PDF...', end=' ', flush=True)
    file_path = parent_dir / 'error-distribution.pdf'
    plt.savefig(
        file_path,
        facecolor='white',
        transparent=False,
        pad_inches=0,
    )
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nDone! This took {time.time() - script_start:.1f} seconds.\n')

"""
Create a bar plot comparing the median relative error (MRE) of our
method to the MRE of the two baselines for different datasets and
numbers of fitting parameters.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from itertools import product
from typing import Dict

import time

from matplotlib.patches import Patch

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ml4ptp.paths import get_experiments_dir
from ml4ptp.plotting import CBF_COLORS, set_fontsize


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

def collect_results_our_method() -> Dict[str, Dict[int, float]]:
    """
    Auxiliary method to collect the results for our method.
    Note: We need to compute the median over the MRE (which itself is
    the mean over the atmospheric layers; comes pre-computed in the
    results HDF file), and then mean-average over the runs.
    """

    # Define parameter combinations for which to collect data
    datasets = ('pyatmos', 'goyal-2020')
    latent_sizes = (1, 2, 3, 4)
    runs = (0, 1, 2)

    # Loop over all experiments and collect the median MRE for each run
    results: Dict[str, Dict[int, float]] = {ds: {} for ds in datasets}
    for dataset, latent_size in product(datasets, latent_sizes):

        # Collect the median MRE for each run
        medians = []
        for run in runs:
            file_path = (
                get_experiments_dir()
                / dataset
                / 'default'
                / f'latent-size-{latent_size}'
                / 'runs'
                / f'run_{run}'
                / 'results_on_test_set.hdf'
            )
            with h5py.File(file_path, 'r') as hdf_file:
                mre = np.array(hdf_file['mre_refined'])
                medians.append(float(np.median(mre)))

        # Compute the mean over the runs
        results[dataset][latent_size] = float(np.mean(medians))

    return dict(results)


def collect_results_pca() -> Dict[str, Dict[int, float]]:
    """
    Auxiliary method to collect the results for the PCA baseline.
    Note: We need to compute the median over the MRE (which itself is
    the mean over the atmospheric layers; comes pre-computed in the
    results HDF file), and then mean-average over the runs.
    """

    # Define parameter combinations for which to collect data
    datasets = ('pyatmos', 'goyal-2020')
    latent_sizes = (2, 3, 4, 5)
    runs = (0, 1, 2)

    # Loop over all experiments and collect the median MRE for each run
    results: Dict[str, Dict[int, float]] = {ds: {} for ds in datasets}
    for dataset, latent_size in product(datasets, latent_sizes):

        # Collect the median MRE for each run (they are all in the same file)
        file_path = (
            get_experiments_dir()
            / dataset
            / 'pca-baseline'
            / 'results_on_test_set.hdf'
        )
        medians = []
        with h5py.File(file_path, 'r') as hdf_file:
            for run in runs:
                key = f'{latent_size}-principal-components/run-{run}/mre'
                mre = np.array(hdf_file[key])
                medians.append(float(np.median(mre)))

        # Compute the mean over the runs
        results[dataset][latent_size] = float(np.mean(medians))

    return dict(results)


def collect_results_polynomial() -> Dict[str, Dict[int, float]]:
    """
    Auxiliary method to collect the results for the polynomial baseline.
    Note: This is slightly simpler because we do not need to take a mean
    average over different runs (because there are no runs).
    """

    # Define parameter combinations for which to collect data
    datasets = ('pyatmos', 'goyal-2020')
    latent_sizes = (2, 3, 4, 5)

    # Loop over the different parameter combinations
    results: Dict[str, Dict[int, float]] = {ds: {} for ds in datasets}
    for dataset, latent_size in product(datasets, latent_sizes):
        file_path = (
            get_experiments_dir()
            / dataset
            / 'polynomial-baseline'
            / 'results_on_test_set.hdf'
        )
        with h5py.File(file_path, 'r') as hdf_file:
            mre = np.array(hdf_file[f'{latent_size}-fit-parameters/mre'])
            results[dataset][latent_size] = float(np.median(mre))

    return dict(results)


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nCREATE MRE TABLE\n', flush=True)

    # -------------------------------------------------------------------------
    # Collect data
    # -------------------------------------------------------------------------

    # Collect the results for each method
    print('Collecting data...', end=' ', flush=True)
    results: Dict[str, Dict[str, Dict[int, float]]] = {
        'polynomial': collect_results_polynomial(),
        'pca': collect_results_pca(),
        'our_method': collect_results_our_method(),
    }
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Create a bar plot
    # -------------------------------------------------------------------------

    # Set default font; define bar width
    mpl.rcParams['font.sans-serif'] = "Arial"
    mpl.rcParams['font.family'] = "sans-serif"
    BAR_WIDTH = 0.2

    print('Creating bar plots...', end=' ', flush=True)

    # Create separate plots for each dataset
    for dataset in ('pyatmos', 'goyal-2020'):

        # Create the figure
        pad_inches = 0.025
        fig, axes = plt.subplots(
            nrows=2,
            figsize=(8.7 / 2.54, 4.5 / 2.54),
            sharey='all',
            height_ratios=[0.5, 4],
        )

        # Add the bars for each latent size and method
        for latent_size, (j, method) in product(
            range(1, 6), enumerate(results.keys())
        ):

            x = latent_size + 1.1 * (j - 1) * BAR_WIDTH
            height = 100 * results[method][dataset].get(latent_size, np.nan)

            # Add the bar
            axes[1].bar(
                x=x,
                height=height,
                width=BAR_WIDTH,
                label=method,
                color=CBF_COLORS[j],
            )

            # Add the text label
            axes[1].text(
                x=x + 0.01,
                y=0.2 if np.isnan(height) else height + 0.2,
                s='n/a' if np.isnan(height) else f'{height:.2f}',
                color=CBF_COLORS[j],
                ha='center',
                va='bottom',
                size=7,
                rotation=90,
                bbox=dict(fc='white', ec='none', boxstyle='square,pad=0'),
            )

        # Add grid (ugly workaround because `ax.grid()` ignores `zorder`)
        for y in np.arange(1, 8).astype(float):
            axes[1].axhline(
                y=y, ls='--', alpha=0.2, color='k', lw=0.25, zorder=-1
            )

        # Set up limits, ticks, labels, etc.
        set_fontsize(axes[1], 8)
        axes[1].spines[['right', 'top', 'left']].set_visible(False)
        axes[1].tick_params('x', width=0.25, length=2, labelsize=7)
        axes[1].tick_params('y', width=0, length=0, labelsize=7)
        axes[1].set_xlim(0.5, 5.5)
        axes[1].set_ylim(0, 7.5)
        axes[1].set_xticks(range(1, 6))
        axes[1].set_yticks(range(0, 8))
        axes[1].set_xlabel('Number of fitting parameters')
        axes[1].set_ylabel('Mean relative error (%)')

        # ---------------------------------------------------------------------
        # Add a legend to the plot
        # ---------------------------------------------------------------------

        labels = ['Polynomials', 'PCA', 'Our method']

        axes[0].axis('off')
        axes[0].legend(
            handles=[
                Patch(fc=CBF_COLORS[i], ec='none', label=method)
                for i, method in enumerate(labels)
            ],
            loc='center',
            fontsize=7,
            frameon=False,
            labelspacing=1,
            ncols=3,
        )

        # ---------------------------------------------------------------------
        # Save the plot as a PDF
        # ---------------------------------------------------------------------

        fig.tight_layout(pad=0)
        fig.savefig(
            f'mre-comparison-{dataset}.pdf',
            bbox_inches='tight',
            pad_inches=pad_inches,
        )
        plt.close(fig)

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nDone! This took {time.time() - script_start:.1f} seconds.\n')

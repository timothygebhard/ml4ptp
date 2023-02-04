"""
Create a bar plot comparing the median relative error (MRE) of our
method to the MRE of the two baselines for different datasets and
numbers of fitting parameters.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Dict

import time

from matplotlib.patches import Patch

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml4ptp.paths import get_experiments_dir
from ml4ptp.plotting import CBF_COLORS, set_fontsize


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------


class NestedDefaultDict(defaultdict):
    """
    Auxiliary class to create arbitrarily nested defaultdicts.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(NestedDefaultDict, *args, **kwargs)

    def __repr__(self) -> str:
        return repr(dict(self))


def collect_results_our_method() -> Dict[str, NestedDefaultDict]:
    """
    Auxiliary method to collect the results for our method.
    """

    # Define parameter combinations for which to collect data
    datasets = ('pyatmos', 'goyal-2020')
    latent_sizes = (1, 2, 3, 4)
    runs = (0, 1, 2)

    # Loop over the different parameter combinations
    medians = NestedDefaultDict()
    for dataset, latent_size, run in product(datasets, latent_sizes, runs):
        file_path = (
            Path('/Users/timothy/mount/mpicluster/projects/ml4ptp/experiments')
            / dataset
            / 'ce-4'
            / f'latent-size-{latent_size}'
            / 'runs'
            / f'run_{run}'
            / 'results_on_test_set.hdf'
        )
        with h5py.File(file_path, 'r') as hdf_file:
            mre = np.array(hdf_file['mre_refined'])
            medians[dataset][latent_size][run] = np.median(mre)

    # Compute the mean and standard deviation over the runs
    means = NestedDefaultDict()
    stds = NestedDefaultDict()
    for dataset, latent_size in product(datasets, latent_sizes):
        means[dataset][latent_size] = np.mean(
            list(medians[dataset][latent_size].values())
        )
        stds[dataset][latent_size] = np.std(
            list(medians[dataset][latent_size].values())
        )

    return {'medians': medians, 'means': means, 'stds': stds}


def collect_results_pca() -> Dict[str, NestedDefaultDict]:
    """
    Auxiliary method to collect the results for the PCA baseline.
    """

    # Define parameter combinations for which to collect data
    datasets = ('pyatmos', 'goyal-2020')
    latent_sizes = (2, 3, 4, 5)
    runs = (0, 1, 2)

    # Loop over the different parameter combinations
    medians = NestedDefaultDict()
    for dataset, latent_size in product(datasets, latent_sizes):
        file_path = (
            get_experiments_dir()
            / dataset
            / 'pca-baseline'
            / 'results_on_test_set.hdf'
        )
        with h5py.File(file_path, 'r') as hdf_file:
            for run in runs:
                key = f'{latent_size}-principal-components/run-{run}/mre'
                mre = np.array(hdf_file[key])
                medians[dataset][latent_size][run] = np.median(mre)

    # Compute the mean and standard deviation over the runs
    means = NestedDefaultDict()
    stds = NestedDefaultDict()
    for dataset, latent_size in product(datasets, latent_sizes):
        means[dataset][latent_size] = np.mean(
            list(medians[dataset][latent_size].values())
        )
        stds[dataset][latent_size] = np.std(
            list(medians[dataset][latent_size].values())
        )

    return {'medians': medians, 'means': means, 'stds': stds}


def collect_results_polynomial() -> Dict[str, NestedDefaultDict]:
    """
    Auxiliary method to collect the results for the polynomial baseline.
    """

    # Define parameter combinations for which to collect data
    datasets = ('pyatmos', 'goyal-2020')
    latent_sizes = (2, 3, 4, 5)

    # Loop over the different parameter combinations
    means = NestedDefaultDict()
    stds = NestedDefaultDict()
    for dataset, latent_size in product(datasets, latent_sizes):
        file_path = (
            get_experiments_dir()
            / dataset
            / 'polynomial-baseline'
            / 'results_on_test_set.hdf'
        )
        with h5py.File(file_path, 'r') as hdf_file:
            key = f'{latent_size}-fit-parameters/mre'
            mre = np.array(hdf_file[key])
            means[dataset][latent_size] = np.median(mre)
            stds[dataset][latent_size] = np.nan

    return {'means': means, 'stds': stds}


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
    pca = collect_results_pca()
    polynomial = collect_results_polynomial()
    our_method = collect_results_our_method()
    print('Done!', flush=True)

    # Collect all our results into a single pandas dataframe
    print('Creating data frame...', end=' ', flush=True)
    df = pd.DataFrame(
        [
            dict(
                dataset_key=dataset_key,
                dataset_name=dataset_name,
                ls=ls,
                method=method_name,
                mean=data['means'][dataset_key].get(ls, np.nan),
            )
            for method_name, data in (
                ('PCA baseline', pca),
                ('Polynomial baseline', polynomial),
                ('Our method', our_method),
            )
            for dataset_key, dataset_name in (
                ('pyatmos', '\pyatmos'),
                ('goyal-2020', '\goyal'),
            )
            for ls in (1, 2, 3, 4, 5)
        ]
    )
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Create a bar plot
    # -------------------------------------------------------------------------

    # Set default font
    mpl.rcParams['font.sans-serif'] = "Arial"
    mpl.rcParams['font.family'] = "sans-serif"

    print('Creating bar plots...', end=' ', flush=True)

    # Create separate plots for each dataset
    for key in ('pyatmos', 'goyal-2020'):

        # Create the figure
        pad_inches = 0.025
        fig, axes = plt.subplots(
            nrows=2,
            figsize=(8 / 2.54, 4.5 / 2.54),
            sharey='all',
            height_ratios=[0.5, 4],
        )
        width = 0.2

        # Disable the first axis (this will be used for the legend)
        axes[0].axis('off')

        # Prepare main axis. Start by adding a horizontal grid:
        ax = axes[1]
        for y in np.arange(1, 8).astype(float):
            ax.axhline(
                y=y, ls='--', alpha=0.5, color='gray', lw=0.25, zorder=-1
            )

        # Remove spines, set lengths and widths of ticks
        ax.spines[['right', 'top', 'left']].set_visible(False)
        ax.tick_params('x', width=0.25, length=2, labelsize=5.5)
        ax.tick_params('y', width=0, length=0, labelsize=5.5)

        # Set the limits and ticks
        ax.set_xlim(0.5, 5.5)
        ax.set_ylim(0, 7.5)
        ax.set_xticks(range(1, 6))
        ax.set_yticks(range(1, 8))

        # Set the labels and fontsize
        ax.set_xlabel('Number of fitting parameters')
        ax.set_ylabel('Mean relative error (in %)')
        set_fontsize(ax, 6.5)

        # Add the bars for each method
        methods = ['Polynomial baseline', 'PCA baseline', 'Our method']
        for j, method in enumerate(methods):
            # Get the data for the current method
            data = df[(df.dataset_key == key) & (df.method == method)]
            x = data.ls.values.astype(float) + 1.1 * (j - 1) * width
            height = 100 * data['mean'].values.astype(float)

            # Add the bars
            bars = ax.bar(
                x=x,
                height=height,
                width=width,
                label=method,
                color=CBF_COLORS[j],
            )

            # Add the text labels with the MRE values for each bar
            for k in range(len(x)):
                # Define label. NaN = the method was not run.
                if np.isnan(height[k]):
                    y = 0.2
                    s = 'n/a'
                else:
                    y = height[k] + 0.2
                    s = f'{height[k]:.2f}'

                # Add the text label
                t = ax.text(
                    x=x[k] + 0.01,
                    y=y,
                    s=s,
                    color=CBF_COLORS[j],
                    ha='center',
                    va='bottom',
                    size=5,
                    rotation=90,
                )
                t.set_bbox(
                    dict(fc='white', ec='none', boxstyle='square,pad=0')
                )

        # Create a custom legend
        axes[0].legend(
            handles=[
                Patch(fc=CBF_COLORS[i], ec='none', label=method)
                for i, method in enumerate(methods)
            ],
            loc='center left',
            fontsize=5.5,
            frameon=False,
            labelspacing=1,
            ncols=3,
        )

        # Save the plot as a PDF
        fig.tight_layout(pad=0)
        fig.savefig(
            f'mre-comparison-{key}.pdf',
            bbox_inches='tight',
            pad_inches=pad_inches,
        )
        plt.close(fig)

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nDone! This took {time.time() - script_start:.1f} seconds.\n')

"""
Create a table with the mean relative error (MRE).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from collections import defaultdict
from itertools import product
from typing import Any, Dict

import time
import warnings

import h5py
import numpy as np
import pandas as pd

from ml4ptp.paths import get_experiments_dir


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
            get_experiments_dir() / dataset / 'default' /
            f'latent-size-{latent_size}' / 'runs' / f'run_{run}' /
            'results_on_test_set.hdf'
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
            get_experiments_dir() / dataset / 'pca-baseline' /
            'results_on_test_set.hdf'
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
            get_experiments_dir() / dataset / 'polynomial-baseline' /
            'results_on_test_set.hdf'
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

    print('Collecting data...', end=' ', flush=True)
    pca = collect_results_pca()
    polynomial = collect_results_polynomial()
    our_method = collect_results_our_method()
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Format everything nicely as a table
    # -------------------------------------------------------------------------

    # Collect all our results into a single pandas dataframe
    print('Creating data frame...', end=' ', flush=True)
    df = pd.DataFrame(
        [
            dict(
                dataset=dataset_name,
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

    # Reshape the dataframe into a pivot table with the results
    pivot = df.pivot(index='method', columns=['dataset', 'ls'], values='mean')

    # Print the pivot table to the console
    print('\n\nResults:\n', flush=True)
    txt_string = pivot.to_string(
        float_format=lambda x: f'{x:.3f}',
        na_rep='---',
        index_names=False,
    )
    print(txt_string)

    # Create a LaTeX table and print it to the console
    print('\n\nLaTeX code:\n', flush=True)
    with warnings.catch_warnings():

        # pandas wants us to use pivot.style.to_latex() here, but it seems
        # that method does not support all the options that we want to use?
        warnings.simplefilter(action='ignore', category=FutureWarning)

        # Rename the indices to make them look nicer in the LaTeX table
        pivot.index = pivot.index.str.replace('ls', '\# of fitting parameters')
        pivot.index = pivot.index.str.replace('dataset', 'Dataset')

        # Create the LaTeX table
        latex_string = pivot.to_latex(
            float_format=lambda x: f'{x:.3f}',
            na_rep='---',
            index_names=False,
            multicolumn_format='c',
            column_format='l' + 'c' * 5 + 'c' * 5,
            escape=False,
        )

        # Add some more space between some columns
        latex_string = latex_string.replace(
            'lcccccccccc',
            'l @{\hskip 10mm} ccccc @{\hskip 10mm} ccccc',
        )

        # Add \cmidrules and "# fit parameters" to the table
        latex_strings = latex_string.split('\n')
        latex_strings.insert(3, r'\cmidrule{2-6} \cmidrule{7-11}')
        latex_strings[4] = latex_strings[4].replace('{}', '\# fit parameters')
        latex_string = '\n'.join(latex_strings)

    # Print the LaTeX table to the console
    print(latex_string)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nDone! This took {time.time() - script_start:.1f} seconds.\n')

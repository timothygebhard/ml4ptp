"""
Create HDF files for the PyATMOS data set.

This script takes a folder with the full FDL PyATMOS data set downloaded
from the exoplanet archive [1] as an input and collects the PT profiles
as well as other model parameters (e.g., the gas concentrations and
fluxes) from it, splits the data into a training and test set, and
saves each of these in an HDF file that is easy to load and work with.

[1]: https://exoplanetarchive.ipac.caltech.edu/docs/fdl_landing.html
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Union
from warnings import warn

import time

from joblib import Parallel, delayed

import h5py
import numpy as np
import pandas as pd

from ml4ptp.utils import (
    get_number_of_available_cores,
    setup_rich_progress_bar,
)


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def collect_data_for_hash(
    input_dir: Path,
    model_hash: str,
) -> Dict[str, Union[str, np.ndarray]]:
    """
    Collect data of a simulation based on its hash.

    Args:
        input_dir: The main PyATMOS folder.
        model_hash: The hash of the simulation for which
            to collect the data.

    Returns:
        A dictionary of numpy arrays with all the data we collected.
    """

    # Initialize results that will be returned by this function
    results: Dict[str, Union[str, np.ndarray]] = dict(hash=model_hash)

    # Find the directory that corresponds to model for the given hash
    folder = 'Dir_alpha' if not (x := model_hash[0]).isdigit() else f'dir_{x}'
    model_dir = input_dir / folder / model_hash

    # Read in the CSV file that contains the PT profile
    file_path = model_dir / 'parsed_clima_final.csv'
    parsed_clima_final = pd.read_csv(
        filepath_or_buffer=file_path, usecols=['P', 'T', 'ALT', 'CONVEC']
    )

    # Add columns to results dictionary
    results['P'] = parsed_clima_final['P'].values
    results['T'] = parsed_clima_final['T'].values
    results['ALT'] = parsed_clima_final['ALT'].values
    results['CONVEC'] = parsed_clima_final['CONVEC'].values

    # Read in the file with the photochemical mixing ratios
    file_path = model_dir / 'parsed_photochem_mixing_ratios.csv'
    parsed_photochem_mixing_ratios = pd.read_csv(filepath_or_buffer=file_path)

    # Collect the columns that we want to keep
    for key in parsed_photochem_mixing_ratios.keys():
        if key in {'Z', 'ALT', 'Unnamed: 0'}:
            continue
        results[key] = parsed_photochem_mixing_ratios[key]

    return results


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print("\nCREATE HDF FILES FOR PYATMOS DATASET\n", flush=True)

    # -------------------------------------------------------------------------
    # Set up argument parser to get path to input directory
    # -------------------------------------------------------------------------

    # Set up parser, add arguments, and parse them
    parser = ArgumentParser()
    parser.add_argument('--input-dir', required=True, type=str)
    parser.add_argument('--output-dir', type=str, default='./output')
    args = parser.parse_args()

    # Define input directory and ensure that it exists.
    # The input directory needs to contain the raw FDL PyATMOS data set.
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        raise RuntimeError('Given input directory does not exist!')

    # Define output directory and sure that it exists
    print('Creating output directory...', end=' ', flush=True)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(exist_ok=True, parents=True)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Read in summary CSV file
    # -------------------------------------------------------------------------

    # Read in the pyatmos_summary.csv file, which contains the basic input
    # and output parameters of all models, as well as the simulation hashes
    print('Reading in pyatmos_summary.csv ...', end=' ', flush=True)
    file_path = input_dir / 'pyatmos_summary.csv'
    summary_df = pd.read_csv(file_path)
    n_models = len(summary_df)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Drop out-of-distribution simulations
    # -------------------------------------------------------------------------

    # We now drop a single row containing a simulation with a PT profile that
    # is completely out-of-distribution and causes problems during training
    print('Dropping out-of-distribution data...', end=' ', flush=True)
    summary_df = summary_df[
        summary_df['hash'] != 'a1313b02c922a93fac37196a112fe3f8'
    ]
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Collect simulation data (in parallel) and combine data frames
    # -------------------------------------------------------------------------

    # Setup progress bar
    progress_bar = setup_rich_progress_bar()

    # Collect simulation data in parallel; convert to DataFrame
    # Note: The main limitation here seems to be I/O --- reading in about
    # 250k files is going to be somewhat slow, even if you parallelize it.
    print('\nCollecting simulation data in parallel:')
    with progress_bar as p:
        results = Parallel(n_jobs=get_number_of_available_cores())(
            delayed(collect_data_for_hash)(input_dir, model_hash)
            for model_hash in p.track(summary_df.hash)
        )
    results_df = pd.DataFrame(results)
    print('', flush=True)

    # Merge the summary data frame (from pyatmos_summary.csv) with the results
    # data frame (collected from simulation run folders) based on the "hash"
    # column which they both should have in common
    print("Merging data frames...", end=' ', flush=True)
    merged_df = pd.merge(summary_df, results_df, on="hash")
    print('Done!\n', flush=True)

    # Sanity check: There should be a total of 124,314 - 1 atmospheres
    # (The -1 is because we dropped one out-of-distribution atmosphere.)
    if len(merged_df) != 124_314 - 1:
        warn(f"Expexted 124,313 atmospheres, but found {len(merged_df):,}!")

    # -------------------------------------------------------------------------
    # Create indices for training and test
    # -------------------------------------------------------------------------

    # Set up a new random number generator
    rng = np.random.RandomState(seed=42)

    # Create indices and shuffle them randomly
    all_idx = np.arange(0, len(merged_df))
    rng.shuffle(all_idx)

    # Define indices for training and test
    train_idx = all_idx[:100_000]
    test_idx = all_idx[100_000:]

    # -------------------------------------------------------------------------
    # Create HDF files for the training, validation and test data sets
    # -------------------------------------------------------------------------

    print("Saving everything to HDF files:", flush=True)

    for file_name, idx in [
        ('train.hdf', train_idx),
        ('test.hdf', test_idx),
    ]:

        print(f'  Creating {file_name} ...', end=' ', flush=True)

        # Create new HDF file
        file_path = output_dir / file_name
        with h5py.File(file_path, "w") as hdf_file:

            # Create a data set for every key
            for key in merged_df.keys():

                # Determine the data type
                dtype = 'S32' if key == 'hash' else float

                # Select the `key` column and the `idx` rows
                # Using `row_stack()` is necessary to ensure that all data
                # sets have compatible dimensions: Some keys are just a single
                # number (like the surface temperature) while others are a 1D
                # array (i.e., one value per layer). We store everything as a
                # 2D array of shape: (n_profiles, dim), where dim is either 1
                # or 101 or 102 [?] (i.e., the number of atmospheric layers).
                data = np.row_stack(merged_df[key].values[idx])
                data = data.astype(dtype)  # type: ignore

                # Create the data set
                hdf_file.create_dataset(name=key, data=data)

        print("Done!", flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f"\nThis took {time.time() - script_start:.1f} seconds!\n")

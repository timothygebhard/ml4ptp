"""
This script takes a folder with the full FDL PyATMOS data set as an
input and collects the PT profiles as well as other model parameters,
such as the gas concentrations and fluxed, from it to save them in a
single HDF file that is easy to load and work with.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Dict

import os
import time

from joblib import Parallel, delayed
from scipy.interpolate import InterpolatedUnivariateSpline
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

import h5py
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_n_jobs() -> int:
    """
    Get the number cores available to the current process (if possible,
    otherwise return some hard-coded default value).
    """

    try:
        n_jobs = len(os.sched_getaffinity(0))
    except AttributeError:
        n_jobs = 8
    return n_jobs


def collect_data_for_hash(
    input_dir: Path,
    model_hash: str,
) -> Dict[str, np.array]:
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
    results = dict(hash=model_hash)

    # Find the directory that corresponds to model for the given hash
    folder = f'Dir_alpha' if not (x := model_hash[0]).isdigit() else f'dir_{x}'
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

    # Convert the altitude ('Z') in the mixing ratio file from cm to km
    parsed_photochem_mixing_ratios['ALT'] = (
        parsed_photochem_mixing_ratios['Z'] / 100_000
    )

    # Define keys of species which we want to interpolate
    excluded = {'Z', 'ALT', 'Unnamed: 0'}
    keys = sorted(
        _ for _ in parsed_photochem_mixing_ratios.keys() if _ not in excluded
    )

    # Loop over keys and interpolate species to target altitude grid
    for key in keys:

        # Set up an interpolator:
        # We will use a cubic spline for the interpolation (`k=3`) and permit
        # extrapolation (`ext=0`), because the minimum of the target altitude
        # grid is slightly smaller than available grid.
        interpolator = InterpolatedUnivariateSpline(
            x=parsed_photochem_mixing_ratios['ALT'],
            y=parsed_photochem_mixing_ratios[key],
            k=3,
            ext=0,
        )

        # Interpolate to the target altitude grid and store
        results[key] = interpolator(parsed_clima_final['ALT'])

    return results


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print("\nCOLLECT PT PROFILES FOR PYATMOS DATASET\n", flush=True)

    # -------------------------------------------------------------------------
    # Read in summary CSV file
    # -------------------------------------------------------------------------

    # Define path to input directory and check if it exists.
    # The input directory needs to contain the raw FDL PyATMOS data set.
    input_dir = (Path('.') / 'input').resolve()
    if not input_dir.exists():
        raise RuntimeError('input directory does not exist!')

    # Read in the pyatmos_summary.csv file, which contains the basic input
    # and output parameters of all models, as well as the simulation hashes
    print('Reading in pyatmos_summary.csv ...', end=' ', flush=True)
    file_path = input_dir / 'pyatmos_summary.csv'
    summary_df = pd.read_csv(file_path)
    n_models = len(summary_df)
    print('Done!', flush=True)

    # Make sure that the output directory exists
    print('Creating output directory...', end=' ', flush=True)
    output_dir = Path('.', 'output')
    output_dir.mkdir(exist_ok=True)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Collect simulation data (in parallel) and combine data frames
    # -------------------------------------------------------------------------

    # Define progress bar
    progress_bar = Progress(
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )

    # Collect simulation data in parallel; convert to DataFrame
    print('\nCollecting simulation data in parallel:')
    with progress_bar as p:
        results = Parallel(n_jobs=get_n_jobs())(
            delayed(collect_data_for_hash)(input_dir, model_hash)
            for model_hash in p.track(summary_df.hash)
        )
    results_df = pd.DataFrame(results)
    print('')

    # Merge the summary data frame with the results data frame based on the
    # "hash" column which they both should have in common
    print("Merging data frames...", end=' ', flush=True)
    merged_df = pd.merge(summary_df, results_df, on="hash")
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Store everything in an output HDF file
    # -------------------------------------------------------------------------

    print("Writing everything to output HDF file...", end=' ', flush=True)

    # Create new output HDF file
    file_path = output_dir / 'pyatmos.hdf'
    with h5py.File(file_path, "w") as hdf_file:

        # Loop over all keys we have collected and create data sets for it
        for key in merged_df.keys():
            dtype = 'S25' if key == 'hash' else float
            hdf_file.create_dataset(
                name=key,
                data=np.row_stack(merged_df[key].values).astype(dtype),
            )

    print("Done!", flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f"\nThis took {time.time() - script_start:.1f} seconds!\n")

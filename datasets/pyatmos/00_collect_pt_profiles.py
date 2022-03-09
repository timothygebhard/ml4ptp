"""
This script takes a folder with the full FDL PyATMOS data set as an
input and collects the PT profiles as well as other model parameters,
such as the gas concentrations and fluxed, from it to save them in a
single HDF file that is easy to load and work with.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
import sys
from pathlib import Path

import time

from tqdm.auto import tqdm

import h5py
import numpy as np
import pandas as pd


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
    input_dir = (Path('..') / 'input').resolve()
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
    output_dir = Path('..', 'output')
    output_dir.mkdir(exist_ok=True)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Collect pressures, temperatures and altitudes
    # -------------------------------------------------------------------------

    # Initialize lists for the arrays / vectors that we will collect
    pressures = []
    temperatures = []
    altitudes = []
    convective = []

    # Loop over all rows in the summary CSV file and collect P, T and ALT
    print("\nCollecting PT profiles from HDF files:", flush=True)
    for _, row in tqdm(summary_df.iterrows(), ncols=80, total=n_models):

        # Find the directory that belongs to the current row
        if row["hash"][0].isdigit():
            model_dir = input_dir / f'dir_{row["hash"][0]}' / row['hash']
        else:
            model_dir = input_dir / f'Dir_alpha' / row['hash']

        # Read in the CSV file that contains the PT profile
        file_path = model_dir / 'parsed_clima_final.csv'
        model_df = pd.read_csv(file_path, usecols=['P', 'T', 'ALT', 'CONVEC'])

        # Get the pressure, temperatures, etc. for this model (= vectors)
        pressures.append(model_df['P'].values)
        temperatures.append(model_df['T'].values)
        altitudes.append(model_df['ALT'].values)
        convective.append(model_df['CONVEC'].values)

    # -------------------------------------------------------------------------
    # Store everything in an output HDF file
    # -------------------------------------------------------------------------

    print("\nWriting everything to output HDF file...", end=' ', flush=True)

    # Create an output HDF file and store everything
    file_path = output_dir / 'pyatmos.hdf'
    with h5py.File(file_path, "w") as hdf_file:

        # First, store the array-valued columns that we collected
        hdf_file.create_dataset(name='P', data=np.asarray(pressures))
        hdf_file.create_dataset(name='T', data=np.asarray(temperatures))
        hdf_file.create_dataset(name='ALT', data=np.asarray(altitudes))
        hdf_file.create_dataset(name='CONVEC', data=np.asarray(convective))

        # Then, store all the columns from the pyatmos_summary.csv file
        for key in summary_df.keys():
            hdf_file.create_dataset(name=key, data=summary_df[key].values)

    print("Done!", flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f"\nThis took {time.time() - script_start:.1f} seconds!\n")

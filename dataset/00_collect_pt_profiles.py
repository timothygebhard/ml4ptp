"""
This script takes a folder with the full FDL PyATMOS data set as an
input and collects the PT profiles from it to save them in a single
HDF file. Specifically, this HDF file will contain one group for each
PyATMOS simulation (whose name is the hash of that simulation), and
each group will contain a data set with the altitudes ("ALT"), the
pressures ("P") and the temperatures ("T").
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import time

from tqdm.auto import tqdm

import h5py
import pandas as pd


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print("\nPREPARE DATASET\n", flush=True)

    # -------------------------------------------------------------------------
    # Collect all PT profiles and associated information in an HDF file
    # -------------------------------------------------------------------------

    # Define path to input directory and check if it exists.
    # The input directory needs to contain the raw FDL PyATMOS data set.
    input_dir = (Path('.') / 'input').resolve()
    if not input_dir.exists():
        raise RuntimeError('input directory does not exist!')

    # Read in the pyatmos_summary.csv file, which contains the basic input and
    # output parameters of all models, as well as the simulation hashes
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

    # Create an output HDF file
    file_path = output_dir / 'pt_profiles.hdf'
    with h5py.File(file_path, "w") as hdf_file:

        print("\nCollecting PT profiles in HDF file:", flush=True)
        for _, row in tqdm(summary_df.iterrows(), ncols=80, total=n_models):

            # Find the directory that belongs to the current row
            if row["hash"][0].isdigit():
                model_dir = input_dir / f'dir_{row["hash"][0]}' / row['hash']
            else:
                model_dir = input_dir / f'Dir_alpha' / row['hash']

            # Read in the CSV file that contains the PT profile
            file_path = model_dir / 'parsed_clima_final.csv'
            model_df = pd.read_csv(file_path)

            # Create a new group in the HDF file for this model
            group = hdf_file.create_group(name=row['hash'])

            # In this group, store the PT profile (and the altitude)
            group.create_dataset(name='P', data=model_df['P'].values)
            group.create_dataset(name='T', data=model_df['T'].values)
            group.create_dataset(name='ALT', data=model_df['ALT'].values)

            # Additionally, store basic model parameters as attributes
            for key, value in row.items():
                if key == 'hash':
                    continue
                group.attrs[key] = value

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f"\nThis took {time.time() - script_start:.1f} seconds!\n")

"""
Quick auxiliary script to get the out-of-distribution PT profile that
we manually remove from the PyATMOS dataset as a separate HDF file.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import time

import h5py
import numpy as np

from ml4ptp.paths import get_datasets_dir
from ml4ptp.pyatmos import collect_data_for_hash


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print("\nCOLLECT OUT-OF-DISTRIBUTION PROFILE\n", flush=True)

    # -------------------------------------------------------------------------
    # Collect out-of-distribution profile
    # -------------------------------------------------------------------------

    # Define the hash of the out-of-distribution simulation
    model_hash = 'a1313b02c922a93fac37196a112fe3f8'

    # Define the input and output directory
    input_dir = get_datasets_dir() / 'pyatmos' / 'input'
    output_dir = get_datasets_dir() / 'pyatmos' / 'output'

    # Collect the data for this simulation
    print('Collecting data for o.o.d. simulation...', end=' ', flush=True)
    df = collect_data_for_hash(input_dir=input_dir, model_hash=model_hash)
    print('Done!', flush=True)

    # Save the data to a file
    print('Saving data to file...', end=' ', flush=True)
    file_path = output_dir / 'ood.hdf'
    with h5py.File(file_path, 'w') as hdf_file:

        # Create a data set for every key and store it with the right dtype
        for key in df.keys():

            dtype = 'S32' if key == 'hash' else float
            data = np.array([df[key]]).astype(dtype)  # type: ignore
            hdf_file.create_dataset(name=key, data=data)

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f"\nThis took {time.time() - script_start:.1f} seconds!\n")

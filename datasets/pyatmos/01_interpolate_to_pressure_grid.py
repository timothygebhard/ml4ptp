"""
Load all PT-profiles and map them to a single pressure grid.
This is useful for methods such as VAEs, where we are working with fixed
temperature vectors that we need to match to pressure values.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import time

from rich.progress import track
from scipy.interpolate import interp1d

import h5py
import numpy as np

from ml4ptp.config import get_dataset_dir


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print("\nINTERPOLATE TO PRESSURE GRID\n", flush=True)

    # -------------------------------------------------------------------------
    # Determine the pressure grid
    # -------------------------------------------------------------------------

    # Define path to the file that holds all the (raw) PT profiles
    file_path = get_dataset_dir() / 'output' / 'pyatmos.hdf'
    if not file_path.exists():
        raise FileNotFoundError(f'{file_path} does not exist!')

    # Open file and compute minimum and maximum of each pressure grid
    print('Determining pressure grid...', end=' ', flush=True)
    with h5py.File(file_path, 'r') as hdf_file:
        pressures = np.array(hdf_file['P'])
        minima = np.min(pressures, axis=1)
        maxima = np.max(pressures, axis=1)

    # Determine interval that is covered by *all* profiles
    p_min = max(minima)
    p_max = min(maxima)

    print('Done!', flush=True)
    print('', flush=True)
    print('p_min:', p_min, flush=True)
    print('p_max:', p_max, flush=True)
    print('', flush=True)

    # Define a pressure grid for interpolation
    # Note: It is *not* obvious how to do this. The PyATMOS data are based on
    # uniformly spaced altitude grids, but the relationship between pressure
    # and altitude is, in general, not straightforward and depends, e.g., on
    # the planet's gravity...
    # Sampling the pressures uniformly in log-space is a simplification that
    # we make to get *something* to work with...
    pressure_grid = np.geomspace(p_min, p_max, 100)

    # -------------------------------------------------------------------------
    # Copy the original pyatmos.hdf file, but remove P and T
    # -------------------------------------------------------------------------

    # Define source (src) and destination (dst) files for copying datasets
    src_file_path = get_dataset_dir() / 'output' / 'pyatmos.hdf'
    dst_file_path = get_dataset_dir() / 'output' / 't_for_fixed_p.hdf'

    with h5py.File(src_file_path, 'r') as src_file:
        with h5py.File(dst_file_path, 'w') as dst_file:

            # Loop over keys in source file to copy them
            for key in src_file.keys():

                # Skip altitudes, pressures and temperatures
                if key in ('ALT', 'P', 'T'):
                    continue

                # Copy the other datasets from source to destination
                print(f'Copying {key}...', end=' ', flush=True)
                src_file.copy(source=key, dest=dst_file)
                print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Interpolate temperatures to pressure_grid
    # -------------------------------------------------------------------------

    # Keep track of interpolated temperatures
    temperatures = []

    # Open the source file, loop over PT arrays and interpolate to grid
    print('\nInterpolating PT profiles:', flush=True)
    with h5py.File(file_path.as_posix(), 'r') as hdf_file:
        for idx in track(range(len(hdf_file['P'])), description=''):

            # Get the values for the pressure and temperature
            pressure = np.array(hdf_file['P'][idx])
            temperature = np.array(hdf_file['T'][idx])

            # Set up an interpolator (use linear interpolation for now)
            interpolator = interp1d(x=pressure, y=temperature, kind='linear')

            # Evaluate the temperatures on our grid and store the result
            temperature = interpolator(pressure_grid)
            temperatures.append(temperature)

    # -------------------------------------------------------------------------
    # Save pressure_grid and interpolated temperatures to output HDF file
    # -------------------------------------------------------------------------

    # Store the interpolated temperatures as an HDF file
    print('\nSaving interpolated temperatures...', end=' ', flush=True)
    with h5py.File(dst_file_path, 'a') as hdf_file:

        hdf_file.create_dataset(
            name='pressure_grid',
            data=pressure_grid,
            dtype=np.float32,
        )
        hdf_file.create_dataset(
            name='temperatures',
            data=np.array(temperatures),
            dtype=np.float32,
        )

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f"\nThis took {time.time() - script_start:.1f} seconds!\n")

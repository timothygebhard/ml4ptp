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

    # We need to open the file, loop over all its contents (= all simulations)
    # and keep track of the respective minimum and maximum pressures, because
    # ultimately we want a grid that fully overlaps with *all* simulations.

    # Define path to the file that holds all the (raw) PT profiles
    file_path = get_dataset_dir() / 'output' / 'pt_profiles.hdf'
    if not file_path.exists():
        raise FileNotFoundError(f'{file_path} does not exist!')

    # Keep track of the pressure values
    p_min = -np.infty
    p_max = np.infty

    # Open the file, loop over all simulations, and update min / max pressure
    print('Finding limits for pressure grid:')
    with h5py.File(file_path, 'r') as hdf_file:
        for key in track(hdf_file.keys(), description=''):
            pressures = hdf_file[key]['P']
            p_min = max(p_min, min(pressures))
            p_max = min(p_max, max(pressures))

    print('')
    print('p_min:', p_min)
    print('p_max:', p_max)
    print('')

    # Define a pressure grid for interpolation
    # Note: It is *not* obvious how to do this. The PyATMOS data are based on
    # uniformly spaced altitude grids, but the relationship between pressure
    # and altitude is, in general, not straightforward and depends, e.g., on
    # the planet's gravity...
    # Sampling the pressures uniformly in log-space is a simplification that
    # we make to get *something* to work with...
    pressure_grid = np.geomspace(p_min, p_max, 100)

    # Store the (interpolated) temperatures and hashes
    temperatures = []
    hashes = []

    # Read in the HDF file and interpolate the altitude-temperature profiles
    print('Interpolating PT profiles:', flush=True)
    with h5py.File(file_path.as_posix(), 'r') as hdf_file:
        for key in track(hdf_file.keys(), description=''):

            # Get the values for the pressure and temperature
            pressure = np.array(hdf_file[key]['P'])
            temperature = np.array(hdf_file[key]['T'])

            # Set up an interpolator (use linear interpolation for now)
            interpolator = interp1d(x=pressure, y=temperature, kind='linear')

            # Evaluate the temperatures on our grid and store the result
            temperature = interpolator(pressure_grid)
            temperatures.append(temperature)

            # Store the hash of the simulation
            hashes.append(key)

    # Store the interpolated temperatures as an HDF file
    print('Saving results to HDF file...', end=' ', flush=True)
    file_path = get_dataset_dir() / 'output' / 't_for_fixed_p.hdf'
    with h5py.File(file_path.as_posix(), 'w') as hdf_file:
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
        hdf_file.create_dataset(
            name='hashes',
            data=np.array(hashes, dtype='S25'),
            dtype='S25',
            
        )
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f"\nThis took {time.time() - script_start:.1f} seconds!\n")

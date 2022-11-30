"""
Create HDF files for the goyal-2020 data set.

This script takes a the folders downloaded from the Google Drive [1] and
collects the PT profiles as well as other model parameters (namely, the
chemical abundances and emission and transmission spectra) from them,
splits the data into a training and test set, and saves each of these
in an HDF file that is easy to load and work with.

[1]: https://drive.google.com/drive/folders/1zCCe6HICuK2nLgnYJFal7W4lyunjU4JE
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import re
import time

from joblib import delayed, Parallel

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

def get_column_names_of_abundances(utilities_dir: Path) -> List[str]:
    """
    Load names of the column in chemical abundances files.
    """

    # Read in the column names from *.txt file
    file_path = utilities_dir / 'chemistry_files_column_names.txt'
    column_data = pd.read_csv(
        filepath_or_buffer=file_path,
        sep='  ',
        engine='python',
        names=['idx', 'name'],
    )

    return list(column_data.name.values.tolist())


def get_identifier(file_name: str) -> str:
    """
    Extract the identifier (= "merge key") from a given file name.

    Example:
    If the file name is "pt-eqpt_XO-2_0.75_+2.3_0.70_model.txt", then
    the identifier is "XO-2_0.75_+2.3_0.70", i.e., a string containing
    the planet and the recirculation factor, the log of the metallicity,
    and the C/O ratio.
    """

    # Define regex and search for identifier
    pattern = r"^(.*eqpt_)(?P<identifier>.*)(_model.txt)$"
    result = re.search(pattern=pattern, string=file_name)

    # If no identifier is found, raise an error
    if (result is None) or (result.group('identifier') is None):
        raise ValueError(f"No valid merge key found in {file_name}!")

    return result.group('identifier')


def get_planet_and_stellar_radius(file_path: Path) -> Tuple[float, float]:
    """
    Get the planet and stellar radius from the first line of the file
    at the given `file_path` (needed for emission spectra).
    """

    # Read in the first line of the given file
    with open(file_path, 'r') as txt_file:
        line = txt_file.readline().strip()

    # Parse the header to get the planetary and stellar radius
    pattern = r"^.*Rp_TOA = (?P<Rp_TOA>(\d|\.)*)\s*Rs = (?P<Rs>(\d|\.)*)$"
    result = re.search(pattern=pattern, string=line)

    # If we found parameters, return them; otherwise, return defaults
    if result is not None:
        planet_radius = float(result.group('Rp_TOA'))
        stellar_radius = float(result.group('Rs'))
    else:
        planet_radius = np.nan
        stellar_radius = np.nan

    return planet_radius, stellar_radius


def apply_function_to_dir(
    target_dir: Path,
    func: Callable[[Path, Any], dict],
    args: Any = (),
) -> pd.DataFrame:
    """
    Auxiliary function to loop over all the text files in a directory
    (and its subdirectories) and apply the given function to them.
    """

    # Get all the *.txt file in the directory's subdirectories
    txt_files = sorted(list(target_dir.glob('*/*.txt')))

    # Process all *.tar files in parallel
    with setup_rich_progress_bar() as progress_bar:
        results = Parallel(n_jobs=get_number_of_available_cores())(
            delayed(func)(tar_file_path, *args)
            for tar_file_path in progress_bar.track(txt_files)
        )

    return pd.DataFrame(results)


def read_pt_profile(file_path: Path, *_: Any) -> dict:
    """
    Read in a PT profile from a *.txt file.
    """

    # Read in the target file using numpy
    content = np.loadtxt(file_path)

    # Store dict with data for this simulation
    data = dict(
        identifier=get_identifier(file_path.name),
        pressure=content[:, 0],
        temperature=content[:, 1],
    )

    return data


def read_chemical_abundances(file_path: Path, column_names: List[str]) -> dict:
    """
    Read in the chemical abundances from a *.txt file.
    """

    # Read in the target file using numpy
    content = np.loadtxt(file_path)

    # Initialize return dictionary
    data: Dict[str, Any] = dict(identifier=get_identifier(file_path.name))

    # Handle missing columns, if needed.
    # Background: There seems to be an issue for HAT-P-25, which is missing
    # the columns for Mg2SiO4(s) and Mg2SiO4(l).
    n_cols = len(column_names)
    new_column_names = column_names.copy()
    if 'HAT-P-25' in file_path.as_posix() and content.shape[1] == n_cols - 2:
        data['Mg2SiO4(s)'] = np.full(content.shape[0], np.nan)
        data['Mg2SiO4(l)'] = np.full(content.shape[0], np.nan)
        new_column_names.remove('Mg2SiO4(s)')
        new_column_names.remove('Mg2SiO4(l)')

    # Loop over "safe" column names and read them from
    for i, column_name in enumerate(new_column_names):
        data[column_name] = content[:, i]

    return data


def read_emission_spectrum(file_path: Path, *_: Any) -> dict:
    """
    Read in an emission spectrum from a *.txt file.
    """

    # Get the planet and the stellar radius from the first line of the file
    planet_radius, stellar_radius = get_planet_and_stellar_radius(file_path)

    # Read in the rest of the file using numpy
    content = np.loadtxt(file_path, skiprows=1)

    # Store dict with data for this simulation
    data = dict(
        identifier=get_identifier(file_path.name),
        emission_wavelength=content[:, 0],
        stellar_flux=content[:, 1],
        planetary_flux=content[:, 2],
        planet_radius=planet_radius,
        stellar_radius=stellar_radius,
    )

    return data


def read_transmission_spectrum(file_path: Path, *_: Any) -> dict:
    """
    Read in a transmission spectrum from a *.txt file.
    """

    # Read in the target file using numpy
    content = np.loadtxt(file_path)

    # Store dict with data for this simulation
    data = dict(
        identifier=get_identifier(file_path.name),
        transmission_wavelength=content[:, 0],
        transit_depth=content[:, 1],
    )

    return data


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print("\nCREATE HDF FILES FOR GOYAL-2020 DATA SET\n", flush=True)

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
    print('Done!\n', flush=True)

    # Define other directories
    pt_profiles_dir = input_dir / 'pressure-temperature-profiles'
    chemical_abundances_dir = input_dir / 'chemical-abundances'
    emission_spectra_dir = input_dir / 'emission-spectra'
    transmission_spectra_dir = input_dir / 'transmission-spectra'
    utilities_dir = input_dir / 'utilities'

    # -------------------------------------------------------------------------
    # Read in all data
    # -------------------------------------------------------------------------

    print('Reading in PT profiles:', flush=True)
    pt_profiles_df = apply_function_to_dir(
        func=read_pt_profile,
        target_dir=pt_profiles_dir,
    )
    print(f'Done! (shape: {pt_profiles_df.shape})', flush=True)

    print('\nReading in chemical abundances:', flush=True)
    column_names = get_column_names_of_abundances(utilities_dir)
    chemical_abundances_df = apply_function_to_dir(
        target_dir=chemical_abundances_dir,
        func=read_chemical_abundances,
        args=(column_names,),
    )
    print(f'Done! (shape: {chemical_abundances_df.shape})', flush=True)

    print('\nReading in emission spectra:', flush=True)
    emission_spectra_df = apply_function_to_dir(
        func=read_emission_spectrum,
        target_dir=emission_spectra_dir,
    )
    print(f'Done! (shape: {emission_spectra_df.shape})', flush=True)

    print('\nReading in transmission spectra:', flush=True)
    transmission_spectra_df = apply_function_to_dir(
        func=read_transmission_spectrum,
        target_dir=transmission_spectra_dir,
    )
    print(f'Done! (shape: {transmission_spectra_df.shape})', flush=True)

    # -------------------------------------------------------------------------
    # Merge data frames
    # -------------------------------------------------------------------------

    # Use `how='inner'` to merge on the *intersection* of the "identifier"
    # fields, that is, only keep rows for which we have all of the following:
    # PT profiles, chemical abundances, emission and transmission spectra.
    # (It seems like there are 3 simulations for which no emission spectrum
    # exists; these are consequently discarded.)
    print('\nMerging data frames...', end=' ', flush=True)
    df = pt_profiles_df
    df = pd.merge(df, chemical_abundances_df, on='identifier', how='inner')
    df = pd.merge(df, emission_spectra_df, on='identifier', how='inner')
    df = pd.merge(df, transmission_spectra_df, on='identifier', how='inner')
    print(f'Done! (shape: {df.shape})', flush=True)

    # -------------------------------------------------------------------------
    # Create indices for training, validation and test
    # -------------------------------------------------------------------------

    print('Creating random indices for split...', end=' ', flush=True)

    # Set up a new random number generator
    rng = np.random.RandomState(seed=42)

    # Create indices and shuffle them randomly
    all_idx = np.arange(0, len(df))
    rng.shuffle(all_idx)

    # Define indices for training, validation and test
    train_idx = all_idx[:10_000]
    test_idx = all_idx[10_000:]

    print('Done!\n', flush=True)

    # -------------------------------------------------------------------------
    # Create HDF files for the training, validation and test data sets
    # -------------------------------------------------------------------------

    output_dir.mkdir(exist_ok=True)

    print("Saving everything to HDF files:", flush=True)

    for file_name, idx in [
        ('train.hdf', train_idx),
        ('test.hdf', test_idx),
    ]:

        print(f'  Creating {file_name} ...', end=' ', flush=True)

        # Create new HDF file
        file_path = output_dir / file_name
        with h5py.File(file_path, "w") as hdf_file:

            # Note: We get the keys from the individual data frames (e.g.,
            # `pt_profiles_df`), but we take the data from the merged data
            # frame to make sure that the data is correctly aligned.
            # Using `np.row_stack()` is needed to get "compatible dimensions"
            # for all keys (some datasets are 1D, some are 2D).

            # Store identifier
            hdf_file.create_dataset(
                name='identifier',
                data=np.row_stack(df['identifier'].values)[idx].astype('S'),
            )

            # Store PT profiles
            group = hdf_file.create_group(name='pt_profiles')
            keys = sorted(set(pt_profiles_df.keys()) - {'identifier'})
            for key in keys:
                data = np.row_stack(df[key].values)[idx].astype(float)
                group.create_dataset(name=key, data=data)

            # Store chemical abundances
            group = hdf_file.create_group(name='chemical_abundances')
            keys = sorted(set(chemical_abundances_df.keys()) - {'identifier'})
            for key in keys:
                data = np.row_stack(df[key].values)[idx].astype(float)
                group.create_dataset(name=key, data=data)

            # Store emission spectra
            group = hdf_file.create_group(name='emission_spectra')
            keys = sorted(set(emission_spectra_df.keys()) - {'identifier'})
            for key in keys:
                data = np.row_stack(df[key].values)[idx].astype(float)
                group.create_dataset(name=key, data=data)

            # Store transmission spectra
            group = hdf_file.create_group(name='transmission_spectra')
            keys = sorted(set(transmission_spectra_df.keys()) - {'identifier'})
            for key in keys:
                data = np.row_stack(df[key].values)[idx].astype(float)
                group.create_dataset(name=key, data=data)

        print("Done!", flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f"\nThis took {time.time() - script_start:.1f} seconds!\n")

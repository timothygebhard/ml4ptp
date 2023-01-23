"""
Methods for preparing the PyATMOS dataset.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# DEFINITIONS
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

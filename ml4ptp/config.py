"""
Functions and utilities related to the general configuration.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import ml4ptp


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_ml4ptp_dir() -> Path:
    return Path(ml4ptp.__file__).parent.parent


def get_dataset_dir() -> Path:
    return get_ml4ptp_dir() / 'dataset'


def get_experiments_dir() -> Path:
    return get_ml4ptp_dir() / 'experiments'

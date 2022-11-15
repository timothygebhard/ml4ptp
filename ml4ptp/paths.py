"""
Methods related to paths.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from os.path import expandvars as expandvars_
from pathlib import Path


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def expandvars(path: Path) -> Path:
    """
    Take a `path`, expand the environmental variables, and return it.

    This function is a thin wrapper around :func:`os.path.expandvars`
    which only supports string arguments (and returns strings).

    Args:
        path: A path that may contain environmental variables, e.g.,
            ``$SOME_DIR``.

    Returns:
        The original ``path`` with environmental variables expanded.
    """

    return Path(expandvars_(path.as_posix()))


def get_datasets_dir() -> Path:
    """
    Return the path to the directory containing the datasets.

    Returns:
        The path to the directory containing the datasets.
    """

    datasets_dir = expandvars(Path('$ML4PTP_DATASETS_DIR'))

    if not datasets_dir.exists():
        raise FileNotFoundError(
            f'The datasets directory does not exist: {datasets_dir}'
        )

    return datasets_dir

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

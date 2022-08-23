"""
General utility functions.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Union

import os

from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

import torch


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_number_of_available_cores(default: int = 8) -> int:
    """
    Get the number cores available to the current process (if possible,
    otherwise return the given default value).

    Args:
        default: The default number of cores that is returned if
            ``os.sched_getaffinity()`` is not available.

    Returns:
        The number of cores available to the current process.
    """

    try:
        return len(os.sched_getaffinity(0))  # type: ignore
    except AttributeError:
        return default


def resolve_gpus(gpus: Union[str, int]) -> int:
    """
    Auxiliary function to resolve the `gpus` argument for the trainer:
    The argument "auto" should give the number of GPUs if CUDA is
    available, and 0 otherwise.
    """

    # If we explicitly specify the number of GPUs, do not overwrite it
    if gpus != 'auto':
        return int(gpus)

    # Otherwise, return either the number of available GPUs, or 0 (for CPU)
    if torch.cuda.is_available():  # pragma: no cover
        return torch.cuda.device_count()
    return 0


def setup_rich_progress_bar() -> Progress:
    """
    Set up a customized progress bar based on ``rich.progress``.

    Returns:
        A progress bar object that can be used as follows:

            with progress_bar as p:
                ...
                for _ in p.track(iterable)
                ...
    """

    return Progress(
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )


def tensor_to_str(tensor: torch.Tensor, n_digits: int = 3) -> str:

    dummy = ', '.join([str(round(float(_), n_digits)) for _ in tensor])
    return f'({dummy})'

"""
General utility functions.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Tuple, Union

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


def get_run_dir(experiment_dir: Path) -> Tuple[Path, Path]:
    """
    For a given `experiment_dir`, return the `runs` subdirectory as
    well as the next / current `run_<X>` directory inside `runs`.

    Args:
        experiment_dir: Path to an experiment directory.

    Returns:
        A tuple of ``Path`` objects, `runs_dir` and `run_dir`.
    """

    # Get the folder where we keep different runs
    runs_dir = experiment_dir / 'runs'
    runs_dir.mkdir(exist_ok=True)

    # Get folder for the *current* run: Check runs_dir for directories that
    # start with 'run_', get all the run numbers, find the maximum, and add 1
    previous_runs = [int(_.name.split('_')[1]) for _ in runs_dir.glob('run_*')]
    number = 0 if not previous_runs else max(previous_runs) + 1
    run_dir = runs_dir / f'run_{number}'
    run_dir.mkdir(exist_ok=True)

    return runs_dir, run_dir


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
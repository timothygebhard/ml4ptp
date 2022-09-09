"""
Create plot of a PT profile from test set.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import List

import argparse
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np

from ml4ptp.config import load_config
from ml4ptp.paths import expandvars
from ml4ptp.plotting import plot_pt_profile


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

def find_run_dirs_with_results(experiment_dir: Path) -> List[Path]:

    runs_dir = experiment_dir / 'runs'
    run_dirs = filter(
        lambda _: (_ / 'results_on_test_set.hdf').exists(),
        runs_dir.glob('run_*')
    )
    return sorted(run_dirs)


def get_cli_args() -> argparse.Namespace:

    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment-dir',
        required=True,
        help='Path to the experiment directory with the config.yaml',
    )
    parser.add_argument(
        '--idx',
        type=int,
        required=True,
        help='Index of the PT profile in the test set.',
    )
    parser.add_argument(
        '--xlim',
        nargs='+',
        type=float,
        default=None,
        help='Limits for the x-axis (temperature in K).',
    )
    parser.add_argument(
        '--ylim',
        nargs='+',
        type=float,
        default=None,
        help='Limits for the y-axis (log10 of pressure in bar).',
    )
    args = parser.parse_args()

    return args


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nCREATE PLOTS FOR EXPERIMENT\n', flush=True)

    # -------------------------------------------------------------------------
    # Get experiment dir and load configuration file
    # -------------------------------------------------------------------------

    # Get CLI arguments, define shortcuts
    args = get_cli_args()
    idx = args.idx

    # Load experiment configuration from YAML
    print('Loading experiment configuration...', end=' ', flush=True)
    experiment_dir = expandvars(Path(args.experiment_dir)).resolve()
    config = load_config(experiment_dir / 'config.yaml')
    plot_config = config['plotting']['pt_profile']
    latent_size = int(config['model']['decoder']['parameters']['latent_size'])
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Determine plot limits
    # -------------------------------------------------------------------------

    # Define limits for x-axis
    if args.xlim is not None:
        xlim = (args.xlim[0], args.xlim[1])
    else:
        xlim = (plot_config['min_T'], plot_config['max_T'])

    # Define limits for y-axis
    if args.ylim is not None:
        ylim = (args.ylim[0], args.ylim[1])
    else:
        ylim = (plot_config['min_log_P'], plot_config['max_log_P'])

    # -------------------------------------------------------------------------
    # Collect data from result HDF files
    # -------------------------------------------------------------------------

    print('Collecting data from result HDF files...', end=' ', flush=True)

    # Find all run directories that have usable results
    run_dirs = find_run_dirs_with_results(experiment_dir)

    # Store profiles that we read from result HDF files
    pt_profiles = []

    # Loop over different file paths (usually those are different runs) and
    # load log_P, T_true and (refined) T_pred from the current HDF file
    file_paths = [_ / 'results_on_test_set.hdf' for _ in run_dirs]
    for i, file_path in enumerate(file_paths):
        with h5py.File(file_path, 'r') as hdf_file:
            log_P = np.array(hdf_file['log_P'][idx]).squeeze()
            T_true = np.array(hdf_file['T_true'][idx]).squeeze()
            T_pred = np.array(hdf_file['T_pred_optimal'][idx]).squeeze()
            pt_profiles.append((log_P, T_true, T_pred))

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Create and save plot
    # -------------------------------------------------------------------------

    # Create the plot of the PT profile
    print('Plotting the PT profile...', end=' ', flush=True)
    fig, ax = plot_pt_profile(
        pt_profiles=pt_profiles,
        latent_size=latent_size,
        xlim=xlim,
        ylim=ylim,
    )
    print('Done!', flush=True)
    print('Saving plot to PDF...', end=' ', flush=True)

    # Create plots folder
    plots_dir = experiment_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Save the PT profile plot
    file_path = plots_dir / f'pt-profile__{idx=}.pdf'
    plt.savefig(file_path, facecolor='white', transparent=False)

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nDone! This took {time.time() - script_start:.1f} seconds.\n')

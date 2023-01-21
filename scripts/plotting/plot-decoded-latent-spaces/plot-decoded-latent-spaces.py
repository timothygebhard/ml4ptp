"""
Create plots of the individual latent spaces, as well as a grid plot
showing the decoded profiles, for an experiment consisting of several
runs with the same settings but different random seeds.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Dict

import argparse
import time

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ml4ptp.onnx import ONNXDecoder
from ml4ptp.paths import expandvars
from ml4ptp.plotting import set_fontsize, disable_ticks


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

def get_cli_arguments() -> argparse.Namespace:
    """
    Get CLI arguments.
    """

    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment-dir',
        default='$ML4PTP_EXPERIMENTS_DIR/goyal-2020/same-data-different-init',
        help='Path to the experiment directory.',
    )
    parser.add_argument(
        '--runs',
        default=[0, 1, 2],
        nargs='+',
        help='List of runs to plot.',
    )
    args = parser.parse_args()

    return args


def plot_latent_space(z: np.ndarray, random_seed: int, color: str) -> None:

    # Create a new figure
    pad_inches = 0.025
    fig, ax = plt.subplots(
        figsize=(
            3.8 / 2.54 - 2 * pad_inches,
            3.8 / 2.54 - 2 * pad_inches,
        ),
    )

    # Plot the scatter plot
    ax.scatter(
        z[:, 0],
        z[:, 1],
        s=1,
        color=color,
        marker='.',
        edgecolors='none',
        rasterized=True,
    )

    # Add title and labels
    ax.set_title(label=f'Random seed {random_seed}', fontweight='bold')
    ax.set_xlabel(r'$z_1$')
    ax.set_ylabel(r'$z_2$')

    # Set linewidth of the frame
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.25)
        ax.xaxis.set_tick_params(width=0.25)
        ax.yaxis.set_tick_params(width=0.25)

    # More plot options
    ax.set_aspect('equal')
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_yticks([-4, -2, 0, 2, 4])
    set_fontsize(ax, 5.5)
    ax.xaxis.label.set_fontsize(6.5)
    ax.yaxis.label.set_fontsize(6.5)
    ax.tick_params('both', length=2, width=0.25, which='major')

    # Save the figure
    file_path = Path(f'latent-space-{random_seed}.pdf')
    fig.tight_layout(pad=0)
    fig.savefig(file_path, dpi=600, bbox_inches='tight', pad_inches=pad_inches)


def plot_decoded_profiles(
    decoders: Dict[int, ONNXDecoder],
) -> None:

    # Define grid and log_P
    grid_size = 7
    grid = np.linspace(-3, 3, grid_size)
    log_P = np.linspace(-6, 2, 100)

    # Create a new figure
    pad_inches = 0.025
    fig, axes = plt.subplots(
        nrows=grid_size,
        ncols=grid_size,
        figsize=(
            12 / 2.54 - 2 * pad_inches,
            12 / 2.54 - 2 * pad_inches,
        ),
    )
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    # Loop over the different decoders
    for i, random_seed in enumerate(decoders.keys()):

        # Get the decoder
        decoder = decoders[random_seed]

        # Loop over the grid
        for row, z_2 in enumerate(grid[::-1]):
            for col, z_1 in enumerate(grid):

                # Get the decoded profile
                z = np.array([[z_1, z_2]])
                T_pred = decoder(z=z, log_P=log_P.reshape(1, -1)).squeeze()

                # Select and prepare the axis
                ax = axes[row, col]
                disable_ticks(ax)

                # Add labels
                if row == 0:
                    ax.set_title(rf'$z_1$ = {z_1}', fontsize=6, pad=5)
                if col == 0:
                    ax.set_ylabel(rf'$z_2$ = {z_2}', fontsize=6, labelpad=5)

                # Set linewidth of the frame
                for axis in ['top', 'bottom', 'left', 'right']:
                    ax.spines[axis].set_linewidth(0.25)
                    ax.xaxis.set_tick_params(width=0.25)
                    ax.yaxis.set_tick_params(width=0.25)

                # Plot the profile
                ls = '-' if np.linalg.norm(z.squeeze()) <= 3 else '--'
                ax.plot(T_pred, log_P, color=f'C{i}', linewidth=1, ls=ls)

                # Additional plot options
                ax.set_xlim(0, 4500)
                ax.set_ylim(2.5, -6.5)

    # Save the figure
    file_path = Path('decoded-profiles.pdf')
    fig.tight_layout(pad=0)
    fig.savefig(file_path, dpi=600, bbox_inches='tight', pad_inches=pad_inches)


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nPLOT LATENT SPACES AND DECODED PT PROFILES\n', flush=True)

    # Get CLI arguments
    args = get_cli_arguments()

    # Set default font for plots
    mpl.rcParams['font.sans-serif'] = "Arial"
    mpl.rcParams['font.family'] = "sans-serif"

    # -------------------------------------------------------------------------
    # Find the different runs and load their latent spaces and decoders
    # -------------------------------------------------------------------------

    print('Loading latent spaces and decoders...', end=' ', flush=True)

    # Get the different run directories
    experiment_dir = expandvars(Path(args.experiment_dir))
    run_dirs = [experiment_dir / 'runs' / f'run_{i}' for i in args.runs]

    # Load the latent spaces and decoders
    latent_spaces: Dict[int, np.ndarray] = {}
    decoders: Dict[int, ONNXDecoder] = {}
    for run_dir, random_seed in zip(run_dirs, args.runs):

        # Load the latent space
        file_path = run_dir / 'results_on_test_set.hdf'
        with h5py.File(file_path, 'r') as hdf_file:
            latent_spaces[random_seed] = np.array(hdf_file['z_refined'])

        # Load the decoder
        file_path = run_dir / 'decoder.onnx'
        decoders[random_seed] = ONNXDecoder(file_path)

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Create plots
    # -------------------------------------------------------------------------

    print('Creating plots of latent spaces...', end=' ', flush=True)

    for i, random_seed in enumerate(latent_spaces.keys()):
        plot_latent_space(
            z=latent_spaces[random_seed],
            random_seed=random_seed,
            color=f'C{i}',
        )

    print('Done!', flush=True)

    print('Creating plots of decoded profiles...', end=' ', flush=True)
    plot_decoded_profiles(decoders=decoders)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.2f} seconds.\n')

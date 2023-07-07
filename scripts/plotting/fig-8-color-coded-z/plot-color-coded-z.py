"""
Create scatter plots of latent variables z colored by some property
of the corresponding atmosphere (e.g., the concentration of some gas).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import argparse
import time

from matplotlib.patches import Rectangle

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ml4ptp.paths import expandvars, get_datasets_dir
from ml4ptp.plotting import set_fontsize, add_colorbar_to_ax


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

def get_cli_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='pyatmos',
        choices=['pyatmos', 'goyal-2020'],
        help='Name of the dataset to use (needed to load test.hdf).',
    )
    parser.add_argument(
        '--key',
        type=str,
        default='T',
        help='HDF key of the property to color the scatter plot by.',
    )
    parser.add_argument(
        '--run-dir',
        type=str,
        required=True,
        help='Path to the directory containing the results_on_test_set.hdf',
    )
    parser.add_argument(
        '--scaling-factor',
        type=float,
        default=1.0,
        help='Scaling factor by which to divide the target before plotting.',
    )
    parser.add_argument(
        '--title',
        type=str,
        default=None,
        help='Title to use for the plot.',
    )
    parser.add_argument(
        '--use-log',
        action='store_true',
        help='Whether to color-code using the log of the property.',
    )
    args = parser.parse_args()

    return args


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nPLOT COLOR-CODED LATENT VARIABLES\n', flush=True)

    # Get CLI arguments
    args = get_cli_arguments()
    print('Received the following CLI arguments:')
    for key, value in vars(args).items():
        print(f'  {key}: {value}')
    print()

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------

    # Load z_refined from results_on_test_set.hdf
    file_path = expandvars(Path(args.run_dir)) / 'results_on_test_set.hdf'
    with h5py.File(file_path, 'r') as hdf_file:
        z_refined = np.array(hdf_file['z_refined'])

    # Load target property from test.hdf
    file_path = get_datasets_dir() / args.dataset / 'output' / 'test.hdf'
    with h5py.File(file_path, 'r') as hdf_file:
        target = np.array(hdf_file[args.key])

    # Apply scaling factor to make colorbar legend nicer
    target /= float(args.scaling_factor)

    # For properties that are not just a single number, we need to aggregate
    if target.ndim > 1:
        target = np.mean(target, axis=1)

    # If we are using the log of the property, take the log
    if args.use_log:
        target = np.log10(target)

    # -------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------

    # Set default font
    mpl.rcParams['font.sans-serif'] = 'Arial'
    mpl.rcParams['font.family'] = 'sans-serif'

    print('Creating plot...', end=' ', flush=True)

    # Create a new figure
    pad_inches = 0.025
    fig, ax = plt.subplots(
        figsize=(
            5.6 / 2.54 - 2 * pad_inches,
            7.0 / 2.54 - 2 * pad_inches,
        ),
    )

    # Plot the scatter plot
    marker_size = 4 if args.dataset == 'pyatmos' else 8
    img = ax.scatter(
        z_refined[:, 0],
        z_refined[:, 1],
        c=target,
        s=marker_size,
        marker='.',
        cmap='viridis',
        edgecolors='none',
        rasterized=True,
    )

    # Set general plot options
    set_fontsize(ax, 7)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.25)
    ax.xaxis.set_tick_params(width=0.25)
    ax.yaxis.set_tick_params(width=0.25)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$z_1$')
    ax.set_ylabel(r'$z_2$')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.xaxis.label.set_fontsize(8)
    ax.yaxis.label.set_fontsize(8)
    ax.tick_params('both', length=1.5, width=0.25, which='major')

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Add colorbar
    # -------------------------------------------------------------------------

    print('Adding colorbar...', end=' ', flush=True)

    cbar = add_colorbar_to_ax(
        img=img,
        fig=fig,
        ax=ax,
        where='top',
    )

    # Add the title to the colorbar
    # The invisible rectangle is a hack to get the same layout for all plots;
    # otherwise, the different heights of the titles affect the plot size due
    # to different sub/superscripts (\vphantom is not supported apparently).
    if args.title is not None:
        cbar.ax.text(
            x=0.5,
            y=4.5,
            s=args.title,
            transform=cbar.ax.transAxes,
            fontsize=8,
            va='center',
            ha='center',
        )
        ax.add_artist(
            Rectangle(
                xy=(0.0, 0),
                width=1.0,
                height=1.3,
                fc='none',
                ec='none',
                alpha=0.5,
                transform=ax.transAxes,
                clip_on=False,
            )
        )

    # Set limits and other options
    cbar.outline.set_linewidth(0.25)
    cbar.ax.xaxis.set_tick_params(width=0.25, length=1.5)

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Save the figure
    # -------------------------------------------------------------------------

    print('Saving figure to PDF...', end=' ', flush=True)

    plots_dir = Path(__file__).resolve().parent / 'plots'
    plots_dir.mkdir(exist_ok=True)

    use_log = 'log_' if args.use_log else ''
    key = args.key.split('/')[-1]
    file_name = f'{args.dataset}__{use_log}{key}.pdf'
    file_path = plots_dir / file_name
    fig.tight_layout(pad=0)
    fig.savefig(file_path, dpi=600, bbox_inches='tight', pad_inches=pad_inches)

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.2f} seconds.\n')

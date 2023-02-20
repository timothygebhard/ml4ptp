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

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ml4ptp.paths import get_datasets_dir
from ml4ptp.plotting import set_fontsize


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

def get_cli_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--aggregate',
        type=str,
        choices=['mean', 'delta_upper_lower'],
        default='mean',
        help='How to aggregate abundance profiles etc. into a single number.',
    )
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
        '--title',
        type=str,
        default=None,
        help='Optional: Title to use for the plot.',
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

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nPLOT COLOR-CODED LATENT VARIABLES\n', flush=True)

    # Get CLI arguments
    args = get_cli_arguments()

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------

    # Load z_refined from results_on_test_set.hdf
    file_path = Path(args.run_dir) / 'results_on_test_set.hdf'
    with h5py.File(file_path, 'r') as hdf_file:
        z_refined = np.array(hdf_file['z_refined'])

    # Load target property from test.hdf
    file_path = get_datasets_dir() / args.dataset / 'output' / 'test.hdf'
    with h5py.File(file_path, 'r') as hdf_file:
        target = np.array(hdf_file[args.key])

    # For properties that are not just a single number, we need to aggregate
    title = ''
    if target.ndim > 1:
        if args.aggregate == 'mean':
            target = np.mean(target, axis=1)
            title = 'Mean '
        elif args.aggregate == 'delta_upper_lower':
            target = (
                np.mean(target[:, 0:10], axis=1) -
                np.mean(target[:, -10:-1], axis=1)
            )
        else:
            raise ValueError(f'Unknown aggregation method: {args.aggregate}')

    # If we are using the log of the property, take the log
    if args.use_log:
        target = np.log10(target)

    # -------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------

    # Set default font
    mpl.rcParams['font.sans-serif'] = "Arial"
    mpl.rcParams['font.family'] = "sans-serif"

    print('Creating plot...', end=' ', flush=True)

    # Create a new figure
    pad_inches = 0.025
    fig, ax = plt.subplots(
        figsize=(
            4.3 / 2.54 - 2 * pad_inches,
            4.3 / 2.54 - 2 * pad_inches,
        ),
    )

    # Set general plot options
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.25)
        ax.xaxis.set_tick_params(width=0.25)
        ax.yaxis.set_tick_params(width=0.25)

    if args.title is not None:
        ax.set_title(args.title)
    else:
        title += ('log$_{{10}}\,$' if args.use_log else '') + args.key
        ax.set_title(
            label=title,
            fontweight='bold',
        )

    ax.set_aspect('equal')
    ax.set_xlabel(r'$z_1$')
    ax.set_ylabel(r'$z_2$')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xticks([-5, -2.5, 0, 2.5, 5])
    ax.set_yticks([-5, -2.5, 0, 2.5, 5])
    set_fontsize(ax, 5.5)
    ax.xaxis.label.set_fontsize(6.5)
    ax.yaxis.label.set_fontsize(6.5)
    ax.tick_params('both', length=2, width=0.25, which='major')

    # Plot the scatter plot
    marker_size = 1 if args.dataset == 'pyatmos' else 4
    ax.scatter(
        z_refined[:, 0],
        z_refined[:, 1],
        c=target,
        s=marker_size,
        marker='.',
        cmap='viridis',
        edgecolors='none',
        rasterized=True,
    )

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

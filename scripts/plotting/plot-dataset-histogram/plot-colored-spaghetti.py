"""
Plot every PT profile in a given dataset, but adjust the color of the
line according to the density of the profiles at this (P, T) coordinate.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Optional

import argparse
import time

from KDEpy import FFTKDE
from matplotlib.collections import LineCollection
from scipy.interpolate import RegularGridInterpolator

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ml4ptp.paths import get_datasets_dir
from ml4ptp.plotting import set_fontsize


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

def get_cli_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        required=True,
        default='pyatmos',
        choices=['pyatmos', 'goyal-2020'],
        help='Name of the dataset for which to plot the histogram.',
    )
    args = parser.parse_args()

    return args


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nPLOT HISTOGRAM OF DATASETS\n', flush=True)

    # Parse command line arguments
    args = get_cli_args()
    dataset = args.dataset

    # -------------------------------------------------------------------------
    # Load the dataset
    # -------------------------------------------------------------------------

    # Type annotations
    excluded_path: Optional[Path]
    excluded_log_P: Optional[np.ndarray]
    excluded_T: Optional[np.ndarray]

    # Define path and keys for the dataset
    if dataset == 'pyatmos':

        train_path = get_datasets_dir() / 'pyatmos' / 'output' / 'train.hdf'
        test_path = get_datasets_dir() / 'pyatmos' / 'output' / 'test.hdf'
        excluded_path = get_datasets_dir() / 'pyatmos' / 'output' / 'ood.hdf'

        P_key = 'P'
        T_key = 'T'

        xlim = (0, 350)
        ylim = (0.25, -5.25)
        yticks = np.arange(-5, 1)

    elif dataset == 'goyal-2020':

        train_path = get_datasets_dir() / 'goyal-2020' / 'output' / 'train.hdf'
        test_path = get_datasets_dir() / 'goyal-2020' / 'output' / 'test.hdf'
        excluded_path = None

        P_key = '/pt_profiles/pressure'
        T_key = '/pt_profiles/temperature'

        xlim = (0, 6500)
        ylim = (4.25, -6.25)
        yticks = np.arange(-6, 5)

    else:
        raise ValueError(f'Unknown dataset "{dataset}"!')

    # Load the dataset: first training, then test set
    print('Loading dataset...', end=' ', flush=True)
    with h5py.File(train_path, 'r') as hdf_file:
        log_P = np.log10(np.array(hdf_file[P_key]))
        T = np.array(hdf_file[T_key])
    with h5py.File(test_path, 'r') as hdf_file:
        log_P = np.concatenate((log_P, np.log10(np.array(hdf_file[P_key]))))
        T = np.concatenate((T, np.array(hdf_file[T_key])))
    print(f'Done! [Size = {T.shape}]', flush=True)

    # Load the excluded dataset, if any
    print('Loading excluded dataset (if any)...', end=' ', flush=True)
    if excluded_path is not None:
        with h5py.File(excluded_path, 'r') as hdf_file:
            excluded_log_P = np.log10(np.array(hdf_file[P_key]))
            excluded_T = np.array(hdf_file[T_key])
    else:
        excluded_log_P = None
        excluded_T = None
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Compute a 2D kernel density estimate of the dataset
    # -------------------------------------------------------------------------

    print('Computing 2D kernel density estimate...', end=' ', flush=True)

    # Create a flattened array of (log_P, T) values
    data = np.array([T.flatten(), log_P.flatten()]).T

    # Grid points in each dimension
    grid_points = 128

    # Compute and evaluate the kernel density estimate
    kde = FFTKDE(bw=0.1)
    grid, points = kde.fit(data).evaluate(grid_points)

    # Reshape the grid to a 2D array
    x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
    z = points.reshape(grid_points, grid_points).T

    # Create an interpolator so that we can evaluate the KDE (i.e., the
    # density of PT profiles) at arbitrary (log_P, T) positions
    density = RegularGridInterpolator((y, x), z, method='linear')

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Create a the colored spaghetti plot
    # -------------------------------------------------------------------------

    print('Creating spaghetti plot...', end=' ', flush=True)

    # Set default font
    mpl.rcParams['font.sans-serif'] = "Arial"
    mpl.rcParams['font.family'] = "sans-serif"

    # Create the figure
    pad_inches = 0.01
    fig, ax = plt.subplots(
        figsize=(8 / 2.54 - 2 * pad_inches, 4.5 / 2.54 - 2 * pad_inches),
    )

    # Set general plot options
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.25)
        ax.xaxis.set_tick_params(width=0.25)
        ax.yaxis.set_tick_params(width=0.25)
    ax.tick_params('both', length=2, width=0.25, which='major')

    # Compute a norm for the colorbar
    norm = plt.Normalize(z.min(), z.max())

    # Loop over all profiles and plot them
    for (x, y) in zip(T, log_P):

        # Compute the density (i.e., the color) of the profile for each
        # (log_P, T) coordinate
        colors = density(np.column_stack([y, x])[1:])

        # Create a line collection
        # Basically, we create a separate line for each segment connecting the
        # points (log_P_{i}, T_{i}) and (log_P_{i+1}, T_{i+1}), and we color
        # each line according to the density of the corresponding PT profile.

        # We start by defining the line segments. See also the following link:
        #   https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/
        #   multicolored_line.html
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create the line collection and add the colors for the segments
        # We need to rasterize this, otherwise the PDF will contain tens of
        # thousands of lines, which is not fun for anybody involved.
        lc = LineCollection(segments, cmap='magma', norm=norm, rasterized=True)
        lc.set_array(colors)

        # Set the line width and opacity
        lc.set_linewidth(0.1)
        lc.set_alpha(0.9)

        # Add the line collection to the plot
        line = ax.add_collection(lc)

    # Plot the excluded dataset, if any
    if excluded_T is not None and excluded_log_P is not None:
        for (x, y) in zip(excluded_T, excluded_log_P):
            ax.plot(x, y, lw=0.3, ls='--', color='LawnGreen')

    # Set the axis labels
    ax.set_xlabel('T (K)')
    ax.set_ylabel(r'$\log_{10}$(P / bar)')

    # Set the axis ticks
    ax.set_yticks(yticks)

    # Set the axis limits
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # Set the colorbar
    mappable = plt.cm.ScalarMappable(cmap='magma', norm=norm)
    cbar = fig.colorbar(mappable, ax=ax, pad=0.01)
    cbar.outline.set_linewidth(0.25)
    cbar.set_label(label='Density of PT profiles in dataset', size=5.5)
    cbar.ax.tick_params(width=0.25, length=2, labelsize=4.5)
    cbar.ax.tick_params('both', length=2, width=0.25, which='major')
    cbar.ax.tick_params('both', length=1, width=0.25, which='minor')

    # Set the font size
    set_fontsize(ax, 5.5)
    ax.xaxis.label.set_fontsize(6.5)
    ax.yaxis.label.set_fontsize(6.5)
    cbar.ax.tick_params(labelsize=5.5)

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Save the figure
    # -------------------------------------------------------------------------

    print('Saving plot...', end=' ', flush=True)

    fig.tight_layout(pad=0)
    fig.savefig(
        f'{dataset}-spaghetti-plot.pdf',
        dpi=1200,
        bbox_inches='tight',
        pad_inches=pad_inches,
    )

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nDone! This took {time.time() - script_start:.1f} seconds.\n')

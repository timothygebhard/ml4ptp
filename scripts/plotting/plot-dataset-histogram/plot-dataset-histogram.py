"""
Create histogram plots of the (training) datasets.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import argparse
import time

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ml4ptp.paths import get_datasets_dir
from ml4ptp.plotting import set_fontsize, add_colorbar_to_ax


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

    # Define path and keys for the dataset
    if dataset == 'pyatmos':

        train_path = get_datasets_dir() / 'pyatmos' / 'output' / 'train.hdf'
        test_path = get_datasets_dir() / 'pyatmos' / 'output' / 'test.hdf'

        P_key = 'P'
        T_key = 'T'

        log_P_bins = np.linspace(-5, 0, 101)
        T_bins = np.linspace(0, 350, 101)
        vmax = 9e4

    elif dataset == 'goyal-2020':

        train_path = get_datasets_dir() / 'goyal-2020' / 'output' / 'train.hdf'
        test_path = get_datasets_dir() / 'goyal-2020' / 'output' / 'test.hdf'

        P_key = '/pt_profiles/pressure'
        T_key = '/pt_profiles/temperature'

        log_P_bins = np.linspace(-6, 4, 101)
        T_bins = np.linspace(0, 6000, 101)
        vmax = 2e3

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

    # -------------------------------------------------------------------------
    # Create a histogram plot
    # -------------------------------------------------------------------------

    print('Creating histogram plot...', end=' ', flush=True)

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

    # Plot the histogram
    hist, xegdes, yedges, img = ax.hist2d(
        x=T.flatten(),
        y=log_P.flatten(),
        bins=(T_bins, log_P_bins),
        cmap='magma',
        norm=mpl.colors.LogNorm(vmin=1, vmax=vmax),
    )

    # Set the axis labels
    ax.set_xlabel('T (K)')
    ax.set_ylabel('log(P / bar)')

    # Set the axis limits
    ax.set_xlim(T_bins[0], T_bins[-1])
    ax.set_ylim(log_P_bins[-1], log_P_bins[0])

    # Set the colorbar
    cbar = add_colorbar_to_ax(img=img, ax=ax, fig=fig)
    cbar.outline.set_linewidth(0.25)
    cbar.ax.tick_params(width=0.25, length=2)
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
        f'{dataset}-histogram.pdf',
        dpi=600,
        bbox_inches='tight',
        pad_inches=pad_inches,
    )

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nDone! This took {time.time() - script_start:.1f} seconds.\n')

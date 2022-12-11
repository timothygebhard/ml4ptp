"""
Create plots of PT profiles and our best approximation of them.
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

from ml4ptp.plotting import set_fontsize


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
        help='Name of the dataset to use.',
    )
    parser.add_argument(
        '--idx',
        nargs='+',
        type=int,
        default=[0],
        help='Index (or indices) of the profile to plot in the HDF file.',
    )
    parser.add_argument(
        '--run-dir',
        type=str,
        required=True,
        default='$ML4PTP_EXPERIMENTS_DIR/pyatmos/default/runs/run_0',
        help='Path to the directory containing the results_on_test_set.hdf',
    )
    parser.add_argument(
        '--sort-by',
        type=str,
        choices=['idx', 'mse'],
        default='mse',
        help=(
            'How to sort the profiles. "idx" keeps the original order from'
            'the test.hdf file. "mse" sorts the profiles by their MSE. This '
            'is useful for plotting profiles with a high / okay / low MSE.'
        ),
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
    print('\nPLOT PT PROFILE\n', flush=True)

    # Get CLI arguments
    args = get_cli_arguments()

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------

    print('Loading data from HDF file...', end=' ', flush=True)

    # Load data from results_on_test_set.hdf
    file_path = Path(args.run_dir) / 'results_on_test_set.hdf'
    with h5py.File(file_path, 'r') as hdf_file:
        log_P = np.array(hdf_file['log_P']).squeeze()
        T_true = np.array(hdf_file['T_true']).squeeze()
        T_pred_refined = np.array(hdf_file['T_pred_refined']).squeeze()
        mse_refined = np.array(hdf_file['mse_refined']).squeeze()
        dim_z = hdf_file['z_refined'].shape[1]

    print('Done!', flush=True)
    print('Sorting profiles...', end=' ', flush=True)

    # If requested, sort the profiles by their MSE
    if args.sort_by == 'mse':
        sort_idx = np.argsort(mse_refined)
        log_P = log_P[sort_idx]
        T_true = T_true[sort_idx]
        T_pred_refined = T_pred_refined[sort_idx]
        mse_refined = mse_refined[sort_idx]

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Create the plot(s)
    # -------------------------------------------------------------------------

    # Set default font
    mpl.rcParams['font.sans-serif'] = "Arial"
    mpl.rcParams['font.family'] = "sans-serif"

    # Loop over indices and create plots
    for idx in args.idx:

        print(f'\nCreating plot (idx={idx})...', end=' ', flush=True)

        # Create a new figure
        pad_inches = 0.025
        fig, ax = plt.subplots(
            figsize=(
                4 / 2.54 - 2 * pad_inches,
                3 / 2.54 - 2 * pad_inches,
            ),
        )

        # Set general plot options
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(0.25)
            ax.xaxis.set_tick_params(width=0.25)
            ax.yaxis.set_tick_params(width=0.25)

        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('log$_{10}$(Pressure / bar)')

        if args.dataset == 'pyatmos':
            ax.set_xlim(0, 400)
            ax.set_ylim(0.5, -5.5)
            ax.set_yticks([-5, -4, -3, -2, -1, 0])
        else:
            ax.set_xlim(0, 5000)
            ax.set_ylim(2.5, -6.5)

        set_fontsize(ax, 5.5)
        ax.xaxis.label.set_fontsize(6.5)
        ax.yaxis.label.set_fontsize(6.5)
        ax.tick_params('both', length=2, width=0.25, which='major')

        # Plot the PT profile
        ax.plot(
            T_true[idx],
            log_P[idx],
            'o',
            color='black',
            markersize=1,
        )
        ax.plot(
            T_pred_refined[idx],
            log_P[idx],
            '-',
            lw=1,
            color='red',
        )

        # Add a text box with dim(z) and the RMSE
        ax.text(
            x=0.95,
            y=0.95,
            s=(
                f'RMSE = {np.sqrt(mse_refined[idx]):.2f}\n'
                f'dim(z) = {dim_z}'
            ),
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=5.5,
        )

        print('Done!', flush=True)
        print('Saving figure to PDF...', end=' ', flush=True)

        plots_dir = Path(__file__).resolve().parent / 'plots'
        plots_dir.mkdir(exist_ok=True)

        file_name = (
            f'{args.dataset}__'
            f'dim-z_{dim_z}__'
            f'idx_{idx}__'
            f'sort-by_{args.sort_by}.pdf'
        )
        file_path = plots_dir / file_name
        fig.tight_layout(pad=0)
        fig.savefig(
            file_path, dpi=600, bbox_inches='tight', pad_inches=pad_inches
        )

        plt.close(fig)

        print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.2f} seconds.\n')

"""
Create the different parts for Figure 7.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import time

from itertools import product
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from KDEpy import FFTKDE
from scipy.stats import norm

from ml4ptp.paths import get_experiments_dir
from ml4ptp.plotting import disable_ticks, CBF_COLORS
from ml4ptp.exporting import PTProfile


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    script_start = time.time()
    print("\nCREATE PLOTS FOR FIGURE 7\n", flush=True)

    # -------------------------------------------------------------------------
    # General preliminaries
    # -------------------------------------------------------------------------

    # Set default font
    mpl.rcParams['font.sans-serif'] = "Arial"
    mpl.rcParams['font.family'] = "sans-serif"

    # Read in the data (results on test set)
    print("Reading in the data...", end=" ", flush=True)

    run_dir = (
        get_experiments_dir()
        / 'goyal-2020'
        / 'default'
        / 'latent-size-2'
        / 'runs'
        / 'run_0'
    )
    file_path = run_dir / 'results_on_test_set.hdf'
    with h5py.File(file_path, 'r') as hdf_file:
        log_P = np.array(hdf_file['log_P'])
        T = np.array(hdf_file['T_true'])
        z = np.array(hdf_file['z_refined'])

    print("Done!", flush=True)

    # Load the trained decoder
    print("Loading the trained decoder...", end=" ", flush=True)
    file_path = run_dir / 'decoder.onnx'
    decoder = PTProfile(file_path.as_posix())
    print("Done!", flush=True)

    # Ensure the output directory exists
    plots_dir = Path(__file__).parent / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Plot the input PT profiles
    # -------------------------------------------------------------------------

    print("Plotting the input PT profiles...", end=" ", flush=True)

    for i, idx in enumerate([0, 51, 68]):

        pad_inches = 0.005
        fig, ax = plt.subplots(
            figsize=(
                2.5 / 2.54 - 2 * pad_inches,
                2.5 / 2.54 - 2 * pad_inches,
            ),
        )

        disable_ticks(ax)

        ax.set_box_aspect(1)

        ax.set_xlim(0, 5000)
        ax.set_ylim(3.5, -6.5)

        ax.set_ylabel(r'$\log_{10}$(P)', fontsize=8, labelpad=3)
        ax.tick_params(
            axis='both',
            which='both',
            left=True,
            labelleft=True,
        )
        ax.set_yticks([-6, -4, -2, 0, 2])
        ax.tick_params(axis='both', which='major', labelsize=7, pad=1)
        ax.tick_params('both', length=1.5, which='major')

        ax.set_xlabel(r'T (K)', fontsize=7, labelpad=3)
        ax.tick_params(
            axis='both',
            which='both',
            bottom=True,
            labelbottom=True,
        )
        ax.set_xticks([0, 4000])
        ax.tick_params(axis='both', which='major', labelsize=7, pad=1)
        ax.tick_params('both', length=1.5, which='major')

        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            item.set_fontsize(8)

        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(0.25)
            ax.xaxis.set_tick_params(width=0.25)
            ax.yaxis.set_tick_params(width=0.25)

        fig.tight_layout(pad=pad_inches)

        ax.plot(T[idx], log_P[idx], '-', lw=0.25, color=CBF_COLORS[i])
        ax.plot(
            T[idx],
            log_P[idx],
            '.',
            ms=4,
            markerfacecolor=CBF_COLORS[i],
            markeredgecolor='none',
        )

        file_path = plots_dir / f'pt-profile-{i}.pdf'
        fig.savefig(file_path, bbox_inches='tight', pad_inches=pad_inches)
        plt.close(fig)

    print("Done!", flush=True)

    # -------------------------------------------------------------------------
    # 2. Plot the latent space
    # -------------------------------------------------------------------------

    print("Plotting the latent space...", end=" ", flush=True)

    pad_inches = 0.025
    fig, axes = plt.subplots(
        ncols=3,
        nrows=3,
        figsize=(
            6 / 2.54 - 2 * pad_inches,
            6 / 2.54 - 2 * pad_inches,
        ),
        width_ratios=[1, 8, 1],
        height_ratios=[1, 8, 1],
    )
    fig.subplots_adjust(wspace=0.075, hspace=0.075)

    for i, j in product(range(3), range(3)):
        if (i, j) != (1, 1):
            disable_ticks(axes[i, j])
            axes[i, j].axis('off')
        else:
            for axis in ['top', 'bottom', 'left', 'right']:
                axes[i, j].spines[axis].set_linewidth(0.25)
                axes[i, j].xaxis.set_tick_params(width=0.25)
                axes[i, j].yaxis.set_tick_params(width=0.25)

    dist = norm(loc=0, scale=1)
    x = np.linspace(-4, 4, 1000)
    y1 = FFTKDE(bw="scott").fit(z[:, 0]).evaluate(x)
    axes[0, 1].axhline(y=0, lw=0.25, c='k')
    axes[0, 1].plot(x, dist.pdf(x), ls=':', c='k', lw=0.25)
    axes[0, 1].plot(x, y1, color=CBF_COLORS[4], lw=0.75)
    axes[0, 1].set_xlim(-4, 4)

    y2 = FFTKDE(bw="scott").fit(z[:, 1]).evaluate(x)
    axes[1, 2].axvline(x=0, lw=0.25, c='k')
    axes[1, 2].plot(dist.pdf(x), x, ls=':', c='k', lw=0.25)
    axes[1, 2].plot(y2, x, color=CBF_COLORS[4], lw=0.75)
    axes[1, 2].set_ylim(-4, 4)

    ax = axes[1, 1]

    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(8)

    ax.tick_params(axis='both', which='major', labelsize=7, pad=2)
    ax.tick_params('both', length=1.5, which='major')

    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_yticks([-4, -2, 0, 2, 4])

    ax.plot(
        z[:, 0],
        z[:, 1],
        '.',
        ms=2,
        markerfacecolor=CBF_COLORS[4],
        markeredgecolor='none',
        alpha=1.0,
    )

    for i, idx in enumerate([0, 51, 68]):
        ax.plot(
            z[idx, 0],
            z[idx, 1],
            '.',
            ms=10,
            markerfacecolor='white',
            markeredgecolor='none',
            alpha=0.7,
        )
        ax.plot(
            z[idx, 0],
            z[idx, 1],
            '.',
            ms=8,
            markerfacecolor=CBF_COLORS[i],
            markeredgecolor='none',
        )

    for z1 in (-2, -1, 0, 1, 2):
        for z2 in (-2, -1, 0, 1, 2):
            ax.plot(z1, z2, '+', ms=5, mew=1, color=CBF_COLORS[5])

    circle = plt.Circle((0, 0), 3.5, fc='none', ec='k', ls='--', lw=0.25)
    ax.add_patch(circle)

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

    file_path = plots_dir / 'latent-space.pdf'
    fig.tight_layout(pad=0)
    plt.savefig(file_path, pad_inches=pad_inches, bbox_inches='tight')
    plt.close(fig)

    print("Done!", flush=True)

    # -------------------------------------------------------------------------
    # Plot the decoded PT profiles
    # -------------------------------------------------------------------------

    print("Plotting the decoded PT profiles...", end=" ", flush=True)

    log_P_grid = np.linspace(-6, 3, 100)

    grid_size = 5
    grid = np.linspace(-2, 2, grid_size)

    fig, axes = plt.subplots(
        ncols=grid_size,
        nrows=grid_size,
        figsize=(
            10 / 2.54 - 2 * pad_inches,
            10 / 2.54 - 2 * pad_inches,
        ),
    )

    # Loop over the grid
    for row, z_2 in enumerate(grid[::-1]):
        for col, z_1 in enumerate(grid):
            # Get the decoded profile
            z = np.array([[z_1, z_2]])
            T_pred = decoder(z=z, log_P=log_P_grid.reshape(1, -1)).squeeze()

            # Select and prepare the axis
            ax = axes[row, col]
            ax2 = None
            disable_ticks(ax)

            # Add labels
            if row == 0:
                ax.set_title(rf'$z_1$ = ${z_1}$', fontsize=8, pad=5)
            if col == 0:
                ax.set_ylabel(r'$\log_{10}$(P)', fontsize=7, labelpad=3)
                ax.tick_params(
                    axis='both',
                    which='both',
                    left=True,
                    labelleft=True,
                )
                ax.set_yticks([-6, -4, -2, 0, 2])
                ax.tick_params(axis='both', which='major', labelsize=7, pad=2)
                ax.tick_params('both', length=1.5, which='major')
            if col == grid_size - 1:
                ax.set_yticks([])
                ax2 = ax.twinx()
                disable_ticks(ax2)
                ax2.set_ylabel(rf'$z_2$ = ${z_2}$', fontsize=8, labelpad=5)
            if row == grid_size - 1:
                ax.set_xlabel(r'T (K)', fontsize=7, labelpad=3)
                ax.tick_params(
                    axis='both',
                    which='both',
                    bottom=True,
                    labelbottom=True,
                )
                ax.set_xticks([0, 4000])
                ax.tick_params(axis='both', which='major', labelsize=7, pad=2)
                ax.tick_params('both', length=1.5, which='major')

            # Set linewidth of the frame
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(0.25)
                ax.xaxis.set_tick_params(width=0.25)
                ax.yaxis.set_tick_params(width=0.25)
                if ax2 is not None:
                    ax2.spines[axis].set_linewidth(0)
                    ax2.xaxis.set_tick_params(width=0)
                    ax2.yaxis.set_tick_params(width=0)
                    ax2.tick_params('both', length=0, width=0, which='major')

            # Plot the profile
            ls = '-' if np.linalg.norm(z.squeeze()) <= 3 else '--'
            ax.plot(T_pred, log_P_grid, color=CBF_COLORS[5], lw=1, ls=ls)

            # Additional plot options
            ax.set_xlim(0, 5000)
            ax.set_ylim(3.5, -6.5)

    file_path = plots_dir / 'decoded.pdf'
    plt.savefig(file_path, pad_inches=pad_inches, bbox_inches='tight')
    plt.close(fig)

    print("Done!", flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f"This took {time.time() - script_start:.1f} seconds!")

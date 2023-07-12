"""
Create figure C1 (closest-matching training set PT profiles for target
PT profiles from Rugheimer & Kaltenegger 2018).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import argparse
import time

from pathlib import Path
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml4ptp.paths import get_datasets_dir
from ml4ptp.plotting import set_fontsize


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_cli_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--profile',
        required=True,
        default='ME',
        choices=['ME', 'NOE'],
        help='Name of the PT profile for which to create the plot.',
    )
    args = parser.parse_args()

    return args


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    script_start = time.time()
    print("\nCREATE FIGURE C1: CLOSEST MATCHES\n", flush=True)

    # Get command line arguments
    args = get_cli_args()
    profile = args.profile

    print(f"Creating plot for: {profile}", flush=True)

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------

    print("Loading data...", end=" ", flush=True)

    # Load training set PT profiles (PyATMOS)
    file_path = get_datasets_dir() / "pyatmos" / "output" / "train.hdf"
    with h5py.File(file_path, "r") as hdf_file:
        T = np.array(hdf_file["T"])
        log_P = np.log10(np.array(hdf_file["P"]))

    # Load target PT profile (Rugheimer & Kaltenegger 2018) as a dataframe
    if profile == "ME":
        file_name = "Modern_Earth_pt_massfractions.csv"
    elif profile == "NOE":
        file_name = "NOE_0.8Ga_Earth_pt_massfractions.csv"
    else:
        raise ValueError(f"Invalid profile: {profile}")

    file_path = get_datasets_dir() / "rugheimer-2018" / file_name
    df = pd.read_csv(file_path, sep=",")

    print("Done!", flush=True)

    # -------------------------------------------------------------------------
    # Compute MSE between target PT profiles and training set PT profiles
    # -------------------------------------------------------------------------

    print("Computing MSE...", end=" ", flush=True)

    # Each training set PT profile is interpolated to the same pressure grid
    # as the target PT profiles to compute the MSE.
    mse = []
    for T_i, log_P_i in list(zip(T, log_P)):
        T_interp = np.interp(
            x=np.log10(df["P(bar)"]).values,
            xp=log_P_i,
            fp=T_i,
        )
        mse.append(np.mean((T_interp - df["T(K)"].values) ** 2))

    # Get indices of training set PT profiles with lowest MSE = best matches
    best_idx = np.argsort(np.array(mse))

    print("Done!", flush=True)

    # -------------------------------------------------------------------------
    # Create plot
    # -------------------------------------------------------------------------

    print("Creating plot...", end=" ", flush=True)

    # Create new figure
    pad_inches = 0.01
    fig, ax = plt.subplots(
        figsize=(8.7 / 2.54 - 2 * pad_inches, 6 / 2.54 - 2 * pad_inches),
    )

    # General plot settings
    set_fontsize(ax, 7)
    ax.set_xlim(0, 350)
    ax.set_ylim(0.5, -5.5)
    ax.set_yticks([0, -1, -2, -3, -4, -5])
    ax.set_xlabel("T (K)", fontsize=8)
    ax.set_ylabel("log$_\mathrm{10}$(P / bar)", fontsize=8)

    # Set width and z-order of the frame
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_zorder(1000)
        ax.spines[axis].set_linewidth(0.25)
        ax.xaxis.set_tick_params(width=0.25)
        ax.yaxis.set_tick_params(width=0.25)

    # Set width and length of the tick marks
    ax.tick_params('x', length=2, width=0.25, which='major')
    ax.tick_params('y', length=2, width=0.25, which='major')
    ax.tick_params('x', length=2, width=0.25, which='minor')
    ax.tick_params('y', length=1, width=0.25, which='minor')

    # Plot all training set PT profiles
    # Enable rasterization to reduce file size
    label: Optional[str] = "All training\nPT profiles"
    for T_i, log_P_i in list(zip(T, log_P)):
        ax.plot(
            T_i,
            log_P_i,
            color="lightgray",
            lw=0.5,
            label=label,
            rasterized=True,
        )
        label = None

    # Plot 100 training set PT profiles with lowest MSE
    label = "Top 100 training\nPT profiles closest\nto target PT profile"
    for i in best_idx[:100]:
        ax.plot(T[i], log_P[i], lw=1, color="C1", label=label)
        label = None

    # Plot target PT profile
    label = "Target PT profile"
    ax.plot(
        df["T(K)"].values,
        np.log10(df["P(bar)"]).values,
        lw=2,
        label=label,
    )

    # Add legend
    ax.legend(loc="lower left", fontsize=7, frameon=False)

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Save figure
    # -------------------------------------------------------------------------

    print("Saving plot to PDF...", end=" ", flush=True)

    fig.tight_layout(pad=0)

    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    file_path = plots_dir / f"closest-matches-{profile}.pdf"
    fig.savefig(file_path, dpi=600, bbox_inches='tight', pad_inches=pad_inches)

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds.\n')

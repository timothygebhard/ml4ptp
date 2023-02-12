"""
For a given (trained) PT profile model, train a normalizing flow that
maps an n-dimensional Gaussian distribution to the latent space of the
PT profile model as estimated from the training set.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import argparse
import time

from corner import corner
from tqdm.auto import tqdm

import h5py
import matplotlib.pyplot as plt
import normflows as nf
import numpy as np

import torch

from ml4ptp.config import load_experiment_config
from ml4ptp.onnx import ONNXEncoder
from ml4ptp.paths import expandvars
from ml4ptp.utils import get_batch_idx


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

def get_cli_args() -> argparse.Namespace:
    """
    Parse the command line arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--batch-size',
        type=int,
        default=1024,
        help='Batch size for training the flow.',
    )
    parser.add_argument(
        '--n-epochs',
        type=int,
        default=100,
        help='Number of training epochs for the flow.',
    )
    parser.add_argument(
        '--noise-scale',
        type=float,
        default=0.02,
        help='Scale of the Gaussian noise added to the latent space.',
    )
    parser.add_argument(
        '--run-dir',
        type=str,
        required=True,
        help='Path to the directory containing the trained model.',
    )

    return parser.parse_args()


def plot_distribution(samples: np.ndarray, file_path: Path) -> None:
    """
    Auxiliary function to plot the distribution of some samples.
    """

    figure = corner(
        data=samples,
        bins=25,
        range=samples.shape[1] * [(-5, 5)],
        plot_density=False,
        plot_contours=False,
    )
    figure.tight_layout()

    plt.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0.1)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nTRAIN NORMALIZING FLOW\n', flush=True)

    # -------------------------------------------------------------------------
    # Get experiment dir and load configuration file
    # -------------------------------------------------------------------------

    # Get CLI arguments
    args = get_cli_args()
    run_dir = Path(args.run_dir).resolve()
    experiment_dir = run_dir.parent.parent

    # Create directory for normalizing flow
    flow_dir = run_dir / 'normalizing-flow'
    flow_dir.mkdir(exist_ok=True)

    # Load the experiment configuration
    file_path = experiment_dir / 'config.yaml'
    config = load_experiment_config(file_path)

    # Set up device (CPU or CUDA)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n', flush=True)

    # -------------------------------------------------------------------------
    # Load training data and trained encoder; compute latent variables
    # -------------------------------------------------------------------------

    # Load training data
    print('Loading training dataset...', end=' ', flush=True)
    file_path = expandvars(Path(config['datamodule']['train_file_path']))
    with h5py.File(file_path, 'r') as hdf_file:
        log_P = np.log10(np.array(hdf_file[config['datamodule']['key_P']]))
        T = np.array(hdf_file[config['datamodule']['key_T']])
    print('Done!', flush=True)

    # Load the trained encoder model
    print('Loading trained encoder...', end=' ', flush=True)
    file_path = run_dir / 'encoder.onnx'
    encoder = ONNXEncoder(path_or_bytes=file_path)
    print('Done!', flush=True)

    # Loop over the training set and compute the latent variables
    print('Computing latent variables...', end=' ', flush=True)
    z_list = []
    for _log_P, _T in zip(log_P, T):
        _z = encoder(log_P=np.atleast_2d(_log_P), T=np.atleast_2d(_T))
        z_list.append(torch.from_numpy(_z).float())
    z = torch.cat(z_list, dim=0).to(device)
    print(f'Done! (z.shape = {tuple(z.shape)})', flush=True)

    # Plot the distribution of the latent variables
    print('Plotting samples from data...', end=' ', flush=True)
    samples_idx = np.random.choice(z.shape[0], size=10_000, replace=False)
    samples_data = z.cpu().numpy()[samples_idx]
    file_path = flow_dir / 'latent-distribution-data.png'
    plot_distribution(samples=samples_data, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Set up the normalizing flow (neural spline flow)
    # -------------------------------------------------------------------------

    print('Setting up neural spline flow...', end=' ', flush=True)

    # Define hyperparameters of the flow
    latent_size = z.shape[1]
    hidden_units = 32
    hidden_layers = 2
    num_layers = 8

    # Construct the flow
    flows = []
    for i in range(num_layers):
        flows += [
            nf.flows.AutoregressiveRationalQuadraticSpline(
                latent_size, hidden_layers, hidden_units
            ),
            nf.flows.LULinearPermute(latent_size),
        ]

    # Define Gaussian base distribution
    base = nf.distributions.DiagGaussian(latent_size, trainable=False)

    # Construct flow model and move model on GPU (if available)
    flow = nf.NormalizingFlow(base, flows).to(device)

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Train the normalizing flow
    # -------------------------------------------------------------------------

    # Set up the optimizer
    optimizer = torch.optim.Adam(flow.parameters(), lr=3e-4, weight_decay=1e-5)

    # Train for the given number of epochs
    print('\nTraining normalizing flow:', flush=True)
    for epoch in range(args.n_epochs):

        # Keep track of all batch losses
        losses = []

        # Loop over the training set in batches
        for batch_idx in tqdm(
            iterable=get_batch_idx(z, args.batch_size),
            ncols=68,
            desc=f'Epoch {epoch + 1}/{args.n_epochs}',
        ):

            # Cast batch_idx to tensor
            idx = torch.from_numpy(batch_idx).to(device)

            # Get training samples: We take a batch of z-values and add some
            # random noise to them. This is done to smoothen / smear out the
            # distribution of the latent variables.
            noise = torch.randn_like(z[idx], device=device)
            x = z[idx] + args.noise_scale * noise
            x = x.to(device)

            # Prepare optimizer and flow for training
            optimizer.zero_grad()
            flow.train()

            # Compute loss on current batch
            loss = flow.forward_kld(x)
            losses.append(loss.item())

            # Do backprop and optimizer step
            if not (torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
                optimizer.step()

        print(f'\033[1A\033[68C Loss: {np.mean(losses):.3f}')

    # -------------------------------------------------------------------------
    # Plot a sample from the trained normalizing flow
    # -------------------------------------------------------------------------

    print('\nPlotting samples from flow...', end=' ', flush=True)
    with torch.no_grad():
        samples_flow, _ = flow.sample(10_000)
    file_path = flow_dir / 'latent-distribution-flow.png'
    plot_distribution(samples=samples_flow.cpu().numpy(), file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Export the trained normalizing flow (with PyTorch)
    # -------------------------------------------------------------------------

    print('Exporting trained flow...', end=' ', flush=True)
    flow.save(path=flow_dir / 'flow.pt')
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nDone! This took {time.time() - script_start:.1f} seconds.\n')

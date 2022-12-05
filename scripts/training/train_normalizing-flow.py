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

import matplotlib.pyplot as plt
import normflows as nf
import numpy as np

import torch

from ml4ptp.config import load_experiment_config
from ml4ptp.data_modules import DataModule
from ml4ptp.utils import get_batch_idx


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

def get_cli_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--run-dir',
        required=True,
        help='Path to the directory containing the trained model.',
    )
    return parser.parse_args()


def plot_distribution(
    samples: np.ndarray,
    file_path: Path,
) -> None:

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

    # Load the experiment configuration
    file_path = experiment_dir / 'config.yaml'
    config = load_experiment_config(file_path)

    # Set up device (CPU or CUDA)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n', flush=True)

    # -------------------------------------------------------------------------
    # Load dataa and trained encoder; compute latent variables
    # -------------------------------------------------------------------------

    # Prepare the congiguration for the data module: We do not really need a
    # validation set, so we set the validation size to 1 sample. We can also
    # increase the batch size and disable `drop_last` to not lose any samples.
    # config['datamodule']['train_size'] = 10_001
    config['datamodule']['val_size'] = 1
    config['datamodule']['batch_size'] = 1024
    config['datamodule']['drop_last'] = False

    # Load the training data set (as a PyTorch Lightning data module)
    print('Loading training dataset...', end=' ', flush=True)
    datamodule = DataModule(**config['datamodule'])
    datamodule.prepare_data()
    dataloader = datamodule.train_dataloader()
    print('Done!', flush=True)

    # Load the trained encoder model
    print('Loading trained encoder...', end=' ', flush=True)
    file_path = run_dir / 'encoder.pt'
    encoder = torch.jit.load(file_path)  # type: ignore
    print('Done!', flush=True)

    # Loop over the training set and compute the latent variables
    print('Computing latent variables...', end=' ', flush=True)
    z_list = []
    for batch in dataloader:
        log_P, T = batch  # type: ignore
        with torch.no_grad():
            z_list.append(encoder(log_P=log_P, T=T))
    z = torch.cat(z_list, dim=0)
    print(f'Done! (z.shape = {tuple(z.shape)})', flush=True)

    # Plot the distribution of the latent variables
    print('Plotting samples from data...', end=' ', flush=True)
    samples_idx = np.random.choice(z.shape[0], size=10_000, replace=False)
    samples_data = z.cpu().numpy()[samples_idx]
    file_path = run_dir / 'latent-distribution-data.png'
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

    # Construct the flow; this is basically taken directly from the examples
    # in the `normflows` package.
    flows = []
    for i in range(num_layers):
        flows += [
            nf.flows.AutoregressiveRationalQuadraticSpline(
                latent_size, hidden_layers, hidden_units
            )
        ]
        flows += [nf.flows.LULinearPermute(latent_size)]

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

    # Set up parameters for training
    n_epochs = 20
    batch_size = 1024
    noise_scale = 0.02

    # Train for the given number of epochs
    print('\nTraining normalizing flow:', flush=True)
    for epoch in range(n_epochs):

        # Keep track of batch losses
        losses = []

        # Loop over the training set in batches
        for batch_idx in tqdm(
            iterable=get_batch_idx(z, batch_size),
            ncols=68,
            desc=f'Epoch {epoch + 1}/{n_epochs}',
        ):

            # Cast batch_idx to tensor
            idx = torch.from_numpy(batch_idx).to(device)

            # Get training samples: We take a batch of z-values and add some
            # random noise to them. This is done to smoothen / smear out the
            # distribution of the latent variables.
            x = z[idx] + noise_scale * torch.randn_like(z[idx])
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
    file_path = run_dir / 'latent-distribution-flow.png'
    plot_distribution(samples=samples_flow.cpu().numpy(), file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Export the trained normalizing flow
    # -------------------------------------------------------------------------

    print('Exporting trained flow...', end=' ', flush=True)
    file_path = run_dir / 'flow.pt'
    torch.save(flow, file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nDone! This took {time.time() - script_start:.1f} seconds.\n')

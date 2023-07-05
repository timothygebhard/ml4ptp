"""
Internal script to download all relevant results files from the cluster
to a local machine. This makes it easier to plot results locally.
"""

import subprocess
import time
from itertools import product
from pathlib import Path

from ml4ptp.paths import get_experiments_dir


if __name__ == "__main__":

    script_start = time.time()
    print("\nDOWNLOAD EXPERIMENT RESULTS FROM CLUSTER\n", flush=True)

    # Get remote and local value of $ML4PTP_EXPERIMENTS_DIR
    remote_cmd = "source ~/.zshrc && env | grep ML4PTP_EXPERIMENTS_DIR"
    cmd = ['ssh', 'mpicluster', 'zsh', '-l', '-c', f"'{remote_cmd}'"]
    result = subprocess.run(cmd, capture_output=True, universal_newlines=True)
    remote_experiments_dir = result.stdout.strip().split('\n')[0].split('=')[1]
    local_experiments_dir = get_experiments_dir()

    print(f"Remote experiments dir: {remote_experiments_dir}")
    print(f"Local experiments dir:  {local_experiments_dir}\n")

    # Define all combinations of datasets, latent sizes, runs and file names.
    # Those are the experiments for which we want to download the results.
    datasets = ['pyatmos', 'goyal-2020']
    latent_sizes = [1, 2, 3, 4]
    n_runs = 3
    file_names = ['decoder.onnx', 'encoder.onnx', 'results_on_test_set.hdf']
    combinations = product(datasets, latent_sizes, range(n_runs), file_names)

    # Loop over all combinations = files to download
    for dataset, latent_size, run, file_name in combinations:

        # Define paths to remote and local files
        remote_path = (
            f'{remote_experiments_dir}/{dataset}/default'
            f'/latent-size-{latent_size}/runs/run_{run}/{file_name}'
        )
        local_path = Path(
            f'{local_experiments_dir}/{dataset}/default'
            f'/latent-size-{latent_size}/runs/run_{run}'
        )
        local_path.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        print(f"Downloading {remote_path} ...", end=" ", flush=True)

        # Run scp command (the `-p` flag preserves modification times)
        cmd = ['scp', '-p', f'mpicluster:{remote_path}', local_path.as_posix()]
        result = subprocess.run(
            cmd,
            capture_output=True,
            universal_newlines=True,
        )

        print(f"Done! ({time.time() - start_time:.2f} seconds)", flush=True)

    print(f"\nThis took {time.time() - script_start:.1f} seconds.\n")

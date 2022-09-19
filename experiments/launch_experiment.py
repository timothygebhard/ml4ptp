"""
Create a HTCondor *.sub file for a given experiment and launch it  as a
job on the cluster.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import argparse
import subprocess
import sys
import time
import traceback

from ml4ptp.paths import expandvars
from ml4ptp.utils import get_run_dir


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

def get_cli_args() -> argparse.Namespace:

    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bid',
        type=int,
        default=5,
        help='How much to bid for the cluster job.',
    )
    parser.add_argument(
        '--blacklist',
        type=str,
        nargs='+',
        default='',
        help='Names of nodes (e.g., "g019") to exclude from running jobs.',
    )
    parser.add_argument(
        '--cpus',
        type=int,
        default=4,
        help='Number of CPUs to request for cluster job.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Create *.sub file, but do not launch job on cluster.',
    )
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='Number of GPUs to request for cluster job.',
    )
    parser.add_argument(
        '--gpu-memory',
        type=int,
        default=8_192,
        help='GPU memory (in MB) to request for cluster job.',
    )
    parser.add_argument(
        '--experiment-dir',
        required=True,
        help='Path to the experiment directory with the config.yaml',
    )
    parser.add_argument(
        '--memory',
        type=int,
        default=16_384,
        help='Memory (in MB) to request for cluster job.',
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for PyTorch, numpy, ....',
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
    print('\nLAUNCH EXPERIMENT ON THE CLUSTER\n', flush=True)

    # -------------------------------------------------------------------------
    # Create a *.sub file for the run
    # -------------------------------------------------------------------------

    # Get command line arguments; resolve experiment_dir
    args = get_cli_args()
    experiment_dir = expandvars(Path(args.experiment_dir)).resolve()

    print('Received the following arguments:\n')
    for key, value in vars(args).items():
        print(f'    {key} = {value}')
    print()

    # Get directory for this run
    run_dir = get_run_dir(experiment_dir=experiment_dir)

    # Create directory for HTCondor files
    print('Creating htcondor directory...', end=' ', flush=True)
    htcondor_dir = run_dir / 'htcondor'
    htcondor_dir.mkdir(exist_ok=True)
    print('Done!', flush=True)

    print('Creating logs directory...', end=' ', flush=True)
    logs_dir = htcondor_dir / 'logs'
    logs_dir.mkdir(exist_ok=True)
    print('Done!', flush=True)

    # Collect arguments for train.py
    arguments = [
        (Path(__file__).parent / 'train.py').as_posix(),
        f'--experiment-dir {args.experiment_dir}',
        f'--run-dir {run_dir.as_posix()}',
        f'--random-seed {args.random_seed}',
    ]

    # Collect requirements
    requirements = []
    if args.gpus > 0 and args.gpu_memory > 0:
        requirements += [f'TARGET.CUDAGlobalMemoryMb > {args.gpu_memory}']
    for node_name in list(args.blacklist):
        print(node_name)
        requirements += [
            f'TARGET.Machine != "{node_name}.internal.cluster.is.localnet"'
        ]
    n_requirements = len(requirements)

    # Construct string from requirements
    requirements_string = ''
    for i, requirement in enumerate(requirements):
        if i == 0:
            requirements_string += f'requirements   = {requirement}'
            requirements_string += ' && \\\n' if n_requirements > 1 else ' \n'
        elif i < n_requirements - 1:
            requirements_string += f'%                {requirement} && \\\n'
        else:
            requirements_string += f'%                {requirement}'

    # Collect the lines for the *.sub file
    lines = f"""
        getenv = true

        executable = {sys.executable}
        arguments  = {' '.join(arguments)}

        output = {(logs_dir / 'htcondor.out.txt').as_posix()}
        error  = {(logs_dir / 'htcondor.err.txt').as_posix()}
        log    = {(logs_dir / 'htcondor.log.txt').as_posix()}

        request_memory = {args.memory}
        request_cpus   = {args.cpus}
        request_gpus   = {args.gpus}

        {requirements_string}

        queue
        """
    lines = '\n'.join(_.strip().replace('%', ' ') for _ in lines.split('\n'))

    # Create the *.sub file in the HTCondor directory
    print('Creating run.sub file...', end=' ', flush=True)
    file_path = htcondor_dir / 'run.sub'
    with open(file_path, 'w') as sub_file:
        sub_file.write(lines)
    print('Done!\n', flush=True)

    # -------------------------------------------------------------------------
    # Launch the job on the cluster
    # -------------------------------------------------------------------------

    # Define the command to launch the job
    cmd = ['condor_submit_bid', str(args.bid), file_path.as_posix()]
    print('Command for launching job:\n')
    print('    ', ' '.join(cmd), '\n')

    # In case of a dry run, we abort here
    if args.dry_run:
        print('Dry run, aborting here!')

    # Otherwise, we actually launch the job
    else:

        print('Launching job on cluster:\n')

        try:

            # Launch a process that runs the command
            p = subprocess.run(args=cmd, capture_output=True)

            # Wait for process to finish; get output
            output = p.stdout.decode()
            print(output)

        except Exception as e:
            print(f'There was a problem: {e}\n')
            print(traceback.format_exc())

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nDone! This took {time.time() - script_start:.1f} seconds.\n')

"""
Create a HTCondor *.sub file for a given experiment and launch it as a
job on the cluster.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import argparse
import time

from ml4ptp.htcondor import SubmitFile, DAGFile, submit_dag
from ml4ptp.paths import expandvars, get_scripts_dir


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

def get_cli_args() -> argparse.Namespace:

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description=(
            'Create HTCondor submit files for training and evaluation, and '
            'a DAG file to control the workflow.\n\n'
            'Useful strings for the --requirements argument:\n'
            '  - "TARGET.Machine != \'g025.internal.cluster.is.localnet\'"\n'
            '  - "TARGET.CUDAGlobalMemoryMb > 15000"\n'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--bid',
        type=int,
        default=5,
        help='How much to bid for the cluster job.',
    )
    parser.add_argument(
        '--cpus',
        type=int,
        default=4,
        help='Number of CPUs to request for cluster job.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Use -force flag when calling condor_submit_dag.',
    )
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='Number of GPUs to request for cluster job.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='If given, create the submit files but do not submit the DAG.',
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
        '--n-evaluation-splits',
        type=int,
        default=128,
        help='Number of parallel jobs for evaluation.',
    )
    parser.add_argument(
        '--no-training',
        action='store_true',
        help='If given, mark the training job as "DONE" in the DAG.',
    )
    parser.add_argument(
        '--random-seeds',
        nargs='+',
        type=int,
        default=[0],
        help='(List of) random seeds; each will be used for a separate run.',
    )
    parser.add_argument(
        '--requirements',
        type=str,
        nargs='+',
        default=['TARGET.CUDAGlobalMemoryMb > 30000'],
        help='One more multuple strings with requirements (e.g., GPU memory).',
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
    # Get command line arguments, prepare runs directory for experiment
    # -------------------------------------------------------------------------

    # Get command line arguments
    args = get_cli_args()

    # Define shortcuts
    experiment_dir = expandvars(Path(args.experiment_dir)).resolve()

    print('Received the following arguments:\n')
    for key, value in vars(args).items():
        print(f'    {key} = {value}')
    print()

    # Get the runs directory
    print('Creating runs directory...', end=' ', flush=True)
    runs_dir = experiment_dir / 'runs'
    runs_dir.mkdir(exist_ok=True)
    print('Done!\n\n', flush=True)

    # -------------------------------------------------------------------------
    # For each random seed, create a run
    # -------------------------------------------------------------------------

    for random_seed in args.random_seeds:

        print(f'Creating run for random seed {random_seed}:', flush=True)

        # Construct the directory for this run
        print('  Creating run directory...', end=' ', flush=True)
        run_dir = runs_dir / f'run_{random_seed}'
        run_dir.mkdir(exist_ok=True)
        print('Done!', flush=True)

        # Create directory for HTCondor files
        print('  Creating htcondor directory...', end=' ', flush=True)
        htcondor_dir = run_dir / 'htcondor'
        htcondor_dir.mkdir(exist_ok=True)
        print('Done!', flush=True)

        # Instantiate a new DAG file
        dag_file = DAGFile()

        # Create submit file for training job and add it to DAG file
        print('  Creating submit file for training...', end=' ', flush=True)
        submit_file = SubmitFile(
            log_dir=htcondor_dir,
            memory=args.memory,
            cpus=args.cpus,
            gpus=args.gpus,
            requirements=args.requirements,
        )
        submit_file.add_job(
            name='training',
            job_script=get_scripts_dir() / 'training' / 'train_pt-profile.py',
            arguments={
                'experiment-dir': experiment_dir.as_posix(),
                'run-dir': run_dir.as_posix(),
                'random-seed': random_seed,
            },
            bid=args.bid,
        )
        file_path = htcondor_dir / 'training.sub'
        submit_file.save(file_path=file_path)
        dag_file.add_submit_file(
            name='training',
            attributes=dict(
                file_path=file_path.as_posix(),
                bid=args.bid,
                done=args.no_training,
            ),
        )
        print('Done!', flush=True)

        # Create submit file for evaluation job and add it to DAG file
        print('  Creating submit file for evaluation...', end=' ', flush=True)
        submit_file = SubmitFile(
            log_dir=htcondor_dir,
            memory=32_768,
            cpus=4,
        )
        submit_file.add_job(
            name='evaluation',
            job_script=(
                get_scripts_dir()
                / 'evaluation'
                / 'evaluate-with-ultranest.py'
            ),
            arguments={
                'experiment-dir': experiment_dir.as_posix(),
                'run-dir': run_dir.as_posix(),
                'n-splits': args.n_evaluation_splits,
                'split-idx': '$(Process)',
                'random-seed': random_seed,
            },
            bid=args.bid,
            queue=int(args.n_evaluation_splits),
        )
        file_path = htcondor_dir / 'evaluation.sub'
        submit_file.save(file_path=file_path)
        dag_file.add_submit_file(
            name='evaluation',
            attributes=dict(file_path=file_path.as_posix(), bid=args.bid),
        )
        print('Done!', flush=True)

        # Create submit file for merging job and add it to DAG file
        print('  Creating submit file for merging...', end=' ', flush=True)
        submit_file = SubmitFile(
            log_dir=htcondor_dir,
            memory=16_384,
            cpus=2,
        )
        submit_file.add_job(
            name='merging',
            job_script=(
                get_scripts_dir()
                / 'evaluation'
                / 'merge-partial-hdf-files.py'
            ),
            arguments={
                'run-dir': run_dir.as_posix(),
            },
            bid=args.bid,
        )
        file_path = htcondor_dir / 'merging.sub'
        submit_file.save(file_path=file_path)
        dag_file.add_submit_file(
            name='merging',
            attributes=dict(file_path=file_path.as_posix(), bid=args.bid),
        )
        print('Done!', flush=True)

        # Add dependencies to DAG file and save file
        print('  Saving DAG file...', end=' ', flush=True)
        dag_file.add_dependency('training', 'evaluation')
        dag_file.add_dependency('evaluation', 'merging')
        file_path = htcondor_dir / 'run.dag'
        dag_file.save(file_path=file_path)
        print('Done!', flush=True)

        # Submit DAG file to cluster
        if not args.dry_run:
            print('  Submitting DAG file...', end=' ', flush=True)
            submit_dag(file_path=file_path, force=args.force)
            print('Done!', flush=True)
        else:
            print('  Skipping submission of DAG file (dry run)!', flush=True)

        print('\n')

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'Done! This took {time.time() - script_start:.1f} seconds.\n')

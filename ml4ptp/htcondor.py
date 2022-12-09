"""
Methods for working with the HTCondor cluster system.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Deque, Dict, List, Optional, Sequence, Set, Union
from pathlib import Path

import subprocess
import sys

from collections import deque


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class SubmitFile:
    """
    A class that provides the methods to comfortably create a submit
    file for the HTCondor cluster.
    """

    def __init__(
        self,
        log_dir: Optional[Union[Path, str]] = None,
        executable: str = sys.executable,
        getenv: bool = True,
        memory: int = 8192,
        cpus: int = 1,
        gpus: int = 0,
        requirements: Sequence[str] = (),
    ):
        """
        Initialize a new submit file object.

        Args:
            log_dir: The path to the folder where the log files for this
                submit file will be stored. If `None` is given, no logs
                will be created and all output will simply be forwarded
                to ``/dev/null``.
            executable: The executable parameter for the submit file.
            memory: How much memory (in MB) to request from the cluster.
            cpus: The number of CPUs to request from the cluster.
            gpus: The number of GPUs to request from the cluster.
            requirements: A sequence (usually: a list) with any
                additional requirements, for example, limitations on the
                available GPU memory, or the machines on which to run
                (e.g., `"Target.CpuFamily =!= 21"` to block certain
                types of nodes from running the job).
        """

        # Store options for this submit file
        self.executable = executable
        self.getenv = getenv
        self.memory = memory
        self.cpus = cpus
        self.gpus = gpus
        self.requirements = requirements
        self.jobs: List[dict] = []

        # Make sure that the log_dir exists
        if log_dir is not None:
            self.log_dir: Optional[Path] = Path(log_dir)
            self.log_dir.mkdir(exist_ok=True, parents=True)
        else:
            self.log_dir = None

    def add_job(
        self,
        name: str,
        job_script: Union[str, Path],
        arguments: Dict[str, str],
        bid: int = 1,
        queue: int = 1,
    ) -> None:

        self.jobs.append(
            dict(
                name=name,
                job_script=Path(job_script).as_posix(),
                arguments=arguments,
                bid=bid,
                queue=queue,
            )
        )

    def __str__(self) -> str:

        # Collect default header for all submit files
        contents = [
            '#' + 78 * '-',
            '# GENERAL JOB REQUIREMENTS',
            '#' + 78 * '-' + '\n',
            f'executable = {self.executable}',
            f'getenv = {self.getenv}\n',
            f'request_memory = {self.memory}',
            f'request_cpus = {self.cpus}\n',
        ]

        # Only add request_gpus parameter if we are actually requesting GPUs
        if self.gpus > 0:
            contents.append(f'request_gpus = {self.gpus}\n')

        # Add requirements (e.g. CUDAGlobalMemory, black hole machines, ...)
        requirements_string = ' && '.join(list(self.requirements))
        if requirements_string != '':
            contents.append(f'requirements = {requirements_string}\n')

        contents.append('#' + 78 * '-' + '\n\n\n')

        # Loop over all jobs and add them to the submit file
        for job in self.jobs:

            # Add some more formatting to the submit files to make them
            # easier to read
            contents.append('#' + 78 * '-')
            contents.append(f'# {job["name"].upper()}')
            contents.append('#' + 78 * '-' + '\n')

            # Add output, error and log file for job
            contents.append('# Logging Information')
            if self.log_dir is not None:
                contents.append(
                    f'output = {self.log_dir.as_posix()}/'
                    f'{job["name"]}.$(Process).out'
                )
                contents.append(
                    f'error = {self.log_dir.as_posix()}/'
                    f'{job["name"]}.$(Process).err'
                )
                contents.append(
                    f'log = {self.log_dir.as_posix()}/'
                    f'{job["name"]}.$(Process).log'
                )
            else:
                contents.append('output = /dev/null')
                contents.append('error = /dev/null')
                contents.append('log = /dev/null')
            contents.append('')

            # Add actual job and arguments
            contents.append('# Actual arguments defining the job')
            arguments = ' '.join(
                [
                    f'--{k} {job["arguments"][k]}'
                    for k in sorted(job['arguments'].keys())
                ]
            )
            contents.append(f'arguments = {job["job_script"]} {arguments}\n')

            # Add the job priority / bid
            contents.append('# Job Priority')
            contents.append(f'priority = {job["bid"] - 1000}\n')

            # Queue the job
            contents.append('# Add this job to the queue')
            contents.append(f'queue {job["queue"]}\n')
            contents.append('#' + 78 * '-' + '\n\n\n')

        return '\n'.join(contents)

    def save(self, file_path: Union[Path, str]) -> None:

        with open(file_path, 'w') as submit_file:
            submit_file.write(self.__str__())


class Node:
    """
    Auxiliary class to represent nodes in a directed acyclic graph.
    """

    def __init__(self, name: str, attributes: dict) -> None:
        self.name = name
        self.attributes = attributes
        self.dependent_nodes: Set[str] = set()

    def add_dependent_node(self, child_node_name: str) -> None:
        self.dependent_nodes.add(child_node_name)


class DAGFile:
    """
    Create submit files for HTCondors DAGman.
    """

    def __init__(self) -> None:
        self.graph: Dict[str, Node] = dict()

    @property
    def nodes(self) -> List[Node]:
        return list(self.graph.values())

    @property
    def node_names(self) -> List[str]:
        return list(self.graph.keys())

    @property
    def dependent_nodes(self) -> Set[Node]:
        dependent_nodes: Set[Node] = set()
        for node in self.nodes:
            dependencies = set(self.graph[_] for _ in node.dependent_nodes)
            dependent_nodes = dependent_nodes.union(dependencies)
        return dependent_nodes

    @property
    def independent_nodes(self) -> Set[Node]:
        return set(self.nodes).difference(self.dependent_nodes)

    def get_node_by_name(self, node_name: str) -> Node:
        return self.graph[node_name]

    def check_validity(self) -> None:

        # Make sure there's at least one independent node as a starting point
        if len(self.independent_nodes) == 0:
            raise ValueError('No independent nodes!')

        # Make sure the graph is acyclic (this will raise a ValueError if not)
        self.topological_sort()

    def add_submit_file(self, name: str, attributes: Optional[dict]) -> None:

        if name in self.node_names:
            raise KeyError(f'Node "{name}" already exists!')

        if attributes is None:
            attributes = dict(file_path=None, bid=1)
        if attributes is not None and 'file_path' not in attributes.keys():
            raise ValueError('attributes is missing file_path!')
        if attributes is not None and 'bid' not in attributes.keys():
            attributes['bid'] = 1

        if attributes is not None:
            node = Node(name=name, attributes=attributes)
            self.graph[name] = node

    def add_dependency(
        self, parent_node_name: str, child_node_name: str
    ) -> None:

        self.graph[parent_node_name].add_dependent_node(child_node_name)
        self.check_validity()

    def topological_sort(self) -> List[Node]:

        in_degree = {}
        for node_name in self.node_names:
            in_degree[node_name] = 0

        for node in self.nodes:
            for dependent_node_name in node.dependent_nodes:
                in_degree[dependent_node_name] += 1

        queue: Deque[str] = deque()
        for node_name in in_degree.keys():
            if in_degree[node_name] == 0:
                queue.appendleft(node_name)

        sorted_nodes = []
        while queue:
            node_name = queue.pop()
            sorted_nodes.append(self.graph[node_name])
            for dependent_node_name in self.graph[node_name].dependent_nodes:
                in_degree[dependent_node_name] -= 1
                if in_degree[dependent_node_name] == 0:
                    queue.appendleft(dependent_node_name)

        if len(sorted_nodes) == len(self.nodes):
            return sorted_nodes
        raise ValueError('Graph is not acyclic!')

    def __str__(self) -> str:

        contents = list()

        # Add all jobs (i.e., nodes); potentially mark them as DONE
        for node in self.topological_sort():
            done = 'DONE' if node.attributes.get('done', False) else ''
            contents.append(
                f'JOB {node.name} {node.attributes["file_path"]} {done}'
            )

        contents.append('')

        # Add all dependencies (i.e, edges)
        for node in self.topological_sort():
            for dependency in node.dependent_nodes:
                contents.append(f'PARENT {node.name} CHILD {dependency}')

        contents.append('')

        # Add bids for submit files
        for node in self.topological_sort():
            bid = node.attributes["bid"] - 1000
            contents.append(f'PRIORITY {node.name} {bid}')

        return '\n'.join(contents)

    def save(self, file_path: Union[Path, str]) -> None:

        with open(file_path, 'w') as dag_file:
            dag_file.write(self.__str__())


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def submit_dag(file_path: Path, force: bool = False) -> None:
    """
    Submit a given DAG file as a workflow on the cluster.

    Args:
        file_path: Path to the *.dag file.
        force: If True, overwrite existing DAG files.
    """

    # Define the command to launch the DAG; add the force flag if requested
    if force:
        cmd = ['condor_submit_dag', '-force', file_path.as_posix()]
    else:
        cmd = ['condor_submit_dag', file_path.as_posix()]

    # Otherwise, we actually launch the workflow
    p = subprocess.run(args=cmd, capture_output=True)

    # Check if the workflow was successfully launched
    if p.returncode != 0:
        raise RuntimeError(
            f'Error launching DAG workflow "{file_path}":\n '
            f'{p.stderr.decode()}'
        )

    return None

"""
Tests for htcondor.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import pytest

from ml4ptp.htcondor import (
    DAGFile,
    SubmitFile,
)


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

@pytest.fixture()
def htcondor_dir(tmp_path: Path) -> Path:
    htcondor_dir = tmp_path / 'htcondor'
    htcondor_dir.mkdir()
    return htcondor_dir


def test__submit_file(htcondor_dir: Path) -> None:

    # Case 1
    submit_file = SubmitFile(
        log_dir=htcondor_dir,
        memory=1024,
        cpus=4,
    )
    submit_file.add_job(
        name='dummy_name',
        job_script='dummy_script',
        arguments={'argument_1': '0'},
        bid=10,
        queue=2,
    )
    file_path = htcondor_dir / 'submit_file.sub'
    submit_file.save(file_path)
    assert file_path.exists()

    # Case 2
    submit_file = SubmitFile(
        log_dir=None,
        requirements=['Some requirements'],
        gpus=2,
    )
    submit_file.add_job(
        name='dummy_name',
        job_script='dummy_script',
        arguments={},
    )
    assert 'request_gpus = 2' in str(submit_file)


def test__dag_file(htcondor_dir: Path) -> None:
    """
    Test `hsr4hci.htcondor.DAGFile`.
    """

    # Case 1
    dag_file = DAGFile()
    dag_file.add_submit_file(name='job_1', attributes=None)
    dag_file.add_submit_file(name='job_2', attributes=None)
    dag_file.add_submit_file(name='job_3', attributes=None)
    dag_file.add_dependency(parent_node_name='job_1', child_node_name='job_2')
    dag_file.add_dependency(parent_node_name='job_2', child_node_name='job_3')
    assert len(dag_file.nodes) == 3
    assert dag_file.node_names == ['job_1', 'job_2', 'job_3']
    assert [_.name for _ in dag_file.independent_nodes] == ['job_1']
    assert sorted([_.name for _ in dag_file.dependent_nodes]) == [
        'job_2',
        'job_3',
    ]

    # Case 2
    file_path = htcondor_dir / 'dag_file.dag'
    dag_file.save(file_path=file_path)
    assert file_path.exists()

    # Case 3
    with pytest.raises(ValueError) as value_error:
        dag_file.add_submit_file(name='job_4', attributes={})
    assert 'attributes is missing file_path' in str(value_error)

    # Case 4
    dag_file.add_submit_file(name='job_4', attributes={'file_path': ''})
    assert dag_file.get_node_by_name('job_4').attributes['bid'] == 1

    # Case 5
    dag_file = DAGFile()
    dag_file.add_submit_file(name='job_1', attributes=None)
    dag_file.add_submit_file(name='job_2', attributes=None)
    dag_file.add_dependency(parent_node_name='job_1', child_node_name='job_2')
    with pytest.raises(ValueError) as value_error:
        dag_file.add_dependency(
            parent_node_name='job_2', child_node_name='job_1'
        )
    assert 'No independent nodes!' in str(value_error)

    # Case 5
    dag_file = DAGFile()
    dag_file.add_submit_file(name='job_1', attributes=None)
    dag_file.add_submit_file(name='job_2', attributes=None)
    dag_file.add_submit_file(name='job_3', attributes=None)
    dag_file.add_dependency(parent_node_name='job_1', child_node_name='job_2')
    dag_file.add_dependency(parent_node_name='job_2', child_node_name='job_3')
    with pytest.raises(ValueError) as value_error:
        dag_file.add_dependency(
            parent_node_name='job_3', child_node_name='job_2'
        )
    assert 'Graph is not acyclic!' in str(value_error)

    # Case 6
    dag_file = DAGFile()
    dag_file.add_submit_file(name='job_1', attributes=None)
    with pytest.raises(KeyError) as key_error:
        dag_file.add_submit_file(name='job_1', attributes=None)
    assert 'Node "job_1" already exists!' in str(key_error)

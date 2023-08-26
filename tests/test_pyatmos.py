"""
Unit tests for pyatmos.py
"""

from pathlib import Path

import numpy as np

from ml4ptp.pyatmos import collect_data_for_hash


def test__collect_data_for_hash(tmp_path: Path) -> None:

    # Create some fake file
    model_hash = "9fa06fb22ed2b3375148850d6a9771d9"
    directory = tmp_path / "dir_9" / model_hash
    directory.mkdir(parents=True)
    file_path = directory / "parsed_clima_final.csv"
    with open(file_path, "w") as f:
        f.write(
            "P,T,ALT,CONVEC\n"
            "1,2,3,4\n"
            "5,6,7,8\n"
            "9,10,11,12\n"
        )
    file_path = directory / "parsed_photochem_mixing_ratios.csv"
    with open(file_path, "w") as f:
        f.write(
            "Z,X\n"
            "1,0\n"
            "2,0\n"
            "3,0\n"
        )

    # Run function
    results = collect_data_for_hash(
        input_dir=tmp_path,
        model_hash=model_hash,
    )

    assert set(results.keys()) == {"hash", "P", "T", "ALT", "CONVEC", "X"}
    assert results["hash"] == model_hash
    assert np.array_equal(results["P"], np.array([1, 5, 9]))
    assert np.array_equal(results["T"], np.array([2, 6, 10]))
    assert np.array_equal(results["ALT"], np.array([3, 7, 11]))
    assert np.array_equal(results["CONVEC"], np.array([4, 8, 12]))
    assert np.array_equal(results["X"], np.array([0, 0, 0]))

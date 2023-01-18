"""
Unit tests for guillot.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np

from ml4ptp.guillot import guillot_profile, fit_profile_with_guillot


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

def test__guillot_profile() -> None:

    P = np.geomspace(1e-6, 1.27e2, 100)
    delta = 6.826884344700991e-05
    gamma = 1.3494876256095905
    T_int = 194.09524648988
    T_equ = 2135.866521371579

    # Case 1: with numba enabled
    T = guillot_profile(P, delta, gamma, T_int, T_equ)
    assert np.isclose(np.mean(T), 2112.3381073472992)

    # Case 2: with numba disabled
    T = guillot_profile.py_func(P, delta, gamma, T_int, T_equ)
    assert np.isclose(np.mean(T), 2112.3381073472992)


def test__fit_profile_with_guillot() -> None:

    # Case 1
    P = np.geomspace(1e-6, 1.27e2, 100)
    delta = 6.826884344700991e-05
    gamma = 1.3494876256095905
    T_int = 194.09524648988
    T_equ = 2135.866521371579
    T_true = guillot_profile(P, delta, gamma, T_int, T_equ)
    result = fit_profile_with_guillot(0, P, T_true, n_runs=4, random_seed=5)

    assert np.isclose(result.mse, 0.0)
    assert np.isclose(result.delta, delta, atol=0.01)
    assert np.isclose(result.gamma, gamma, atol=0.01)
    assert np.isclose(result.T_int, T_int, atol=0.01)
    assert np.isclose(result.T_equ, T_equ, atol=0.01)

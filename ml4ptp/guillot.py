"""
Methods for the PT profile parameterization from Guillot (2010):

    Guillot, T. (2010). "On the radiative equilibrium of irradiated
    planetary atmospheres." Astronomy & Astrophysics, 520, A27.
    DOI: 10.1051/0004-6361/200913396

"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from dataclasses import dataclass

import time

from numba import jit
from scipy.optimize import minimize

import nevergrad as ng
import numpy as np


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

@jit(nopython=True)
def guillot_profile(
    P: np.ndarray,
    delta: float,
    gamma: float,
    T_int: float,
    T_equ: float,
) -> np.ndarray:
    """
    Guillot (2010) model for the PT profile.

    This implementation is taken directly from the petitRADTRANS
    implementation (commit hash: 0dfc339a) available at:

        https://gitlab.com/mauricemolli/petitRADTRANS/-/blob/
            master/petitRADTRANS/physics.py

    We are using the modified version which replaces `kappa_IR / grav`
    by `delta` (see `guillot_global_ret()` in the petitRADTRANS code),
    as the former cannot be uniquely determined by the fit.

    Using numba's `@jit` decorator gives a (basically free) speed-up of
    a factor of ~6, from ca. 36 ms to ca. 6 ms for a single profile.

    Args:
        P: A numpy array of floats with the input pressures in bars.
        delta: The ratio of `kappa_IR` (i.e., the infrared opacity in
            units of cm^2 / s) to `grav` (i.e., the planetary surface
            gravity in units of cm / s^2).
        gamma: The ratio between the visual and infrared opacity.
        T_int: The planetary internal temperature in units of K.
        T_equ: The planetary equilibrium temperature in units of K.

    Returns:
        A numpy array of floats with the predicted temperatures in K.
    """

    # All parameters are positive
    delta = np.abs(delta)
    gamma = np.abs(gamma)
    T_int = np.abs(T_int)
    T_equ = np.abs(T_equ)

    # Compute the temperature profile
    # Note: If the `np.exp()` function is giving overflow errors, try using
    # `np.exp(np.clip(-gamma * tau * 3.0**0.5, -88.72, 88.72))` instead.
    tau = P * 1e6 * delta
    T_irr = T_equ * np.sqrt(2.0)
    T = (
        0.75 * T_int**4.0 * (2.0 / 3.0 + tau)
        + 0.75
        * T_irr**4.0
        / 4.0
        * (
            2.0 / 3.0
            + 1.0 / gamma / 3.0**0.5
            + (gamma / 3.0**0.5 - 1.0 / 3.0**0.5 / gamma)
            * np.exp(
                -gamma * tau * 3.0**0.5
            )
        )
    ) ** 0.25

    return np.asarray(T)


@dataclass
class FitResult:

    idx: int
    P_true: np.ndarray
    T_true: np.ndarray
    T_pred: np.ndarray
    delta: float = np.nan
    gamma: float = np.nan
    T_int: float = np.nan
    T_equ: float = np.nan
    mse: float = np.nan
    mre: float = np.nan
    runtime: float = np.nan


def fit_profile_with_guillot(
    idx: int,
    P_true: np.ndarray,
    T_true: np.ndarray,
    n_runs: int = 100,
    random_seed: int = 42,
) -> FitResult:

    # Preliminaries
    start_time = time.time()
    np.random.seed(random_seed)

    # Define the optimization target = MSE loss between true and predicted T
    def loss(x: np.ndarray) -> float:
        delta, gamma, T_int, T_equ = x
        mse = np.mean(
            (T_true - guillot_profile(P_true, delta, gamma, T_int, T_equ)) ** 2
        )
        return float(np.clip(mse / 1_000, 0, 1e6))

    # Keep track of the best-fit parameters
    best_x = np.zeros(4)
    best_mse = np.inf

    # Run the optimization multiple times
    for _ in range(n_runs):

        # Get initial guess
        optimizer = ng.optimizers.TwoPointsDE(
            parametrization=4, budget=1000
        )
        recommendation = optimizer.minimize(loss)

        # Refine the initial guess
        x = minimize(loss, x0=recommendation.value, method='Powell').x
        x = np.abs(x)
        mse = loss(x)

        # If the current guess is better than the best guess so far, keep it
        if mse < best_mse:
            best_x = x
            best_mse = mse

    # Compute best-fit PT profile and error metrics
    T_pred = guillot_profile(P_true, *best_x)
    mse = float(np.mean((T_true - T_pred) ** 2))
    mre = float(np.mean(np.abs(T_true - T_pred) / T_true))

    fit_result = FitResult(
        idx=idx,
        P_true=P_true,
        T_true=T_true,
        T_pred=T_pred,
        delta=best_x[0],
        gamma=best_x[1],
        T_int=best_x[2],
        T_equ=best_x[3],
        mse=mse,
        mre=mre,
        runtime=(time.time() - start_time),
    )

    return fit_result

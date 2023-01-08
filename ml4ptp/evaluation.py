"""
Functions for evaluation trained models.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from sys import stderr

import logging
import time
import traceback

from scipy.special import erf, erfinv

import numpy as np
import ultranest

from ml4ptp.onnx import ONNXEncoder, ONNXDecoder
from ml4ptp.timeout import timelimit, TimeoutException


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """
    Simple dataclass for storing the results of an evaluation.
    """

    idx: int
    log_P: np.ndarray
    T_true: np.ndarray
    z_initial: np.ndarray = np.empty(0)
    z_refined: np.ndarray = np.empty(0)
    T_pred_initial: np.ndarray = np.empty(0)
    T_pred_refined: np.ndarray = np.empty(0)
    mre_initial: float = np.nan
    mre_refined: float = np.nan
    mse_initial: float = np.nan
    mse_refined: float = np.nan
    niter: int = -1
    ncall: int = -1
    success: int = 0
    runtime: float = np.nan
    timeout : int = 0


def find_optimal_z_with_ultranest(
    log_P: np.ndarray,
    T_true: np.ndarray,
    idx: int,
    encoder_bytes: bytes,
    decoder_bytes: bytes,
    random_seed: int,
    n_live_points: int = 400,
    n_max_calls: int = 500_000,
    timeout: int = 600,
    prior: str = 'uniform',
    limit: float = 4.0,
) -> EvaluationResult:
    """
    Use nested sampling to find the optimal latent variable `z` for a
    given PT profile.

    Args:
        log_P: The logarithm the pressure value of the profile.
        T_true: The temperature value of the profile.
        idx: The index of the profile.
        encoder_bytes: The ONNX encoder model as a byte string.
        decoder_bytes: The ONNX decoder model as a byte string.
        random_seed: The random seed to use.
        n_live_points: The number of live points to use.
        n_max_calls: The maximum number of calls to the likelihood
            function.
        timeout: The maximum number of seconds for the nested sampling.
        prior: The prior to use. Can be either 'uniform' or 'gaussian',
            where the latter is a truncated Gaussian prior with mean 0
            and standard deviation 1.
        limit: The limit of the prior. For a uniform prior, this is the
            range of the prior: `[-limit, limit]`. For the (truncated)
            normal prior, this is the maximum norm of the proposed `z`.

    Returns:
        An `EvaluationResult` instance containing the results of the
        evaluation using nested sampling.
    """

    # Start the stopwatch
    start_time = time.time()

    # Fix the shapes of the inputs
    log_P = log_P.reshape(1, -1)
    T_true = T_true.reshape(1, -1)

    # Load encoder and decoder from byte strings
    encoder = ONNXEncoder(encoder_bytes)
    decoder = ONNXDecoder(decoder_bytes)

    # Prepare result object
    result = EvaluationResult(
        idx=idx,
        log_P=log_P,
        T_true=T_true,
    )

    # -------------------------------------------------------------------------
    # Get initial guess for z from encoder and compute error
    # -------------------------------------------------------------------------

    # Get initial guess for z from encoder, and compute predicted T
    z_initial = encoder(log_P=log_P, T=T_true)
    T_pred_initial = decoder(log_P=log_P, z=z_initial)

    # Compute mean squared error (MSE) and mean relative error (MRE)
    mse_initial = float(np.mean((T_pred_initial - T_true) ** 2))
    mre_initial = float(np.mean(np.abs(T_pred_initial - T_true) / T_true))

    # Store results
    result.z_initial = z_initial
    result.T_pred_initial = T_pred_initial
    result.mse_initial = mse_initial
    result.mre_initial = mre_initial

    # -------------------------------------------------------------------------
    # Define prior and likelihood
    # -------------------------------------------------------------------------

    def prior_transform(cube: np.ndarray) -> np.ndarray:
        """
        Transformation of the unit hypercube to the prior.
        Supports either a uniform prior or a truncated Gaussian prior.
        """

        z = cube.copy()
        for i in range(z.shape[1]):

            if prior == 'uniform':
                z[:, i] = 2 * limit * (z[:, i] - 0.5)

            elif prior == 'gaussian':
                factor = erf(np.sqrt(2) * limit / 2)
                z[:, i] = np.sqrt(2) * erfinv((2 * z[:, i] - 1) * factor)

            else:
                raise ValueError(f'Invalid prior "{prior}"!')

        return z

    def likelihood(z: np.ndarray) -> np.ndarray:
        """
        Likelihood for comparing PT profiles (= negative MSE).
        """

        log_P_ = np.tile(log_P, (z.shape[0], 1))
        T_true_ = np.tile(T_true, (z.shape[0], 1))

        T_pred = decoder(log_P=log_P_, z=z)
        mse = np.asarray(np.mean((T_true_ - T_pred) ** 2, axis=1))

        if np.isnan(mse).any():  # pragma: no cover
            return -1e300 * np.ones_like(mse)
        else:
            return -mse

    # -------------------------------------------------------------------------
    # Set up nested sampling and run
    # -------------------------------------------------------------------------

    # Disable extensive logging from ultranest
    logger = logging.getLogger("ultranest")
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.WARNING)

    # Run ultranest
    # In a few, rare cases, this fails with rather creative, hard-to-debug
    # errors, so we try it multiple times, and if it keeps failing, we return
    # the initial guess.
    n_tries = 0
    while n_tries < 3:

        try:

            # Set random seed
            np.random.seed(random_seed + n_tries)

            # Set up sampler
            # This needs to re-created for each try, otherwise we do not get
            # different results for the different tries
            sampler = ultranest.ReactiveNestedSampler(
                param_names=[f'z{i}' for i in range(z_initial.shape[1])],
                loglike=likelihood,
                transform=prior_transform,
                vectorized=True,
            )

            # Run sampler
            # We limit the runtime to make sure that the evaluation will
            # terminate in a reasonable and predictable amount of time.
            with timelimit(timeout):

                # `frac_remain=0.05` seems to help with cases where the
                # likelihood has a plateau, which results in many live points
                # being removed without replacement, which then leads to an
                # error when `nlive` becomes zero.
                # noinspection PyTypeChecker
                posterior = sampler.run(
                    min_num_live_points=n_live_points,
                    show_status=False,
                    viz_callback=False,
                    max_ncalls=n_max_calls,
                    frac_remain=0.05,
                )

            # If we get here, the evaluation was successful and did not time
            # out, so we can continue with the results we have obtained, and
            # break out of the loop
            z_refined = np.asarray(posterior['maximum_likelihood']['point'])
            result.z_refined = z_refined.squeeze()
            result.ncall = int(posterior['ncall'])
            result.niter = int(posterior['niter'])
            result.success = int(
                posterior['insertion_order_MWW_test']['converged']
            )
            break

        except Exception as e:  # pragma: no cover

            # Check what kind of error has brought us here:

            # If we have reached a time limit, we abort the evaluation and
            # return the initial guess, because trying again will probably
            # not help ...
            if isinstance(e, TimeoutException):

                print(f'Evaluation timed out! (profile {idx})', file=stderr)
                z_refined = z_initial
                result.z_refined = z_refined.squeeze()
                result.timeout = 1
                break

            # Otherwise, we can log the error and try again
            else:

                print(
                    f'\nError in ultranest (profile {idx}): {str(e)}\n',
                    file=stderr
                )
                print(traceback.format_exc(), file=stderr)
                n_tries += 1

    # The `else` clause is executed if we do not `break` out of the while loop,
    # that is, if we did NOT find a solution with nested sampling, and did NOT
    # time out. In this case, we return the initial guess.
    else:  # pragma: no cover

        print(
            f'Failed to run ultranest! Using initial guess for profile {idx}',
            file=stderr
        )

        z_refined = z_initial
        result.z_refined = z_refined.squeeze()

    # -------------------------------------------------------------------------
    # Decode refined z and compute error
    # -------------------------------------------------------------------------

    # Decode refined z
    T_pred_refined = decoder(log_P=log_P, z=np.atleast_2d(z_refined))

    # Compute mean squared error (MSE) and mean relative error (MRE)
    mse_refined = float(np.mean((T_pred_refined - T_true) ** 2))
    mre_refined = float(np.mean(np.abs(T_pred_refined - T_true) / T_true))

    # Store results
    result.T_pred_refined = T_pred_refined
    result.mse_refined = mse_refined
    result.mre_refined = mre_refined
    result.runtime = float(time.time() - start_time)

    return result

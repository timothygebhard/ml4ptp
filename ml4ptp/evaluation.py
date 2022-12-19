"""
Functions for evaluation trained models.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Dict, Union

import logging
import traceback

import numpy as np
import ultranest

from ml4ptp.onnx import ONNXEncoder, ONNXDecoder


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

def find_optimal_z_with_ultranest(
    log_P: np.ndarray,
    T_true: np.ndarray,
    idx: int,
    encoder_bytes: bytes,
    decoder_bytes: bytes,
    random_seed: int,
    n_live_points: int = 400,
    n_max_calls: int = 500_000,
) -> Dict[str, Union[int, float, np.ndarray]]:
    """
    Use nested sampling to find the optimal latent variable `z` for a
    given PT profile.
    """

    # Fix the shapes of the inputs
    log_P = log_P.reshape(1, -1)
    T_true = T_true.reshape(1, -1)

    # Load encoder and decoder from byte strings
    encoder = ONNXEncoder(encoder_bytes)
    decoder = ONNXDecoder(decoder_bytes)

    # Prepare dict with results
    results: Dict[str, Union[int, float, np.ndarray]] = dict(
        idx=idx,
        log_P=log_P,
        T_true=T_true,
    )

    # -------------------------------------------------------------------------
    # Get initial guess for z from encoder and compute error
    # -------------------------------------------------------------------------

    # Get initial guess for z from encoder
    z_initial = encoder(log_P=log_P, T=T_true)

    print('\n\n')
    print('z_initial', z_initial, z_initial.shape)

    T_pred_initial = decoder(log_P=log_P, z=z_initial)

    # Compute mean squared error (MSE) and mean relative error (MRE)
    mse_initial = float(np.mean((T_pred_initial - T_true) ** 2))
    mre_initial = float(np.mean(np.abs(T_pred_initial - T_true) / T_true))

    # Save results
    results['z_initial'] = z_initial
    results['T_pred_initial'] = T_pred_initial
    results['mse_initial'] = mse_initial
    results['mre_initial'] = mre_initial

    # -------------------------------------------------------------------------
    # Define prior and likelihood
    # -------------------------------------------------------------------------

    def prior(cube: np.ndarray) -> np.ndarray:
        """
        Prior for z. (Currently uniform in [-5, 5].)
        """

        z = cube.copy()
        for i in range(z.shape[1]):
            z[:, i] = 10 * (z[:, i] - 0.5)

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
                transform=prior,
                vectorized=True,
            )

            # `frac_remain=0.05` seems to help with cases where the likelihood
            # has a plateau, which results in many live points being removed
            # without replacement, which then leads to an error when `nlive`
            # becomes zero.
            # noinspection PyTypeChecker
            result = sampler.run(
                min_num_live_points=n_live_points,
                show_status=False,
                viz_callback=False,
                max_ncalls=n_max_calls,
                frac_remain=0.05,
            )

            z_refined = np.asarray(result['maximum_likelihood']['point'])
            results['z_refined'] = z_refined.squeeze()
            results['ncall'] = int(result['ncall'])
            results['niter'] = int(result['niter'])
            results['success'] = int(
                result['insertion_order_MWW_test']['converged']
            )

            # If we get here, we have successfully run the sampler, so we can
            # break out of the loop
            break

        except Exception as e:  # pragma: no cover

            print(f'\nError in ultranest (profile {idx}): {str(e)}\n')
            print(traceback.format_exc())
            n_tries += 1

    # The `else` clause is executed if we do not `break` out of the while loop,
    # that is, if we did NOT find a solution with nested sampling.
    else:  # pragma: no cover

        print('\n\nFailed to run ultranest!')
        print(f'Using initial guess for profile {idx}.\n\n')

        z_refined = z_initial
        results['z_refined'] = z_refined.squeeze()
        results['ncall'] = -1
        results['niter'] = -1
        results['success'] = 0

    # -------------------------------------------------------------------------
    # Decode refined z and compute error
    # -------------------------------------------------------------------------

    # Decode refined z
    T_pred_refined = decoder(log_P=log_P, z=np.atleast_2d(z_refined))

    # Compute mean squared error (MSE) and mean relative error (MRE)
    mse_refined = float(np.mean((T_pred_refined - T_true) ** 2))
    mre_refined = float(np.mean(np.abs(T_pred_refined - T_true) / T_true))

    # Save results
    results['T_pred_refined'] = T_pred_refined
    results['mse_refined'] = mse_refined
    results['mre_refined'] = mre_refined

    return results

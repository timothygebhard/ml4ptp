"""
Implementation of the MMD metric from Gretton et al. (2012).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Optional

import torch


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

def compute_mmd(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute the MMD metric (assuming a Gaussian kernel) between two
    samples `x` and `y`, as given by eq. (3) in Gretton et al. (2012):

        A. Gretton, K. Borgwardt, M. Rasch, B. Schölkopf, and A. Smola:
        "A Kernel Two-Sample Test."
        Journal of Machine Learning Research 13, 723-773 (2012).
        https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf

    Note: Technically, this functions returns the *squared* MMD, which
    may still be negative (because it is an unbiased estimator).

    Args:
        x: A tensor of shape (n, d) containing the first sample.
        y: A tensor of shape (m, d) containing the second sample.
        sigma: The bandwidth of the Gaussian kernel. If `None`, the
            median heuristic is used to estimate `sigma` automatically.

    Returns:
        The unbiased squared MMD between `x` and `y`.
    """

    # Define shortcuts for lengths of samples
    m = len(x)
    n = len(y)

    # If no sigma is given, compute it from the samples using the median
    # heuristic from Gretton et al. (2012), where sigma is given as one half
    # of the median of the distances between the points in the combined sample.
    if sigma is None:
        distances = torch.pdist(torch.cat([x, y], dim=0))
        sigma = float(distances.median() / 2)

    # Compute term 1: K(X, X)
    # - `dists` contains a matrix of shape (n, n) with the pairwise Euclidean
    #   distances between the points in `x`.
    # - The Gaussian kernel is given by:
    #   k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
    # - The "-m" term takes care of the x_i != x_j in the double sum by
    #   removing the diagonal, which is always 1 after exponentiation.
    distances = torch.cdist(x, x)
    k_xx = torch.exp((-1 / (2 * sigma**2)) * distances ** 2).sum() - m

    # Compute term 2: K(Y, Y)
    distances = torch.cdist(y, y)
    k_yy = torch.exp((-1 / (2 * sigma**2)) * distances ** 2).sum() - n

    # Compute term 3: K(X, Y)
    distances = torch.cdist(x, y)
    k_xy = torch.exp((-1 / (2 * sigma**2)) * distances ** 2).sum()

    # Combined the three terms to compute the unbiased MMD estimator
    mmd = (
        1 / (m * (m - 1)) * k_xx
        + 1 / (n * (n - 1)) * k_yy
        - 2 / (m * n) * k_xy
    )

    return mmd

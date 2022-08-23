"""
Utility functions that have to do with kernels.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import torch


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

def gaussian_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)

    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)

    kernel_input: torch.Tensor = (tiled_x - tiled_y).pow(2).mean(2)

    return torch.exp(-kernel_input / float(dim))


def compute_mmd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    x_kernel = gaussian_kernel(x, x)
    y_kernel = gaussian_kernel(y, y)
    xy_kernel = gaussian_kernel(x, y)

    return x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()

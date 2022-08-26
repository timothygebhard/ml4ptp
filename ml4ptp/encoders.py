"""
Define encoder architectures.
"""


# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Callable

import torch
import torch.nn as nn

from ml4ptp.layers import Mean
from ml4ptp.mixins import NormalizerMixin


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

class Encoder(nn.Module, NormalizerMixin):

    def __init__(
        self,
        latent_size: int,
        layer_size: int,
        T_mean: float,
        T_std: float,
    ) -> None:

        super().__init__()

        # Store constructor arguments
        self.latent_size = latent_size
        self.layer_size = layer_size
        self.T_mean = T_mean
        self.T_std = T_std

        # Define encoder architecture
        self.layers: Callable[[torch.Tensor], torch.Tensor] = nn.Sequential(
            nn.Conv1d(
                in_channels=2,
                out_channels=self.layer_size,
                kernel_size=(4,),
            ),
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=self.layer_size,
                out_channels=self.layer_size,
                kernel_size=(4,),
            ),
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=self.layer_size,
                out_channels=self.latent_size,
                kernel_size=(4,),
            ),
            Mean(dim=2),
        )

    def forward(self, log_P: torch.Tensor, T: torch.Tensor) -> torch.Tensor:

        # Normalize temperatures and construct encoder input
        normalized_T = self.normalize(T)
        encoder_input = torch.stack((log_P, normalized_T), dim=1)

        # Compute forward pass through encoder to get latent variable z
        z = self.layers(encoder_input)

        return z

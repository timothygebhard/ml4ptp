"""
Define decoder architectures.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from math import prod
from typing import Any, Callable, Dict

import torch
import torch.nn as nn

from ml4ptp.layers import get_mlp_layers, get_activation, Identity
from ml4ptp.mixins import NormalizerMixin


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

class Decoder(nn.Module, NormalizerMixin):
    """
    A standard decoder architecture where the conditioning is achieved
    by concatenating each input (i.e., a pressure values) with `z`.

    Arguments:
        latent_size: The size of the latent space.
        layer_size: The size of the hidden layers of the decoder.
        n_layers: The number of hidden layers of the decoder.
        normalization: A dictionary containing the normalization
            parameters for the pressure and temperature.
        activation: The activation function to use in the decoder.
    """

    def __init__(
        self,
        latent_size: int,
        layer_size: int,
        n_layers: int,
        normalization: Dict[str, Any],
        activation: str = 'leaky_relu',
    ) -> None:

        super().__init__()

        # Store constructor arguments
        self.latent_size = latent_size
        self.layer_size = layer_size
        self.n_layers = n_layers
        self.normalization = normalization

        # Define decoder architecture
        # Note: The `+ 1` on the input is for the (log) pressure at which we
        # want to evaluate the profile represented by this decoder.
        self.layers: torch.nn.Sequential = get_mlp_layers(
            input_size=latent_size + 1,
            n_layers=n_layers,
            layer_size=layer_size,
            output_size=1,
            activation=activation,
            final_tanh=False,
        )

    def forward(self, z: torch.Tensor, log_P: torch.Tensor) -> torch.Tensor:

        # Reminder / step-by-step explanation:
        #  * z has shape `(batch_size, latent_size)`
        #  * log_P has shape `(batch_size, grid_size)`
        #  * The decoder takes inputs of size `latent_size + 1`
        #  * We now need to combine `z` and `log_P` into a single tensor of
        #    shape `(batch_size * grid_size, latent_size + 1)` that we can
        #    give to the decoder
        #  * After passing through the decoder, we want to reshape the output
        #    to `(batch_size, grid_size)` again = one T for each log_P value

        # Get batch size and grid size
        batch_size, grid_size = log_P.shape

        # Repeat z so that we can concatenate it with every pressure value.
        # This changes the shape of z:
        #   (batch_size, latent_size) -> (batch_size, grid_size, latent_size)
        z = z.unsqueeze(1).repeat(1, grid_size, 1)

        # Reshape the pressure grid and z into the batch dimension. They now
        # show have shapes:
        #   p_flat: (batch_size * grid_size, 1)
        #   z_flat: (batch_size * grid_size, latent_size)
        log_P_flat = log_P.reshape(batch_size * grid_size, 1)
        z_flat = z.reshape(batch_size * grid_size, self.latent_size)

        # Normalize log_P and and concatenate to z to form decoder input.
        # The shape should now be:
        #   (batch_size * n_points, 1 + z_dim)
        decoder_input = torch.concat(
            tensors=(
                self.normalize_log_P(log_P_flat),
                z_flat,
            ),
            dim=1,
        )

        # Send through decoder. The decoder output will have shape:
        #   (batch_size * grid_size, 1)
        decoded: torch.Tensor = self.layers(decoder_input)

        # Reshape the decoder output back to to the original shape of log_P
        T_pred = decoded.reshape(batch_size, grid_size)

        # Undo normalization (i.e., transform back to Kelvin)
        T_pred = self.normalize_T(T_pred, undo=True)

        return T_pred


class SkipConnectionsDecoder(nn.Module, NormalizerMixin):
    """
    Similar to the standard decoder, but with skip connections: The
    latent variable is concatenated to the output of each layer.

    Arguments:
        latent_size: The size of the latent space.
        layer_size: The size of the hidden layers of the decoder.
        n_layers: The number of hidden layers of the decoder.
        normalization: A dictionary containing the normalization
            parameters for the pressure and temperature.
        activation: The activation function to use in the decoder.
    """

    def __init__(
        self,
        latent_size: int,
        layer_size: int,
        n_layers: int,
        normalization: Dict[str, Any],
        activation: str = 'leaky_relu',
    ) -> None:

        super().__init__()

        # Store constructor arguments
        self.latent_size = latent_size
        self.layer_size = layer_size
        self.n_layers = n_layers
        self.normalization = normalization
        self.activation_function = get_activation(activation)

        # Collect list of layers without activation functions, and excluding
        # the final layer (see below)
        self.layers = nn.ModuleList(
            [nn.Linear(latent_size + 1, layer_size - latent_size)]
        )
        for i in range(n_layers):
            self.layers.append(nn.Linear(layer_size, layer_size - latent_size))

        # Define final layer
        self.final_layer = nn.Linear(layer_size, 1)

    def forward(self, z: torch.Tensor, log_P: torch.Tensor) -> torch.Tensor:

        # Most of this is analogous to the standard `Decoder` class, so see
        # there for a more detailed explanation.

        # Get batch size and grid size
        batch_size, grid_size = log_P.shape

        # Repeat z so that we can concatenate it with every pressure value.
        # This changes the shape of z:
        #   (batch_size, latent_size) -> (batch_size, grid_size, latent_size)
        z = z.unsqueeze(1).repeat(1, grid_size, 1)

        # Reshape the pressure grid and z into the batch dimension.
        # They now show have shapes:
        #   p_flat: (batch_size * grid_size, 1)
        #   z_flat: (batch_size * grid_size, latent_size)
        log_P_flat = log_P.reshape(batch_size * grid_size, 1)
        z_flat = z.reshape(batch_size * grid_size, self.latent_size)

        # Normalize log_P to create the first input to the decoder
        x = self.normalize_log_P(log_P_flat)

        # Send through decoder (manually, so that we can take care of the skip
        # connections and concatenate z to the output of each layer)
        for layer in self.layers:

            # Before sending through the layer, concatenate the latent
            # variable to the output of the previous layer
            x = torch.cat(tensors=(x, z_flat), dim=1)

            # Send through layer and apply activation function
            x = layer(x)
            x = self.activation_function(x)

        # Send through final layer (no activation function)
        x = torch.cat((x, z_flat), dim=1)
        x = self.final_layer(x)

        # Reshape the decoder output back to to the original shape of log_P
        T_pred = x.reshape(batch_size, grid_size)

        # Undo normalization (i.e., transform back to Kelvin)
        T_pred = self.normalize_T(T_pred, undo=True)

        return T_pred


class HypernetDecoder(nn.Module, NormalizerMixin):
    """
    A more sophisticated decoder model that uses a hypernetwork to turn
    the latent variable `z` into the weights and biases of a decoder.

    Arguments:
        latent_size: The size of the latent variable `z`.
        normalization: A dictionary containing the normalization
            parameters for the pressure and temperature.
        hypernet_layer_size: The size of the (hidden) layers in the
            hypernetwork.
        decoder_layer_size: The size of the (hidden) layers in the
            decoder.
        hypernet_n_layers: The number of layers in the hypernetwork.
        decoder_n_layers: The number of layers in the decoder.
        hypernet_activation: The activation function to use in the
            hypernetwork.
        decoder_activation: The activation function to use in the
            decoder.
    """

    def __init__(
        self,
        latent_size: int,
        hypernet_layer_size: int,
        decoder_layer_size: int,
        hypernet_n_layers: int,
        decoder_n_layers: int,
        normalization: Dict[str, Any],
        hypernet_activation: str = 'leaky_relu',
        decoder_activation: str = 'siren',
    ) -> None:

        super().__init__()

        # Store constructor arguments
        self.latent_size = latent_size
        self.hypernet_layer_size = hypernet_layer_size
        self.decoder_layer_size = decoder_layer_size
        self.hypernet_n_layers = hypernet_n_layers
        self.decoder_n_layers = decoder_n_layers
        self.hypernet_activation = hypernet_activation
        self.decoder_activation = decoder_activation
        self.normalization = normalization

        # Compute sizes of the weight and bias tensors for the decoder
        self.weight_sizes, self.bias_sizes = self._get_weight_and_bias_sizes()

        # Collect activation functions for each layer
        self.activations = nn.ModuleList(
            [
                get_activation(self.decoder_activation)
                for _ in range(decoder_n_layers + 1)
            ]
            + [Identity()]  # Final layer has no activation function
        )

        # Make sure that all lengths match
        dummy = [self.weight_sizes, self.bias_sizes, self.activations]
        assert len(set(map(len, dummy))) == 1

        # Compute the total number of weights that the hypernet needs to output
        output_size = sum(prod(_) for _ in self.weight_sizes)
        output_size += sum(self.bias_sizes)

        # Define hypernet architecture
        self.hypernet: Callable[[torch.Tensor], torch.Tensor] = get_mlp_layers(
            input_size=latent_size,
            n_layers=hypernet_n_layers,
            layer_size=hypernet_layer_size,
            output_size=output_size,
            activation=hypernet_activation,
            final_tanh=False,
        )

    def _get_weight_and_bias_sizes(self) -> tuple:

        weight_sizes = (
            [(1, self.decoder_layer_size)]
            + [
                (self.decoder_layer_size, self.decoder_layer_size)
                for _ in range(self.decoder_n_layers)
            ]
            + [(self.decoder_layer_size, 1)]
        )
        bias_sizes = (
            [1]
            + [self.decoder_layer_size for _ in range(self.decoder_n_layers)]
            + [1]
        )

        return weight_sizes, bias_sizes

    def forward(self, z: torch.Tensor, log_P: torch.Tensor) -> torch.Tensor:

        # Get batch size and grid size
        batch_size, grid_size = log_P.shape

        # Repeat z so that we can concatenate it with every pressure value.
        # This changes the shape of z:
        #   (batch_size, latent_size) -> (batch_size, grid_size, latent_size)
        z = z.unsqueeze(1).repeat(1, grid_size, 1)

        # Reshape the pressure grid and z into the batch dimension.
        # They now show have shapes:
        #   p_flat: (batch_size * grid_size, 1)
        #   z_flat: (batch_size * grid_size, latent_size)
        log_P_flat = log_P.reshape(batch_size * grid_size, 1)
        z_flat = z.reshape(batch_size * grid_size, self.latent_size)

        # Normalize log_P to create the first input to the decoder
        x = self.normalize_log_P(log_P_flat)

        # Get weights and biases from hypernet
        weights_and_biases = self.hypernet(z_flat)

        # Loop over weights and biases and construct linear layers from them
        # which correspond to the decoder
        first_layer_flag = True
        for weight_size, bias_size, activation in zip(
            self.weight_sizes, self.bias_sizes, self.activations
        ):

            # Compute the number of weights in this layer
            n = prod(weight_size)

            # Get the weights and biases for this layer
            weight = weights_and_biases[:, :n]
            bias = weights_and_biases[:, n : n + bias_size]
            weights_and_biases = weights_and_biases[:, n + bias_size :]

            # Reshape weight and bias
            weight = weight.reshape(batch_size * grid_size, *weight_size)
            bias = bias.reshape(batch_size * grid_size, bias_size)

            # Compute the pass through the linear layer
            x = torch.einsum('bi,bij->bj', x, weight) + bias

            # On the first layer, increase the frequency of the sine activation
            if first_layer_flag and self.decoder_activation == 'siren':
                x *= 30
                first_layer_flag = False

            # Apply activation function
            x = activation(x)

        # Reshape the decoder output to (batch_size, grid_size), that is,
        # the original shape of `log_p`
        T_pred = x.reshape(batch_size, grid_size)

        # Undo normalization (i.e., transform back to Kelvin)
        T_pred = self.normalize_T(T_pred, undo=True)

        return T_pred

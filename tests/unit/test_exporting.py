"""
Unit tests for exporting.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import onnx
import pytest
import torch

from ml4ptp.encoders import (
    CNPEncoder,
    ConvolutionalEncoder,
    MLPEncoder,
    ModifiedMLPEncoder,
)
from ml4ptp.decoders import (
    Decoder,
    HypernetDecoder,
    SkipConnectionsDecoder,
)
from ml4ptp.onnx import ONNXEncoder, ONNXDecoder
from ml4ptp.exporting import (
    export_encoder_with_onnx,
    export_decoder_with_onnx,
    export_model_with_torchscript,
    PTProfile,
)


# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------

@pytest.fixture()
def normalization() -> Dict[str, Any]:
    return dict(
        normalization='whiten',
        T_offset=0,
        T_factor=1,
        log_P_offset=0,
        log_P_factor=1,
    )


@pytest.fixture()
def mlp_encoder(normalization: Dict[str, Any]) -> MLPEncoder:
    return MLPEncoder(
        input_size=101,
        latent_size=2,
        layer_size=32,
        n_layers=3,
        normalization=normalization,
    )


@pytest.fixture()
def modified_mlp_encoder(normalization: Dict[str, Any]) -> ModifiedMLPEncoder:
    return ModifiedMLPEncoder(
        input_size=101,
        latent_size=2,
        layer_size=32,
        n_layers=3,
        normalization=normalization,
    )


@pytest.fixture()
def convolutional_encoder(
    normalization: Dict[str, Any],
) -> ConvolutionalEncoder:
    return ConvolutionalEncoder(
        input_size=101,
        latent_size=2,
        cnn_n_layers=2,
        cnn_n_channels=64,
        cnn_kernel_size=1,
        mlp_layer_size=32,
        mlp_n_layers=3,
        normalization=normalization,
        batch_norm=False,
    )


@pytest.fixture()
def cnp_encoder(normalization: Dict[str, Any]) -> CNPEncoder:
    return CNPEncoder(
        latent_size=2,
        layer_size=32,
        n_layers=3,
        normalization=normalization,
    )


@pytest.fixture()
def decoder(normalization: Dict[str, Any]) -> Decoder:
    return Decoder(
        latent_size=2,
        layer_size=32,
        n_layers=2,
        normalization=normalization,
        activation='leaky_relu',
    )


@pytest.fixture()
def skip_connections_decoder(
    normalization: Dict[str, Any],
) -> SkipConnectionsDecoder:
    return SkipConnectionsDecoder(
        latent_size=2,
        layer_size=32,
        n_layers=2,
        normalization=normalization,
        activation='leaky_relu',
    )


@pytest.fixture()
def hypernet_decoder(normalization: Dict[str, Any]) -> HypernetDecoder:
    return HypernetDecoder(
        latent_size=2,
        normalization=normalization,
        hypernet_layer_size=32,
        hypernet_n_layers=2,
        decoder_layer_size=32,
        decoder_n_layers=2,
        hypernet_activation='leaky_relu',
        decoder_activation='siren',
    )


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

def test__export_encoder_with_onnx(
    tmp_path: Path,
    mlp_encoder: MLPEncoder,
    modified_mlp_encoder: ModifiedMLPEncoder,
    cnp_encoder: CNPEncoder,
    convolutional_encoder: ConvolutionalEncoder,
) -> None:

    # Define inputs for all test cases
    # Note: The grid_size cannot change
    log_P_1 = torch.rand(32, 101)
    T_1 = torch.rand(32, 101)
    log_P_2 = torch.rand(13, 101)
    T_2 = torch.rand(13, 101)

    # Loop over different encoders and test that we can export and load them
    for encoder_original in [
        cnp_encoder,
        convolutional_encoder,
        mlp_encoder,
        modified_mlp_encoder,
    ]:

        # Ensure that the encoder is in eval mode
        encoder_original.eval()

        # Export model with ONNX
        file_name = f'exported_{encoder_original.__class__.__name__}.onnx'
        file_path = tmp_path / file_name
        export_encoder_with_onnx(
            model=encoder_original,
            example_inputs=dict(log_P=log_P_1, T=T_1),
            file_path=file_path,
        )

        # Load model to run ONNX checks
        encoder_loaded = onnx.load(file_path.as_posix())
        onnx.checker.check_model(encoder_loaded)

        # Load model into a ONNX runtime wrapper (for inference)
        encoder_loaded = ONNXEncoder(path_or_bytes=file_path)

        # Case 0 (check that we can also load from a bytes string)
        # Note: It's not clear if we can also check that the two different ways
        # of loading the model give the same results (except for passing some
        # inputs through the model and checking that the outputs are the same).
        encoder_bytes = onnx.load(file_path.as_posix()).SerializeToString()
        assert isinstance(ONNXEncoder(encoder_bytes), ONNXEncoder)

        # Case 1 (original input size)
        z_original = encoder_original(log_P=log_P_1, T=T_1).detach().numpy()
        z_loaded = encoder_loaded(log_P=log_P_1.numpy(), T=T_1.numpy())
        assert np.allclose(z_original, z_loaded, atol=1e-6)

        # Case 2 (different input size)
        z_original = encoder_original(log_P=log_P_2, T=T_2).detach().numpy()
        z_loaded = encoder_loaded(log_P=log_P_2.numpy(), T=T_2.numpy())
        assert np.allclose(z_original, z_loaded, atol=1e-6)


def test__export_decoder_with_onnx(
    tmp_path: Path,
    decoder: Decoder,
    skip_connections_decoder: SkipConnectionsDecoder,
    hypernet_decoder: HypernetDecoder,
) -> None:

    # Define inputs for all test cases
    z_1 = torch.rand(32, 2)
    z_2 = torch.rand(13, 2)
    log_P_1 = torch.rand(32, 51)
    log_P_2 = torch.rand(13, 51)

    # Loop over different decoders and test that we can export and load them
    for decoder_original in [
        decoder,
        skip_connections_decoder,
        hypernet_decoder,
    ]:

        # Ensure that the decoder is in eval mode
        decoder_original.eval()

        # Export model with ONNX
        file_name = f'exported_{decoder_original.__class__.__name__}.onnx'
        file_path = tmp_path / file_name
        export_decoder_with_onnx(
            model=decoder_original,
            example_inputs=dict(z=z_1, log_P=log_P_1),
            file_path=file_path,
        )

        # Load model to run ONNX checks
        decoder_loaded = onnx.load(file_path.as_posix())
        onnx.checker.check_model(decoder_loaded)

        # Load model into a ONNX runtime wrapper (for inference)
        decoder_loaded = ONNXDecoder(path_or_bytes=file_path)

        # Case 0 (check that we can also load from a bytes string)
        # Note: It's not clear if we can also check that the two different ways
        # of loading the model give the same results (except for passing some
        # inputs through the model and checking that the outputs are the same).
        decoder_bytes = onnx.load(file_path.as_posix()).SerializeToString()
        assert isinstance(ONNXDecoder(decoder_bytes), ONNXDecoder)

        # Case 1 (original input size)
        T_pred_original = decoder_original(log_P=log_P_1, z=z_1).detach()
        T_pred_loaded = decoder_loaded(log_P=log_P_1.numpy(), z=z_1.numpy())
        assert np.allclose(T_pred_original.numpy(), T_pred_loaded, atol=1e-6)

        # Case 2 (different input size)
        T_pred_original = decoder_original(log_P=log_P_2, z=z_2).detach()
        T_pred_loaded = decoder_loaded(log_P=log_P_2.numpy(), z=z_2.numpy())
        assert np.allclose(T_pred_original.numpy(), T_pred_loaded, atol=1e-6)


def test__export_encoder_with_torchscript(
    tmp_path: Path,
    mlp_encoder: MLPEncoder,
    modified_mlp_encoder: ModifiedMLPEncoder,
    cnp_encoder: CNPEncoder,
    convolutional_encoder: ConvolutionalEncoder,
) -> None:

    # Define inputs for all test cases
    # Note: The grid_size cannot change
    log_P_1 = torch.rand(32, 101)
    T_1 = torch.rand(32, 101)
    log_P_2 = torch.rand(13, 101)
    T_2 = torch.rand(13, 101)

    # Loop over different encoders and test that we can export and load them
    for encoder_original in [
        cnp_encoder,
        convolutional_encoder,
        mlp_encoder,
        modified_mlp_encoder,
    ]:

        # Ensure that the encoder is in eval mode
        encoder_original.eval()

        # Export model with TorchScript; load saved model
        file_name = f'exported_{encoder_original.__class__.__name__}.pt'
        file_path = tmp_path / file_name
        export_model_with_torchscript(
            model=encoder_original,
            example_inputs=(log_P_1, T_1),
            file_path=file_path,
        )
        encoder_loaded = torch.jit.load(file_path.as_posix())  # type: ignore

        # Case 1 (original input size)
        with torch.no_grad():
            z_original = encoder_original(log_P=log_P_1, T=T_1).numpy()
            z_loaded = encoder_loaded(log_P=log_P_1, T=T_1).numpy()
            assert np.allclose(z_original, z_loaded, atol=1e-6)

        # Case 2 (different input size)
        with torch.no_grad():
            z_original = encoder_original(log_P=log_P_2, T=T_2).numpy()
            z_loaded = encoder_loaded(log_P=log_P_2, T=T_2).numpy()
            assert np.allclose(z_original, z_loaded, atol=1e-6)


def test__export_decoder_with_torchscript(
    tmp_path: Path,
    decoder: Decoder,
    skip_connections_decoder: SkipConnectionsDecoder,
    hypernet_decoder: HypernetDecoder,
) -> None:

    # Define inputs for all test cases
    z_1 = torch.rand(32, 2)
    z_2 = torch.rand(13, 2)
    log_P_1 = torch.rand(32, 51)
    log_P_2 = torch.rand(13, 51)

    # Loop over different decoders and test that we can export and load them
    for decoder_original in [
        decoder,
        skip_connections_decoder,
        hypernet_decoder,
    ]:

        # Ensure that the decoder is in eval mode
        decoder_original.eval()

        # Export model with TorchScript; load saved model
        file_name = f'exported_{decoder_original.__class__.__name__}.pt'
        file_path = tmp_path / file_name
        export_model_with_torchscript(
            model=decoder_original,
            example_inputs=(z_1, log_P_1),
            file_path=file_path,
        )
        decoder_loaded = torch.jit.load(file_path.as_posix())  # type: ignore

        # Case 1 (original input size)
        with torch.no_grad():
            T_pred_original = decoder_original(log_P=log_P_1, z=z_1).numpy()
            T_pred_loaded = decoder_loaded(log_P=log_P_1, z=z_1).numpy()
            assert np.allclose(T_pred_original, T_pred_loaded, atol=1e-6)

        # Case 2 (different input size)
        with torch.no_grad():
            T_pred_original = decoder_original(log_P=log_P_2, z=z_2).numpy()
            T_pred_loaded = decoder_loaded(log_P=log_P_2, z=z_2).numpy()
            assert np.allclose(T_pred_original, T_pred_loaded, atol=1e-6)


def test__pt_profile(
    decoder: Decoder,
    tmp_path: Path,
) -> None:

    # Define inputs for all test cases
    batch_size = 32
    grid_size = 51
    latent_size = decoder.latent_size

    # Export the model with ONNX
    file_path = tmp_path / 'decoder.onnx'
    export_decoder_with_onnx(
        model=decoder,
        example_inputs=dict(
            z=torch.randn(batch_size, latent_size),
            log_P=torch.randn(batch_size, grid_size),
        ),
        file_path=file_path,
    )

    # Load the model into the PTProfile wrapper
    pt_profile = PTProfile(file_path.as_posix())

    # Some type annotations
    z: np.ndarray
    log_P: Union[np.ndarray, float]

    # Case 1: Check that we can reconstruct the latent size from ONNX
    assert pt_profile.latent_size == 2

    # Case 2
    z = np.array([0.0, 0.0])
    log_P = np.linspace(0, 5, grid_size)
    T_pred = pt_profile(z=z, log_P=log_P)
    assert T_pred.shape == (grid_size,)

    # Case 3
    z = np.array([[0.0, 0.0], [1.0, 1.0]])
    log_P = np.array(
        [np.linspace(0, 5, 10), np.linspace(0, 5, 10)]
    )
    T_pred = pt_profile(z=z, log_P=log_P)
    assert T_pred.shape == (2, 10)

    # Case 4
    z = np.array([0.0, 0.0, 0.0])
    log_P = np.linspace(0, 5, grid_size)
    with pytest.raises(ValueError) as value_error:
        pt_profile(z=z, log_P=log_P)
    assert 'z must be 2-dimensional!' in str(value_error)

    # Case 5
    z = np.array([[0.0, 0.0], [1.0, 1.0]])
    log_P = np.linspace(0, 5, grid_size)
    with pytest.raises(ValueError) as value_error:
        pt_profile(z=z, log_P=log_P)
    assert 'Batch size of z and log_P must match!' in str(value_error)

    # Case 6
    z = np.array([0.0, 0.0])
    log_P = 3.0
    T_pred = pt_profile(z=z, log_P=log_P)
    assert T_pred.shape == (1, )

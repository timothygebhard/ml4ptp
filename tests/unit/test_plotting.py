"""
Unit tests for plotting.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from matplotlib.colorbar import Colorbar

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from ml4ptp.plotting import (
    add_colorbar_to_ax,
    plot_profile_to_tensorboard,
    plot_z_to_tensorboard,
    set_fontsize,
    disable_ticks,
)


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

def test__add_colorbar_to_ax() -> None:

    fig, ax = plt.subplots()
    img = ax.imshow(np.random.normal(0, 1, (10, 10)))

    # Case 1
    cbar = add_colorbar_to_ax(img=img, fig=fig, ax=ax, where='right')
    assert isinstance(cbar, Colorbar)

    # Case 2
    cbar = add_colorbar_to_ax(img=img, fig=fig, ax=ax, where='left')
    assert isinstance(cbar, Colorbar)

    # Case 3
    cbar = add_colorbar_to_ax(img=img, fig=fig, ax=ax, where='top')
    assert isinstance(cbar, Colorbar)

    # Case 4
    cbar = add_colorbar_to_ax(img=img, fig=fig, ax=ax, where='bottom')
    assert isinstance(cbar, Colorbar)

    # Case 5
    with pytest.raises(ValueError) as error:
        add_colorbar_to_ax(img=img, fig=fig, ax=ax, where='illegal')
    assert 'Illegal value for `where`' in str(error)

    plt.close()


def test__set_fontsize() -> None:

    fig, ax = plt.subplots()
    set_fontsize(ax, 6)
    plt.close(fig)


def test__plot_profile_to_tensorboard() -> None:

    figure = plot_profile_to_tensorboard(
        z=torch.randn(17, 5),
        log_P=torch.randn(17, 19),
        T_pred=torch.randn(17, 19),
        T_true=torch.randn(17, 19),
        plotting_config=dict(
            min_T=0,
            max_T=100,
            min_log_P=-5,
            max_log_P=5,
        ),
    )
    assert isinstance(figure, plt.Figure)
    plt.close(figure)


def test__plot_z_to_tensorboard() -> None:

    figure = plot_z_to_tensorboard(z=torch.randn(17, 3))
    assert isinstance(figure, plt.Figure)
    plt.close(figure)


def test__disable_ticks() -> None:

    fig, ax = plt.subplots()
    disable_ticks(ax)
    plt.close(fig)

"""
Unit tests for mixins.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import pytest
import torch

from ml4ptp.mixins import NormalizerMixin


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

def test__normalizer_mixin() -> None:

    # Case 1: NormalizerMixin cannot be used directly
    normalizer_mixin = NormalizerMixin()

    with pytest.raises(AttributeError) as attribute_error:
        normalizer_mixin.normalize(T=torch.Tensor([0]), undo=False)
    assert "'NormalizerMixin' object has no attribute" in str(attribute_error)

    with pytest.raises(AttributeError) as attribute_error:
        normalizer_mixin.normalize(T=torch.Tensor([0]), undo=True)
    assert "'NormalizerMixin' object has no attribute" in str(attribute_error)

    # Case 2: Create Dummy class that uses NormalizerMixin
    class Dummy(NormalizerMixin):
        def __init__(self) -> None:
            self.T_mean = 1.0
            self.T_std = 2.0

    dummy = Dummy()
    assert torch.equal(
        dummy.normalize(T=torch.Tensor([3.0, 5.0, 7.0]), undo=False),
        torch.Tensor([1.0, 2.0, 3.0]),
    )
    assert torch.equal(
        dummy.normalize(T=torch.Tensor([1.0, 2.0, 3.0]), undo=True),
        torch.Tensor([3.0, 5.0, 7.0]),
    )

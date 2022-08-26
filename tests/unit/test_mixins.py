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

    # Case 1
    normalizer_mixin = NormalizerMixin()
    with pytest.raises(AttributeError) as attribute_error:
        normalizer_mixin.normalize(T=torch.Tensor([0]), undo=False)
    assert "'NormalizerMixin' object has no attribute" in str(attribute_error)

    # Case 2
    normalizer_mixin = NormalizerMixin()
    with pytest.raises(AttributeError) as attribute_error:
        normalizer_mixin.normalize(T=torch.Tensor([0]), undo=True)
    assert "'NormalizerMixin' object has no attribute" in str(attribute_error)

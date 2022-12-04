"""
Unit tests for mixins.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Any, Dict

import pytest
import torch

from ml4ptp.mixins import NormalizerMixin


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

def test__normalizer_mixin() -> None:

    torch.manual_seed(42)

    class Dummy(NormalizerMixin):
        def __init__(self, normalization: Dict[str, Any]) -> None:
            self.normalization = normalization

    # -------------------------------------------------------------------------
    # Case 1: NormalizerMixin cannot be used directly
    # -------------------------------------------------------------------------

    normalizer_mixin = NormalizerMixin()

    with pytest.raises(AttributeError) as attribute_error:
        normalizer_mixin.normalize_T(T=torch.Tensor([0]), undo=False)
    assert "'NormalizerMixin' object has no attribute" in str(attribute_error)

    with pytest.raises(AttributeError) as attribute_error:
        normalizer_mixin.normalize_T(T=torch.Tensor([0]), undo=True)
    assert "'NormalizerMixin' object has no attribute" in str(attribute_error)

    with pytest.raises(AttributeError) as attribute_error:
        normalizer_mixin.normalize_log_P(log_P=torch.Tensor([0]), undo=False)
    assert "'NormalizerMixin' object has no attribute" in str(attribute_error)

    with pytest.raises(AttributeError) as attribute_error:
        normalizer_mixin.normalize_log_P(log_P=torch.Tensor([0]), undo=True)
    assert "'NormalizerMixin' object has no attribute" in str(attribute_error)

    # -------------------------------------------------------------------------
    # Case 2: Test the normalization of T
    # -------------------------------------------------------------------------

    normalization = dict(
        normalization='whiten',
        T_offset=1.0,
        T_factor=2.0,
        log_P_offset=1.0,
        log_P_factor=2.0,
    )
    dummy = Dummy(normalization)

    assert torch.equal(
        dummy.normalize_T(T=torch.Tensor([3.0, 5.0, 7.0]), undo=False),
        torch.Tensor([1.0, 2.0, 3.0]),
    )
    assert torch.equal(
        dummy.normalize_T(T=torch.Tensor([1.0, 2.0, 3.0]), undo=True),
        torch.Tensor([3.0, 5.0, 7.0]),
    )

    # -------------------------------------------------------------------------
    # Case 2: Test the normalization of log_P
    # -------------------------------------------------------------------------
    
    assert torch.equal(
        dummy.normalize_log_P(log_P=torch.Tensor([3.0, 5.0, 7.0]), undo=False),
        torch.Tensor([1.0, 2.0, 3.0]),
    )
    assert torch.equal(
        dummy.normalize_log_P(log_P=torch.Tensor([1.0, 2.0, 3.0]), undo=True),
        torch.Tensor([3.0, 5.0, 7.0]),
    )

    # -------------------------------------------------------------------------
    # Case 3: Check assertions
    # -------------------------------------------------------------------------

    normalization = dict(
        normalization='minmax',
        T_offset=0.0,
        T_factor=0.0,
        log_P_offset=0.0,
        log_P_factor=0.0,
    )
    dummy = Dummy(normalization)

    with pytest.raises(AssertionError) as assertion_error:
        dummy.normalize_log_P(log_P=torch.Tensor([0]))
    assert "factor must not be zero" in str(assertion_error)

    with pytest.raises(AssertionError) as assertion_error:
        dummy.normalize_T(T=torch.Tensor([0]))
    assert "factor must not be zero" in str(assertion_error)

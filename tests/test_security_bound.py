"""Executable checks for the finite-query computational reduction."""

from __future__ import annotations

from fractions import Fraction

import pytest

from fedmerit.model import SecurityProfile
from fedmerit.security import reference_computational_bound


def test_reference_profile_bound_is_exact_and_exceeds_94_bits() -> None:
    profile = SecurityProfile()
    bound = reference_computational_bound(profile)

    assert bound.probe_hiding == Fraction(1_000 * (1 << 40), 1 << 256)
    assert bound.hash_collision == Fraction(
        (1 << 32) * ((1 << 32) - 1), 2 * (1 << 256)
    )
    assert bound.signature_forgery == Fraction(1 << 32, 1 << 128)
    assert bound.beacon_bias == Fraction(1 << 32, 1 << 128)
    assert bound.security_bits > 94


@pytest.mark.parametrize("field", ["signature_security_bits", "beacon_bias_security_bits"])
def test_reference_profile_rejects_invalid_assumption_bits(field: str) -> None:
    kwargs = {field: 0}
    with pytest.raises(ValueError, match="positive integer"):
        reference_computational_bound(SecurityProfile(), **kwargs)

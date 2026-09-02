"""Executable checks for the finite-query computational reduction."""

from __future__ import annotations

from fractions import Fraction

import pytest

from fedmerit.model import SecurityProfile
from fedmerit.security import reference_computational_bound


def test_reference_profile_bound_is_exact_and_exceeds_85_bits() -> None:
    profile = SecurityProfile()
    bound = reference_computational_bound(profile)

    hiding = 1_000 * 4_096 * (1 << 40)
    total_hashes = hiding + (1 << 32)
    assert bound.probe_hiding == Fraction(hiding, 1 << 256)
    assert bound.hash_collision == Fraction(
        total_hashes * (total_hashes - 1), 2 * (1 << 256)
    )
    assert bound.signature_forgery == Fraction(64 * (1 << 32), 1 << 128)
    assert bound.beacon_bias == Fraction(1_000 * (1 << 32), 1 << 128)
    assert bound.security_bits > 85


def test_reference_bound_scales_with_catalog_and_key_caps() -> None:
    base = SecurityProfile()
    fewer_leaves = SecurityProfile(max_catalog_leaves=1)
    fewer_keys = SecurityProfile(max_verification_keys=1)
    assert (
        reference_computational_bound(fewer_leaves).probe_hiding
        < reference_computational_bound(base).probe_hiding
    )
    assert (
        reference_computational_bound(fewer_keys).signature_forgery
        < reference_computational_bound(base).signature_forgery
    )


@pytest.mark.parametrize(
    "field", ["signature_security_bits", "beacon_bias_security_bits"]
)
def test_reference_profile_rejects_invalid_assumption_bits(field: str) -> None:
    kwargs = {field: 0}
    with pytest.raises(ValueError, match="positive integer"):
        reference_computational_bound(SecurityProfile(), **kwargs)

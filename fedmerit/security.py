"""Concrete reference bounds for the FedMERIT computational reduction.

This module expands the registered per-leaf, per-key, and per-successor query
caps over one finite lineage. It does not estimate empirical cryptanalytic
strength: the Ed25519 and beacon terms are explicit deployment assumptions.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from fractions import Fraction

from .model import SecurityProfile


@dataclass(frozen=True)
class ComputationalBound:
    """Exact union-bound components for one finite protocol lineage."""

    probe_hiding: Fraction
    hash_collision: Fraction
    signature_forgery: Fraction
    beacon_bias: Fraction

    @property
    def total(self) -> Fraction:
        return (
            self.probe_hiding
            + self.hash_collision
            + self.signature_forgery
            + self.beacon_bias
        )

    @property
    def security_bits(self) -> float:
        """Return ``-log2(total)`` without converting the fraction to float."""
        numerator = self.total.numerator
        denominator = self.total.denominator
        return math.log2(denominator) - math.log2(numerator)

    def as_dict(self) -> dict[str, object]:
        def encode(value: Fraction) -> dict[str, object]:
            return {
                "numerator": value.numerator,
                "denominator": value.denominator,
                "log2_probability": (
                    -math.inf
                    if value == 0
                    else math.log2(value.numerator) - math.log2(value.denominator)
                ),
            }

        return {
            "components": {
                "probe_hiding": encode(self.probe_hiding),
                "hash_collision": encode(self.hash_collision),
                "signature_forgery": encode(self.signature_forgery),
                "beacon_bias": encode(self.beacon_bias),
            },
            "total": encode(self.total),
            "security_bits": self.security_bits,
        }


def reference_computational_bound(
    profile: SecurityProfile,
    *,
    signature_security_bits: int = 128,
    beacon_bias_security_bits: int = 128,
) -> ComputationalBound:
    """Instantiate the paper's finite-query reduction with exact rationals.

    ``max_hash_queries`` is a per-leaf random-oracle query cap.  The reduction
    charges every attempt and every registered catalog leaf, then includes those
    hiding queries in the one SHA-256 collision universe.  Signature queries are
    capped per authorized key and beacon queries per successor challenge; the
    profile's key and handover caps make those resources lineage-scoped.
    """
    for name, bits in (
        ("signature_security_bits", signature_security_bits),
        ("beacon_bias_security_bits", beacon_bias_security_bits),
    ):
        if isinstance(bits, bool) or not isinstance(bits, int) or bits <= 0:
            raise ValueError(f"{name} must be a positive integer")

    digest_space = 1 << profile.security_parameter_bits
    hiding_query_work = (
        profile.max_attempts * profile.max_catalog_leaves * profile.max_hash_queries
    )
    total_hash_queries = hiding_query_work + profile.max_collision_queries
    probe_hiding = Fraction(
        hiding_query_work,
        digest_space,
    )
    hash_collision = Fraction(
        total_hash_queries * (total_hash_queries - 1),
        2 * digest_space,
    )
    signature_forgery = Fraction(
        profile.max_verification_keys * profile.max_signature_queries,
        1 << signature_security_bits,
    )
    beacon_bias = Fraction(
        profile.max_attempts * profile.max_beacon_queries,
        1 << beacon_bias_security_bits,
    )
    bound = ComputationalBound(
        probe_hiding,
        hash_collision,
        signature_forgery,
        beacon_bias,
    )
    if bound.total >= 1:
        raise ValueError("reference query caps do not provide a non-trivial bound")
    return bound


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the exact FedMERIT reference-profile reduction bound"
    )
    parser.add_argument("--signature-security-bits", type=int, default=128)
    parser.add_argument("--beacon-security-bits", type=int, default=128)
    args = parser.parse_args()
    profile = SecurityProfile()
    bound = reference_computational_bound(
        profile,
        signature_security_bits=args.signature_security_bits,
        beacon_bias_security_bits=args.beacon_security_bits,
    )
    print(
        json.dumps(
            {
                "profile": asdict(profile),
                "assumptions": {
                    "signature_security_bits": args.signature_security_bits,
                    "beacon_bias_security_bits": args.beacon_security_bits,
                },
                **bound.as_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

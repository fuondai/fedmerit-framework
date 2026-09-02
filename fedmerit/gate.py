"""Durable one-use probe release, risk allocation, and deterministic replay."""

from __future__ import annotations

import hashlib
import math
import secrets
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Protocol

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .canonical import canonical_bytes, digest, merkle_path, verify_merkle_path
from .model import (
    BeaconFixationReservation,
    BeaconRound,
    Candidate,
    CommitProbe,
    EvaluationPolicy,
    LinearModelArtifact,
    ProbeGroup,
    RiskSchedule,
    SamplingFrame,
    SamplingFrameCommitment,
    SamplingFrameEntry,
    SignedSamplingFrame,
    SignedSamplingFrameCommitment,
    SignedBeaconRound,
    SignedBeaconFixationReservation,
    SourcePartition,
    UINT32_MAX,
    ZERO_HASH,
)


def _risk_exponent(n_groups: int, epsilon: float, gamma: float) -> Fraction:
    if (
        n_groups <= 0
        or epsilon <= 0
        or gamma < 0
        or not math.isfinite(epsilon)
        or not math.isfinite(gamma)
    ):
        raise ValueError("invalid risk-bound arguments")
    total = Fraction.from_float(float(epsilon)) + Fraction.from_float(float(gamma))
    return Fraction(n_groups, 2) * total * total


def _exp_neg_taylor_interval(
    exponent: Fraction, term_limit: int, precision_bits: int
) -> tuple[Fraction, Fraction]:
    """Enclose exp(-exponent) with directed fixed-point interval arithmetic."""
    if exponent < 0 or term_limit < 2:
        raise ValueError("invalid exponential interval arguments")
    if exponent == 0:
        return Fraction(1), Fraction(1)
    squarings = 0
    reduced = exponent
    while reduced > Fraction(1, 2):
        reduced /= 2
        squarings += 1
    guard_bits = 16 + math.ceil(math.log2(term_limit + 1))
    scale = 1 << (precision_bits + squarings + guard_bits)
    reduced_lower_scaled = reduced.numerator * scale // reduced.denominator
    reduced_upper_scaled = -(-(reduced.numerator * scale) // reduced.denominator)

    def alternating_interval_scaled(value_scaled: int) -> tuple[int, int]:
        term_lower = scale
        term_upper = scale
        partial_lower = scale
        partial_upper = scale
        series_lower = 0
        series_upper = scale
        for index in range(1, term_limit + 1):
            denominator = scale * index
            term_lower = term_lower * value_scaled // denominator
            term_upper = -(-(term_upper * value_scaled) // denominator)
            if index % 2:
                partial_lower -= term_upper
                partial_upper -= term_lower
                series_lower = partial_lower
            else:
                partial_lower += term_lower
                partial_upper += term_upper
                series_upper = partial_upper
        if term_limit % 2:
            denominator = scale * (term_limit + 1)
            term_upper = -(-(term_upper * value_scaled) // denominator)
            series_upper = partial_upper + term_upper
        return max(0, series_lower), min(scale, series_upper)

    lower_scaled, _ = alternating_interval_scaled(reduced_upper_scaled)
    _, upper_scaled = alternating_interval_scaled(reduced_lower_scaled)
    for _ in range(squarings):
        lower_scaled = lower_scaled * lower_scaled // scale
        upper_scaled = -(-(upper_scaled * upper_scaled) // scale)
    lower = Fraction(lower_scaled, scale)
    upper = Fraction(upper_scaled, scale)
    if not Fraction(0) <= lower <= upper <= Fraction(1):
        raise ArithmeticError("alternating-series enclosure failed")
    return lower, upper


def risk_bound_interval(
    n_groups: int, epsilon: float, gamma: float, *, precision_bits: int = 192
) -> tuple[Fraction, Fraction]:
    """Return a rational enclosure of the Hoeffding exponential."""
    exponent = _risk_exponent(n_groups, epsilon, gamma)
    target_width = Fraction(1, 1 << precision_bits)
    term_limit = 16
    maximum_term_limit = max(384, precision_bits + 32)
    while True:
        lower, upper = _exp_neg_taylor_interval(exponent, term_limit, precision_bits)
        if upper - lower <= target_width:
            return lower, upper
        if term_limit == maximum_term_limit:
            break
        term_limit = min(
            maximum_term_limit,
            math.ceil(term_limit * 3 / 2),
        )
    raise ArithmeticError("risk-bound interval did not reach the requested precision")


def risk_bound(n_groups: int, epsilon: float, gamma: float) -> float:
    """Binary64 display midpoint; protocol decisions use the rational enclosure."""
    lower, upper = risk_bound_interval(n_groups, epsilon, gamma)
    return float((lower + upper) / 2)


def risk_is_satisfied(
    n_groups: int, epsilon: float, gamma: float, alpha: float
) -> bool:
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    threshold = Fraction.from_float(float(alpha))
    magnitude_bits = max(
        0,
        threshold.denominator.bit_length() - threshold.numerator.bit_length(),
    )
    precision_bits = max(192, magnitude_bits + 96)
    precision_limit = magnitude_bits + 4096
    while True:
        lower, upper = risk_bound_interval(
            n_groups, epsilon, gamma, precision_bits=precision_bits
        )
        if upper <= threshold:
            return True
        if lower > threshold:
            return False
        if precision_bits == precision_limit:
            break
        precision_bits = min(precision_bits * 2, precision_limit)
    raise ArithmeticError(
        f"risk threshold is unresolved at {precision_limit}-bit precision"
    )


def required_groups(alpha: float, epsilon: float, gamma: float) -> int:
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    if epsilon + gamma <= 0:
        raise ValueError("epsilon + gamma must be positive")
    if not all(math.isfinite(value) for value in (alpha, epsilon, gamma)):
        raise ValueError("risk parameters must be finite")
    total = Fraction.from_float(float(epsilon)) + Fraction.from_float(float(gamma))
    with localcontext() as context:
        context.prec = 64
        alpha_decimal = Decimal.from_float(float(alpha))
        total_decimal = Decimal(total.numerator) / Decimal(total.denominator)
        rough_estimate = -2 * alpha_decimal.ln() / (total_decimal * total_decimal)
        integer_digits = max(1, rough_estimate.adjusted() + 1)
    with localcontext() as context:
        context.prec = integer_digits + 80
        alpha_decimal = Decimal.from_float(float(alpha))
        total_decimal = Decimal(total.numerator) / Decimal(total.denominator)
        estimate_decimal = -2 * alpha_decimal.ln() / (total_decimal * total_decimal)
        estimate = max(
            1,
            int(estimate_decimal.to_integral_value(rounding=ROUND_CEILING)),
        )

    if risk_is_satisfied(estimate, epsilon, gamma, alpha):
        if estimate == 1 or not risk_is_satisfied(estimate - 1, epsilon, gamma, alpha):
            return estimate
        upper = estimate - 1
        step = 1
        lower = max(0, upper - step)
        while lower > 0 and risk_is_satisfied(lower, epsilon, gamma, alpha):
            upper = lower
            step *= 2
            lower = max(0, estimate - step)
    else:
        lower = estimate
        step = 1
        upper = estimate + step
        while not risk_is_satisfied(upper, epsilon, gamma, alpha):
            lower = upper
            step *= 2
            upper = estimate + step
    while lower + 1 < upper:
        midpoint = (lower + upper) // 2
        if risk_is_satisfied(midpoint, epsilon, gamma, alpha):
            upper = midpoint
        else:
            lower = midpoint
    return upper


class SourceManifestReservation(Protocol):
    """Canonical lineage fences used when a probe is released."""

    def reserve_risk_allocation(
        self,
        schedule: RiskSchedule,
        allocation_index: int,
        *,
        fixation_hash: str,
    ) -> None:
        """Spend one schedule index before the local probe is retired."""

    def reserve_source_manifests(
        self,
        source_manifest_hashes: tuple[str, ...],
        *,
        context_hash: str,
        fixation_hash: str,
        probe_id_hash: str,
    ) -> None:
        """Reserve source manifests before the local probe is retired."""

    def observe_beacon_head(
        self,
        signed_beacon_head: SignedBeaconRound,
        *,
        beacon_public_key: Ed25519PublicKey,
        signed_frame: SignedSamplingFrame,
        frame_public_key: Ed25519PublicKey,
    ) -> None:
        """Advance the authoritative beacon watcher state."""


class RiskScheduleRegistration(Protocol):
    """Canonical context-lineage registry for one lifetime risk schedule."""

    def register_risk_schedule(self, schedule: RiskSchedule) -> None:
        """Freeze the exact schedule against the live context and audit head."""

    def reserve_beacon_successor(
        self,
        beacon_public_key_hash: str,
        beacon_id: str,
        round_number: int,
        parent_round_hash: str,
        *,
        fixation_hash: str,
    ) -> None:
        """Reserve one successor globally across all local risk-ledger replicas."""

    def observe_beacon_head(
        self,
        signed_beacon_head: SignedBeaconRound,
        *,
        beacon_public_key: Ed25519PublicKey,
        signed_frame: SignedSamplingFrame,
        frame_public_key: Ed25519PublicKey,
    ) -> None:
        """Advance the authoritative beacon watcher state."""


class RiskLedger:
    """SQLite-backed finite schedule registry; each index can fund one release."""

    def __init__(self, path: str | Path) -> None:
        if str(path) == ":memory:":
            raise ValueError("RiskLedger must use durable storage, not :memory:")
        self.path = str(path)
        with closing(self._connect()) as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS schedules(
                    schedule_hash TEXT PRIMARY KEY, context_hash TEXT UNIQUE NOT NULL,
                    anchor_receipt_hash TEXT NOT NULL, lifetime_delta REAL NOT NULL,
                    schedule_blob BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS allocations(
                    schedule_hash TEXT NOT NULL, allocation_index INTEGER NOT NULL,
                    fixation_hash TEXT, beacon_parent_round INTEGER,
                    beacon_parent_hash TEXT,
                    PRIMARY KEY(schedule_hash, allocation_index)
                );
                CREATE TABLE IF NOT EXISTS beacon_heads(
                    beacon_public_key_hash TEXT PRIMARY KEY, beacon_id TEXT NOT NULL,
                    round_number INTEGER NOT NULL, round_hash TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS beacon_successor_reservations(
                    beacon_public_key_hash TEXT NOT NULL,
                    round_number INTEGER NOT NULL,
                    parent_round_hash TEXT NOT NULL,
                    fixation_hash TEXT UNIQUE NOT NULL,
                    PRIMARY KEY(beacon_public_key_hash, round_number)
                );
                CREATE TABLE IF NOT EXISTS source_manifest_ledger(
                    source_manifest_hash TEXT PRIMARY KEY,
                    context_hash TEXT NOT NULL,
                    fixation_hash TEXT NOT NULL,
                    probe_id_hash TEXT NOT NULL
                );
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(allocations)")}
            if "beacon_parent_round" not in columns:
                db.execute(
                    "ALTER TABLE allocations ADD COLUMN beacon_parent_round INTEGER"
                )
            if "beacon_parent_hash" not in columns:
                db.execute("ALTER TABLE allocations ADD COLUMN beacon_parent_hash TEXT")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def register(
        self,
        schedule: RiskSchedule,
        *,
        audit_registry: RiskScheduleRegistration,
    ) -> None:
        if any(
            not risk_is_satisfied(
                item.group_count, item.epsilon, item.gamma, item.alpha
            )
            for item in schedule.allocations
        ):
            raise ValueError("risk schedule contains an under-provisioned allocation")
        audit_registry.register_risk_schedule(schedule)
        blob = canonical_bytes(schedule)
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute(
                "SELECT schedule_hash, schedule_blob FROM schedules WHERE context_hash=?",
                (schedule.context_hash,),
            ).fetchone()
            if prior is not None:
                if prior[0] != schedule.schedule_hash or bytes(prior[1]) != blob:
                    db.rollback()
                    raise ValueError(
                        "a different lifetime schedule is already frozen for this context"
                    )
            db.execute(
                "INSERT OR IGNORE INTO schedules VALUES(?,?,?,?,?)",
                (
                    schedule.schedule_hash,
                    schedule.context_hash,
                    schedule.anchor_receipt_hash,
                    schedule.lifetime_delta,
                    blob,
                ),
            )
            for index in range(len(schedule.allocations)):
                db.execute(
                    "INSERT OR IGNORE INTO allocations"
                    "(schedule_hash,allocation_index,fixation_hash,"
                    "beacon_parent_round,beacon_parent_hash) "
                    "VALUES(?,?,NULL,NULL,NULL)",
                    (schedule.schedule_hash, index),
                )
            db.commit()

    def observe_beacon_head(
        self,
        signed_beacon_head: SignedBeaconRound,
        *,
        audit_registry: RiskScheduleRegistration,
        beacon_public_key: Ed25519PublicKey,
        signed_frame: SignedSamplingFrame,
        frame_public_key: Ed25519PublicKey,
    ) -> None:
        """Durably advance one authenticated beacon head without allowing rollback."""
        if not verify_sampling_frame(signed_frame, frame_public_key):
            raise ValueError("beacon observation needs an authenticated sealed catalog")
        frame = signed_frame.frame
        raw_beacon_key = _raw_public_key(beacon_public_key)
        beacon_key_hash = hashlib.sha256(raw_beacon_key).hexdigest()
        beacon_head = signed_beacon_head.round
        if (
            frame.beacon_public_key_hash != beacon_key_hash
            or frame.beacon_id != beacon_head.beacon_id
            or not verify_beacon_round(signed_beacon_head, beacon_public_key)
        ):
            raise ValueError("beacon head does not match the signed catalog")
        audit_registry.observe_beacon_head(
            signed_beacon_head,
            beacon_public_key=beacon_public_key,
            signed_frame=signed_frame,
            frame_public_key=frame_public_key,
        )
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute(
                "SELECT beacon_id, round_number, round_hash FROM beacon_heads "
                "WHERE beacon_public_key_hash=?",
                (beacon_key_hash,),
            ).fetchone()
            if prior is None:
                exact_checkpoint = (
                    beacon_head.round_number == frame.beacon_checkpoint_round
                    and beacon_head.round_hash == frame.beacon_checkpoint_hash
                )
                immediate_checkpoint_successor = (
                    beacon_head.round_number == frame.beacon_checkpoint_round + 1
                    and beacon_head.previous_round_hash == frame.beacon_checkpoint_hash
                )
                if not (exact_checkpoint or immediate_checkpoint_successor):
                    db.rollback()
                    raise ValueError(
                        "first beacon head must equal the signed checkpoint or its "
                        "immediate authenticated successor"
                    )
            else:
                prior_id, prior_number, prior_hash = (
                    str(prior[0]),
                    int(prior[1]),
                    str(prior[2]),
                )
                exact_redelivery = (
                    prior_id == beacon_head.beacon_id
                    and prior_number == beacon_head.round_number
                    and prior_hash == beacon_head.round_hash
                )
                immediate_successor = (
                    prior_id == beacon_head.beacon_id
                    and beacon_head.round_number == prior_number + 1
                    and beacon_head.previous_round_hash == prior_hash
                )
                if not (exact_redelivery or immediate_successor):
                    db.rollback()
                    raise ValueError(
                        "authenticated beacon head must be an exact redelivery "
                        "or the immediate hash-linked successor"
                    )
            db.execute(
                "INSERT INTO beacon_heads VALUES(?,?,?,?) "
                "ON CONFLICT(beacon_public_key_hash) DO UPDATE SET "
                "beacon_id=excluded.beacon_id, round_number=excluded.round_number, "
                "round_hash=excluded.round_hash",
                (
                    beacon_key_hash,
                    beacon_head.beacon_id,
                    beacon_head.round_number,
                    beacon_head.round_hash,
                ),
            )
            db.commit()

    def consume(
        self,
        candidate: Candidate,
        schedule: RiskSchedule,
        *,
        audit_registry: RiskScheduleRegistration,
        beacon_public_key: Ed25519PublicKey,
        signed_frame: SignedSamplingFrame,
        frame_public_key: Ed25519PublicKey,
    ) -> None:
        allocation = schedule.allocation(candidate.risk_schedule_index)
        if (
            schedule.context_hash != candidate.context_hash
            or schedule.schedule_hash != candidate.risk_schedule_hash
            or allocation != candidate.risk
        ):
            raise ValueError(
                "candidate risk fields do not match the registered schedule allocation"
            )
        if candidate.risk_schedule_index == 0 and (
            candidate.previous_receipt_hash != schedule.anchor_receipt_hash
        ):
            raise ValueError("first schedule entry does not extend the activation head")
        if not risk_is_satisfied(
            allocation.group_count,
            allocation.epsilon,
            allocation.gamma,
            allocation.alpha,
        ):
            raise ValueError(
                "predeclared allocation does not satisfy its risk inequality"
            )
        if not verify_sampling_frame(signed_frame, frame_public_key):
            raise ValueError("candidate fixation needs an authenticated sealed catalog")
        frame = signed_frame.frame
        raw_beacon_key = _raw_public_key(beacon_public_key)
        beacon_key_hash = hashlib.sha256(raw_beacon_key).hexdigest()
        if (
            frame.frame_hash != candidate.sampling_frame_hash
            or frame.catalog_root != candidate.sealed_catalog_root
            or frame.beacon_public_key_hash != beacon_key_hash
            or candidate.source_partition.partition_hash
            not in frame.source_partition_hashes
        ):
            raise ValueError(
                "candidate fixation does not match its authenticated catalog/beacon key"
            )
        if len(frame.entries) > candidate.evaluation_policy.security_profile.max_catalog_leaves:
            raise ValueError("sealed catalog exceeds the registered security cap")
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            beacon_head = db.execute(
                "SELECT beacon_id, round_number, round_hash FROM beacon_heads "
                "WHERE beacon_public_key_hash=?",
                (beacon_key_hash,),
            ).fetchone()
            if (
                beacon_head is None
                or beacon_head[0] != frame.beacon_id
                or int(beacon_head[1]) + 1 != candidate.beacon_round
                or beacon_head[2] != candidate.beacon_parent_hash
            ):
                db.rollback()
                raise ValueError(
                    "candidate does not bind the durable authenticated beacon head"
                )
            schedule_row = db.execute(
                "SELECT context_hash, anchor_receipt_hash, schedule_blob FROM schedules "
                "WHERE schedule_hash=?",
                (schedule.schedule_hash,),
            ).fetchone()
            if (
                schedule_row is None
                or schedule_row[0] != schedule.context_hash
                or schedule_row[1] != schedule.anchor_receipt_hash
                or bytes(schedule_row[2]) != canonical_bytes(schedule)
            ):
                db.rollback()
                raise ValueError("risk schedule is not the frozen registered schedule")
            row = db.execute(
                "SELECT fixation_hash, beacon_parent_round, beacon_parent_hash "
                "FROM allocations "
                "WHERE schedule_hash=? AND allocation_index=?",
                (schedule.schedule_hash, candidate.risk_schedule_index),
            ).fetchone()
            if row is None:
                db.rollback()
                raise ValueError("risk schedule/index was not predeclared")
            if row[0] is not None:
                db.rollback()
                if row[0] == candidate.fixation_hash:
                    if row[1] != int(beacon_head[1]) or row[2] != beacon_head[2]:
                        raise ValueError(
                            "stored fixation does not precede beacon release"
                        )
                    return
                raise ValueError("risk schedule allocation has already been spent")
            reservation = db.execute(
                "SELECT parent_round_hash, fixation_hash "
                "FROM beacon_successor_reservations "
                "WHERE beacon_public_key_hash=? AND round_number=?",
                (beacon_key_hash, candidate.beacon_round),
            ).fetchone()
            expected_reservation = (
                candidate.beacon_parent_hash,
                candidate.fixation_hash,
            )
            audit_registry.reserve_beacon_successor(
                beacon_key_hash,
                frame.beacon_id,
                candidate.beacon_round,
                candidate.beacon_parent_hash,
                fixation_hash=candidate.fixation_hash,
            )
            if reservation is None:
                try:
                    db.execute(
                        "INSERT INTO beacon_successor_reservations VALUES(?,?,?,?)",
                        (
                            beacon_key_hash,
                            candidate.beacon_round,
                            candidate.beacon_parent_hash,
                            candidate.fixation_hash,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    db.rollback()
                    raise ValueError(
                        "beacon successor is already reserved by another fixation"
                    ) from exc
            elif (str(reservation[0]), str(reservation[1])) != expected_reservation:
                db.rollback()
                raise ValueError(
                    "beacon successor is already reserved by another fixation"
                )
            db.execute(
                "UPDATE allocations SET fixation_hash=?, beacon_parent_round=?, "
                "beacon_parent_hash=? "
                "WHERE schedule_hash=? AND allocation_index=?",
                (
                    candidate.fixation_hash,
                    int(beacon_head[1]),
                    beacon_head[2],
                    schedule.schedule_hash,
                    candidate.risk_schedule_index,
                ),
            )
            db.commit()

    def is_consumed(
        self, schedule_hash: str, index: int, fixation_hash: str | None = None
    ) -> bool:
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT fixation_hash FROM allocations WHERE schedule_hash=? AND allocation_index=?",
                (schedule_hash, index),
            ).fetchone()
        return bool(
            row
            and row[0] is not None
            and (fixation_hash is None or row[0] == fixation_hash)
        )

    def fixation_precedes_beacon(self, candidate: Candidate) -> bool:
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT fixation_hash, beacon_parent_round, beacon_parent_hash "
                "FROM allocations "
                "WHERE schedule_hash=? AND allocation_index=?",
                (candidate.risk_schedule_hash, candidate.risk_schedule_index),
            ).fetchone()
        return bool(
            row
            and row[0] == candidate.fixation_hash
            and row[1] is not None
            and int(row[1]) + 1 == candidate.beacon_round
            and row[2] == candidate.beacon_parent_hash
        )

    def reserve_source_manifests(
        self,
        source_manifest_hashes: tuple[str, ...],
        *,
        context_hash: str,
        fixation_hash: str,
        probe_id_hash: str,
    ) -> None:
        """Reserve source manifests across model-successor catalogs.

        The risk ledger is shared by successor stores, so a fresh SQLite
        inventory cannot silently make a previously released source shard
        eligible again.  The same release may be retried idempotently; any
        other fixation or probe owner is rejected before the store retires its
        local rows.  Reservations intentionally happen before local retirement
        and are not rolled back across databases after a crash.
        """
        manifests = tuple(source_manifest_hashes)
        if not manifests or len(set(manifests)) != len(manifests):
            raise ValueError("source manifest reservation must be non-empty and unique")
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            for manifest in manifests:
                prior = db.execute(
                    "SELECT context_hash, fixation_hash, probe_id_hash "
                    "FROM source_manifest_ledger WHERE source_manifest_hash=?",
                    (manifest,),
                ).fetchone()
                if prior is not None and tuple(prior) != (
                    context_hash,
                    fixation_hash,
                    probe_id_hash,
                ):
                    db.rollback()
                    raise ValueError(
                        "source manifest has already been reserved by another release"
                    )
            db.executemany(
                "INSERT OR IGNORE INTO source_manifest_ledger "
                "(source_manifest_hash,context_hash,fixation_hash,probe_id_hash) "
                "VALUES(?,?,?,?)",
                (
                    (manifest, context_hash, fixation_hash, probe_id_hash)
                    for manifest in manifests
                ),
            )
            db.commit()


def _selected_shard_root(probe: CommitProbe) -> str:
    return probe.commitment


def _sampling_frame_fields(
    frame: SamplingFrame | SamplingFrameCommitment,
) -> dict[str, object]:
    commitment = frame.public_commitment if isinstance(frame, SamplingFrame) else frame
    return {
        "domain": "fedmerit-sealed-catalog-v3",
        "frame_commitment": commitment,
    }


def sign_sampling_frame(
    frame: SamplingFrame, private_key: Ed25519PrivateKey
) -> SignedSamplingFrame:
    return SignedSamplingFrame(
        frame, private_key.sign(canonical_bytes(_sampling_frame_fields(frame)))
    )


def verify_sampling_frame(
    signed_frame: SignedSamplingFrame, public_key: Ed25519PublicKey
) -> bool:
    try:
        public_key.verify(
            signed_frame.signature,
            canonical_bytes(_sampling_frame_fields(signed_frame.frame)),
        )
    except InvalidSignature:
        return False
    return True


def verify_sampling_frame_commitment(
    signed_frame: SignedSamplingFrameCommitment,
    public_key: Ed25519PublicKey,
) -> bool:
    try:
        public_key.verify(
            signed_frame.signature,
            canonical_bytes(_sampling_frame_fields(signed_frame.commitment)),
        )
    except InvalidSignature:
        return False
    return True


def _beacon_round_fields(round_value: BeaconRound) -> dict[str, object]:
    return {"domain": "fedmerit-beacon-round-v2", "round": round_value}


def _beacon_reservation_fields(
    reservation: BeaconFixationReservation,
) -> dict[str, object]:
    return {
        "domain": "fedmerit-beacon-fixation-reservation-v1",
        "reservation": reservation,
    }


def _beacon_randomness(
    entropy_secret: bytes,
    signed_reservation: SignedBeaconFixationReservation,
) -> bytes:
    """Derive a successor value from service-held entropy, not public fields."""
    return hashlib.sha256(
        canonical_bytes(
            {
                "domain": "fedmerit-beacon-randomness-v2",
                "entropy_secret": entropy_secret,
                "reservation_hash": signed_reservation.reservation.reservation_hash,
                "reservation_signature": signed_reservation.signature,
            }
        )
    ).digest()


def _sign_beacon_round(
    round_value: BeaconRound,
    private_key: Ed25519PrivateKey,
    fixation_reservation: SignedBeaconFixationReservation | None = None,
) -> SignedBeaconRound:
    return SignedBeaconRound(
        round_value,
        private_key.sign(canonical_bytes(_beacon_round_fields(round_value))),
        fixation_reservation,
    )


def sign_beacon_round(
    round_value: BeaconRound, private_key: Ed25519PrivateKey
) -> SignedBeaconRound:
    """Sign an unreserved checkpoint/head observation.

    Randomness-bearing successor rounds used for probe selection are deliberately
    unavailable through this helper.  They must be produced by
    :class:`BeaconService`, whose durable reservation precedes randomness
    generation.
    """
    if round_value.fixation_hash != ZERO_HASH:
        raise ValueError(
            "post-fixation successor rounds must be finalized by BeaconService"
        )
    return _sign_beacon_round(round_value, private_key)


def verify_beacon_fixation_reservation(
    signed_reservation: SignedBeaconFixationReservation,
    public_key: Ed25519PublicKey,
) -> bool:
    try:
        public_key.verify(
            signed_reservation.signature,
            canonical_bytes(
                _beacon_reservation_fields(signed_reservation.reservation)
            ),
        )
    except InvalidSignature:
        return False
    return True


def verify_beacon_round(
    signed_round: SignedBeaconRound, public_key: Ed25519PublicKey
) -> bool:
    try:
        public_key.verify(
            signed_round.signature,
            canonical_bytes(_beacon_round_fields(signed_round.round)),
        )
    except InvalidSignature:
        return False
    round_value = signed_round.round
    if round_value.fixation_hash == ZERO_HASH:
        return signed_round.fixation_reservation is None
    reservation = signed_round.fixation_reservation
    return bool(
        reservation is not None
        and verify_beacon_fixation_reservation(reservation, public_key)
        and reservation.reservation.beacon_id == round_value.beacon_id
        and reservation.reservation.round_number == round_value.round_number
        and reservation.reservation.parent_round_hash
        == round_value.previous_round_hash
        and reservation.reservation.fixation_hash == round_value.fixation_hash
        and reservation.reservation.reservation_hash
        == round_value.reservation_hash
    )


class BeaconService:
    """Durable two-phase beacon service for one authenticated hash chain.

    ``reserve_fixation`` accepts only an allocation already consumed by the
    durable risk ledger.  ``finalize_successor`` then generates randomness
    internally, in the same transaction that marks the unique successor final.
    The signed reservation travels with the round, so a release verifier can
    check the pre-randomness fixation binding without trusting caller timing.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        beacon_id: str,
        checkpoint: SignedBeaconRound,
        private_key: Ed25519PrivateKey | None = None,
        entropy_seed: bytes | None = None,
    ) -> None:
        if str(path) == ":memory:":
            raise ValueError("BeaconService must use durable storage, not :memory:")
        if not isinstance(beacon_id, str) or not beacon_id.strip():
            raise ValueError("beacon_id must be a non-empty string")
        if private_key is not None and not isinstance(private_key, Ed25519PrivateKey):
            raise TypeError("private_key must be an Ed25519 private key")
        if entropy_seed is not None and (
            not isinstance(entropy_seed, (bytes, bytearray))
            or len(entropy_seed) != 32
        ):
            raise ValueError("entropy_seed must contain exactly 256 bits")
        self.path = str(path)
        self.beacon_id = beacon_id
        provided_raw = None
        if private_key is not None:
            provided_raw = private_key.private_bytes(
                serialization.Encoding.Raw,
                serialization.PrivateFormat.Raw,
                serialization.NoEncryption(),
            )
        with closing(self._connect()) as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS beacon_service_key(
                    id INTEGER PRIMARY KEY CHECK(id=1), private_key BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS beacon_service_entropy(
                    id INTEGER PRIMARY KEY CHECK(id=1), entropy_secret BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS beacon_service_state(
                    id INTEGER PRIMARY KEY CHECK(id=1), beacon_id TEXT NOT NULL,
                    round_number INTEGER NOT NULL, round_hash TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS beacon_fixation_reservations(
                    round_number INTEGER PRIMARY KEY,
                    parent_round_hash TEXT NOT NULL,
                    fixation_hash TEXT UNIQUE NOT NULL,
                    reservation_signature BLOB NOT NULL,
                    randomness BLOB,
                    round_signature BLOB,
                    round_hash TEXT
                );
            """)
            key_row = db.execute(
                "SELECT private_key FROM beacon_service_key WHERE id=1"
            ).fetchone()
            if key_row is None:
                raw_key = provided_raw or Ed25519PrivateKey.generate().private_bytes(
                    serialization.Encoding.Raw,
                    serialization.PrivateFormat.Raw,
                    serialization.NoEncryption(),
                )
                db.execute("INSERT INTO beacon_service_key VALUES(1,?)", (raw_key,))
            else:
                raw_key = bytes(key_row[0])
                if provided_raw is not None and provided_raw != raw_key:
                    raise ValueError("beacon service is bound to a different signing key")
            service_key = Ed25519PrivateKey.from_private_bytes(raw_key)
            entropy_row = db.execute(
                "SELECT entropy_secret FROM beacon_service_entropy WHERE id=1"
            ).fetchone()
            provided_entropy = None if entropy_seed is None else bytes(entropy_seed)
            if entropy_row is None:
                entropy_secret = provided_entropy or secrets.token_bytes(32)
                db.execute(
                    "INSERT INTO beacon_service_entropy VALUES(1,?)",
                    (entropy_secret,),
                )
            else:
                entropy_secret = bytes(entropy_row[0])
                if provided_entropy is not None and provided_entropy != entropy_secret:
                    raise ValueError("beacon service is bound to different entropy")
            if len(entropy_secret) != 32:
                raise ValueError("stored beacon entropy must contain exactly 256 bits")
            if (
                checkpoint.round.beacon_id != beacon_id
                or not verify_beacon_round(checkpoint, service_key.public_key())
            ):
                raise ValueError("beacon checkpoint is invalid for this service")
            state = db.execute(
                "SELECT beacon_id, round_number, round_hash "
                "FROM beacon_service_state WHERE id=1"
            ).fetchone()
            checkpoint_state = (
                beacon_id,
                checkpoint.round.round_number,
                checkpoint.round.round_hash,
            )
            if state is None:
                db.execute(
                    "INSERT INTO beacon_service_state VALUES(1,?,?,?)",
                    checkpoint_state,
                )
            elif tuple(state) != checkpoint_state:
                raise ValueError(
                    "beacon checkpoint does not equal the durable service head"
                )
        self._private_key = Ed25519PrivateKey.from_private_bytes(raw_key)
        self._entropy_secret = entropy_secret

    @classmethod
    def bootstrap(
        cls,
        path: str | Path,
        *,
        beacon_id: str,
        checkpoint: BeaconRound,
        private_key: Ed25519PrivateKey | None = None,
        entropy_seed: bytes | None = None,
    ) -> tuple["BeaconService", SignedBeaconRound]:
        """Create a service and sign its checkpoint in one operation.

        The service keeps the signing key in its durable boundary and exposes
        only ``public_key`` as part of its supported API.  ``private_key`` and
        ``entropy_seed`` are retained solely for deterministic test fixtures and
        offline benchmark replays.
        """
        service_key = private_key or Ed25519PrivateKey.generate()
        signed_checkpoint = _sign_beacon_round(checkpoint, service_key)
        service = cls(
            path,
            beacon_id=beacon_id,
            checkpoint=signed_checkpoint,
            private_key=service_key,
            entropy_seed=entropy_seed,
        )
        return service, signed_checkpoint

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30, isolation_level=None)

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self._private_key.public_key()

    def reserve_fixation(
        self,
        candidate: Candidate,
        *,
        risk_ledger: RiskLedger,
    ) -> SignedBeaconFixationReservation:
        """Sign and persist the candidate binding before randomness exists."""
        if not isinstance(risk_ledger, RiskLedger):
            raise TypeError("risk_ledger must be a RiskLedger")
        if not risk_ledger.fixation_precedes_beacon(candidate):
            raise ValueError("candidate must be durably fixed before beacon reservation")
        reservation = BeaconFixationReservation(
            self.beacon_id,
            candidate.beacon_round,
            candidate.beacon_parent_hash,
            candidate.fixation_hash,
        )
        fields = canonical_bytes(_beacon_reservation_fields(reservation))
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            state = db.execute(
                "SELECT beacon_id, round_number, round_hash "
                "FROM beacon_service_state WHERE id=1"
            ).fetchone()
            row = db.execute(
                "SELECT parent_round_hash, fixation_hash, reservation_signature "
                "FROM beacon_fixation_reservations WHERE round_number=?",
                (candidate.beacon_round,),
            ).fetchone()
            if row is not None:
                if (str(row[0]), str(row[1])) != (
                    candidate.beacon_parent_hash,
                    candidate.fixation_hash,
                ):
                    db.rollback()
                    raise ValueError("beacon successor is reserved for another fixation")
                signature = bytes(row[2])
            else:
                if state != (
                    self.beacon_id,
                    candidate.beacon_round - 1,
                    candidate.beacon_parent_hash,
                ):
                    db.rollback()
                    raise ValueError("candidate does not extend the durable beacon head")
                signature = self._private_key.sign(fields)
                try:
                    db.execute(
                        "INSERT INTO beacon_fixation_reservations"
                        "(round_number,parent_round_hash,fixation_hash,"
                        "reservation_signature,randomness,round_signature,round_hash) "
                        "VALUES(?,?,?,?,NULL,NULL,NULL)",
                        (
                            candidate.beacon_round,
                            candidate.beacon_parent_hash,
                            candidate.fixation_hash,
                            signature,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    db.rollback()
                    raise ValueError(
                        "fixation is already bound to another beacon successor"
                    ) from exc
            db.commit()
        return SignedBeaconFixationReservation(reservation, signature)

    def finalize_successor(
        self,
        signed_reservation: SignedBeaconFixationReservation,
    ) -> SignedBeaconRound:
        """Generate and finalize the unique successor for a durable reservation."""
        if not verify_beacon_fixation_reservation(
            signed_reservation, self.public_key
        ):
            raise ValueError("beacon fixation reservation signature is invalid")
        reservation = signed_reservation.reservation
        if reservation.beacon_id != self.beacon_id:
            raise ValueError("beacon reservation belongs to another service")
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            state = db.execute(
                "SELECT beacon_id, round_number, round_hash "
                "FROM beacon_service_state WHERE id=1"
            ).fetchone()
            row = db.execute(
                "SELECT parent_round_hash, fixation_hash, reservation_signature,"
                "randomness,round_signature,round_hash "
                "FROM beacon_fixation_reservations WHERE round_number=?",
                (reservation.round_number,),
            ).fetchone()
            expected_state = (
                self.beacon_id,
                reservation.round_number - 1,
                reservation.parent_round_hash,
            )
            if row is None or (
                str(row[0]), str(row[1]), bytes(row[2])
            ) != (
                reservation.parent_round_hash,
                reservation.fixation_hash,
                signed_reservation.signature,
            ):
                db.rollback()
                raise ValueError("beacon reservation is absent from durable state")
            if row[3] is not None:
                randomness = bytes(row[3])
                round_signature = bytes(row[4])
                expected_round_hash = str(row[5])
            else:
                if tuple(state) != expected_state:
                    db.rollback()
                    raise ValueError("beacon reservation no longer extends the live head")
                randomness = _beacon_randomness(
                    self._entropy_secret,
                    signed_reservation,
                )
                round_value = BeaconRound(
                    reservation.beacon_id,
                    reservation.round_number,
                    reservation.parent_round_hash,
                    randomness,
                    reservation.fixation_hash,
                    reservation.reservation_hash,
                )
                round_signature = self._private_key.sign(
                    canonical_bytes(_beacon_round_fields(round_value))
                )
                expected_round_hash = round_value.round_hash
                db.execute(
                    "UPDATE beacon_fixation_reservations SET randomness=?,"
                    "round_signature=?,round_hash=? WHERE round_number=?",
                    (
                        randomness,
                        round_signature,
                        expected_round_hash,
                        reservation.round_number,
                    ),
                )
                db.execute(
                    "UPDATE beacon_service_state SET round_number=?,round_hash=? "
                    "WHERE id=1",
                    (reservation.round_number, expected_round_hash),
                )
            db.commit()
        round_value = BeaconRound(
            reservation.beacon_id,
            reservation.round_number,
            reservation.parent_round_hash,
            randomness,
            reservation.fixation_hash,
            reservation.reservation_hash,
        )
        if round_value.round_hash != expected_round_hash:
            raise ValueError("durable beacon transcript hash is inconsistent")
        return SignedBeaconRound(
            round_value,
            round_signature,
            signed_reservation,
        )


def _raw_public_key(public_key: Ed25519PublicKey) -> bytes:
    return public_key.public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )


def _draw_index(
    *,
    frame_hash: str,
    fixation_hash: str,
    beacon_randomness: bytes,
    population_size: int,
) -> tuple[int, int]:
    """Map an authenticated post-fixation beacon value without modulo bias."""
    if population_size <= 0:
        raise ValueError("draw population must be non-empty")
    modulus = 1 << 256
    limit = modulus - modulus % population_size
    for counter in range(UINT32_MAX + 1):
        value = int.from_bytes(
            hashlib.sha256(
                canonical_bytes(
                    {
                        "domain": "fedmerit-beacon-draw-v1",
                        "frame_hash": frame_hash,
                        "fixation_hash": fixation_hash,
                        "beacon_randomness": beacon_randomness,
                        "counter": counter,
                    }
                )
            ).digest(),
            "big",
        )
        if value < limit:
            return value % population_size, counter
    raise RuntimeError("deterministic rejection sampler exhausted uint32 counter")


def _release_fields(
    candidate: Candidate,
    selected_entry: SamplingFrameEntry,
    signed_frame: SignedSamplingFrame | SignedSamplingFrameCommitment,
    signed_beacon_round: SignedBeaconRound,
    beacon_public_key: bytes,
    commit_probe_commitment: str,
    eligible_probe_id_hashes: tuple[str, ...],
    draw_counter: int,
) -> dict[str, object]:
    """The release hash and signature cover exactly this domain-separated map."""
    frame = (
        signed_frame.frame.public_commitment
        if isinstance(signed_frame, SignedSamplingFrame)
        else signed_frame.commitment
    )
    return {
        "domain": "fedmerit-release-v1",
        "candidate_hash": candidate.candidate_hash,
        "fixation_hash": candidate.fixation_hash,
        "context_hash": candidate.context_hash,
        "probe_policy_hash": candidate.probe_policy_hash,
        "risk_schedule_hash": candidate.risk_schedule_hash,
        "risk_schedule_index": candidate.risk_schedule_index,
        "epsilon": candidate.risk.epsilon,
        "gamma": candidate.risk.gamma,
        "alpha": candidate.risk.alpha,
        "group_count": candidate.risk.group_count,
        "previous_receipt_hash": candidate.previous_receipt_hash,
        "source_partition_hash": candidate.source_partition.partition_hash,
        "sampling_frame_hash": frame.frame_hash,
        "sealed_catalog_root": frame.catalog_root,
        "sampling_frame_signature_hash": hashlib.sha256(
            signed_frame.signature
        ).hexdigest(),
        "beacon_round_hash": signed_beacon_round.round.round_hash,
        "beacon_signature_hash": hashlib.sha256(
            signed_beacon_round.signature
        ).hexdigest(),
        "beacon_public_key_hash": hashlib.sha256(beacon_public_key).hexdigest(),
        "eligible_probe_id_hashes_hash": digest(
            {
                "domain": "fedmerit-eligible-probes-v1",
                "probe_id_hashes": eligible_probe_id_hashes,
            }
        ),
        "draw_counter": draw_counter,
        "commit_probe_commitment": commit_probe_commitment,
        "selected_probe_id_hash": selected_entry.probe_id_hash,
    }


@dataclass(frozen=True)
class PublicProbeRelease:
    """Public release token; it contains a catalog leaf but no raw probe opening."""

    signed_sampling_frame: SignedSamplingFrameCommitment
    candidate_hash: str
    fixation_hash: str
    sealed_catalog_root: str
    selected_catalog_entry: SamplingFrameEntry
    catalog_membership_path: tuple[tuple[str, bool], ...]
    signed_beacon_round: SignedBeaconRound
    beacon_public_key: bytes
    eligible_probe_id_hashes: tuple[str, ...]
    draw_counter: int
    release_hash: str
    signature: bytes

    def __post_init__(self) -> None:
        if len(self.beacon_public_key) != 32 or len(self.signature) != 64:
            raise ValueError(
                "beacon public key and release signature have invalid size"
            )
        for name in (
            "candidate_hash",
            "fixation_hash",
            "sealed_catalog_root",
            "release_hash",
        ):
            value = getattr(self, name)
            if len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 digest")
            bytes.fromhex(value)
        ids = self.eligible_probe_id_hashes
        if not ids or ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise ValueError("eligible probe hashes must be distinct and ascending")
        for probe_id_hash in ids:
            if len(probe_id_hash) != 64:
                raise ValueError("eligible probe hash must be a SHA-256 digest")
            bytes.fromhex(probe_id_hash)
        if self.selected_catalog_entry.probe_id_hash not in ids:
            raise ValueError("selected catalog entry is absent from the draw set")
        path = tuple(self.catalog_membership_path)
        object.__setattr__(self, "catalog_membership_path", path)
        for sibling_hash, sibling_is_left in path:
            if len(sibling_hash) != 64 or not isinstance(sibling_is_left, bool):
                raise ValueError("catalog membership path contains an invalid step")
            bytes.fromhex(sibling_hash)
        if (
            isinstance(self.draw_counter, bool)
            or not isinstance(self.draw_counter, int)
            or not 0 <= self.draw_counter <= UINT32_MAX
        ):
            raise ValueError("draw_counter must be uint32")

    @property
    def commit_probe_commitment(self) -> str:
        return self.selected_catalog_entry.payload_commitment


@dataclass(frozen=True)
class ProbeRelease:
    probe: CommitProbe
    signed_sampling_frame: SignedSamplingFrame
    candidate_hash: str
    fixation_hash: str
    sealed_catalog_root: str
    commit_probe_commitment: str
    signed_beacon_round: SignedBeaconRound
    beacon_public_key: bytes
    eligible_probe_id_hashes: tuple[str, ...]
    draw_counter: int
    release_hash: str
    signature: bytes

    def __post_init__(self) -> None:
        if len(self.beacon_public_key) != 32 or len(self.signature) != 64:
            raise ValueError(
                "beacon public key and release signature have invalid size"
            )
        if len(self.commit_probe_commitment) != 64:
            raise ValueError("commit_probe_commitment must be a SHA-256 digest")
        bytes.fromhex(self.commit_probe_commitment)
        if self.commit_probe_commitment != _selected_shard_root(self.probe):
            raise ValueError("commit_probe_commitment does not bind the released probe")
        if self.probe.frame_entry not in self.signed_sampling_frame.frame.entries:
            raise ValueError("released probe does not open a sealed-catalog leaf")
        if self.sealed_catalog_root != self.signed_sampling_frame.frame.catalog_root:
            raise ValueError("release catalog root differs from its signed frame")
        ids = self.eligible_probe_id_hashes
        if not ids or ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise ValueError("eligible probe hashes must be distinct and ascending")
        for probe_id_hash in ids:
            if len(probe_id_hash) != 64:
                raise ValueError("eligible probe hash must be a SHA-256 digest")
            bytes.fromhex(probe_id_hash)
        if self.probe.probe_id_hash not in ids:
            raise ValueError("selected probe is absent from the eligible draw set")
        if (
            isinstance(self.draw_counter, bool)
            or not isinstance(self.draw_counter, int)
            or not 0 <= self.draw_counter <= UINT32_MAX
        ):
            raise ValueError("draw_counter must be uint32")

    @property
    def selected_shard_root(self) -> str:
        """Compatibility name for the pre-fixation payload commitment."""
        return self.commit_probe_commitment

    @property
    def public_release(self) -> PublicProbeRelease:
        entries = self.signed_sampling_frame.frame.entries
        selected_index = entries.index(self.probe.frame_entry)
        path = merkle_path(
            [
                {"domain": "fedmerit-sealed-catalog-leaf-v2", "entry": entry}
                for entry in entries
            ],
            selected_index,
        )
        return PublicProbeRelease(
            self.signed_sampling_frame.public_commitment,
            self.candidate_hash,
            self.fixation_hash,
            self.sealed_catalog_root,
            self.probe.frame_entry,
            path,
            self.signed_beacon_round,
            self.beacon_public_key,
            self.eligible_probe_id_hashes,
            self.draw_counter,
            self.release_hash,
            self.signature,
        )


class CommitProbeStore:
    """Select an eligible probe internally and retire it in one SQLite transaction."""

    def __init__(
        self,
        probes: list[CommitProbe],
        partitions: list[SourcePartition],
        signed_frame: SignedSamplingFrame,
        frame_public_key: Ed25519PublicKey,
        path: str | Path,
        *,
        store_private_key: Ed25519PrivateKey | None = None,
    ) -> None:
        if str(path) == ":memory:":
            raise ValueError("probe retirement must use durable storage, not :memory:")
        if len({probe.probe_id for probe in probes}) != len(probes):
            raise ValueError("probe identifiers must be unique")
        if len({probe.sealing_nonce for probe in probes}) != len(probes):
            raise ValueError("sealed catalog requires a unique nonce per probe")
        group_ids = [group.group_id for probe in probes for group in probe.groups]
        manifests = [
            group.source_manifest_hash for probe in probes for group in probe.groups
        ]
        if len(set(group_ids)) != len(group_ids) or len(set(manifests)) != len(
            manifests
        ):
            raise ValueError(
                "source groups and manifests must be globally unique in the inventory"
            )
        partition_map = {item.partition_hash: item for item in partitions}
        if not partition_map or len(partition_map) != len(partitions):
            raise ValueError(
                "source partitions must be non-empty and uniquely committed"
            )
        if not verify_sampling_frame(signed_frame, frame_public_key):
            raise ValueError("sampling-frame authority signature is invalid")
        if tuple(sorted(partition_map)) != signed_frame.frame.source_partition_hashes:
            raise ValueError(
                "signed sampling frame does not exactly bind the source partitions"
            )
        frame_entries = tuple(
            sorted(
                (probe.frame_entry for probe in probes), key=lambda x: x.probe_id_hash
            )
        )
        if signed_frame.frame.entries != frame_entries:
            raise ValueError(
                "signed sampling frame does not exactly describe the private inventory"
            )
        self.path = str(path)
        self._probes = {probe.probe_id: probe for probe in probes}
        self._partitions = partition_map
        self._signed_frame = signed_frame
        self._frame_public_key = frame_public_key
        provided_raw = None
        if store_private_key is not None:
            if not isinstance(store_private_key, Ed25519PrivateKey):
                raise TypeError("store_private_key must be an Ed25519 private key")
            provided_raw = store_private_key.private_bytes(
                serialization.Encoding.Raw,
                serialization.PrivateFormat.Raw,
                serialization.NoEncryption(),
            )
        with closing(self._connect()) as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS probes(
                    probe_id TEXT PRIMARY KEY, context_hash TEXT NOT NULL,
                    policy_hash TEXT NOT NULL, group_count INTEGER NOT NULL,
                    commitment TEXT NOT NULL, consumed INTEGER NOT NULL DEFAULT 0,
                    fixation_hash TEXT, release_shard_root TEXT,
                    release_beacon_round_hash TEXT, release_beacon_signature BLOB,
                    release_beacon_public_key BLOB, release_eligible_ids BLOB,
                    release_draw_counter INTEGER, release_hash TEXT,
                    release_signature BLOB
                );
                CREATE TABLE IF NOT EXISTS probe_groups(
                    source_manifest_hash TEXT PRIMARY KEY, group_id TEXT UNIQUE NOT NULL,
                    probe_id TEXT NOT NULL, consumed INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS source_partitions(
                    partition_hash TEXT PRIMARY KEY, partition_blob BLOB NOT NULL
                );
            CREATE TABLE IF NOT EXISTS store_key(id INTEGER PRIMARY KEY CHECK(id=1), private_key BLOB NOT NULL);
            """)
            existing_key_row = db.execute(
                "SELECT private_key FROM store_key WHERE id=1"
            ).fetchone()
            if (
                existing_key_row is not None
                and provided_raw is not None
                and bytes(existing_key_row[0]) != provided_raw
            ):
                raise ValueError(
                    "probe store is already bound to a different signing key"
                )
            for probe in probes:
                row = db.execute(
                    "SELECT commitment FROM probes WHERE probe_id=?", (probe.probe_id,)
                ).fetchone()
                if row is not None and row[0] != probe.commitment:
                    raise ValueError(
                        "probe id is already bound to different sealed content"
                    )
                db.execute(
                    "INSERT OR IGNORE INTO probes VALUES(?,?,?,?,?,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL)",
                    (
                        probe.probe_id,
                        probe.context_hash,
                        probe.probe_policy_hash,
                        len(probe.groups),
                        probe.commitment,
                    ),
                )
                for group in probe.groups:
                    group_row = db.execute(
                        "SELECT group_id, probe_id FROM probe_groups WHERE source_manifest_hash=?",
                        (group.source_manifest_hash,),
                    ).fetchone()
                    if group_row is not None and group_row != (
                        group.group_id,
                        probe.probe_id,
                    ):
                        raise ValueError(
                            "source manifest is already bound to another inventory group"
                        )
                    db.execute(
                        "INSERT OR IGNORE INTO probe_groups VALUES(?,?,?,0)",
                        (group.source_manifest_hash, group.group_id, probe.probe_id),
                    )
            for partition_hash, partition in partition_map.items():
                blob = canonical_bytes(partition)
                partition_row = db.execute(
                    "SELECT partition_blob FROM source_partitions WHERE partition_hash=?",
                    (partition_hash,),
                ).fetchone()
                if partition_row is not None and bytes(partition_row[0]) != blob:
                    raise ValueError(
                        "source partition hash is bound to different content"
                    )
                db.execute(
                    "INSERT OR IGNORE INTO source_partitions VALUES(?,?)",
                    (partition_hash, blob),
                )
            key_row = db.execute(
                "SELECT private_key FROM store_key WHERE id=1"
            ).fetchone()
            if key_row is None:
                raw = provided_raw or Ed25519PrivateKey.generate().private_bytes(
                    serialization.Encoding.Raw,
                    serialization.PrivateFormat.Raw,
                    serialization.NoEncryption(),
                )
                db.execute("INSERT INTO store_key VALUES(1,?)", (raw,))
            else:
                raw = bytes(key_row[0])
        self._private_key = Ed25519PrivateKey.from_private_bytes(raw)

    @classmethod
    def successor(
        cls,
        previous: "CommitProbeStore",
        *,
        probes: list[CommitProbe],
        partitions: list[SourcePartition],
        signed_frame: SignedSamplingFrame,
        path: str | Path,
    ) -> "CommitProbeStore":
        """Create a fresh catalog while preserving the context trust roots.

        A committed model successor inherits the previous context's verifier
        trust.  The successor catalog therefore needs a new inventory/database
        but the same store and frame signing identities; silently generating a
        new key would make the next otherwise-valid commit unverifiable.
        ``signed_frame`` is checked with the predecessor frame public key, so
        callers must sign the successor frame with the same frame authority.
        """
        if not isinstance(previous, cls):
            raise TypeError("previous must be a CommitProbeStore")
        return cls(
            probes,
            partitions,
            signed_frame,
            previous._frame_public_key,
            path,
            store_private_key=previous._private_key,
        )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30, isolation_level=None)

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self._private_key.public_key()

    def release(
        self,
        candidate: Candidate,
        *,
        signed_beacon_round: SignedBeaconRound,
        beacon_public_key: Ed25519PublicKey,
        schedule: RiskSchedule,
        risk_ledger: RiskLedger,
        audit_registry: SourceManifestReservation,
    ) -> ProbeRelease:
        """Use a post-fixation beacon round and retire one eligible probe."""
        if (
            candidate.sampling_frame_hash != self._signed_frame.frame.frame_hash
            or candidate.sealed_catalog_root != self._signed_frame.frame.catalog_root
        ):
            raise ValueError("candidate does not bind the signed sealed catalog")
        if (
            self._signed_frame.frame.context_hash != candidate.context_hash
            or self._signed_frame.frame.policy_hash != candidate.probe_policy_hash
        ):
            raise ValueError(
                "sampling frame does not match candidate context and policy"
            )
        raw_beacon_key = _raw_public_key(beacon_public_key)
        beacon_round = signed_beacon_round.round
        if (
            hashlib.sha256(raw_beacon_key).hexdigest()
            != self._signed_frame.frame.beacon_public_key_hash
            or beacon_round.beacon_id != self._signed_frame.frame.beacon_id
            or beacon_round.round_number != candidate.beacon_round
            or beacon_round.previous_round_hash != candidate.beacon_parent_hash
            or beacon_round.fixation_hash != candidate.fixation_hash
            or not verify_beacon_round(signed_beacon_round, beacon_public_key)
        ):
            raise ValueError("beacon transcript is not the frame-bound future round")
        if (
            schedule.schedule_hash != candidate.risk_schedule_hash
            or not risk_ledger.fixation_precedes_beacon(candidate)
        ):
            raise ValueError("candidate was not durably fixed before beacon release")
        audit_registry.observe_beacon_head(
            signed_beacon_round,
            beacon_public_key=beacon_public_key,
            signed_frame=self._signed_frame,
            frame_public_key=self._frame_public_key,
        )
        excluded = set(candidate.excluded_source_manifests).union(
            self._signed_frame.frame.exclusion_source_manifest_hashes
        )
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            catalog_entries = {
                entry.probe_id_hash: entry for entry in self._signed_frame.frame.entries
            }
            stored_probes = db.execute(
                "SELECT probe_id, commitment FROM probes ORDER BY probe_id"
            ).fetchall()
            if len(stored_probes) != len(self._probes) or any(
                probe_id not in self._probes
                or commitment != self._probes[probe_id].commitment
                or catalog_entries.get(self._probes[probe_id].probe_id_hash)
                != self._probes[probe_id].frame_entry
                for probe_id, commitment in stored_probes
            ):
                db.rollback()
                raise ValueError("sealed catalog inventory changed after fixation")
            partition_row = db.execute(
                "SELECT partition_blob FROM source_partitions WHERE partition_hash=?",
                (candidate.source_partition.partition_hash,),
            ).fetchone()
            if partition_row is None or bytes(partition_row[0]) != canonical_bytes(
                candidate.source_partition
            ):
                db.rollback()
                raise ValueError(
                    "candidate source partition is not registered in the probe store"
                )
            recovered = db.execute(
                "SELECT probe_id, release_shard_root, release_beacon_round_hash, "
                "release_beacon_signature, release_beacon_public_key, "
                "release_eligible_ids, release_draw_counter, release_hash, "
                "release_signature FROM probes "
                "WHERE consumed=1 AND fixation_hash=?",
                (candidate.fixation_hash,),
            ).fetchone()
            if recovered is not None:
                recovered_eligible_ids = tuple(
                    bytes(recovered[5]).decode("ascii").split(",")
                )
                if (
                    recovered[2] != beacon_round.round_hash
                    or bytes(recovered[3]) != signed_beacon_round.signature
                    or bytes(recovered[4]) != raw_beacon_key
                    or recovered_eligible_ids != candidate.eligible_probe_id_hashes
                ):
                    db.rollback()
                    raise ValueError("retry supplied a different beacon transcript")
                probe = self._probes[recovered[0]]
                try:
                    audit_registry.reserve_risk_allocation(
                        schedule,
                        candidate.risk_schedule_index,
                        fixation_hash=candidate.fixation_hash,
                    )
                    audit_registry.reserve_source_manifests(
                        tuple(group.source_manifest_hash for group in probe.groups),
                        context_hash=candidate.context_hash,
                        fixation_hash=candidate.fixation_hash,
                        probe_id_hash=probe.probe_id_hash,
                    )
                except ValueError:
                    db.rollback()
                    raise
                db.commit()
                return ProbeRelease(
                    probe,
                    self._signed_frame,
                    candidate.candidate_hash,
                    candidate.fixation_hash,
                    candidate.sealed_catalog_root,
                    recovered[1],
                    signed_beacon_round,
                    raw_beacon_key,
                    recovered_eligible_ids,
                    int(recovered[6]),
                    recovered[7],
                    bytes(recovered[8]),
                )
            rows = db.execute(
                "SELECT probe_id FROM probes WHERE consumed=0 AND context_hash=? "
                "AND policy_hash=? AND group_count=? ORDER BY probe_id",
                (
                    candidate.context_hash,
                    candidate.probe_policy_hash,
                    candidate.risk.group_count,
                ),
            ).fetchall()
            eligible = sorted(
                (
                    self._probes[row[0]]
                    for row in rows
                    if not excluded.intersection(
                        group.source_manifest_hash
                        for group in self._probes[row[0]].groups
                    )
                ),
                key=lambda probe: probe.probe_id_hash,
            )
            if not eligible:
                db.rollback()
                raise ValueError(
                    "no unused, context/policy/count/source-disjoint probe is eligible"
                )
            eligible_hashes = tuple(probe.probe_id_hash for probe in eligible)
            if eligible_hashes != candidate.eligible_probe_id_hashes:
                db.rollback()
                raise ValueError(
                    "post-fixation eligible set differs from the candidate commitment"
                )
            selected_index, draw_counter = _draw_index(
                frame_hash=self._signed_frame.frame.frame_hash,
                fixation_hash=candidate.fixation_hash,
                beacon_randomness=beacon_round.randomness,
                population_size=len(eligible),
            )
            probe = eligible[selected_index]
            probe_id = probe.probe_id
            shard_root = _selected_shard_root(probe)
            selected_entry = catalog_entries.get(probe.probe_id_hash)
            if selected_entry != probe.frame_entry or (
                selected_entry.payload_commitment != shard_root
            ):
                db.rollback()
                raise ValueError("selected probe does not open its sealed-catalog leaf")
            source_manifest_hashes = tuple(
                group.source_manifest_hash for group in probe.groups
            )
            try:
                audit_registry.reserve_risk_allocation(
                    schedule,
                    candidate.risk_schedule_index,
                    fixation_hash=candidate.fixation_hash,
                )
                audit_registry.reserve_source_manifests(
                    source_manifest_hashes,
                    context_hash=candidate.context_hash,
                    fixation_hash=candidate.fixation_hash,
                    probe_id_hash=probe.probe_id_hash,
                )
                risk_ledger.reserve_source_manifests(
                    source_manifest_hashes,
                    context_hash=candidate.context_hash,
                    fixation_hash=candidate.fixation_hash,
                    probe_id_hash=probe.probe_id_hash,
                )
            except ValueError:
                db.rollback()
                raise
            fields = _release_fields(
                candidate,
                selected_entry,
                self._signed_frame,
                signed_beacon_round,
                raw_beacon_key,
                shard_root,
                eligible_hashes,
                draw_counter,
            )
            payload = canonical_bytes(fields)
            release_hash = digest(fields)
            signature = self._private_key.sign(payload)
            updated = db.execute(
                "UPDATE probes SET consumed=1, fixation_hash=?, release_shard_root=?, "
                "release_beacon_round_hash=?, release_beacon_signature=?, "
                "release_beacon_public_key=?, release_eligible_ids=?, "
                "release_draw_counter=?, "
                "release_hash=?, release_signature=? "
                "WHERE probe_id=? AND consumed=0",
                (
                    candidate.fixation_hash,
                    shard_root,
                    beacon_round.round_hash,
                    signed_beacon_round.signature,
                    raw_beacon_key,
                    ",".join(eligible_hashes).encode("ascii"),
                    draw_counter,
                    release_hash,
                    signature,
                    probe_id,
                ),
            ).rowcount
            if updated != 1:
                db.rollback()
                raise RuntimeError("atomic probe retirement lost a concurrent race")
            consumed_groups = db.execute(
                "UPDATE probe_groups SET consumed=1 WHERE probe_id=? AND consumed=0",
                (probe_id,),
            ).rowcount
            if consumed_groups != len(self._probes[probe_id].groups):
                db.rollback()
                raise RuntimeError(
                    "probe contains a source group retired by another release"
                )
            db.commit()
        return ProbeRelease(
            probe,
            self._signed_frame,
            candidate.candidate_hash,
            candidate.fixation_hash,
            candidate.sealed_catalog_root,
            shard_root,
            signed_beacon_round,
            raw_beacon_key,
            eligible_hashes,
            draw_counter,
            release_hash,
            signature,
        )

    def is_consumed(self, probe_id: str) -> bool:
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT consumed FROM probes WHERE probe_id=?", (probe_id,)
            ).fetchone()
        if row is None:
            raise KeyError(probe_id)
        return bool(row[0])


def verify_public_release(
    release: PublicProbeRelease,
    candidate: Candidate,
    public_key: Ed25519PublicKey,
    frame_public_key: Ed25519PublicKey,
) -> bool:
    if (
        release.candidate_hash != candidate.candidate_hash
        or release.fixation_hash != candidate.fixation_hash
    ):
        return False
    frame = release.signed_sampling_frame.commitment
    if (
        not verify_sampling_frame_commitment(
            release.signed_sampling_frame, frame_public_key
        )
        or frame.frame_hash != candidate.sampling_frame_hash
        or frame.catalog_root != candidate.sealed_catalog_root
        or release.sealed_catalog_root != candidate.sealed_catalog_root
        or frame.context_hash != candidate.context_hash
        or frame.policy_hash != candidate.probe_policy_hash
        or candidate.source_partition.partition_hash not in frame.source_partition_hashes
        or not verify_merkle_path(
            {
                "domain": "fedmerit-sealed-catalog-leaf-v2",
                "entry": release.selected_catalog_entry,
            },
            release.catalog_membership_path,
            frame.catalog_root,
        )
    ):
        return False
    selected_entry = release.selected_catalog_entry
    if (
        selected_entry.context_hash != candidate.context_hash
        or selected_entry.policy_hash != candidate.probe_policy_hash
        or selected_entry.group_count != candidate.risk.group_count
    ):
        return False
    if release.commit_probe_commitment == candidate.score_probe_commitment:
        return False
    try:
        beacon_public_key = Ed25519PublicKey.from_public_bytes(
            release.beacon_public_key
        )
    except ValueError:
        return False
    beacon_round = release.signed_beacon_round.round
    if (
        hashlib.sha256(release.beacon_public_key).hexdigest()
        != frame.beacon_public_key_hash
        or beacon_round.beacon_id != frame.beacon_id
        or beacon_round.round_number != candidate.beacon_round
        or beacon_round.previous_round_hash != candidate.beacon_parent_hash
        or beacon_round.fixation_hash != candidate.fixation_hash
        or not verify_beacon_round(release.signed_beacon_round, beacon_public_key)
    ):
        return False
    frame_ids = set(frame.catalog_id_hashes)
    if release.eligible_probe_id_hashes != candidate.eligible_probe_id_hashes or any(
        item not in frame_ids for item in release.eligible_probe_id_hashes
    ):
        return False
    selected_index, draw_counter = _draw_index(
        frame_hash=frame.frame_hash,
        fixation_hash=candidate.fixation_hash,
        beacon_randomness=beacon_round.randomness,
        population_size=len(release.eligible_probe_id_hashes),
    )
    if (
        draw_counter != release.draw_counter
        or release.eligible_probe_id_hashes[selected_index]
        != selected_entry.probe_id_hash
    ):
        return False
    fields = _release_fields(
        candidate,
        selected_entry,
        release.signed_sampling_frame,
        release.signed_beacon_round,
        release.beacon_public_key,
        release.commit_probe_commitment,
        release.eligible_probe_id_hashes,
        release.draw_counter,
    )
    if digest(fields) != release.release_hash:
        return False
    try:
        public_key.verify(release.signature, canonical_bytes(fields))
    except InvalidSignature:
        return False
    return True


def verify_release(
    release: ProbeRelease,
    candidate: Candidate,
    public_key: Ed25519PublicKey,
    frame_public_key: Ed25519PublicKey,
) -> bool:
    """Authorized verification of the catalog membership and raw-payload opening."""
    if not verify_public_release(
        release.public_release, candidate, public_key, frame_public_key
    ):
        return False
    if release.probe.frame_entry != release.public_release.selected_catalog_entry:
        return False
    if release.commit_probe_commitment != release.probe.commitment:
        return False
    excluded = set(candidate.excluded_source_manifests).union(
        release.signed_sampling_frame.frame.exclusion_source_manifest_hashes
    )
    if excluded.intersection(
        group.source_manifest_hash for group in release.probe.groups
    ):
        return False
    return len(release.probe.groups) == candidate.risk.group_count


def _as_decimal(value: float) -> Decimal:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("replay inputs must be finite")
    return Decimal.from_float(number)


def _sigmoid(value: Decimal, *, logit_clamp: int) -> Decimal:
    clamp = Decimal(logit_clamp)
    if value >= clamp:
        return Decimal(1)
    if value <= -clamp:
        return Decimal(0)
    if value >= 0:
        return Decimal(1) / (Decimal(1) + (-value).exp())
    exp_value = value.exp()
    return exp_value / (Decimal(1) + exp_value)


def _loss_quanta(value: Decimal, *, quanta_per_unit: int) -> int:
    bounded = min(Decimal(1), max(Decimal(0), value))
    scaled = bounded * Decimal(quanta_per_unit)
    return int(scaled.to_integral_value(rounding=ROUND_HALF_EVEN))


def _group_brier_quanta(
    model: LinearModelArtifact,
    features: tuple[tuple[float, ...], ...],
    labels: tuple[int, ...],
    policy: EvaluationPolicy,
) -> int:
    width = len(model.weights) - 1
    if any(len(row) != width for row in features):
        raise ValueError("model and raw probe feature dimensions differ")
    with localcontext() as context:
        context.prec = policy.decimal_precision
        context.rounding = ROUND_HALF_EVEN
        weighted_loss = Decimal(0)
        total_weight = Decimal(0)
        for row, label in zip(features, labels, strict=True):
            if model.feature_mean:
                transformed = tuple(
                    (_as_decimal(value) - _as_decimal(mean)) / _as_decimal(scale)
                    for value, mean, scale in zip(
                        row, model.feature_mean, model.feature_scale, strict=True
                    )
                )
            else:
                transformed = tuple(_as_decimal(value) for value in row)
            logit = sum(
                (
                    x * _as_decimal(weight)
                    for x, weight in zip(transformed, model.weights[:-1], strict=True)
                ),
                _as_decimal(model.weights[-1]),
            )
            prediction = _sigmoid(
                logit, logit_clamp=policy.sigmoid_logit_clamp
            )
            weight = _as_decimal(policy.class_weights[label])
            weighted_loss += weight * (prediction - Decimal(label)) ** 2
            total_weight += weight
        quanta_per_unit = int(Decimal(1) / Decimal(policy.group_loss_quantum))
        return _loss_quanta(
            weighted_loss / total_weight, quanta_per_unit=quanta_per_unit
        )


def paired_model_loss_difference_exact(
    before_model: LinearModelArtifact,
    after_model: LinearModelArtifact,
    groups: tuple[ProbeGroup, ...],
    policy: EvaluationPolicy,
) -> Fraction:
    """Replay the exact registered paired loss for two concrete model artifacts."""
    if not groups:
        raise ValueError("paired replay requires at least one source group")
    if tuple(group.group_id for group in groups) != tuple(
        sorted(group.group_id for group in groups)
    ):
        raise ValueError("probe group order is not canonical")
    differences: list[int] = []
    for group in groups:
        before = _group_brier_quanta(
            before_model, group.features, group.labels, policy
        )
        after = _group_brier_quanta(
            after_model, group.features, group.labels, policy
        )
        differences.append(after - before)
    quanta_per_unit = int(Decimal(1) / Decimal(policy.group_loss_quantum))
    return Fraction(sum(differences), len(differences) * quanta_per_unit)


def paired_loss_difference_exact(candidate: Candidate, probe: CommitProbe) -> Fraction:
    """Return the exact mean of the policy-quantized paired group losses."""
    policy = candidate.evaluation_policy
    if probe.probe_policy_hash != policy.policy_hash:
        raise ValueError("probe and candidate evaluator policies differ")
    return paired_model_loss_difference_exact(
        candidate.before_model,
        candidate.after_model,
        probe.groups,
        policy,
    )


def paired_loss_difference(candidate: Candidate, probe: CommitProbe) -> float:
    """Return the canonical binary64 display of the exact paired mean."""
    result = float(paired_loss_difference_exact(candidate, probe))
    if not math.isfinite(result) or not -1 <= result <= 1:
        raise ValueError("deterministic replay produced an invalid bounded difference")
    return result


def gate_decision(
    candidate: Candidate, probe: CommitProbe
) -> tuple[Fraction, float, str]:
    """Compare the exact paired rational with the exact binary64 risk threshold."""
    exact = paired_loss_difference_exact(candidate, probe)
    display = float(exact)
    threshold = -Fraction.from_float(float(candidate.risk.gamma))
    return exact, display, "commit" if exact <= threshold else "reject"

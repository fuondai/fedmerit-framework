"""Quorum issuance and artifact-level verification of FedMERIT receipts."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .gate import (
    PublicProbeRelease,
    ProbeRelease,
    RiskLedger,
    gate_decision,
    risk_is_satisfied,
    verify_beacon_round,
    verify_public_release,
    verify_release,
    verify_sampling_frame,
)
from .canonical import canonical_bytes, digest
from .model import (
    Candidate,
    EvaluationPolicy,
    LinearModelArtifact,
    Receipt,
    ReceiptCore,
    RiskSchedule,
    SignedBeaconRound,
    SignedSamplingFrame,
    StateContext,
    UINT32_MAX,
    WitnessSignature,
    ZERO_HASH,
)


def _raw_public_key(public_key: Ed25519PublicKey) -> bytes:
    return public_key.public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )


def _require_digest(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA-256 digest") from exc
    return value.lower()


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class VerificationTrust:
    """Context-scoped verifier keys and Byzantine threshold."""

    f: int
    witness_public_keys: tuple[bytes, ...]
    store_public_key: bytes
    frame_public_key: bytes
    roster_epoch: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.f, bool) or not isinstance(self.f, int) or self.f < 0:
            raise ValueError("f must be a non-negative integer")
        if len(self.witness_public_keys) != 3 * self.f + 1:
            raise ValueError("witness trust must contain exactly 3f+1 keys")
        if len(self.witness_public_keys) > 64:
            raise ValueError("witness trust exceeds the registered key cap")
        if (
            any(len(key) != 32 for key in self.witness_public_keys)
            or len(set(self.witness_public_keys)) != len(self.witness_public_keys)
        ):
            raise ValueError("witness trust keys must be distinct Ed25519 public keys")
        if len(self.store_public_key) != 32 or len(self.frame_public_key) != 32:
            raise ValueError("store and frame trust roots must be Ed25519 public keys")
        if (
            isinstance(self.roster_epoch, bool)
            or not isinstance(self.roster_epoch, int)
            or not 0 <= self.roster_epoch < 1 << 32
        ):
            raise ValueError("roster_epoch must be an unsigned 32-bit integer")

    @property
    def authority_certificate_hash(self) -> str:
        """Bind the roster epoch, threshold, and every verification key."""
        return digest(self)

    @classmethod
    def from_keys(
        cls,
        public_keys: Sequence[Ed25519PublicKey],
        *,
        f: int,
        store_public_key: Ed25519PublicKey,
        frame_public_key: Ed25519PublicKey,
        roster_epoch: int = 0,
    ) -> "VerificationTrust":
        return cls(
            f=f,
            witness_public_keys=tuple(_raw_public_key(key) for key in public_keys),
            store_public_key=_raw_public_key(store_public_key),
            frame_public_key=_raw_public_key(frame_public_key),
            roster_epoch=roster_epoch,
        )


def _handover_fields(
    *,
    previous_context_hash: str,
    successor_context: StateContext,
    previous_receipt_hash: str,
    installed_model_hash: str,
    installed_model_version: int,
    roster_epoch: int,
) -> dict[str, object]:
    return {
        "domain": "fedmerit-context-handover-v1",
        "previous_context_hash": previous_context_hash,
        "successor_context": successor_context,
        "previous_receipt_hash": previous_receipt_hash,
        "installed_model_hash": installed_model_hash,
        "installed_model_version": installed_model_version,
        "roster_epoch": roster_epoch,
    }


@dataclass(frozen=True)
class HandoverAuthorization:
    """Old-roster quorum authorization for one immediate context successor."""

    previous_context_hash: str
    successor_context: StateContext
    previous_receipt_hash: str
    installed_model_hash: str
    installed_model_version: int
    roster_epoch: int
    witness_count: int
    signatures: tuple[WitnessSignature, ...]

    def __post_init__(self) -> None:
        for name in (
            "previous_context_hash",
            "previous_receipt_hash",
            "installed_model_hash",
        ):
            _require_digest(getattr(self, name), name)
        for name in ("installed_model_version", "roster_epoch", "witness_count"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value >= 1 << 32
            ):
                raise ValueError(f"{name} must be uint32")
        if self.witness_count == 0:
            raise ValueError("handover witness_count must be positive")
        signatures = tuple(self.signatures)
        object.__setattr__(self, "signatures", signatures)
        if (
            any(not isinstance(item, WitnessSignature) for item in signatures)
        ):
            raise ValueError("handover signatures must contain witness signatures")
        indices = tuple(item.witness_index for item in signatures)
        if (
            indices != tuple(sorted(indices))
            or len(indices) != len(set(indices))
            or any(index >= self.witness_count for index in indices)
        ):
            raise ValueError("handover signatures need distinct ascending roster indices")

    @property
    def signing_bytes(self) -> bytes:
        return canonical_bytes(
            _handover_fields(
                previous_context_hash=self.previous_context_hash,
                successor_context=self.successor_context,
                previous_receipt_hash=self.previous_receipt_hash,
                installed_model_hash=self.installed_model_hash,
                installed_model_version=self.installed_model_version,
                roster_epoch=self.roster_epoch,
            )
        )

    @property
    def authorization_hash(self) -> str:
        return hashlib.sha256(self.signing_bytes).hexdigest()


def verify_handover_authorization(
    authorization: HandoverAuthorization,
    trust: VerificationTrust,
) -> bool:
    """Verify that at least 2f+1 old-roster members signed one transition."""
    if (
        authorization.roster_epoch != trust.roster_epoch
        or authorization.witness_count != len(trust.witness_public_keys)
        or len(authorization.signatures) < 2 * trust.f + 1
    ):
        return False
    seen: set[int] = set()
    try:
        for item in authorization.signatures:
            if item.witness_index in seen:
                return False
            key = Ed25519PublicKey.from_public_bytes(
                trust.witness_public_keys[item.witness_index]
            )
            key.verify(item.signature, authorization.signing_bytes)
            seen.add(item.witness_index)
    except (IndexError, InvalidSignature, ValueError):
        return False
    return len(seen) >= 2 * trust.f + 1


def _verification_trust_from_blob(payload: bytes) -> VerificationTrust:
    """Decode the canonical trust record stored by ``AuditRegistry``."""
    try:
        value = json.loads(payload)
        return VerificationTrust(
            f=int(value["f"]),
            witness_public_keys=tuple(
                bytes.fromhex(item["hex"]) for item in value["witness_public_keys"]
            ),
            store_public_key=bytes.fromhex(value["store_public_key"]["hex"]),
            frame_public_key=bytes.fromhex(value["frame_public_key"]["hex"]),
            roster_epoch=int(value["roster_epoch"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("stored verification trust has invalid canonical bytes") from exc


def _replay_core(
    candidate: Candidate,
    release: ProbeRelease,
    *,
    store_public_key: Ed25519PublicKey,
    frame_public_key: Ed25519PublicKey,
    schedule: RiskSchedule,
    risk_ledger: RiskLedger,
    audit_registry: "AuditRegistry",
) -> ReceiptCore:
    audit_registry.validate_candidate(candidate)
    audit_registry.require_risk_schedule(schedule)
    allocation = schedule.allocation(candidate.risk_schedule_index)
    if (
        schedule.context_hash != candidate.context_hash
        or schedule.schedule_hash != candidate.risk_schedule_hash
        or allocation != candidate.risk
    ):
        raise ValueError("candidate does not match its predeclared risk schedule")
    if not risk_ledger.is_consumed(
        schedule.schedule_hash, candidate.risk_schedule_index, candidate.fixation_hash
    ) or not risk_ledger.fixation_precedes_beacon(candidate):
        raise ValueError(
            "risk allocation was not fixed before its committed beacon round"
        )
    if not verify_release(release, candidate, store_public_key, frame_public_key):
        raise ValueError("probe release token is invalid or not bound to the fixation")
    if not risk_is_satisfied(
        candidate.risk.group_count,
        candidate.risk.epsilon,
        candidate.risk.gamma,
        candidate.risk.alpha,
    ):
        raise ValueError(
            "precommitted source-group count does not satisfy the risk allocation"
        )
    _, delta_hat, decision = gate_decision(candidate, release.probe)
    return ReceiptCore(
        candidate.context_hash,
        candidate.before_model_hash,
        candidate.after_model_hash,
        candidate.fixation_hash,
        candidate.contributor_root,
        candidate.score_probe_commitment,
        candidate.probe_policy_hash,
        release.release_hash,
        candidate.risk_schedule_hash,
        candidate.previous_receipt_hash,
        candidate.risk.group_count,
        candidate.risk.epsilon,
        candidate.risk.gamma,
        candidate.risk.alpha,
        delta_hat,
        decision,
    )


@dataclass(frozen=True)
class Witness:
    witness_index: int
    private_key: Ed25519PrivateKey
    state_path: str

    @classmethod
    def open(
        cls,
        witness_index: int,
        state_path: str | Path,
        *,
        private_key: Ed25519PrivateKey | None = None,
    ) -> "Witness":
        path = Path(state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path, timeout=30, isolation_level=None)) as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS witness_key(
                    id INTEGER PRIMARY KEY CHECK(id=1), witness_index INTEGER UNIQUE NOT NULL,
                    private_key BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS signed_scopes(
                    context_hash TEXT NOT NULL, before_model_hash TEXT NOT NULL,
                    previous_receipt_hash TEXT NOT NULL, receipt_hash TEXT NOT NULL,
                    PRIMARY KEY(context_hash, before_model_hash, previous_receipt_hash)
                );
                CREATE TABLE IF NOT EXISTS signed_handovers(
                    previous_context_hash TEXT PRIMARY KEY,
                    authorization_hash TEXT NOT NULL
                );
            """)
            row = db.execute(
                "SELECT witness_index, private_key FROM witness_key WHERE id=1"
            ).fetchone()
            if row is None:
                private_key = private_key or Ed25519PrivateKey.generate()
                raw = private_key.private_bytes(
                    serialization.Encoding.Raw,
                    serialization.PrivateFormat.Raw,
                    serialization.NoEncryption(),
                )
                db.execute(
                    "INSERT INTO witness_key VALUES(1,?,?)", (witness_index, raw)
                )
            else:
                if int(row[0]) != witness_index:
                    raise ValueError(
                        "witness state belongs to a different witness index"
                    )
                stored = bytes(row[1])
                if private_key is not None:
                    supplied = private_key.private_bytes(
                        serialization.Encoding.Raw,
                        serialization.PrivateFormat.Raw,
                        serialization.NoEncryption(),
                    )
                    if supplied != stored:
                        raise ValueError("witness state is bound to another private key")
                private_key = Ed25519PrivateKey.from_private_bytes(stored)
        return cls(witness_index, private_key, str(path))

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.state_path, timeout=30, isolation_level=None)

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self.private_key.public_key()

    def _sign_core(self, core: ReceiptCore) -> WitnessSignature:
        scope = (core.context_hash, core.before_model_hash, core.previous_receipt_hash)
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT receipt_hash FROM signed_scopes WHERE context_hash=? "
                "AND before_model_hash=? AND previous_receipt_hash=?",
                scope,
            ).fetchone()
            if row is not None and row[0] != core.receipt_hash:
                db.rollback()
                raise ValueError(
                    "witness refuses a conflicting core for this attempt scope"
                )
            db.execute(
                "INSERT OR IGNORE INTO signed_scopes VALUES(?,?,?,?)",
                (*scope, core.receipt_hash),
            )
            db.commit()
        return WitnessSignature(
            self.witness_index, self.private_key.sign(core.to_bytes())
        )

    def authorize_handover(self, signing_bytes: bytes) -> WitnessSignature:
        """Sign at most one successor transition from a context head."""
        try:
            fields = json.loads(signing_bytes)
            if canonical_bytes(fields) != bytes(signing_bytes):
                raise ValueError("handover authorization is not canonically encoded")
            if fields.get("domain") != "fedmerit-context-handover-v1":
                raise ValueError("handover authorization domain is invalid")
            previous_context_hash = str(fields["previous_context_hash"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("handover authorization has invalid canonical bytes") from exc
        _require_digest(previous_context_hash, "previous_context_hash")
        authorization_hash = hashlib.sha256(signing_bytes).hexdigest()
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute(
                "SELECT authorization_hash FROM signed_handovers "
                "WHERE previous_context_hash=?",
                (previous_context_hash,),
            ).fetchone()
            if prior is not None and str(prior[0]) != authorization_hash:
                db.rollback()
                raise ValueError(
                    "witness refuses conflicting handovers from one context head"
                )
            db.execute(
                "INSERT OR IGNORE INTO signed_handovers VALUES(?,?)",
                (previous_context_hash, authorization_hash),
            )
            db.commit()
        return WitnessSignature(
            self.witness_index,
            self.private_key.sign(signing_bytes),
        )

    def attest(
        self,
        candidate: Candidate,
        release: ProbeRelease,
        *,
        store_public_key: Ed25519PublicKey,
        frame_public_key: Ed25519PublicKey,
        schedule: RiskSchedule,
        risk_ledger: RiskLedger,
        audit_registry: "AuditRegistry",
    ) -> tuple[ReceiptCore, WitnessSignature]:
        core = _replay_core(
            candidate,
            release,
            store_public_key=store_public_key,
            frame_public_key=frame_public_key,
            schedule=schedule,
            risk_ledger=risk_ledger,
            audit_registry=audit_registry,
        )
        return core, self._sign_core(core)


class CertificateAuthority:
    """Reference witness quorum; issuance derives all gate fields from fixation/release."""

    def __init__(self, witnesses: list[Witness], f: int, *, roster_epoch: int = 0) -> None:
        if isinstance(f, bool) or not isinstance(f, int) or f < 0:
            raise ValueError("f must be a non-negative integer")
        if len(witnesses) != 3 * f + 1:
            raise ValueError("witness set must contain exactly 3f+1 members")
        indices = tuple(w.witness_index for w in witnesses)
        if indices != tuple(range(len(witnesses))):
            raise ValueError("witness indices must be contiguous from zero")
        public_keys = tuple(_raw_public_key(w.public_key) for w in witnesses)
        if len(set(public_keys)) != len(public_keys):
            raise ValueError("witness public keys must be distinct")
        self.witnesses = tuple(witnesses)
        self.f = f
        if (
            isinstance(roster_epoch, bool)
            or not isinstance(roster_epoch, int)
            or not 0 <= roster_epoch < 1 << 32
        ):
            raise ValueError("roster_epoch must be an unsigned 32-bit integer")
        self.roster_epoch = roster_epoch

    @classmethod
    def persistent(
        cls,
        directory: str | Path,
        f: int = 1,
        *,
        private_keys: Sequence[Ed25519PrivateKey] | None = None,
        roster_epoch: int = 0,
    ) -> "CertificateAuthority":
        if isinstance(f, bool) or not isinstance(f, int) or f < 0:
            raise ValueError("f must be a non-negative integer")
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        if private_keys is not None and len(private_keys) != 3 * f + 1:
            raise ValueError("private_keys must contain exactly 3f+1 keys")
        return cls(
            [
                Witness.open(
                    i,
                    root / f"witness-{i}.sqlite3",
                    private_key=None if private_keys is None else private_keys[i],
                )
                for i in range(3 * f + 1)
            ],
            f,
            roster_epoch=roster_epoch,
        )

    @property
    def public_keys(self) -> tuple[Ed25519PublicKey, ...]:
        return tuple(w.public_key for w in self.witnesses)

    def issue(
        self,
        candidate: Candidate,
        release: ProbeRelease,
        *,
        store_public_key: Ed25519PublicKey,
        frame_public_key: Ed25519PublicKey,
        schedule: RiskSchedule,
        risk_ledger: RiskLedger,
        audit_registry: "AuditRegistry",
    ) -> Receipt:
        """Collect independently replayed attestations from a threshold of witnesses."""
        trust = VerificationTrust.from_keys(
            self.public_keys,
            f=self.f,
            store_public_key=store_public_key,
            frame_public_key=frame_public_key,
            roster_epoch=self.roster_epoch,
        )
        audit_registry.require_verification_trust(
            candidate.context_hash,
            candidate.state_context.authority_certificate_hash,
            trust,
        )
        threshold = 2 * self.f + 1
        buckets: dict[str, tuple[ReceiptCore, list[WitnessSignature]]] = {}
        for witness in self.witnesses:
            try:
                replayed, signature = witness.attest(
                    candidate,
                    release,
                    store_public_key=store_public_key,
                    frame_public_key=frame_public_key,
                    schedule=schedule,
                    risk_ledger=risk_ledger,
                    audit_registry=audit_registry,
                )
                if signature.witness_index != witness.witness_index:
                    continue
                witness.public_key.verify(signature.signature, replayed.to_bytes())
            except Exception:
                continue
            key = replayed.receipt_hash
            bucket = buckets.setdefault(key, (replayed, []))
            if bucket[0] != replayed:
                continue
            bucket[1].append(signature)
            if len(bucket[1]) == threshold:
                receipt = Receipt(
                    replayed,
                    len(self.witnesses),
                    tuple(sorted(bucket[1], key=lambda item: item.witness_index)),
                )
                break
        else:
            raise ValueError("fewer than 2f+1 witnesses returned one valid replay core")
        audit_registry.record_issued_receipt(
            receipt,
            candidate=candidate,
            schedule=schedule,
            verification_trust=trust,
        )
        return receipt

    def authorize_handover(
        self,
        *,
        previous_context_hash: str,
        successor_context: StateContext,
        previous_receipt_hash: str,
        installed_model_hash: str,
        installed_model_version: int,
    ) -> HandoverAuthorization:
        """Collect an old-roster quorum over one fully bound handover tuple."""
        fields = _handover_fields(
            previous_context_hash=previous_context_hash,
            successor_context=successor_context,
            previous_receipt_hash=previous_receipt_hash,
            installed_model_hash=installed_model_hash,
            installed_model_version=installed_model_version,
            roster_epoch=self.roster_epoch,
        )
        payload = canonical_bytes(fields)
        threshold = 2 * self.f + 1
        signatures: list[WitnessSignature] = []
        for witness in self.witnesses:
            try:
                signature = witness.authorize_handover(payload)
                witness.public_key.verify(signature.signature, payload)
            except Exception:
                continue
            signatures.append(signature)
            if len(signatures) == threshold:
                break
        if len(signatures) < threshold:
            raise ValueError("fewer than 2f+1 witnesses authorized the handover")
        return HandoverAuthorization(
            previous_context_hash,
            successor_context,
            previous_receipt_hash,
            installed_model_hash,
            installed_model_version,
            self.roster_epoch,
            len(self.witnesses),
            tuple(sorted(signatures, key=lambda item: item.witness_index)),
        )


def _core_matches_candidate(
    core: ReceiptCore,
    candidate: Candidate,
    release: ProbeRelease | PublicProbeRelease,
) -> bool:
    return (
        core.context_hash == candidate.context_hash
        and core.before_model_hash == candidate.before_model_hash
        and core.after_model_hash == candidate.after_model_hash
        and core.fixation_hash == candidate.fixation_hash
        and core.contributor_root == candidate.contributor_root
        and core.score_probe_commitment == candidate.score_probe_commitment
        and core.probe_policy_hash == candidate.probe_policy_hash
        and core.release_hash == release.release_hash
        and core.risk_schedule_hash == candidate.risk_schedule_hash
        and core.previous_receipt_hash == candidate.previous_receipt_hash
        and core.source_group_count == candidate.risk.group_count
        and core.epsilon == candidate.risk.epsilon
        and core.gamma == candidate.risk.gamma
        and core.alpha == candidate.risk.alpha
    )


def _verify_public_receipt(
    receipt: Receipt,
    public_keys: Sequence[Ed25519PublicKey],
    *,
    f: int,
    release: PublicProbeRelease,
    candidate: Candidate,
    store_public_key: Ed25519PublicKey,
    frame_public_key: Ed25519PublicKey,
    schedule: RiskSchedule,
    risk_ledger: RiskLedger,
    audit_registry: "AuditRegistry",
    roster_epoch: int = 0,
) -> bool:
    if f < 0 or len(public_keys) != 3 * f + 1:
        return False
    trust = VerificationTrust.from_keys(
        public_keys,
        f=f,
        store_public_key=store_public_key,
        frame_public_key=frame_public_key,
        roster_epoch=roster_epoch,
    )
    audit_registry.require_verification_trust(
        candidate.context_hash,
        candidate.state_context.authority_certificate_hash,
        trust,
    )
    encoded_public_keys = trust.witness_public_keys
    if len(set(encoded_public_keys)) != len(encoded_public_keys):
        return False
    if receipt.witness_count != len(public_keys):
        return False
    if len(receipt.signatures) < 2 * f + 1:
        return False
    audit_registry.validate_candidate(candidate, receipt_hash=receipt.receipt_hash)
    audit_registry.require_risk_schedule(schedule)
    core = receipt.core
    allocation = schedule.allocation(candidate.risk_schedule_index)
    if (
        schedule.context_hash != candidate.context_hash
        or schedule.schedule_hash != candidate.risk_schedule_hash
        or allocation != candidate.risk
    ):
        return False
    if not risk_ledger.is_consumed(
        schedule.schedule_hash, candidate.risk_schedule_index, candidate.fixation_hash
    ) or not risk_ledger.fixation_precedes_beacon(candidate):
        return False
    if not verify_public_release(
        release, candidate, store_public_key, frame_public_key
    ):
        return False
    if not _core_matches_candidate(core, candidate, release):
        return False
    if not risk_is_satisfied(
        core.source_group_count, core.epsilon, core.gamma, core.alpha
    ):
        return False
    seen: set[int] = set()
    for item in receipt.signatures:
        if item.witness_index in seen or item.witness_index >= len(public_keys):
            return False
        public_keys[item.witness_index].verify(item.signature, core.to_bytes())
        seen.add(item.witness_index)
    return len(seen) >= 2 * f + 1


def verify_public_receipt(
    receipt: Receipt,
    public_keys: Sequence[Ed25519PublicKey],
    *,
    f: int,
    release: PublicProbeRelease,
    candidate: Candidate,
    store_public_key: Ed25519PublicKey,
    frame_public_key: Ed25519PublicKey,
    schedule: RiskSchedule,
    risk_ledger: RiskLedger,
    audit_registry: "AuditRegistry",
    roster_epoch: int = 0,
) -> bool:
    """Verify public bindings and quorum without receiving the raw probe opening."""
    try:
        return _verify_public_receipt(
            receipt,
            public_keys,
            f=f,
            release=release,
            candidate=candidate,
            store_public_key=store_public_key,
            frame_public_key=frame_public_key,
            schedule=schedule,
            risk_ledger=risk_ledger,
            audit_registry=audit_registry,
            roster_epoch=roster_epoch,
        )
    except Exception:
        return False


def _verify_receipt(
    receipt: Receipt,
    public_keys: Sequence[Ed25519PublicKey],
    *,
    f: int,
    release: ProbeRelease,
    candidate: Candidate,
    store_public_key: Ed25519PublicKey,
    frame_public_key: Ed25519PublicKey,
    schedule: RiskSchedule,
    risk_ledger: RiskLedger,
    audit_registry: "AuditRegistry",
    roster_epoch: int = 0,
) -> bool:
    if not _verify_public_receipt(
        receipt,
        public_keys,
        f=f,
        release=release.public_release,
        candidate=candidate,
        store_public_key=store_public_key,
        frame_public_key=frame_public_key,
        schedule=schedule,
        risk_ledger=risk_ledger,
        audit_registry=audit_registry,
        roster_epoch=roster_epoch,
    ):
        return False
    if not verify_release(release, candidate, store_public_key, frame_public_key):
        return False
    _, replay, decision = gate_decision(candidate, release.probe)
    # Equality is intentional: signed binary64 is the deterministic evaluator output.
    return receipt.core.delta_hat == replay and receipt.core.decision == decision


def verify_receipt(
    receipt: Receipt,
    public_keys: Sequence[Ed25519PublicKey],
    *,
    f: int,
    release: ProbeRelease,
    candidate: Candidate,
    store_public_key: Ed25519PublicKey,
    frame_public_key: Ed25519PublicKey,
    schedule: RiskSchedule,
    risk_ledger: RiskLedger,
    audit_registry: "AuditRegistry",
    roster_epoch: int = 0,
) -> bool:
    """Authorized raw-probe replay; return false for every malformed artifact."""

    try:
        return _verify_receipt(
            receipt,
            public_keys,
            f=f,
            release=release,
            candidate=candidate,
            store_public_key=store_public_key,
            frame_public_key=frame_public_key,
            schedule=schedule,
            risk_ledger=risk_ledger,
            audit_registry=audit_registry,
            roster_epoch=roster_epoch,
        )
    except Exception:
        return False


def verify_receipt_bytes(
    payload: bytes,
    *,
    witness_count: int,
    public_keys: Sequence[Ed25519PublicKey],
    f: int,
    release: ProbeRelease,
    candidate: Candidate,
    store_public_key: Ed25519PublicKey,
    frame_public_key: Ed25519PublicKey,
    schedule: RiskSchedule,
    risk_ledger: RiskLedger,
    audit_registry: "AuditRegistry",
    roster_epoch: int = 0,
) -> bool:
    try:
        receipt = Receipt.from_bytes(payload, witness_count=witness_count)
    except Exception:
        return False
    return verify_receipt(
        receipt,
        public_keys,
        f=f,
        release=release,
        candidate=candidate,
        store_public_key=store_public_key,
        frame_public_key=frame_public_key,
        schedule=schedule,
        risk_ledger=risk_ledger,
        audit_registry=audit_registry,
        roster_epoch=roster_epoch,
    )


class AuditRegistry:
    """Durable authenticated receipt head and installed-model registry."""

    def __init__(
        self,
        path: str | Path,
        *,
        genesis_model: LinearModelArtifact,
        initial_context: StateContext,
        evaluation_policy: EvaluationPolicy,
        verification_trust: VerificationTrust | None = None,
    ) -> None:
        if str(path) == ":memory:":
            raise ValueError("audit registry must use durable storage")
        if not isinstance(genesis_model, LinearModelArtifact):
            raise TypeError("genesis_model must be a LinearModelArtifact")
        genesis_model_hash = genesis_model.artifact_hash
        genesis_model_blob = genesis_model.artifact_bytes
        if initial_context.policy_hash != evaluation_policy.policy_hash:
            raise ValueError("initial context does not authorize the evaluation policy")
        self.path = str(path)
        self._evaluation_policy = evaluation_policy
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS audit_state(
                    id INTEGER PRIMARY KEY CHECK(id=1), head TEXT NOT NULL,
                    installed_model_hash TEXT NOT NULL, genesis_model_hash TEXT NOT NULL,
                    installed_model_version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS serving_model(
                    id INTEGER PRIMARY KEY CHECK(id=1),
                    model_hash TEXT NOT NULL,
                    model_version INTEGER NOT NULL,
                    artifact_blob BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS context_head(
                    id INTEGER PRIMARY KEY CHECK(id=1),
                    context_hash TEXT NOT NULL,
                    authority_certificate_hash TEXT NOT NULL,
                    policy_hash TEXT NOT NULL,
                    twin_id TEXT NOT NULL,
                    state_version INTEGER NOT NULL,
                    model_version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evaluation_policies(
                    policy_hash TEXT PRIMARY KEY, policy_blob BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS registered_contexts(
                    context_hash TEXT PRIMARY KEY, context_blob BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS verification_trust(
                    context_hash TEXT PRIMARY KEY, trust_blob BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS issued_receipts(
                    receipt_hash TEXT PRIMARY KEY, context_hash TEXT NOT NULL,
                    before_model_hash TEXT NOT NULL, after_model_hash TEXT NOT NULL,
                    previous_receipt_hash TEXT NOT NULL,
                    fixation_hash TEXT NOT NULL, schedule_hash TEXT NOT NULL,
                    schedule_index INTEGER NOT NULL,
                    UNIQUE(context_hash, before_model_hash, previous_receipt_hash),
                    UNIQUE(schedule_hash, schedule_index)
                );
                CREATE TABLE IF NOT EXISTS receipts(
                    receipt_hash TEXT PRIMARY KEY, context_hash TEXT NOT NULL,
                    before_model_hash TEXT NOT NULL, after_model_hash TEXT NOT NULL,
                    previous_receipt_hash TEXT NOT NULL,
                    fixation_hash TEXT NOT NULL, schedule_hash TEXT NOT NULL,
                    schedule_index INTEGER NOT NULL,
                    UNIQUE(context_hash, before_model_hash, previous_receipt_hash),
                    UNIQUE(schedule_hash, schedule_index)
                );
                CREATE TABLE IF NOT EXISTS retired_source_manifests(
                    source_manifest_hash TEXT PRIMARY KEY,
                    context_hash TEXT NOT NULL,
                    fixation_hash TEXT NOT NULL,
                    probe_id_hash TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS risk_schedules(
                    schedule_hash TEXT PRIMARY KEY,
                    context_hash TEXT UNIQUE NOT NULL,
                    anchor_receipt_hash TEXT NOT NULL,
                    lifetime_delta REAL NOT NULL,
                    schedule_blob BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS lineage_risk_budget(
                    id INTEGER PRIMARY KEY CHECK(id=1),
                    twin_id TEXT UNIQUE NOT NULL,
                    anchor_receipt_hash TEXT NOT NULL,
                    lifetime_delta REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS spent_risk_allocations(
                    schedule_hash TEXT NOT NULL,
                    allocation_index INTEGER NOT NULL,
                    context_hash TEXT NOT NULL,
                    fixation_hash TEXT NOT NULL,
                    PRIMARY KEY(schedule_hash, allocation_index)
                );
                CREATE TABLE IF NOT EXISTS beacon_successor_reservations(
                    beacon_public_key_hash TEXT NOT NULL,
                    round_number INTEGER NOT NULL,
                    parent_round_hash TEXT NOT NULL,
                    fixation_hash TEXT UNIQUE NOT NULL,
                    PRIMARY KEY(beacon_public_key_hash, round_number)
                );
                CREATE TABLE IF NOT EXISTS beacon_heads(
                    beacon_public_key_hash TEXT PRIMARY KEY,
                    beacon_id TEXT NOT NULL,
                    round_number INTEGER NOT NULL,
                    round_hash TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS lineage_security_caps(
                    id INTEGER PRIMARY KEY CHECK(id=1),
                    twin_id TEXT UNIQUE NOT NULL,
                    profile_blob BLOB NOT NULL,
                    max_attempts INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS risk_schedule_usage(
                    schedule_hash TEXT PRIMARY KEY,
                    attempt_count INTEGER NOT NULL CHECK(attempt_count > 0)
                );
                CREATE TABLE IF NOT EXISTS protocol_events(
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    context_hash TEXT NOT NULL,
                    object_hash TEXT NOT NULL,
                    detail_hash TEXT NOT NULL,
                    previous_event_hash TEXT NOT NULL,
                    event_hash TEXT UNIQUE NOT NULL,
                    UNIQUE(event_type, object_hash)
                );
                CREATE TRIGGER IF NOT EXISTS protocol_events_no_update
                BEFORE UPDATE ON protocol_events
                BEGIN
                    SELECT RAISE(ABORT, 'protocol event journal is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS protocol_events_no_delete
                BEFORE DELETE ON protocol_events
                BEGIN
                    SELECT RAISE(ABORT, 'protocol event journal is append-only');
                END;
            """)
            context_columns = tuple(
                row[1] for row in db.execute("PRAGMA table_info(context_head)")
            )
            receipt_columns = tuple(
                row[1] for row in db.execute("PRAGMA table_info(receipts)")
            )
            audit_columns = tuple(
                row[1] for row in db.execute("PRAGMA table_info(audit_state)")
            )
            if (
                "policy_hash" not in context_columns
                or "after_model_hash" not in receipt_columns
            ):
                raise ValueError(
                    "legacy audit registry requires explicit schema migration"
                )
            for column, column_type in (
                ("twin_id", "TEXT"),
                ("state_version", "INTEGER"),
                ("model_version", "INTEGER"),
            ):
                if column not in context_columns:
                    db.execute(
                        f"ALTER TABLE context_head ADD COLUMN {column} {column_type}"
                    )
            if "installed_model_version" not in audit_columns:
                db.execute(
                    "ALTER TABLE audit_state ADD COLUMN installed_model_version INTEGER"
                )
                db.execute(
                    "UPDATE audit_state SET installed_model_version=? "
                    "WHERE installed_model_version IS NULL",
                    (initial_context.model_version,),
                )
            row = db.execute(
                "SELECT genesis_model_hash FROM audit_state WHERE id=1"
            ).fetchone()
            if row is None:
                db.execute(
                    "INSERT INTO audit_state "
                    "(id, head, installed_model_hash, genesis_model_hash, "
                    "installed_model_version) VALUES(1,?,?,?,?)",
                    (
                        ZERO_HASH,
                        genesis_model_hash,
                        genesis_model_hash,
                        initial_context.model_version,
                    ),
                )
            elif row[0] != genesis_model_hash:
                raise ValueError(
                    "audit registry was initialized for another genesis model"
                )
            audit_model = db.execute(
                "SELECT installed_model_hash, installed_model_version "
                "FROM audit_state WHERE id=1"
            ).fetchone()
            serving_model = db.execute(
                "SELECT model_hash, model_version, artifact_blob "
                "FROM serving_model WHERE id=1"
            ).fetchone()
            expected_serving = (str(audit_model[0]), int(audit_model[1]))
            if serving_model is None:
                if expected_serving[0] != genesis_model_hash:
                    raise ValueError(
                        "serving bytes are missing for a non-genesis installed model"
                    )
                db.execute(
                    "INSERT INTO serving_model "
                    "(id, model_hash, model_version, artifact_blob) VALUES(1,?,?,?)",
                    (*expected_serving, genesis_model_blob),
                )
            else:
                serving_blob = None if serving_model[2] is None else bytes(serving_model[2])
                if (
                    (str(serving_model[0]), int(serving_model[1])) != expected_serving
                    or serving_blob is None
                    or _digest_bytes(serving_blob) != str(serving_model[0])
                ):
                    raise ValueError(
                        "serving model bytes and authenticated audit state disagree"
                    )
            context_row = db.execute(
                "SELECT context_hash, authority_certificate_hash, policy_hash, "
                "twin_id, state_version, model_version FROM context_head WHERE id=1"
            ).fetchone()
            if context_row is None:
                db.execute(
                    "INSERT INTO context_head "
                    "(id, context_hash, authority_certificate_hash, policy_hash, "
                    "twin_id, state_version, model_version) VALUES(1,?,?,?,?,?,?)",
                    (
                        initial_context.context_hash,
                        initial_context.authority_certificate_hash,
                        evaluation_policy.policy_hash,
                        initial_context.twin_id,
                        initial_context.state_version,
                        initial_context.model_version,
                    ),
                )
            else:
                expected_context_row = (
                    initial_context.context_hash,
                    initial_context.authority_certificate_hash,
                    evaluation_policy.policy_hash,
                    initial_context.twin_id,
                    initial_context.state_version,
                    initial_context.model_version,
                )
                if context_row[:3] != expected_context_row[:3]:
                    raise ValueError(
                        "provided initial context does not match the provisioned live context"
                    )
                if context_row[3:] == (None, None, None):
                    db.execute(
                        "UPDATE context_head SET twin_id=?, state_version=?, model_version=? "
                        "WHERE id=1",
                        expected_context_row[3:],
                    )
                elif context_row[3:] != expected_context_row[3:]:
                    raise ValueError(
                        "stored context identity or version disagrees with its live descriptor"
                    )
            policy_blob = canonical_bytes(evaluation_policy)
            registered = db.execute(
                "SELECT policy_blob FROM evaluation_policies WHERE policy_hash=?",
                (evaluation_policy.policy_hash,),
            ).fetchone()
            if registered is not None and bytes(registered[0]) != policy_blob:
                raise ValueError("evaluation policy hash is bound to different content")
            db.execute(
                "INSERT OR IGNORE INTO evaluation_policies VALUES(?,?)",
                (evaluation_policy.policy_hash, policy_blob),
            )
            context_blob = canonical_bytes(initial_context)
            registered_context = db.execute(
                "SELECT context_blob FROM registered_contexts WHERE context_hash=?",
                (initial_context.context_hash,),
            ).fetchone()
            if registered_context is not None and bytes(registered_context[0]) != context_blob:
                raise ValueError("context hash is bound to different content")
            db.execute(
                "INSERT OR IGNORE INTO registered_contexts VALUES(?,?)",
                (initial_context.context_hash, context_blob),
            )
            if verification_trust is not None:
                self._provision_trust_locked(
                    db,
                    initial_context.context_hash,
                    initial_context.authority_certificate_hash,
                    verification_trust,
                )
            for schedule_hash, schedule_blob in db.execute(
                "SELECT schedule_hash, schedule_blob FROM risk_schedules"
            ).fetchall():
                try:
                    attempt_count = len(json.loads(bytes(schedule_blob))["allocations"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("registered risk schedule has invalid canonical bytes") from exc
                if attempt_count <= 0:
                    raise ValueError("registered risk schedule has no finite allocations")
                db.execute(
                    "INSERT OR IGNORE INTO risk_schedule_usage VALUES(?,?)",
                    (str(schedule_hash), attempt_count),
                )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30, isolation_level=None)

    @staticmethod
    def _append_event_locked(
        db: sqlite3.Connection,
        *,
        event_type: str,
        context_hash: str,
        object_hash: str,
        details: object,
    ) -> str:
        """Append one idempotent, hash-linked protocol event in the caller transaction."""
        if not event_type or not event_type.isascii():
            raise ValueError("event_type must be non-empty ASCII")
        _require_digest(context_hash, "event context_hash")
        _require_digest(object_hash, "event object_hash")
        detail_hash = digest(details)
        prior = db.execute(
            "SELECT context_hash, detail_hash, event_hash FROM protocol_events "
            "WHERE event_type=? AND object_hash=?",
            (event_type, object_hash),
        ).fetchone()
        if prior is not None:
            if (str(prior[0]), str(prior[1])) != (context_hash, detail_hash):
                raise ValueError("protocol event identity is bound to different content")
            return str(prior[2])
        predecessor = db.execute(
            "SELECT event_hash FROM protocol_events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        previous_event_hash = ZERO_HASH if predecessor is None else str(predecessor[0])
        event_hash = digest(
            {
                "domain": "fedmerit-protocol-event-v1",
                "previous_event_hash": previous_event_hash,
                "event_type": event_type,
                "context_hash": context_hash,
                "object_hash": object_hash,
                "detail_hash": detail_hash,
            }
        )
        db.execute(
            "INSERT INTO protocol_events "
            "(event_type,context_hash,object_hash,detail_hash,previous_event_hash,event_hash) "
            "VALUES(?,?,?,?,?,?)",
            (
                event_type,
                context_hash,
                object_hash,
                detail_hash,
                previous_event_hash,
                event_hash,
            ),
        )
        return event_hash

    @property
    def protocol_events(self) -> tuple[dict[str, object], ...]:
        """Return the canonical event chain for audit and recovery checks."""
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT sequence,event_type,context_hash,object_hash,detail_hash,"
                "previous_event_hash,event_hash FROM protocol_events ORDER BY sequence"
            ).fetchall()
        return tuple(
            {
                "sequence": int(row[0]),
                "event_type": str(row[1]),
                "context_hash": str(row[2]),
                "object_hash": str(row[3]),
                "detail_hash": str(row[4]),
                "previous_event_hash": str(row[5]),
                "event_hash": str(row[6]),
            }
            for row in rows
        )

    def protocol_event_chain_valid(self) -> bool:
        """Verify sequence continuity and every stored event-chain digest."""
        previous = ZERO_HASH
        for expected_sequence, event in enumerate(self.protocol_events, start=1):
            if (
                event["sequence"] != expected_sequence
                or event["previous_event_hash"] != previous
            ):
                return False
            expected_hash = digest(
                {
                    "domain": "fedmerit-protocol-event-v1",
                    "previous_event_hash": previous,
                    "event_type": event["event_type"],
                    "context_hash": event["context_hash"],
                    "object_hash": event["object_hash"],
                    "detail_hash": event["detail_hash"],
                }
            )
            if event["event_hash"] != expected_hash:
                return False
            previous = expected_hash
        return True

    def reserve_beacon_successor(
        self,
        beacon_public_key_hash: str,
        beacon_id: str,
        round_number: int,
        parent_round_hash: str,
        *,
        fixation_hash: str,
    ) -> None:
        """Burn one beacon successor for one fixation at the authoritative head.

        Risk ledgers may be replicated or recreated, so successor exclusivity
        cannot live only in a local ledger database. A failed downstream release
        deliberately leaves this reservation spent.
        """
        if not isinstance(beacon_id, str) or not beacon_id.strip():
            raise ValueError("beacon_id must be a non-empty string")
        for name, value in (
            ("beacon_public_key_hash", beacon_public_key_hash),
            ("parent_round_hash", parent_round_hash),
            ("fixation_hash", fixation_hash),
        ):
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
            try:
                bytes.fromhex(value)
            except ValueError as exc:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest") from exc
            if value != value.lower():
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if (
            isinstance(round_number, bool)
            or not isinstance(round_number, int)
            or not 0 <= round_number <= UINT32_MAX
        ):
            raise ValueError("round_number must be uint32")
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            head = db.execute(
                "SELECT beacon_id, round_number, round_hash FROM beacon_heads "
                "WHERE beacon_public_key_hash=?",
                (beacon_public_key_hash,),
            ).fetchone()
            if (
                head is None
                or str(head[0]) != beacon_id
                or int(head[1]) + 1 != round_number
                or str(head[2]) != parent_round_hash
            ):
                db.rollback()
                raise ValueError(
                    "fixation must reserve the immediate successor of the "
                    "authoritative beacon head"
                )
            row = db.execute(
                "SELECT parent_round_hash, fixation_hash "
                "FROM beacon_successor_reservations "
                "WHERE beacon_public_key_hash=? AND round_number=?",
                (beacon_public_key_hash, round_number),
            ).fetchone()
            expected = (parent_round_hash, fixation_hash)
            if row is None:
                try:
                    db.execute(
                        "INSERT INTO beacon_successor_reservations VALUES(?,?,?,?)",
                        (
                            beacon_public_key_hash,
                            round_number,
                            parent_round_hash,
                            fixation_hash,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    db.rollback()
                    raise ValueError(
                        "beacon successor is already reserved by another fixation"
                    ) from exc
            elif (str(row[0]), str(row[1])) != expected:
                db.rollback()
                raise ValueError(
                    "beacon successor is already reserved by another fixation"
                )
            context_hash = str(
                db.execute("SELECT context_hash FROM context_head WHERE id=1").fetchone()[0]
            )
            self._append_event_locked(
                db,
                event_type="beacon-successor-reserved",
                context_hash=context_hash,
                object_hash=fixation_hash,
                details={
                    "beacon_public_key_hash": beacon_public_key_hash,
                    "beacon_id": beacon_id,
                    "round_number": round_number,
                    "parent_round_hash": parent_round_hash,
                },
            )
            db.commit()

    def observe_beacon_head(
        self,
        signed_beacon_head: SignedBeaconRound,
        *,
        beacon_public_key: Ed25519PublicKey,
        signed_frame: SignedSamplingFrame,
        frame_public_key: Ed25519PublicKey,
    ) -> None:
        """Advance the canonical watcher state shared by every risk ledger."""
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
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute(
                "SELECT beacon_id, round_number, round_hash FROM beacon_heads "
                "WHERE beacon_public_key_hash=?",
                (beacon_key_hash,),
            ).fetchone()
            if prior is None:
                valid = (
                    beacon_head.round_number == frame.beacon_checkpoint_round
                    and beacon_head.round_hash == frame.beacon_checkpoint_hash
                )
            else:
                valid = (
                    str(prior[0]) == beacon_head.beacon_id
                    and int(prior[1]) == beacon_head.round_number
                    and str(prior[2]) == beacon_head.round_hash
                ) or (
                    str(prior[0]) == beacon_head.beacon_id
                    and beacon_head.round_number == int(prior[1]) + 1
                    and beacon_head.previous_round_hash == str(prior[2])
                )
            if not valid:
                db.rollback()
                raise ValueError(
                    "authoritative beacon head must begin at the signed checkpoint "
                    "and advance one hash-linked round"
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
            context_hash = str(
                db.execute("SELECT context_hash FROM context_head WHERE id=1").fetchone()[0]
            )
            self._append_event_locked(
                db,
                event_type="beacon-head-observed",
                context_hash=context_hash,
                object_hash=digest(
                    {
                        "context_hash": context_hash,
                        "round_hash": beacon_head.round_hash,
                    }
                ),
                details={
                    "beacon_public_key_hash": beacon_key_hash,
                    "beacon_id": beacon_head.beacon_id,
                    "round_number": beacon_head.round_number,
                    "round_hash": beacon_head.round_hash,
                },
            )
            db.commit()

    @property
    def head(self) -> str:
        with closing(self._connect()) as db:
            return str(
                db.execute("SELECT head FROM audit_state WHERE id=1").fetchone()[0]
            )

    @property
    def installed_model_hash(self) -> str:
        with closing(self._connect()) as db:
            return str(
                db.execute(
                    "SELECT installed_model_hash FROM audit_state WHERE id=1"
                ).fetchone()[0]
            )

    @property
    def installed_model_version(self) -> int:
        with closing(self._connect()) as db:
            return int(
                db.execute(
                    "SELECT installed_model_version FROM audit_state WHERE id=1"
                ).fetchone()[0]
            )

    @property
    def serving_model_snapshot(self) -> tuple[str, int, bytes | None]:
        """Return the model atomically installed with the authenticated head."""
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT model_hash, model_version, artifact_blob "
                "FROM serving_model WHERE id=1"
            ).fetchone()
        if row is None:
            raise RuntimeError("serving model is not initialized")
        blob = None if row[2] is None else bytes(row[2])
        return str(row[0]), int(row[1]), blob

    @staticmethod
    def _provision_trust_locked(
        db: sqlite3.Connection,
        context_hash: str,
        authority_certificate_hash: str,
        trust: VerificationTrust,
    ) -> None:
        if not isinstance(trust, VerificationTrust):
            raise TypeError("verification_trust must be a VerificationTrust")
        expected_certificate = _require_digest(
            authority_certificate_hash, "authority_certificate_hash"
        )
        if trust.authority_certificate_hash != expected_certificate:
            raise ValueError(
                "authority certificate does not bind the supplied verification roots"
            )
        trust_blob = canonical_bytes(trust)
        existing = db.execute(
            "SELECT trust_blob FROM verification_trust WHERE context_hash=?",
            (context_hash,),
        ).fetchone()
        if existing is not None and bytes(existing[0]) != trust_blob:
            raise ValueError("context is already bound to different verification roots")
        db.execute(
            "INSERT OR IGNORE INTO verification_trust VALUES(?,?)",
            (context_hash, trust_blob),
        )

    def provision_verification_trust(
        self,
        trust: VerificationTrust,
        *,
        context_hash: str | None = None,
    ) -> None:
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            selected = context_hash
            if selected is None:
                selected = str(
                    db.execute(
                        "SELECT context_hash FROM context_head WHERE id=1"
                    ).fetchone()[0]
                )
            live = db.execute(
                "SELECT context_hash, authority_certificate_hash "
                "FROM context_head WHERE id=1"
            ).fetchone()
            if live is None or selected != str(live[0]):
                db.rollback()
                raise ValueError("verification roots may only provision the live context")
            if db.execute(
                "SELECT 1 FROM registered_contexts WHERE context_hash=?", (selected,)
            ).fetchone() is None:
                db.rollback()
                raise ValueError("verification roots require a registered context")
            self._provision_trust_locked(db, selected, str(live[1]), trust)
            db.commit()

    def require_verification_trust(
        self,
        context_hash: str,
        authority_certificate_hash: str,
        trust: VerificationTrust,
    ) -> None:
        if trust.authority_certificate_hash != authority_certificate_hash:
            raise ValueError("verification roots do not open the authority certificate")
        expected = canonical_bytes(trust)
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT trust_blob FROM verification_trust WHERE context_hash=?",
                (context_hash,),
            ).fetchone()
        if row is None or bytes(row[0]) != expected:
            raise ValueError("verification keys are not authorized by the context")

    @property
    def context_head(self) -> tuple[str, str]:
        """Return the live context and authority certificate digests."""
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT context_hash, authority_certificate_hash FROM context_head "
                "WHERE id=1"
            ).fetchone()
        if row is None:
            raise RuntimeError("context head is not initialized")
        return str(row[0]), str(row[1])

    def evaluation_policy_registered(self, policy: EvaluationPolicy) -> bool:
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT policy_blob FROM evaluation_policies WHERE policy_hash=?",
                (policy.policy_hash,),
            ).fetchone()
        return row is not None and bytes(row[0]) == canonical_bytes(policy)

    def provision_lineage_risk_budget(self, lifetime_delta: float) -> None:
        """Freeze one risk envelope that survives context handovers.

        The envelope belongs to the invariant twin identity rather than a domain
        context.  Context-specific schedules may rotate after handover, but their
        declared lifetime budgets must fit jointly inside this envelope.
        """

        if isinstance(lifetime_delta, bool) or not isinstance(
            lifetime_delta, (int, float)
        ):
            raise ValueError("lineage lifetime budget must be a binary64 scalar")
        lifetime_delta = float(lifetime_delta)
        if not math.isfinite(lifetime_delta) or not 0 < lifetime_delta < 1:
            raise ValueError("lineage lifetime budget must lie in (0,1)")
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            twin_id = str(
                db.execute("SELECT twin_id FROM context_head WHERE id=1").fetchone()[0]
            )
            context_hash = str(
                db.execute("SELECT context_hash FROM context_head WHERE id=1").fetchone()[0]
            )
            head = str(db.execute("SELECT head FROM audit_state WHERE id=1").fetchone()[0])
            profile = self._evaluation_policy.security_profile
            profile_blob = canonical_bytes(profile)
            prior = db.execute(
                "SELECT twin_id, anchor_receipt_hash, lifetime_delta "
                "FROM lineage_risk_budget WHERE id=1"
            ).fetchone()
            expected = (twin_id, head, lifetime_delta)
            if prior is not None:
                if (str(prior[0]), str(prior[1]), float(prior[2])) != expected:
                    db.rollback()
                    raise ValueError("lineage risk budget is already frozen")
                caps = db.execute(
                    "SELECT twin_id, profile_blob, max_attempts "
                    "FROM lineage_security_caps WHERE id=1"
                ).fetchone()
                expected_caps = (twin_id, profile_blob, profile.max_attempts)
                if caps is None:
                    db.execute(
                        "INSERT INTO lineage_security_caps VALUES(1,?,?,?)",
                        expected_caps,
                    )
                elif (str(caps[0]), bytes(caps[1]), int(caps[2])) != expected_caps:
                    db.rollback()
                    raise ValueError("lineage security profile is already frozen")
                self._append_event_locked(
                    db,
                    event_type="lineage-budget-provisioned",
                    context_hash=context_hash,
                    object_hash=digest({"twin_id": twin_id, "anchor": head}),
                    details={"lifetime_delta": lifetime_delta, "profile": profile},
                )
                db.commit()
                return
            if db.execute("SELECT 1 FROM risk_schedules LIMIT 1").fetchone() is not None:
                db.rollback()
                raise ValueError("lineage risk budget must precede every schedule")
            db.execute(
                "INSERT INTO lineage_risk_budget VALUES(1,?,?,?)", expected
            )
            db.execute(
                "INSERT INTO lineage_security_caps VALUES(1,?,?,?)",
                (twin_id, profile_blob, profile.max_attempts),
            )
            self._append_event_locked(
                db,
                event_type="lineage-budget-provisioned",
                context_hash=context_hash,
                object_hash=digest({"twin_id": twin_id, "anchor": head}),
                details={"lifetime_delta": lifetime_delta, "profile": profile},
            )
            db.commit()

    @property
    def lineage_risk_budget(self) -> tuple[str, str, float] | None:
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT twin_id, anchor_receipt_hash, lifetime_delta "
                "FROM lineage_risk_budget WHERE id=1"
            ).fetchone()
        if row is None:
            return None
        return str(row[0]), str(row[1]), float(row[2])

    @staticmethod
    def _require_risk_schedule_locked(
        db: sqlite3.Connection, schedule: RiskSchedule
    ) -> None:
        row = db.execute(
            "SELECT context_hash, anchor_receipt_hash, lifetime_delta, schedule_blob "
            "FROM risk_schedules WHERE schedule_hash=?",
            (schedule.schedule_hash,),
        ).fetchone()
        expected = (
            schedule.context_hash,
            schedule.anchor_receipt_hash,
            schedule.lifetime_delta,
            canonical_bytes(schedule),
        )
        if row is None or (
            str(row[0]), str(row[1]), float(row[2]), bytes(row[3])
        ) != expected:
            raise ValueError("risk schedule is not canonically registered")

    def register_risk_schedule(self, schedule: RiskSchedule) -> None:
        """Freeze exactly one lifetime schedule for the live context lineage."""
        if not isinstance(schedule, RiskSchedule):
            raise TypeError("schedule must be a RiskSchedule")
        blob = canonical_bytes(schedule)
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute(
                "SELECT schedule_hash, anchor_receipt_hash, lifetime_delta, "
                "schedule_blob FROM risk_schedules WHERE context_hash=?",
                (schedule.context_hash,),
            ).fetchone()
            if prior is not None:
                exact = (
                    str(prior[0]) == schedule.schedule_hash
                    and str(prior[1]) == schedule.anchor_receipt_hash
                    and float(prior[2]) == schedule.lifetime_delta
                    and bytes(prior[3]) == blob
                )
                if not exact:
                    db.rollback()
                    raise ValueError(
                        "a different lifetime schedule is already frozen for this context"
                    )
                self._append_event_locked(
                    db,
                    event_type="risk-schedule-registered",
                    context_hash=schedule.context_hash,
                    object_hash=schedule.schedule_hash,
                    details={
                        "anchor_receipt_hash": schedule.anchor_receipt_hash,
                        "lifetime_delta": schedule.lifetime_delta,
                        "attempt_count": len(schedule.allocations),
                    },
                )
                db.commit()
                return
            live_context = str(
                db.execute(
                    "SELECT context_hash FROM context_head WHERE id=1"
                ).fetchone()[0]
            )
            head = str(
                db.execute("SELECT head FROM audit_state WHERE id=1").fetchone()[0]
            )
            if schedule.context_hash != live_context:
                db.rollback()
                raise ValueError("risk schedule does not belong to the live context")
            if schedule.anchor_receipt_hash != head:
                db.rollback()
                raise ValueError(
                    "risk schedule is not anchored to the authenticated audit head"
                )
            lineage = db.execute(
                "SELECT twin_id, lifetime_delta FROM lineage_risk_budget WHERE id=1"
            ).fetchone()
            if lineage is None:
                db.rollback()
                raise ValueError(
                    "lineage risk budget must be provisioned before every schedule"
                )
            live_twin = str(
                db.execute(
                    "SELECT twin_id FROM context_head WHERE id=1"
                ).fetchone()[0]
            )
            if str(lineage[0]) != live_twin:
                db.rollback()
                raise ValueError("lineage risk budget belongs to another twin")
            caps = db.execute(
                "SELECT twin_id, max_attempts FROM lineage_security_caps WHERE id=1"
            ).fetchone()
            if caps is None or str(caps[0]) != live_twin:
                db.rollback()
                raise ValueError("lineage security profile must precede every schedule")
            attempts_used = int(
                db.execute(
                    "SELECT COALESCE(SUM(attempt_count),0) FROM risk_schedule_usage"
                ).fetchone()[0]
            )
            if attempts_used + len(schedule.allocations) > int(caps[1]):
                db.rollback()
                raise ValueError(
                    "risk schedules exceed the cross-handover lifetime attempt cap"
                )
            allocated = sum(
                (
                    Fraction.from_float(float(row[0]))
                    for row in db.execute(
                        "SELECT lifetime_delta FROM risk_schedules"
                    ).fetchall()
                ),
                Fraction(),
            )
            proposed = allocated + Fraction.from_float(schedule.lifetime_delta)
            if proposed > Fraction.from_float(float(lineage[1])):
                db.rollback()
                raise ValueError(
                    "context schedule exceeds the cross-handover lineage budget"
                )
            try:
                db.execute(
                    "INSERT INTO risk_schedules VALUES(?,?,?,?,?)",
                    (
                        schedule.schedule_hash,
                        schedule.context_hash,
                        schedule.anchor_receipt_hash,
                        schedule.lifetime_delta,
                        blob,
                    ),
                )
                db.execute(
                    "INSERT INTO risk_schedule_usage VALUES(?,?)",
                    (schedule.schedule_hash, len(schedule.allocations)),
                )
            except sqlite3.IntegrityError as exc:
                db.rollback()
                raise ValueError("conflicting canonical risk schedule") from exc
            self._append_event_locked(
                db,
                event_type="risk-schedule-registered",
                context_hash=schedule.context_hash,
                object_hash=schedule.schedule_hash,
                details={
                    "anchor_receipt_hash": schedule.anchor_receipt_hash,
                    "lifetime_delta": schedule.lifetime_delta,
                    "attempt_count": len(schedule.allocations),
                },
            )
            db.commit()

    def require_risk_schedule(self, schedule: RiskSchedule) -> None:
        """Require byte-exact membership in the canonical schedule registry."""
        with closing(self._connect()) as db:
            self._require_risk_schedule_locked(db, schedule)

    @staticmethod
    def _require_risk_allocation_locked(
        db: sqlite3.Connection,
        schedule: RiskSchedule,
        allocation_index: int,
        fixation_hash: str,
    ) -> None:
        row = db.execute(
            "SELECT context_hash, fixation_hash FROM spent_risk_allocations "
            "WHERE schedule_hash=? AND allocation_index=?",
            (schedule.schedule_hash, allocation_index),
        ).fetchone()
        if row != (schedule.context_hash, fixation_hash):
            raise ValueError("risk allocation is not canonically spent for this fixation")

    def reserve_risk_allocation(
        self,
        schedule: RiskSchedule,
        allocation_index: int,
        *,
        fixation_hash: str,
    ) -> None:
        """Durably spend one canonical schedule entry before probe retirement."""
        schedule.allocation(allocation_index)
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            self._require_risk_schedule_locked(db, schedule)
            prior = db.execute(
                "SELECT context_hash, fixation_hash FROM spent_risk_allocations "
                "WHERE schedule_hash=? AND allocation_index=?",
                (schedule.schedule_hash, allocation_index),
            ).fetchone()
            owner = (schedule.context_hash, fixation_hash)
            if prior is not None:
                if tuple(prior) != owner:
                    db.rollback()
                    raise ValueError(
                        "risk allocation has already been spent by another fixation"
                    )
                self._append_event_locked(
                    db,
                    event_type="risk-allocation-spent",
                    context_hash=schedule.context_hash,
                    object_hash=fixation_hash,
                    details={
                        "schedule_hash": schedule.schedule_hash,
                        "allocation_index": allocation_index,
                    },
                )
                db.commit()
                return
            db.execute(
                "INSERT INTO spent_risk_allocations VALUES(?,?,?,?)",
                (
                    schedule.schedule_hash,
                    allocation_index,
                    schedule.context_hash,
                    fixation_hash,
                ),
            )
            self._append_event_locked(
                db,
                event_type="risk-allocation-spent",
                context_hash=schedule.context_hash,
                object_hash=fixation_hash,
                details={
                    "schedule_hash": schedule.schedule_hash,
                    "allocation_index": allocation_index,
                },
            )
            db.commit()

    def handover(
        self,
        *,
        state_context: StateContext,
        evaluation_policy: EvaluationPolicy,
        verification_trust: VerificationTrust | None = None,
        authorization: HandoverAuthorization | None = None,
        authorizer: CertificateAuthority | None = None,
    ) -> None:
        """Atomically replace the live authority head after old-roster approval."""
        if state_context.policy_hash != evaluation_policy.policy_hash:
            raise ValueError("successor context does not authorize its policy")
        if authorization is not None and authorizer is not None:
            raise ValueError("supply authorization or authorizer, not both")
        if authorization is None and authorizer is not None:
            authorization = authorizer.authorize_handover(
                previous_context_hash=self.context_head[0],
                successor_context=state_context,
                previous_receipt_hash=self.head,
                installed_model_hash=self.installed_model_hash,
                installed_model_version=self.installed_model_version,
            )
        policy_blob = canonical_bytes(evaluation_policy)
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            live = db.execute(
                "SELECT context_hash, authority_certificate_hash, twin_id, state_version "
                "FROM context_head WHERE id=1"
            ).fetchone()
            if live is None:
                db.rollback()
                raise RuntimeError("context head is not initialized")
            live_context_hash, live_authority, live_twin, live_state_version = live
            if state_context.twin_id != live_twin:
                db.rollback()
                raise ValueError("successor context must preserve the twin identity")
            if state_context.state_version != int(live_state_version) + 1:
                db.rollback()
                raise ValueError(
                    "handover requires the immediate successor state version"
                )
            audit_state = db.execute(
                "SELECT head, installed_model_hash, installed_model_version "
                "FROM audit_state WHERE id=1"
            ).fetchone()
            previous_receipt_hash = str(audit_state[0])
            installed_model_hash = str(audit_state[1])
            installed_model_version = int(audit_state[2])
            if state_context.model_version != installed_model_version:
                db.rollback()
                raise ValueError(
                    "handover model version must equal the installed model version"
                )
            old_trust_row = db.execute(
                "SELECT trust_blob FROM verification_trust WHERE context_hash=?",
                (live_context_hash,),
            ).fetchone()
            if old_trust_row is not None:
                if authorization is None:
                    db.rollback()
                    raise ValueError(
                        "live context requires an old-roster handover authorization"
                    )
                expected_authorization = (
                    str(live_context_hash),
                    state_context,
                    previous_receipt_hash,
                    installed_model_hash,
                    installed_model_version,
                )
                actual_authorization = (
                    authorization.previous_context_hash,
                    authorization.successor_context,
                    authorization.previous_receipt_hash,
                    authorization.installed_model_hash,
                    authorization.installed_model_version,
                )
                old_trust = _verification_trust_from_blob(bytes(old_trust_row[0]))
                if (
                    actual_authorization != expected_authorization
                    or not verify_handover_authorization(authorization, old_trust)
                ):
                    db.rollback()
                    raise ValueError(
                        "handover authorization does not bind the live transition"
                    )
            elif authorization is not None:
                db.rollback()
                raise ValueError(
                    "handover authorization cannot be checked without prior trust roots"
                )
            lineage_caps = db.execute(
                "SELECT twin_id, profile_blob FROM lineage_security_caps WHERE id=1"
            ).fetchone()
            if lineage_caps is not None and (
                str(lineage_caps[0]) != state_context.twin_id
                or bytes(lineage_caps[1])
                != canonical_bytes(evaluation_policy.security_profile)
            ):
                db.rollback()
                raise ValueError(
                    "handover must preserve the lineage security profile"
                )
            registered = db.execute(
                "SELECT policy_blob FROM evaluation_policies WHERE policy_hash=?",
                (evaluation_policy.policy_hash,),
            ).fetchone()
            if registered is not None and bytes(registered[0]) != policy_blob:
                db.rollback()
                raise ValueError("evaluation policy hash is bound to different content")
            db.execute(
                "INSERT OR IGNORE INTO evaluation_policies VALUES(?,?)",
                (evaluation_policy.policy_hash, policy_blob),
            )
            context_blob = canonical_bytes(state_context)
            existing_context = db.execute(
                "SELECT context_blob FROM registered_contexts WHERE context_hash=?",
                (state_context.context_hash,),
            ).fetchone()
            if existing_context is not None and bytes(existing_context[0]) != context_blob:
                db.rollback()
                raise ValueError("context hash is bound to different content")
            db.execute(
                "INSERT OR IGNORE INTO registered_contexts VALUES(?,?)",
                (state_context.context_hash, context_blob),
            )
            if verification_trust is None:
                inherited = db.execute(
                    "SELECT trust_blob FROM verification_trust WHERE context_hash=?",
                    (live_context_hash,),
                ).fetchone()
                if inherited is not None:
                    if _digest_bytes(bytes(inherited[0])) != state_context.authority_certificate_hash:
                        db.rollback()
                        raise ValueError(
                            "successor authority certificate does not bind inherited roots"
                        )
                    existing_trust = db.execute(
                        "SELECT trust_blob FROM verification_trust WHERE context_hash=?",
                        (state_context.context_hash,),
                    ).fetchone()
                    if existing_trust is not None and bytes(existing_trust[0]) != bytes(
                        inherited[0]
                    ):
                        db.rollback()
                        raise ValueError(
                            "successor context is bound to different verification roots"
                        )
                    db.execute(
                        "INSERT OR IGNORE INTO verification_trust VALUES(?,?)",
                        (state_context.context_hash, bytes(inherited[0])),
                    )
            else:
                self._provision_trust_locked(
                    db,
                    state_context.context_hash,
                    state_context.authority_certificate_hash,
                    verification_trust,
                )
            db.execute(
                "UPDATE context_head SET context_hash=?, authority_certificate_hash=?, "
                "policy_hash=?, twin_id=?, state_version=?, model_version=? "
                "WHERE id=1",
                (
                    state_context.context_hash,
                    state_context.authority_certificate_hash,
                    evaluation_policy.policy_hash,
                    state_context.twin_id,
                    state_context.state_version,
                    state_context.model_version,
                ),
            )
            self._append_event_locked(
                db,
                event_type="context-handover",
                context_hash=state_context.context_hash,
                object_hash=state_context.context_hash,
                details={
                    "previous_context_hash": str(live_context_hash),
                    "twin_id": state_context.twin_id,
                    "state_version": state_context.state_version,
                    "model_version": state_context.model_version,
                    "authorization_hash": (
                        None
                        if authorization is None
                        else authorization.authorization_hash
                    ),
                },
            )
            db.commit()
        self._evaluation_policy = evaluation_policy

    def validate_candidate(
        self, candidate: Candidate, *, receipt_hash: str | None = None
    ) -> None:
        with closing(self._connect()) as db:
            head, installed = db.execute(
                "SELECT head, installed_model_hash FROM audit_state WHERE id=1"
            ).fetchone()
            live_context, live_authority, live_policy = db.execute(
                "SELECT context_hash, authority_certificate_hash, policy_hash FROM context_head "
                "WHERE id=1"
            ).fetchone()
            policy_row = db.execute(
                "SELECT policy_blob FROM evaluation_policies WHERE policy_hash=?",
                (candidate.evaluation_policy.policy_hash,),
            ).fetchone()
            if policy_row is None or bytes(policy_row[0]) != canonical_bytes(
                candidate.evaluation_policy
            ):
                raise ValueError("candidate evaluation policy is not registered")
            context_row = db.execute(
                "SELECT context_blob FROM registered_contexts WHERE context_hash=?",
                (candidate.context_hash,),
            ).fetchone()
            if context_row is None or bytes(context_row[0]) != canonical_bytes(
                candidate.state_context
            ):
                raise ValueError("candidate context is not registered")
            historical = None
            if receipt_hash is not None:
                historical = db.execute(
                    "SELECT context_hash, before_model_hash, after_model_hash, "
                    "previous_receipt_hash, fixation_hash, schedule_hash, schedule_index "
                    "FROM issued_receipts WHERE receipt_hash=?",
                    (receipt_hash,),
                ).fetchone()
                expected_historical = (
                    candidate.context_hash,
                    candidate.before_model_hash,
                    candidate.after_model_hash,
                    candidate.previous_receipt_hash,
                    candidate.fixation_hash,
                    candidate.risk_schedule_hash,
                    candidate.risk_schedule_index,
                )
                if historical is not None:
                    if historical == expected_historical:
                        return
                    raise ValueError(
                        "receipt hash is bound to another historical issuance"
                    )
                historical = db.execute(
                    "SELECT context_hash, before_model_hash, after_model_hash, "
                    "previous_receipt_hash, fixation_hash, schedule_hash, schedule_index "
                    "FROM receipts WHERE receipt_hash=?",
                    (receipt_hash,),
                ).fetchone()
                if historical is not None:
                    if historical == expected_historical:
                        return
                    raise ValueError("receipt hash is bound to another historical append")
            candidate_authority = candidate.state_context.authority_certificate_hash
            if (
                live_context != candidate.context_hash
                or live_authority != candidate_authority
                or live_policy != candidate.evaluation_policy.policy_hash
            ):
                raise ValueError("candidate does not match the live context head")
            installed_model_version = int(
                db.execute(
                    "SELECT installed_model_version FROM audit_state WHERE id=1"
                ).fetchone()[0]
            )
            if candidate.state_context.model_version != installed_model_version:
                raise ValueError("candidate model version differs from installed state")
            if (
                candidate.previous_receipt_hash == head
                and candidate.before_model_hash == installed
            ):
                return
        if candidate.previous_receipt_hash != head:
            raise ValueError("candidate does not extend the authenticated audit head")
        raise ValueError("candidate before-model differs from installed chain state")

    def record_issued_receipt(
        self,
        receipt: Receipt,
        *,
        candidate: Candidate,
        schedule: RiskSchedule,
        verification_trust: VerificationTrust,
    ) -> None:
        core = receipt.core
        if (
            core.context_hash != candidate.context_hash
            or core.before_model_hash != candidate.before_model_hash
            or core.after_model_hash != candidate.after_model_hash
            or core.previous_receipt_hash != candidate.previous_receipt_hash
            or core.fixation_hash != candidate.fixation_hash
            or core.risk_schedule_hash != schedule.schedule_hash
            or schedule.context_hash != candidate.context_hash
            or schedule.allocation(candidate.risk_schedule_index) != candidate.risk
        ):
            raise ValueError("issued receipt is not bound to the candidate and schedule")
        expected = (
            candidate.context_hash,
            candidate.before_model_hash,
            candidate.after_model_hash,
            candidate.previous_receipt_hash,
            candidate.fixation_hash,
            candidate.risk_schedule_hash,
            candidate.risk_schedule_index,
        )
        if (
            verification_trust.authority_certificate_hash
            != candidate.state_context.authority_certificate_hash
        ):
            raise ValueError("verification roots do not open the authority certificate")
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            self._require_risk_schedule_locked(db, schedule)
            self._require_risk_allocation_locked(
                db,
                schedule,
                candidate.risk_schedule_index,
                candidate.fixation_hash,
            )
            trust_row = db.execute(
                "SELECT trust_blob FROM verification_trust WHERE context_hash=?",
                (candidate.context_hash,),
            ).fetchone()
            if trust_row is None or bytes(trust_row[0]) != canonical_bytes(
                verification_trust
            ):
                db.rollback()
                raise ValueError("verification keys are not authorized by the context")
            live = db.execute(
                "SELECT context_hash, authority_certificate_hash, policy_hash "
                "FROM context_head WHERE id=1"
            ).fetchone()
            if live != (
                candidate.context_hash,
                candidate.state_context.authority_certificate_hash,
                candidate.evaluation_policy.policy_hash,
            ):
                db.rollback()
                raise ValueError("candidate lost the live context before issuance")
            head, installed, installed_version = db.execute(
                "SELECT head, installed_model_hash, installed_model_version "
                "FROM audit_state WHERE id=1"
            ).fetchone()
            if (
                candidate.previous_receipt_hash != head
                or candidate.before_model_hash != installed
                or candidate.state_context.model_version != int(installed_version)
            ):
                db.rollback()
                raise ValueError("candidate lost the installed state before issuance")
            existing = db.execute(
                "SELECT context_hash, before_model_hash, after_model_hash, "
                "previous_receipt_hash, fixation_hash, schedule_hash, schedule_index "
                "FROM issued_receipts WHERE receipt_hash=?",
                (receipt.receipt_hash,),
            ).fetchone()
            if existing is not None:
                if existing != expected:
                    db.rollback()
                    raise ValueError("receipt hash is bound to another issuance")
                self._append_event_locked(
                    db,
                    event_type="receipt-issued",
                    context_hash=candidate.context_hash,
                    object_hash=receipt.receipt_hash,
                    details={
                        "fixation_hash": candidate.fixation_hash,
                        "schedule_hash": schedule.schedule_hash,
                        "schedule_index": candidate.risk_schedule_index,
                    },
                )
                db.commit()
                return
            try:
                db.execute(
                    "INSERT INTO issued_receipts VALUES(?,?,?,?,?,?,?,?)",
                    (receipt.receipt_hash, *expected),
                )
            except sqlite3.IntegrityError as exc:
                db.rollback()
                raise ValueError("conflicting issued attempt or risk allocation") from exc
            self._append_event_locked(
                db,
                event_type="receipt-issued",
                context_hash=candidate.context_hash,
                object_hash=receipt.receipt_hash,
                details={
                    "fixation_hash": candidate.fixation_hash,
                    "schedule_hash": schedule.schedule_hash,
                    "schedule_index": candidate.risk_schedule_index,
                },
            )
            db.commit()

    @staticmethod
    def _reserve_source_manifests_locked(
        db: sqlite3.Connection,
        source_manifest_hashes: tuple[str, ...],
        *,
        context_hash: str,
        fixation_hash: str,
        probe_id_hash: str,
    ) -> None:
        manifests = tuple(source_manifest_hashes)
        if not manifests or len(set(manifests)) != len(manifests):
            raise ValueError("released source manifests must be non-empty and unique")
        owner = (context_hash, fixation_hash, probe_id_hash)
        for manifest in manifests:
            prior = db.execute(
                "SELECT context_hash, fixation_hash, probe_id_hash "
                "FROM retired_source_manifests WHERE source_manifest_hash=?",
                (manifest,),
            ).fetchone()
            if prior is not None and tuple(prior) != owner:
                raise ValueError(
                    "source manifest has already been retired by another release"
                )
        db.executemany(
            "INSERT OR IGNORE INTO retired_source_manifests "
            "(source_manifest_hash,context_hash,fixation_hash,probe_id_hash) "
            "VALUES(?,?,?,?)",
            ((manifest, *owner) for manifest in manifests),
        )

    def reserve_source_manifests(
        self,
        source_manifest_hashes: tuple[str, ...],
        *,
        context_hash: str,
        fixation_hash: str,
        probe_id_hash: str,
    ) -> None:
        """Reserve released source manifests before local probe retirement.

        This is the lineage-wide fence.  It deliberately commits independently
        of a probe-store transaction so a release followed by a handover cannot
        make the same source evidence available to a successor catalog.
        """
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                self._reserve_source_manifests_locked(
                    db,
                    source_manifest_hashes,
                    context_hash=context_hash,
                    fixation_hash=fixation_hash,
                    probe_id_hash=probe_id_hash,
                )
            except Exception:
                db.rollback()
                raise
            self._append_event_locked(
                db,
                event_type="source-manifests-retired",
                context_hash=context_hash,
                object_hash=digest(
                    {"fixation_hash": fixation_hash, "probe_id_hash": probe_id_hash}
                ),
                details={
                    "fixation_hash": fixation_hash,
                    "probe_id_hash": probe_id_hash,
                    "source_manifest_hashes": tuple(source_manifest_hashes),
                },
            )
            db.commit()

    def _append(
        self,
        receipt: Receipt,
        *,
        candidate: Candidate,
        schedule: RiskSchedule,
        source_manifest_hashes: tuple[str, ...],
        probe_id_hash: str,
    ) -> None:
        core = receipt.core
        if core.fixation_hash != candidate.fixation_hash:
            raise ValueError(
                "receipt does not belong to the disclosed candidate fixation"
            )
        if (
            schedule.context_hash != core.context_hash
            or schedule.schedule_hash != core.risk_schedule_hash
        ):
            raise ValueError("receipt risk schedule is not registered for audit")
        if schedule.allocation(candidate.risk_schedule_index) != candidate.risk:
            raise ValueError("receipt risk allocation differs from the finite schedule")
        if (
            core.context_hash != candidate.context_hash
            or core.before_model_hash != candidate.before_model_hash
            or core.after_model_hash != candidate.after_model_hash
            or core.previous_receipt_hash != candidate.previous_receipt_hash
        ):
            raise ValueError("receipt state fields differ from the candidate")
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            self._require_risk_schedule_locked(db, schedule)
            self._require_risk_allocation_locked(
                db,
                schedule,
                candidate.risk_schedule_index,
                candidate.fixation_hash,
            )
            issued = db.execute(
                "SELECT context_hash, before_model_hash, after_model_hash, "
                "previous_receipt_hash, fixation_hash, schedule_hash, schedule_index "
                "FROM issued_receipts WHERE receipt_hash=?",
                (receipt.receipt_hash,),
            ).fetchone()
            expected_issued = (
                core.context_hash,
                core.before_model_hash,
                core.after_model_hash,
                core.previous_receipt_hash,
                core.fixation_hash,
                schedule.schedule_hash,
                candidate.risk_schedule_index,
            )
            if issued != expected_issued:
                db.rollback()
                raise ValueError("receipt is not a registered issuance")
            existing = db.execute(
                "SELECT context_hash, before_model_hash, after_model_hash, "
                "previous_receipt_hash, fixation_hash, schedule_hash, schedule_index "
                "FROM receipts WHERE receipt_hash=?",
                (receipt.receipt_hash,),
            ).fetchone()
            expected_existing = (
                core.context_hash,
                core.before_model_hash,
                core.after_model_hash,
                core.previous_receipt_hash,
                core.fixation_hash,
                schedule.schedule_hash,
                candidate.risk_schedule_index,
            )
            if existing is not None:
                if existing != expected_existing:
                    db.rollback()
                    raise ValueError(
                        "receipt hash is bound to another historical append"
                    )
                installed_state = db.execute(
                    "SELECT installed_model_hash, installed_model_version "
                    "FROM audit_state WHERE id=1"
                ).fetchone()
                self._append_event_locked(
                    db,
                    event_type="receipt-appended",
                    context_hash=core.context_hash,
                    object_hash=receipt.receipt_hash,
                    details={
                        "decision": core.decision,
                        "previous_receipt_hash": core.previous_receipt_hash,
                        "installed_model_hash": str(installed_state[0]),
                        "installed_model_version": int(installed_state[1]),
                        "successor_context_hash": (
                            candidate.state_context.model_successor().context_hash
                            if core.decision == "commit"
                            else None
                        ),
                    },
                )
                db.commit()
                return
            live_context, live_authority, live_policy = db.execute(
                "SELECT context_hash, authority_certificate_hash, policy_hash FROM context_head "
                "WHERE id=1"
            ).fetchone()
            candidate_authority = candidate.state_context.authority_certificate_hash
            if (
                live_context != candidate.context_hash
                or live_authority != candidate_authority
                or live_policy != candidate.evaluation_policy.policy_hash
            ):
                db.rollback()
                raise ValueError("CheckAppend context head mismatch")
            head, installed, installed_version = db.execute(
                "SELECT head, installed_model_hash, installed_model_version "
                "FROM audit_state WHERE id=1"
            ).fetchone()
            if (
                core.previous_receipt_hash != head
                or core.before_model_hash != installed
            ):
                db.rollback()
                raise ValueError(
                    "receipt does not extend the current authenticated state"
                )
            if candidate.state_context.model_version != int(installed_version):
                db.rollback()
                raise ValueError("receipt model version differs from installed state")
            serving = db.execute(
                "SELECT model_hash, model_version, artifact_blob "
                "FROM serving_model WHERE id=1"
            ).fetchone()
            before_blob = candidate.before_model.artifact_bytes
            if (
                serving is None
                or str(serving[0]) != installed
                or int(serving[1]) != int(installed_version)
                or serving[2] is None
                or bytes(serving[2]) != before_blob
            ):
                db.rollback()
                raise ValueError(
                    "serving artifact differs from the authenticated before-model"
                )
            successor_context = None
            inherited_trust = None
            if core.decision == "commit":
                successor_context = candidate.state_context.model_successor()
                if successor_context.model_version != int(installed_version) + 1:
                    db.rollback()
                    raise ValueError("model-successor version is not consecutive")
                inherited_trust = db.execute(
                    "SELECT trust_blob FROM verification_trust WHERE context_hash=?",
                    (candidate.context_hash,),
                ).fetchone()
                if inherited_trust is None:
                    db.rollback()
                    raise ValueError(
                        "committed model successor requires inherited verification roots"
                    )
                if (
                    _digest_bytes(bytes(inherited_trust[0]))
                    != candidate.state_context.authority_certificate_hash
                ):
                    db.rollback()
                    raise ValueError(
                        "live authority certificate does not bind inherited roots"
                    )
                successor_blob = canonical_bytes(successor_context)
                registered_successor = db.execute(
                    "SELECT context_blob FROM registered_contexts WHERE context_hash=?",
                    (successor_context.context_hash,),
                ).fetchone()
                if (
                    registered_successor is not None
                    and bytes(registered_successor[0]) != successor_blob
                ):
                    db.rollback()
                    raise ValueError("successor context hash is bound to different content")
                successor_trust = db.execute(
                    "SELECT trust_blob FROM verification_trust WHERE context_hash=?",
                    (successor_context.context_hash,),
                ).fetchone()
                if (
                    successor_trust is not None
                    and bytes(successor_trust[0]) != bytes(inherited_trust[0])
                ):
                    db.rollback()
                    raise ValueError(
                        "model successor is bound to different verification roots"
                    )
            self._reserve_source_manifests_locked(
                db,
                source_manifest_hashes,
                context_hash=core.context_hash,
                fixation_hash=core.fixation_hash,
                probe_id_hash=probe_id_hash,
            )
            try:
                db.execute(
                    "INSERT INTO receipts VALUES(?,?,?,?,?,?,?,?)",
                    (
                        receipt.receipt_hash,
                        core.context_hash,
                        core.before_model_hash,
                        core.after_model_hash,
                        core.previous_receipt_hash,
                        core.fixation_hash,
                        schedule.schedule_hash,
                        candidate.risk_schedule_index,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                db.rollback()
                raise ValueError("conflicting scope or reused risk allocation") from exc
            installed_next = (
                core.after_model_hash
                if core.decision == "commit"
                else core.before_model_hash
            )
            installed_version_next = int(installed_version) + (
                1 if core.decision == "commit" else 0
            )
            serving_blob_next = (
                candidate.after_model.artifact_bytes
                if core.decision == "commit"
                else before_blob
            )
            updated_serving = db.execute(
                "UPDATE serving_model SET model_hash=?, model_version=?, "
                "artifact_blob=? WHERE id=1 AND model_hash=? AND model_version=?",
                (
                    installed_next,
                    installed_version_next,
                    serving_blob_next,
                    installed,
                    installed_version,
                ),
            ).rowcount
            if updated_serving != 1:
                db.rollback()
                raise ValueError("serving-model compare-and-swap lost a concurrent race")
            if successor_context is not None:
                db.execute(
                    "INSERT OR IGNORE INTO registered_contexts VALUES(?,?)",
                    (
                        successor_context.context_hash,
                        canonical_bytes(successor_context),
                    ),
                )
                db.execute(
                    "INSERT OR IGNORE INTO verification_trust VALUES(?,?)",
                    (
                        successor_context.context_hash,
                        bytes(inherited_trust[0]),
                    ),
                )
                db.execute(
                    "UPDATE context_head SET context_hash=?, "
                    "authority_certificate_hash=?, policy_hash=?, twin_id=?, "
                    "state_version=?, model_version=? WHERE id=1",
                    (
                        successor_context.context_hash,
                        successor_context.authority_certificate_hash,
                        successor_context.policy_hash,
                        successor_context.twin_id,
                        successor_context.state_version,
                        successor_context.model_version,
                    ),
                )
            db.execute(
                "UPDATE audit_state SET head=?, installed_model_hash=?, "
                "installed_model_version=? WHERE id=1",
                (receipt.receipt_hash, installed_next, installed_version_next),
            )
            self._append_event_locked(
                db,
                event_type="receipt-appended",
                context_hash=core.context_hash,
                object_hash=receipt.receipt_hash,
                details={
                    "decision": core.decision,
                    "previous_receipt_hash": core.previous_receipt_hash,
                    "installed_model_hash": installed_next,
                    "installed_model_version": installed_version_next,
                    "successor_context_hash": (
                        None
                        if successor_context is None
                        else successor_context.context_hash
                    ),
                },
            )
            db.commit()

    def verify_and_append(
        self,
        receipt: Receipt,
        public_keys: Sequence[Ed25519PublicKey],
        *,
        f: int,
        release: ProbeRelease,
        candidate: Candidate,
        store_public_key: Ed25519PublicKey,
        frame_public_key: Ed25519PublicKey,
        schedule: RiskSchedule,
        risk_ledger: RiskLedger,
        roster_epoch: int = 0,
    ) -> bool:
        """Verify, install, and append at one SQLite linearization point.

        The registry is the authoritative serving store for the reference
        implementation.  Its model bytes, model version, context head, receipt
        head, and receipt row change in the same transaction.  External serving
        systems must provide an adapter with equivalent compare-and-swap and
        crash-atomic semantics rather than supplying an unauthenticated read-back.
        """
        if not verify_receipt(
            receipt,
            public_keys,
            f=f,
            release=release,
            candidate=candidate,
            store_public_key=store_public_key,
            frame_public_key=frame_public_key,
            schedule=schedule,
            risk_ledger=risk_ledger,
            audit_registry=self,
            roster_epoch=roster_epoch,
        ):
            return False
        try:
            self._append(
                receipt,
                candidate=candidate,
                schedule=schedule,
                source_manifest_hashes=tuple(
                    group.source_manifest_hash for group in release.probe.groups
                ),
                probe_id_hash=release.probe.probe_id_hash,
            )
        except (ValueError, sqlite3.Error):
            return False
        return True

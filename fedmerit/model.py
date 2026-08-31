"""Immutable protocol objects and the compact FedMERIT wire format."""

from __future__ import annotations

import hashlib
import hmac
import math
import secrets
import struct
from dataclasses import dataclass, field
from fractions import Fraction

from .canonical import canonical_bytes, digest, merkle_root


ZERO_HASH = "0" * 64
UINT32_MAX = (1 << 32) - 1
RECEIPT_CORE_BYTES = 357
RECEIPT_VERSION = 1
_DECISION_TO_BYTE = {"commit": 0x10, "reject": 0x11}
_BYTE_TO_DECISION = {value: key for key, value in _DECISION_TO_BYTE.items()}


def _required(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _sha256(name: str, value: str) -> None:
    _required(name, value)
    if len(value) != 64 or value != value.lower():
        raise ValueError(f"{name} must be a SHA-256 hexadecimal digest")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA-256 hexadecimal digest") from exc


def _uint32(name: str, value: int) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= UINT32_MAX
    ):
        raise ValueError(f"{name} must be an unsigned 32-bit integer")


def _finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite binary64")


def _binary64(name: str, value: float) -> float:
    """Normalize one public scalar to its canonical binary64 value."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a binary64 scalar")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a binary64 scalar") from exc
    _finite(name, number)
    return number


@dataclass(frozen=True)
class EvaluationPolicy:
    """Canonical evaluator contract committed by the state context."""

    policy_id: str
    loss: str = "brier"
    preprocessing: str = "artifact-affine-standardization-v1"
    decimal_precision: int = 80
    rounding: str = "round-half-even"
    sigmoid_logit_clamp: int = 80
    group_loss_quantum: str = "0.000000000001"
    group_order: str = "ascending-group-id"
    missing_value_rule: str = "reject"
    class_weights: tuple[float, float] = (1.0, 1.0)
    group_reduction: str = "class-weighted-row-mean-then-clip-[0,1]"

    def __post_init__(self) -> None:
        _required("policy_id", self.policy_id)
        required_values = {
            "loss": (self.loss, "brier"),
            "preprocessing": (
                self.preprocessing,
                "artifact-affine-standardization-v1",
            ),
            "rounding": (self.rounding, "round-half-even"),
            "group_loss_quantum": (self.group_loss_quantum, "0.000000000001"),
            "group_order": (self.group_order, "ascending-group-id"),
            "missing_value_rule": (self.missing_value_rule, "reject"),
            "group_reduction": (
                self.group_reduction,
                "class-weighted-row-mean-then-clip-[0,1]",
            ),
        }
        for name, (actual, expected) in required_values.items():
            if actual != expected:
                raise ValueError(f"unsupported {name}: expected {expected}")
        if self.decimal_precision != 80:
            raise ValueError("reference policy requires Decimal precision 80")
        if type(self.sigmoid_logit_clamp) is not int or self.sigmoid_logit_clamp != 80:
            raise ValueError("reference policy requires sigmoid logit clamp 80")
        class_weights = tuple(float(weight) for weight in self.class_weights)
        object.__setattr__(self, "class_weights", class_weights)
        if len(class_weights) != 2:
            raise ValueError("binary Brier policy needs two class weights")
        for weight in class_weights:
            _finite("class weight", float(weight))
            if weight <= 0:
                raise ValueError("class weights must be positive")

    @property
    def policy_hash(self) -> str:
        return digest(self)


@dataclass(frozen=True)
class StateContext:
    twin_id: str
    domain_id: str
    state_version: int
    sensor_schema_hash: str
    policy_hash: str
    model_version: int
    authority_certificate_hash: str

    def __post_init__(self) -> None:
        for name in ("twin_id", "domain_id"):
            _required(name, getattr(self, name))
        for name in ("sensor_schema_hash", "policy_hash", "authority_certificate_hash"):
            _sha256(name, getattr(self, name))
        if self.authority_certificate_hash == ZERO_HASH:
            raise ValueError("authority_certificate_hash cannot be the zero digest")
        _uint32("state_version", self.state_version)
        _uint32("model_version", self.model_version)

    @property
    def context_hash(self) -> str:
        return digest(self)

    def model_successor(self) -> StateContext:
        """Return the deterministic context after one committed model update."""
        if self.model_version == UINT32_MAX:
            raise ValueError("model version cannot advance beyond uint32")
        return StateContext(
            twin_id=self.twin_id,
            domain_id=self.domain_id,
            state_version=self.state_version,
            sensor_schema_hash=self.sensor_schema_hash,
            policy_hash=self.policy_hash,
            model_version=self.model_version + 1,
            authority_certificate_hash=self.authority_certificate_hash,
        )


@dataclass(frozen=True)
class LinearModelArtifact:
    """Concrete reference artifact: preprocessing plus logistic weights, bias last."""

    weights: tuple[float, ...]
    feature_mean: tuple[float, ...] = ()
    feature_scale: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        weights = tuple(float(value) for value in self.weights)
        feature_mean = tuple(float(value) for value in self.feature_mean)
        feature_scale = tuple(float(value) for value in self.feature_scale)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "feature_mean", feature_mean)
        object.__setattr__(self, "feature_scale", feature_scale)
        if len(weights) < 2:
            raise ValueError(
                "model artifact needs at least one feature weight and a bias"
            )
        width = len(weights) - 1
        if bool(feature_mean) != bool(feature_scale):
            raise ValueError(
                "feature_mean and feature_scale must both be present or both absent"
            )
        if feature_mean and (
            len(feature_mean) != width or len(feature_scale) != width
        ):
            raise ValueError("preprocessing vectors must match the model feature width")
        if any(value <= 0 for value in feature_scale):
            raise ValueError("feature scales must be positive")
        for value in weights + feature_mean + feature_scale:
            _finite("model weight", value)

    @property
    def artifact_hash(self) -> str:
        return digest(
            {
                "format": "fedmerit-linear-logit-v1",
                "weights": self.weights,
                "feature_mean": self.feature_mean,
                "feature_scale": self.feature_scale,
            }
        )


@dataclass(frozen=True)
class ProbeGroup:
    """Raw rows for one source group; rows remain paired across both models."""

    group_id: str
    source_manifest_hash: str
    features: tuple[tuple[float, ...], ...]
    labels: tuple[int, ...]

    def __post_init__(self) -> None:
        _required("group_id", self.group_id)
        _sha256("source_manifest_hash", self.source_manifest_hash)
        features = tuple(
            tuple(float(value) for value in row) for row in self.features
        )
        labels = tuple(self.labels)
        if any(isinstance(label, bool) or not isinstance(label, int) for label in labels):
            raise ValueError("reference evaluator accepts integer binary labels only")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "labels", labels)
        if not features or len(features) != len(labels):
            raise ValueError(
                "a group needs equally sized non-empty feature and label rows"
            )
        width = len(features[0])
        if width == 0 or any(len(row) != width for row in features):
            raise ValueError(
                "all feature rows in a group must have one fixed positive width"
            )
        if any(label not in (0, 1) for label in labels):
            raise ValueError("reference evaluator accepts binary labels only")
        for row in features:
            for value in row:
                _finite("feature", value)


@dataclass(frozen=True)
class RiskAllocation:
    epsilon: float
    gamma: float
    alpha: float
    group_count: int

    def __post_init__(self) -> None:
        for name in ("epsilon", "gamma", "alpha"):
            object.__setattr__(self, name, _binary64(name, getattr(self, name)))
        if self.epsilon <= 0 or self.gamma < 0 or not 0 < self.alpha < 1:
            raise ValueError(
                "risk allocation requires epsilon > 0, gamma >= 0, and alpha in (0,1)"
            )
        _uint32("group_count", self.group_count)
        if self.group_count == 0:
            raise ValueError("group_count must be uint32 and positive")


@dataclass(frozen=True)
class RiskSchedule:
    """Finite context budget anchored before the first covered proposal."""

    schedule_id: str
    context_hash: str
    anchor_receipt_hash: str
    lifetime_delta: float
    allocations: tuple[RiskAllocation, ...]

    def __post_init__(self) -> None:
        _required("schedule_id", self.schedule_id)
        _sha256("context_hash", self.context_hash)
        _sha256("anchor_receipt_hash", self.anchor_receipt_hash)
        object.__setattr__(
            self,
            "lifetime_delta",
            _binary64("lifetime_delta", self.lifetime_delta),
        )
        if not 0 < self.lifetime_delta < 1:
            raise ValueError("lifetime_delta must lie in (0,1)")
        allocations = tuple(self.allocations)
        if any(not isinstance(item, RiskAllocation) for item in allocations):
            raise TypeError("risk schedule allocations must be RiskAllocation objects")
        object.__setattr__(self, "allocations", allocations)
        if not allocations or len(allocations) > UINT32_MAX:
            raise ValueError("risk schedule must be finite and non-empty")
        # Compare the exact rational embeddings of the encoded binary64 values.
        # Decimal arithmetic would otherwise inherit the caller's ambient
        # precision and could round an over-budget schedule into acceptance.
        allocated = sum(
            (Fraction.from_float(float(item.alpha)) for item in allocations),
            Fraction(),
        )
        if allocated > Fraction.from_float(float(self.lifetime_delta)):
            raise ValueError("risk schedule allocations exceed the lifetime budget")

    @property
    def schedule_hash(self) -> str:
        return digest(self)

    def allocation(self, index: int) -> RiskAllocation:
        _uint32("risk_schedule_index", index)
        try:
            return self.allocations[index]
        except IndexError as exc:
            raise ValueError(
                "risk schedule index is outside the predeclared horizon"
            ) from exc


@dataclass(frozen=True)
class SourcePartition:
    """Complete proposal-source exclusion manifest registered by the authority."""

    context_hash: str
    contributor_root: str
    score_probe_commitment: str
    policy_hash: str
    source_manifest_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "context_hash",
            "contributor_root",
            "score_probe_commitment",
            "policy_hash",
        ):
            _sha256(name, getattr(self, name))
        manifests = tuple(self.source_manifest_hashes)
        object.__setattr__(self, "source_manifest_hashes", manifests)
        if not manifests:
            raise ValueError("source partition must contain proposal-source manifests")
        if manifests != tuple(sorted(manifests)):
            raise ValueError("source manifests must use canonical ascending order")
        if len(set(manifests)) != len(manifests):
            raise ValueError("source manifests must be distinct")
        for item in manifests:
            _sha256("source_manifest_hash", item)

    @property
    def partition_hash(self) -> str:
        return digest(self)


@dataclass(frozen=True)
class SamplingFrameEntry:
    """Public leaf binding one opaque identifier to sealed probe content."""

    probe_id_hash: str
    context_hash: str
    policy_hash: str
    group_count: int
    collection_window_start: str
    collection_window_end: str
    source_handle_hash: str
    payload_commitment: str

    def __post_init__(self) -> None:
        for name in (
            "probe_id_hash",
            "context_hash",
            "policy_hash",
            "source_handle_hash",
            "payload_commitment",
        ):
            _sha256(name, getattr(self, name))
        _uint32("group_count", self.group_count)
        if self.group_count == 0:
            raise ValueError("sampling-frame entries must contain at least one group")
        _required("collection_window_start", self.collection_window_start)
        _required("collection_window_end", self.collection_window_end)
        if self.collection_window_start >= self.collection_window_end:
            raise ValueError("sampling-frame collection window must be increasing")


@dataclass(frozen=True)
class SamplingFrame:
    """Signed sealed catalog whose leaves disclose no raw probe payload."""

    frame_id: str
    context_hash: str
    policy_hash: str
    entries: tuple[SamplingFrameEntry, ...]
    beacon_id: str
    beacon_public_key_hash: str
    exclusion_source_manifest_hashes: tuple[str, ...] = ()
    selection_policy: str = "beacon-sha256-rejection-sampling-v1"

    def __post_init__(self) -> None:
        _required("frame_id", self.frame_id)
        _required("beacon_id", self.beacon_id)
        _sha256("context_hash", self.context_hash)
        _sha256("policy_hash", self.policy_hash)
        _sha256("beacon_public_key_hash", self.beacon_public_key_hash)
        if self.selection_policy != "beacon-sha256-rejection-sampling-v1":
            raise ValueError("unsupported sampling-frame selection policy")
        entries = tuple(self.entries)
        if any(not isinstance(entry, SamplingFrameEntry) for entry in entries):
            raise TypeError("sampling-frame entries must be SamplingFrameEntry objects")
        exclusions = tuple(self.exclusion_source_manifest_hashes)
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "exclusion_source_manifest_hashes", exclusions)
        ids = tuple(entry.probe_id_hash for entry in entries)
        if not ids or ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise ValueError("sampling-frame entries must have distinct ascending IDs")
        commitments = tuple(entry.payload_commitment for entry in entries)
        if len(commitments) != len(set(commitments)):
            raise ValueError("sealed-catalog payload commitments must be distinct")
        if any(
            entry.context_hash != self.context_hash
            or entry.policy_hash != self.policy_hash
            for entry in entries
        ):
            raise ValueError("sampling-frame entries must share context and policy")
        if exclusions != tuple(sorted(exclusions)) or len(exclusions) != len(
            set(exclusions)
        ):
            raise ValueError("sampling-frame exclusions must be distinct and ascending")
        for item in exclusions:
            _sha256("excluded_source_manifest_hash", item)

    @property
    def catalog_root(self) -> str:
        return merkle_root(
            [
                {"domain": "fedmerit-sealed-catalog-leaf-v1", "entry": entry}
                for entry in self.entries
            ]
        )

    @property
    def frame_hash(self) -> str:
        return digest(
            {
                "domain": "fedmerit-sealed-catalog-frame-v2",
                "catalog_root": self.catalog_root,
                "frame": self,
            }
        )


@dataclass(frozen=True)
class SignedSamplingFrame:
    """Authority signature over one immutable sealed catalog."""

    frame: SamplingFrame
    signature: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.signature, (bytes, bytearray)):
            raise TypeError("sampling-frame signature must be bytes")
        signature = bytes(self.signature)
        object.__setattr__(self, "signature", signature)
        if len(signature) != 64:
            raise ValueError("sampling-frame Ed25519 signature must be 64 bytes")


@dataclass(frozen=True)
class BeaconRound:
    """Public randomness-beacon value authenticated under the frame-bound key."""

    beacon_id: str
    round_number: int
    previous_round_hash: str
    randomness: bytes

    def __post_init__(self) -> None:
        _required("beacon_id", self.beacon_id)
        _uint32("round_number", self.round_number)
        if self.round_number == 0:
            raise ValueError("beacon round must be positive")
        _sha256("previous_round_hash", self.previous_round_hash)
        if not isinstance(self.randomness, (bytes, bytearray)):
            raise TypeError("beacon randomness must be bytes")
        randomness = bytes(self.randomness)
        object.__setattr__(self, "randomness", randomness)
        if len(randomness) != 32:
            raise ValueError("beacon randomness must be exactly 32 bytes")

    @property
    def round_hash(self) -> str:
        return digest({"domain": "fedmerit-beacon-round-v1", "round": self})


@dataclass(frozen=True)
class SignedBeaconRound:
    round: BeaconRound
    signature: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.signature, (bytes, bytearray)):
            raise TypeError("beacon transcript signature must be bytes")
        signature = bytes(self.signature)
        object.__setattr__(self, "signature", signature)
        if len(signature) != 64:
            raise ValueError("beacon transcript signature must be 64 bytes")


@dataclass(frozen=True)
class Candidate:
    """Candidate fixation. Every risk/probe field is fixed before release."""

    context_hash: str
    state_context: StateContext
    before_model: LinearModelArtifact
    after_model: LinearModelArtifact
    contributor_root: str
    score_probe_commitment: str
    source_partition: SourcePartition
    evaluation_policy: EvaluationPolicy
    sampling_frame_hash: str
    sealed_catalog_root: str
    eligible_probe_id_hashes: tuple[str, ...]
    risk_schedule_hash: str
    risk_schedule_index: int
    risk: RiskAllocation
    beacon_parent_hash: str
    beacon_round: int
    previous_receipt_hash: str = ZERO_HASH

    def __post_init__(self) -> None:
        for name in (
            "context_hash",
            "contributor_root",
            "score_probe_commitment",
            "sampling_frame_hash",
            "sealed_catalog_root",
            "beacon_parent_hash",
            "risk_schedule_hash",
            "previous_receipt_hash",
        ):
            _sha256(name, getattr(self, name))
        _uint32("risk_schedule_index", self.risk_schedule_index)
        _uint32("beacon_round", self.beacon_round)
        if self.beacon_round == 0:
            raise ValueError("candidate must bind a future positive beacon round")
        eligible_ids = tuple(self.eligible_probe_id_hashes)
        object.__setattr__(self, "eligible_probe_id_hashes", eligible_ids)
        if (
            not eligible_ids
            or eligible_ids != tuple(sorted(eligible_ids))
            or len(eligible_ids) != len(set(eligible_ids))
        ):
            raise ValueError(
                "candidate eligible probe IDs must be distinct and ascending"
            )
        for probe_id_hash in eligible_ids:
            _sha256("eligible_probe_id_hash", probe_id_hash)
        if self.state_context.context_hash != self.context_hash:
            raise ValueError(
                "candidate context hash does not match the disclosed state context"
            )
        if self.state_context.policy_hash != self.evaluation_policy.policy_hash:
            raise ValueError(
                "candidate policy is not authorized by the disclosed state context"
            )
        if (
            self.source_partition.context_hash != self.context_hash
            or self.source_partition.contributor_root != self.contributor_root
            or self.source_partition.score_probe_commitment
            != self.score_probe_commitment
            or self.source_partition.policy_hash != self.evaluation_policy.policy_hash
        ):
            raise ValueError("source partition is not bound to this proposal context")
        if len(self.before_model.weights) != len(self.after_model.weights):
            raise ValueError(
                "before and after artifacts must use the same feature width"
            )
        if (
            self.before_model.feature_mean != self.after_model.feature_mean
            or self.before_model.feature_scale != self.after_model.feature_scale
        ):
            raise ValueError(
                "before and after artifacts must use identical preprocessing"
            )

    @property
    def before_model_hash(self) -> str:
        return self.before_model.artifact_hash

    @property
    def after_model_hash(self) -> str:
        return self.after_model.artifact_hash

    @property
    def candidate_hash(self) -> str:
        return digest(self)

    @property
    def context_policy_hash(self) -> str:
        return self.evaluation_policy.policy_hash

    @property
    def probe_policy_hash(self) -> str:
        return self.evaluation_policy.policy_hash

    @property
    def excluded_source_manifests(self) -> tuple[str, ...]:
        return self.source_partition.source_manifest_hashes

    @property
    def fixation_hash(self) -> str:
        return digest(
            {
                "domain": "fedmerit-fixation-v1",
                "context_hash": self.context_hash,
                "before_model_hash": self.before_model_hash,
                "after_model_hash": self.after_model_hash,
                "candidate_hash": self.candidate_hash,
                "contributor_root": self.contributor_root,
                "score_probe_commitment": self.score_probe_commitment,
                "source_partition_hash": self.source_partition.partition_hash,
                "context_policy_hash": self.context_policy_hash,
                "probe_policy_hash": self.probe_policy_hash,
                "evaluation_policy": self.evaluation_policy,
                "sampling_frame_hash": self.sampling_frame_hash,
                "sealed_catalog_root": self.sealed_catalog_root,
                "eligible_probe_id_hashes": self.eligible_probe_id_hashes,
                "risk_schedule_hash": self.risk_schedule_hash,
                "risk_schedule_index": self.risk_schedule_index,
                "epsilon": self.risk.epsilon,
                "gamma": self.risk.gamma,
                "alpha": self.risk.alpha,
                "group_count": self.risk.group_count,
                "beacon_parent_hash": self.beacon_parent_hash,
                "beacon_round": self.beacon_round,
                "previous_receipt_hash": self.previous_receipt_hash,
            }
        )

    @property
    def signing_scope(self) -> tuple[str, str, str]:
        return self.context_hash, self.before_model_hash, self.previous_receipt_hash


@dataclass(frozen=True)
class CommitProbe:
    """Sealed raw grouped probe with a 256-bit HMAC opening key."""

    probe_id: str
    context_hash: str
    probe_policy_hash: str
    groups: tuple[ProbeGroup, ...]
    collection_window_start: str
    collection_window_end: str
    source_handle_hash: str
    sealing_nonce: bytes = field(default_factory=lambda: secrets.token_bytes(32))

    def __post_init__(self) -> None:
        _required("probe_id", self.probe_id)
        _sha256("context_hash", self.context_hash)
        _sha256("probe_policy_hash", self.probe_policy_hash)
        _sha256("source_handle_hash", self.source_handle_hash)
        groups = tuple(self.groups)
        if any(not isinstance(group, ProbeGroup) for group in groups):
            raise TypeError("commit-probe groups must be ProbeGroup objects")
        object.__setattr__(self, "groups", groups)
        if not isinstance(self.sealing_nonce, (bytes, bytearray)):
            raise TypeError("sealing_nonce must be bytes")
        sealing_nonce = bytes(self.sealing_nonce)
        object.__setattr__(self, "sealing_nonce", sealing_nonce)
        if len(sealing_nonce) != 32:
            raise ValueError("sealing_nonce must contain 256 bits")
        _required("collection_window_start", self.collection_window_start)
        _required("collection_window_end", self.collection_window_end)
        if self.collection_window_start >= self.collection_window_end:
            raise ValueError("commit-probe collection window must be increasing")
        ids = tuple(group.group_id for group in groups)
        manifests = tuple(group.source_manifest_hash for group in groups)
        if not ids or ids != tuple(sorted(ids)):
            raise ValueError("probe groups must be non-empty and sorted by group_id")
        if len(set(ids)) != len(ids) or len(set(manifests)) != len(manifests):
            raise ValueError("probe group ids and source manifests must be distinct")

    @property
    def commitment(self) -> str:
        payload = canonical_bytes(
            {
                "domain": "fedmerit-probe-payload-commitment-v2",
                "probe_id_hash": self.probe_id_hash,
                "payload": {
                    "probe_id": self.probe_id,
                    "context_hash": self.context_hash,
                    "probe_policy_hash": self.probe_policy_hash,
                    "groups": self.groups,
                    "collection_window_start": self.collection_window_start,
                    "collection_window_end": self.collection_window_end,
                    "source_handle_hash": self.source_handle_hash,
                },
            }
        )
        return hmac.new(self.sealing_nonce, payload, hashlib.sha256).hexdigest()

    @property
    def probe_id_hash(self) -> str:
        payload = canonical_bytes(
            {
                "domain": "fedmerit-opaque-probe-id-v2",
                "probe_id": self.probe_id,
            }
        )
        return hmac.new(self.sealing_nonce, payload, hashlib.sha256).hexdigest()

    @property
    def frame_entry(self) -> SamplingFrameEntry:
        return SamplingFrameEntry(
            self.probe_id_hash,
            self.context_hash,
            self.probe_policy_hash,
            len(self.groups),
            self.collection_window_start,
            self.collection_window_end,
            self.source_handle_hash,
            self.commitment,
        )


@dataclass(frozen=True)
class ReceiptCore:
    """Version-1 357-byte core: 10 digests, uint32, 4 binary64, decision byte."""

    context_hash: str
    before_model_hash: str
    after_model_hash: str
    fixation_hash: str
    contributor_root: str
    score_probe_commitment: str
    probe_policy_hash: str
    release_hash: str
    risk_schedule_hash: str
    previous_receipt_hash: str
    source_group_count: int
    epsilon: float
    gamma: float
    alpha: float
    delta_hat: float
    decision: str

    DIGEST_FIELDS = (
        "context_hash",
        "before_model_hash",
        "after_model_hash",
        "fixation_hash",
        "contributor_root",
        "score_probe_commitment",
        "probe_policy_hash",
        "release_hash",
        "risk_schedule_hash",
        "previous_receipt_hash",
    )

    def __post_init__(self) -> None:
        for name in self.DIGEST_FIELDS:
            _sha256(name, getattr(self, name))
        if self.decision not in _DECISION_TO_BYTE:
            raise ValueError("decision must be commit or reject")
        _uint32("source_group_count", self.source_group_count)
        if self.source_group_count == 0:
            raise ValueError("source_group_count must be a positive uint32")
        for name in ("epsilon", "gamma", "alpha", "delta_hat"):
            object.__setattr__(self, name, _binary64(name, getattr(self, name)))
        if self.epsilon <= 0 or self.gamma < 0 or not 0 < self.alpha < 1:
            raise ValueError("invalid risk fields")
        if not -1 <= self.delta_hat <= 1:
            raise ValueError("paired loss difference must lie in [-1,1]")

    def to_bytes(self) -> bytes:
        payload = b"".join(
            bytes.fromhex(getattr(self, name)) for name in self.DIGEST_FIELDS
        )
        payload += struct.pack(
            ">IddddB",
            self.source_group_count,
            self.epsilon,
            self.gamma,
            self.alpha,
            self.delta_hat,
            _DECISION_TO_BYTE[self.decision],
        )
        if len(payload) != RECEIPT_CORE_BYTES:
            raise AssertionError("ReceiptCore encoder violated the 357-byte contract")
        return payload

    @classmethod
    def from_bytes(cls, payload: bytes) -> "ReceiptCore":
        if len(payload) != RECEIPT_CORE_BYTES:
            raise ValueError("version-1 ReceiptCore must be exactly 357 bytes")
        digests = [payload[i : i + 32].hex() for i in range(0, 320, 32)]
        count, epsilon, gamma, alpha, delta_hat, tag = struct.unpack(
            ">IddddB", payload[320:]
        )
        try:
            decision = _BYTE_TO_DECISION[tag]
        except KeyError as exc:
            version = tag >> 4
            raise ValueError(
                f"unsupported ReceiptCore version/decision byte 0x{tag:02x} (version {version})"
            ) from exc
        return cls(*digests, count, epsilon, gamma, alpha, delta_hat, decision)

    @property
    def receipt_hash(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()


@dataclass(frozen=True)
class WitnessSignature:
    witness_index: int
    signature: bytes

    def __post_init__(self) -> None:
        _uint32("witness_index", self.witness_index)
        if not isinstance(self.signature, (bytes, bytearray)):
            raise TypeError("Ed25519 signature must be bytes")
        signature = bytes(self.signature)
        object.__setattr__(self, "signature", signature)
        if len(signature) != 64:
            raise ValueError("Ed25519 signatures must be exactly 64 bytes")


@dataclass(frozen=True)
class Receipt:
    core: ReceiptCore
    witness_count: int
    signatures: tuple[WitnessSignature, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _uint32("witness_count", self.witness_count)
        if self.witness_count == 0:
            raise ValueError("witness_count must be positive")
        signatures = tuple(self.signatures)
        if any(not isinstance(item, WitnessSignature) for item in signatures):
            raise TypeError("signatures must be WitnessSignature objects")
        object.__setattr__(self, "signatures", signatures)
        indices = tuple(item.witness_index for item in signatures)
        if indices != tuple(sorted(indices)) or len(indices) != len(set(indices)):
            raise ValueError("signatures must have distinct ascending witness indices")
        if any(index >= self.witness_count for index in indices):
            raise ValueError("signature index outside witness bitmap")

    @property
    def receipt_hash(self) -> str:
        return self.core.receipt_hash

    def to_bytes(self) -> bytes:
        bitmap = bytearray((self.witness_count + 7) // 8)
        for item in self.signatures:
            bitmap[item.witness_index // 8] |= 1 << (item.witness_index % 8)
        return (
            self.core.to_bytes()
            + bytes(bitmap)
            + b"".join(item.signature for item in self.signatures)
        )

    @classmethod
    def from_bytes(cls, payload: bytes, *, witness_count: int) -> "Receipt":
        _uint32("witness_count", witness_count)
        if witness_count == 0:
            raise ValueError("witness_count must be positive")
        bitmap_len = (witness_count + 7) // 8
        if len(payload) < RECEIPT_CORE_BYTES + bitmap_len:
            raise ValueError("truncated certificate")
        core = ReceiptCore.from_bytes(payload[:RECEIPT_CORE_BYTES])
        bitmap = payload[RECEIPT_CORE_BYTES : RECEIPT_CORE_BYTES + bitmap_len]
        if witness_count % 8 and bitmap[-1] & ~((1 << (witness_count % 8)) - 1):
            raise ValueError("signer bitmap has bits outside the witness set")
        indices = [i for i in range(witness_count) if bitmap[i // 8] & (1 << (i % 8))]
        signature_bytes = payload[RECEIPT_CORE_BYTES + bitmap_len :]
        if len(signature_bytes) != 64 * len(indices):
            raise ValueError("signature payload length disagrees with bitmap")
        signatures = tuple(
            WitnessSignature(i, signature_bytes[64 * j : 64 * (j + 1)])
            for j, i in enumerate(indices)
        )
        return cls(core, witness_count, signatures)

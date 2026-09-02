"""End-to-end FedMERIT benchmark on the UCI UR3 CobotOps dataset.

The benchmark treats operation cycles as non-IID source groups. Proposal,
validation, commit-probe, and audit cycles are disjoint for every seed. Every
candidate is evaluated through a signed sealed frame, a post-fixation beacon,
one-use risk consumption, 2f+1 witness replay, and atomic ``CheckAppend``. Raw
metrics are read back from the resulting receipt and installed serving state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from fedmerit import (
    AuditRegistry,
    BeaconRound,
    Candidate,
    CertificateAuthority,
    CommitProbe,
    CommitProbeStore,
    ContributorLeaf,
    EvaluationPolicy,
    LinearModelArtifact,
    ProbeGroup,
    RiskAllocation,
    RiskLedger,
    RiskSchedule,
    SamplingFrame,
    SourcePartition,
    StateContext,
    VerificationTrust,
    contributor_merkle_root,
    paired_model_loss_difference_exact,
    required_groups,
    risk_bound,
    sign_beacon_round,
    sign_sampling_frame,
)
from fedmerit.canonical import digest
from fedmerit.gate import BeaconService


FEATURE_COLUMNS = (
    "Current_J0",
    "Temperature_T0",
    "Current_J1",
    "Temperature_J1",
    "Current_J2",
    "Temperature_J2",
    "Current_J3",
    "Temperature_J3",
    "Current_J4",
    "Temperature_J4",
    "Current_J5",
    "Temperature_J5",
    "Speed_J0",
    "Speed_J1",
    "Speed_J2",
    "Speed_J3",
    "Speed_J4",
    "Speed_J5",
    "Tool_current",
)
DATASET_DOI = "10.24432/C5J891"
ALPHA = 0.10
EPSILON = 0.35
GAMMA = 0.0
# This secondary threshold is an operational stress-test marker; the certified
# contract still uses EPSILON.  Keeping both metrics avoids conflating a useful
# workload-level diagnostic with the theorem's declared-harm event.
OPERATIONAL_HARM = 0.05
COMMIT_GROUPS = required_groups(ALPHA, EPSILON, GAMMA)


@dataclass(frozen=True)
class CycleGroup:
    cycle_id: int
    raw_x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class _UnavailableWitness:
    """Benchmark fault: a roster member that never returns an attestation."""

    witness_index: int
    public_key: Ed25519PublicKey

    def attest(self, *_: object, **__: object) -> None:
        raise TimeoutError("injected unavailable witness")


def _deterministic_private_key(scope: str, role: str) -> Ed25519PrivateKey:
    material = hashlib.sha256(
        f"fedmerit-ur3-benchmark:{scope}:{role}".encode("utf-8")
    ).digest()
    return Ed25519PrivateKey.from_private_bytes(material)


def _cycle_manifest_hash(cycle_id: int) -> str:
    return hashlib.sha256(
        f"uci-ur3-cobotops:{DATASET_DOI}:cycle:{cycle_id}".encode()
    ).hexdigest()


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits, -80.0, 80.0)
    return np.where(
        clipped >= 0,
        1.0 / (1.0 + np.exp(-clipped)),
        np.exp(clipped) / (1.0 + np.exp(clipped)),
    )


def _predict(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    return _sigmoid(x @ weights[:-1] + weights[-1])


def _balanced_accuracy(weights: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    predicted = _predict(weights, x) >= 0.5
    recalls = []
    for label in (0, 1):
        mask = y == label
        if np.any(mask):
            recalls.append(float(np.mean(predicted[mask] == bool(label))))
    return float(np.mean(recalls))


def _brier(weights: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((_predict(weights, x) - y) ** 2))


def _local_update(
    weights: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float,
    epochs: int,
    positive_weight: float,
) -> np.ndarray:
    result = weights.copy()
    sample_weights = np.where(y == 1, positive_weight, 1.0)
    normalizer = float(np.sum(sample_weights))
    for _ in range(epochs):
        error = (_predict(result, x) - y) * sample_weights
        gradient = np.empty_like(result)
        gradient[:-1] = x.T @ error / normalizer + 1e-4 * result[:-1]
        gradient[-1] = float(np.sum(error) / normalizer)
        result -= learning_rate * gradient
    return result - weights


def _weighted_mean(updates: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    return np.average(updates, axis=0, weights=sizes)


def _coordinate_median(updates: np.ndarray, _: np.ndarray) -> np.ndarray:
    return np.median(updates, axis=0)


def _krum(updates: np.ndarray, _: np.ndarray, faults: int) -> np.ndarray:
    count = len(updates)
    neighbours = count - faults - 2
    if neighbours <= 0:
        raise ValueError("Krum requires n > f + 2")
    distances = np.sum((updates[:, None, :] - updates[None, :, :]) ** 2, axis=2)
    scores = []
    for index in range(count):
        ordered = np.sort(np.delete(distances[index], index))
        scores.append(float(np.sum(ordered[:neighbours])))
    return updates[int(np.argmin(scores))]


def _fltrust(
    updates: np.ndarray, sizes: np.ndarray, root_update: np.ndarray
) -> np.ndarray:
    root_norm = float(np.linalg.norm(root_update))
    if root_norm == 0:
        return np.zeros_like(root_update)
    norms = np.linalg.norm(updates, axis=1)
    cosine = (updates @ root_update) / np.maximum(norms * root_norm, 1e-15)
    trust = np.maximum(cosine, 0.0)
    clipped = updates * np.minimum(1.0, root_norm / np.maximum(norms, 1e-15))[:, None]
    weights = trust * sizes
    if float(np.sum(weights)) == 0:
        return np.zeros_like(root_update)
    return np.average(clipped, axis=0, weights=weights)


def _fedval(
    before: np.ndarray,
    updates: np.ndarray,
    sizes: np.ndarray,
    score_x: np.ndarray,
    score_y: np.ndarray,
) -> np.ndarray:
    losses = np.asarray(
        [_brier(before + update, score_x, score_y) for update in updates]
    )
    scale = max(float(np.std(losses)), 1e-6)
    validation_weights = np.exp(-(losses - float(np.min(losses))) / scale)
    return np.average(updates, axis=0, weights=validation_weights * sizes)


def _flshield_cluster(
    before: np.ndarray,
    updates: np.ndarray,
    sizes: np.ndarray,
    validator_groups: tuple[tuple[np.ndarray, np.ndarray], ...],
    *,
    seed: int,
) -> np.ndarray:
    """Tabular instantiation of FLShield's cluster representative path.

    Validators are honest in this experiment. K-Means plus silhouette selection
    creates representative models; classwise loss-impact scores rank them. The
    accepted clusters map back to individual updates before norm clipping and
    aggregation, matching the paper's five-stage path.
    """
    best_labels = None
    best_score = -math.inf
    max_clusters = min(6, len(updates) - 1)
    for clusters in range(2, max_clusters + 1):
        labels = KMeans(
            n_clusters=clusters,
            n_init=10,
            random_state=seed,
        ).fit_predict(updates)
        if len(set(labels.tolist())) < 2:
            continue
        score = float(silhouette_score(updates, labels))
        if score > best_score:
            best_labels = labels
            best_score = score
    if best_labels is None:
        return _coordinate_median(updates, sizes)

    cluster_ids = sorted(set(best_labels.tolist()))
    representative_scores = []
    for cluster_id in cluster_ids:
        member_mask = best_labels == cluster_id
        representative = np.average(
            updates[member_mask], axis=0, weights=sizes[member_mask]
        )
        impacts = []
        for x, y in validator_groups:
            per_class = []
            for label in (0, 1):
                mask = y == label
                if np.any(mask):
                    per_class.append(
                        _brier(before, x[mask], y[mask])
                        - _brier(before + representative, x[mask], y[mask])
                    )
            if per_class:
                impacts.append(min(per_class))
        representative_scores.append(float(np.median(impacts)))
    keep_count = max(1, math.ceil(len(cluster_ids) / 2))
    keep_clusters = {
        cluster_ids[index] for index in np.argsort(representative_scores)[-keep_count:]
    }
    accepted = np.asarray(
        [index for index, label in enumerate(best_labels) if label in keep_clusters]
    )
    accepted_updates = updates[accepted]
    accepted_sizes = sizes[accepted]
    norms = np.linalg.norm(accepted_updates, axis=1)
    clip = float(np.median(norms))
    clipped = (
        accepted_updates * np.minimum(1.0, clip / np.maximum(norms, 1e-15))[:, None]
    )
    return np.average(clipped, axis=0, weights=accepted_sizes)


def _load_groups(path: Path) -> tuple[CycleGroup, ...]:
    frame = pd.read_excel(path, engine="openpyxl")
    required = set(FEATURE_COLUMNS) | {"cycle ", "Robot_ProtectiveStop", "grip_lost"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")
    selected = frame[list(FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    cycle = pd.to_numeric(frame["cycle "], errors="coerce")
    stop = pd.to_numeric(frame["Robot_ProtectiveStop"], errors="coerce")
    grip = pd.to_numeric(frame["grip_lost"], errors="coerce")
    valid = selected.notna().all(axis=1) & cycle.notna() & stop.notna() & grip.notna()
    selected = selected.loc[valid]
    labels = ((stop.loc[valid] > 0) | (grip.loc[valid] > 0)).astype(int)
    cycles = cycle.loc[valid].astype(int)
    groups = []
    for cycle_id in sorted(cycles.unique().tolist()):
        mask = cycles == cycle_id
        groups.append(
            CycleGroup(
                int(cycle_id),
                selected.loc[mask].to_numpy(dtype=np.float64),
                labels.loc[mask].to_numpy(dtype=np.int64),
            )
        )
    if len(groups) < 4 * 40:
        raise ValueError("benchmark requires at least 160 distinct operation cycles")
    return tuple(groups)


def _stack(
    groups: tuple[CycleGroup, ...], mean: np.ndarray, scale: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.vstack([group.raw_x for group in groups])
    labels = np.concatenate([group.y for group in groups])
    return (raw - mean) / scale, labels


def _probe_groups(groups: tuple[CycleGroup, ...]) -> tuple[ProbeGroup, ...]:
    return tuple(
        ProbeGroup(
            f"cycle-{group.cycle_id:04d}",
            _cycle_manifest_hash(group.cycle_id),
            tuple(tuple(float(value) for value in row) for row in group.raw_x),
            tuple(int(value) for value in group.y),
        )
        for group in sorted(groups, key=lambda item: item.cycle_id)
    )


def _execute_protocol_trial(
    root: Path,
    *,
    trial_id: str,
    before_artifact: LinearModelArtifact,
    after_artifact: LinearModelArtifact,
    proposal_groups: tuple[CycleGroup, ...],
    proposal_updates: np.ndarray,
    proposal_sizes: np.ndarray,
    score_groups: tuple[CycleGroup, ...],
    commit_groups: tuple[CycleGroup, ...],
    policy: EvaluationPolicy,
    exercise_append_race: bool = False,
) -> dict[str, object]:
    """Execute one isolated candidate through the complete reference protocol."""
    if len(proposal_groups) != len(proposal_updates) or len(proposal_groups) != len(
        proposal_sizes
    ):
        raise ValueError("proposal groups, updates, and sizes must align")
    if len(commit_groups) != COMMIT_GROUPS:
        raise ValueError(
            "commit group count differs from the registered risk allocation"
        )
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter_ns()
    scope = digest(
        {
            "domain": "fedmerit-ur3-trial-v1",
            "trial_id": trial_id,
            "before_model_hash": before_artifact.artifact_hash,
            "after_model_hash": after_artifact.artifact_hash,
        }
    )
    frame_key = _deterministic_private_key(scope, "frame")
    store_key = _deterministic_private_key(scope, "store")
    beacon_key = _deterministic_private_key(scope, "beacon")
    witness_keys = tuple(
        _deterministic_private_key(scope, f"witness-{index}") for index in range(4)
    )
    persistent_authority = CertificateAuthority.persistent(
        root / "witnesses", f=1, private_keys=witness_keys
    )
    unavailable_index = 0
    witnesses = list(persistent_authority.witnesses)
    witnesses[unavailable_index] = _UnavailableWitness(
        unavailable_index, witnesses[unavailable_index].public_key
    )
    authority = CertificateAuthority(witnesses, f=1)
    trust = VerificationTrust.from_keys(
        authority.public_keys,
        f=1,
        store_public_key=store_key.public_key(),
        frame_public_key=frame_key.public_key(),
    )
    context = StateContext(
        f"ur3-twin-{scope[:16]}",
        "uci-ur3-cobotops",
        0,
        digest(
            {
                "domain": "fedmerit-ur3-feature-schema-v1",
                "feature_count": len(before_artifact.weights) - 1,
            }
        ),
        policy.policy_hash,
        0,
        trust.authority_certificate_hash,
    )

    total_size = float(np.sum(proposal_sizes))
    if total_size <= 0:
        raise ValueError("proposal sample sizes must be positive")
    contributor_rows = sorted(
        zip(proposal_groups, proposal_updates, proposal_sizes, strict=True),
        key=lambda item: item[0].cycle_id,
    )
    contributor_leaves = tuple(
        ContributorLeaf(
            f"client-cycle-{group.cycle_id:04d}",
            digest(
                {
                    "domain": "fedmerit-ur3-client-update-v1",
                    "cycle_id": group.cycle_id,
                    "update": tuple(float(value) for value in update),
                }
            ),
            float(size) / total_size,
        )
        for group, update, size in contributor_rows
    )
    contributor_root = contributor_merkle_root(contributor_leaves)
    score_probe_groups = _probe_groups(score_groups)
    score_probe_commitment = digest(
        {
            "domain": "fedmerit-ur3-score-probe-v1",
            "groups": score_probe_groups,
        }
    )
    partition = SourcePartition(
        context.context_hash,
        contributor_root,
        score_probe_commitment,
        policy.policy_hash,
        tuple(
            sorted(_cycle_manifest_hash(group.cycle_id) for group in proposal_groups)
        ),
        tuple(sorted(_cycle_manifest_hash(group.cycle_id) for group in score_groups)),
    )
    probe = CommitProbe(
        f"commit-probe-{scope[:16]}",
        context.context_hash,
        policy.policy_hash,
        _probe_groups(commit_groups),
        "2026-01-01T00:00:00Z",
        "2026-01-02T00:00:00Z",
        digest({"domain": "fedmerit-ur3-source-handle-v1", "scope": scope}),
        hashlib.sha256(f"fedmerit-ur3-probe-nonce:{scope}".encode()).digest(),
    )

    checkpoint = sign_beacon_round(
        BeaconRound(
            "ur3-benchmark-beacon",
            1,
            digest({"domain": "fedmerit-ur3-beacon-genesis-v1"}),
            hashlib.sha256(f"fedmerit-ur3-beacon:{scope}:1".encode()).digest(),
        ),
        beacon_key,
    )
    beacon_key_bytes = beacon_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    frame = SamplingFrame(
        f"frame-{scope[:16]}",
        context.context_hash,
        policy.policy_hash,
        (probe.frame_entry,),
        checkpoint.round.beacon_id,
        hashlib.sha256(beacon_key_bytes).hexdigest(),
        (partition.partition_hash,),
        checkpoint.round.round_number,
        checkpoint.round.round_hash,
    )
    signed_frame = sign_sampling_frame(frame, frame_key)
    allocation = RiskAllocation(EPSILON, GAMMA, ALPHA, COMMIT_GROUPS)
    schedule = RiskSchedule(
        f"schedule-{scope[:16]}",
        context.context_hash,
        "0" * 64,
        0.20,
        (allocation,),
    )
    candidate = Candidate(
        context.context_hash,
        context,
        before_artifact,
        after_artifact,
        contributor_root,
        score_probe_commitment,
        partition,
        policy,
        frame.frame_hash,
        frame.catalog_root,
        (probe.probe_id_hash,),
        schedule.schedule_hash,
        0,
        allocation,
        checkpoint.round.round_hash,
        2,
    )
    registry = AuditRegistry(
        root / "audit.sqlite3",
        genesis_model=before_artifact,
        initial_context=context,
        evaluation_policy=policy,
        verification_trust=trust,
    )
    registry.provision_lineage_risk_budget(schedule.lifetime_delta)
    ledger = RiskLedger(root / "risk.sqlite3")
    ledger.register(schedule, audit_registry=registry)
    ledger.observe_beacon_head(
        checkpoint,
        audit_registry=registry,
        beacon_public_key=beacon_key.public_key(),
        signed_frame=signed_frame,
        frame_public_key=frame_key.public_key(),
    )
    ledger.consume(
        candidate,
        schedule,
        audit_registry=registry,
        beacon_public_key=beacon_key.public_key(),
        signed_frame=signed_frame,
        frame_public_key=frame_key.public_key(),
    )

    competing_weights = list(after_artifact.weights)
    competing_weights[0] = math.nextafter(competing_weights[0], math.inf)
    competing_candidate = replace(
        candidate,
        after_model=LinearModelArtifact(
            tuple(competing_weights),
            after_artifact.feature_mean,
            after_artifact.feature_scale,
        ),
    )
    competing_fixation_blocked = False
    try:
        ledger.consume(
            competing_candidate,
            schedule,
            audit_registry=registry,
            beacon_public_key=beacon_key.public_key(),
            signed_frame=signed_frame,
            frame_public_key=frame_key.public_key(),
        )
    except ValueError:
        competing_fixation_blocked = True
    if not competing_fixation_blocked:
        raise RuntimeError("risk allocation accepted a competing candidate fixation")

    beacon_service = BeaconService(
        root / "beacon.sqlite3",
        beacon_id=checkpoint.round.beacon_id,
        checkpoint=checkpoint,
        private_key=beacon_key,
        entropy_seed=hashlib.sha256(f"fedmerit-ur3-entropy:{scope}".encode()).digest(),
    )
    fixation_reservation = beacon_service.reserve_fixation(
        candidate,
        risk_ledger=ledger,
    )
    successor = beacon_service.finalize_successor(fixation_reservation)
    store = CommitProbeStore(
        [probe],
        [partition],
        signed_frame,
        frame_key.public_key(),
        root / "probe.sqlite3",
        store_private_key=store_key,
    )
    release = store.release(
        candidate,
        signed_beacon_round=successor,
        beacon_public_key=beacon_key.public_key(),
        schedule=schedule,
        risk_ledger=ledger,
        audit_registry=registry,
    )
    issue_started = time.perf_counter_ns()
    receipt = authority.issue(
        candidate,
        release,
        store_public_key=store.public_key,
        frame_public_key=frame_key.public_key(),
        schedule=schedule,
        risk_ledger=ledger,
        audit_registry=registry,
    )
    certificate_issue_ms = (time.perf_counter_ns() - issue_started) / 1_000_000.0

    def append_once() -> bool:
        return registry.verify_and_append(
            receipt,
            authority.public_keys,
            f=authority.f,
            release=release,
            candidate=candidate,
            store_public_key=store.public_key,
            frame_public_key=frame_key.public_key(),
            schedule=schedule,
            risk_ledger=ledger,
        )

    append_attempts = 2 if exercise_append_race else 1
    if exercise_append_race:
        with ThreadPoolExecutor(max_workers=2) as executor:
            append_results = tuple(executor.map(lambda _: append_once(), range(2)))
    else:
        append_results = (append_once(),)
    if not all(append_results):
        raise RuntimeError("valid receipt failed atomic CheckAppend")

    expected_model = (
        after_artifact if receipt.core.decision == "commit" else before_artifact
    )
    expected_version = 1 if receipt.core.decision == "commit" else 0
    serving_hash, serving_version, serving_blob = registry.serving_model_snapshot
    serving_bytes_verified = (
        serving_hash == expected_model.artifact_hash
        and serving_version == expected_version
        and serving_blob == expected_model.artifact_bytes
    )
    state_verified = (
        registry.head == receipt.receipt_hash
        and registry.installed_model_hash == expected_model.artifact_hash
        and registry.installed_model_version == expected_version
        and serving_bytes_verified
    )
    if not state_verified:
        raise RuntimeError("receipt decision and installed serving state diverged")
    event_types = {str(event["event_type"]) for event in registry.protocol_events}
    required_events = {
        "risk-schedule-registered",
        "beacon-head-observed",
        "beacon-successor-reserved",
        "risk-allocation-spent",
        "source-manifests-retired",
        "receipt-issued",
        "receipt-appended",
    }
    event_chain_valid = (
        registry.protocol_event_chain_valid() and required_events.issubset(event_types)
    )
    if not event_chain_valid:
        raise RuntimeError("protocol event journal is incomplete or invalid")
    protocol_e2e_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    threshold = 2 * authority.f + 1
    return {
        "receipt_decision": receipt.core.decision,
        "receipt_delta": receipt.core.delta_hat,
        "receipt_hash": receipt.receipt_hash,
        "append_succeeded": int(all(append_results)),
        "append_attempts": append_attempts,
        "append_successes": int(sum(append_results)),
        "installed_candidate": int(
            registry.installed_model_hash == after_artifact.artifact_hash
        ),
        "installed_model_version": registry.installed_model_version,
        "serving_bytes_verified": int(serving_bytes_verified),
        "risk_consumed": int(
            ledger.is_consumed(schedule.schedule_hash, 0, candidate.fixation_hash)
        ),
        "probe_consumed": int(store.is_consumed(probe.probe_id)),
        "competing_fixation_blocked": int(competing_fixation_blocked),
        "witness_faults_injected": 1,
        "witness_signatures": len(receipt.signatures),
        "witness_threshold": threshold,
        "witness_quorum_met": int(len(receipt.signatures) >= threshold),
        "event_chain_valid": int(event_chain_valid),
        "protocol_event_count": len(registry.protocol_events),
        "catalog_leaves": len(frame.entries),
        "certificate_issue_ms": certificate_issue_ms,
        "protocol_e2e_ms": protocol_e2e_ms,
    }


def _mean_ci(values: pd.Series, *, binary: bool = False) -> tuple[float, float]:
    array = values.to_numpy(dtype=float)
    mean = float(np.mean(array))
    if binary:
        # Wilson's interval remains informative at 0 and 1, unlike the
        # normal approximation's zero-width interval at the boundaries.
        size = len(array)
        z = 1.96
        denominator = 1.0 + z * z / size
        center = (mean + z * z / (2.0 * size)) / denominator
        radius = (
            z
            * math.sqrt(mean * (1.0 - mean) / size + z * z / (4.0 * size * size))
            / denominator
        )
        return mean, max(center + radius - mean, mean - (center - radius))
    half_width = (
        0.0
        if len(array) < 2
        else 1.96 * float(np.std(array, ddof=1)) / math.sqrt(len(array))
    )
    return mean, half_width


def _run_seed(
    groups: tuple[CycleGroup, ...],
    *,
    seed: int,
    pretrain_rounds: int,
    clients_per_round: int,
    byzantine_fraction: float,
    protocol_root: Path,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    train = tuple(groups[index] for index in order[:120])
    score = tuple(groups[index] for index in order[120:160])
    commit_pool = tuple(groups[index] for index in order[160:200])
    audit = tuple(groups[index] for index in order[200:])
    commit = tuple(
        commit_pool[index]
        for index in sorted(rng.choice(40, COMMIT_GROUPS, replace=False))
    )

    train_raw = np.vstack([group.raw_x for group in train])
    mean = np.mean(train_raw, axis=0)
    scale = np.std(train_raw, axis=0)
    scale[scale < 1e-12] = 1.0
    train_xy = {group.cycle_id: _stack((group,), mean, scale) for group in train}
    score_x, score_y = _stack(score, mean, scale)
    audit_x, audit_y = _stack(audit, mean, scale)
    positives = int(sum(int(np.sum(group.y)) for group in train))
    rows = int(sum(len(group.y) for group in train))
    positive_weight = max(1.0, (rows - positives) / max(positives, 1))

    weights = np.zeros(len(FEATURE_COLUMNS) + 1, dtype=np.float64)
    for _ in range(pretrain_rounds):
        selected = rng.choice(len(train), clients_per_round, replace=False)
        updates = []
        sizes = []
        for index in selected:
            group = train[int(index)]
            x, y = train_xy[group.cycle_id]
            updates.append(
                _local_update(
                    weights,
                    x,
                    y,
                    learning_rate=0.08,
                    epochs=3,
                    positive_weight=positive_weight,
                )
            )
            sizes.append(len(y))
        weights += _weighted_mean(np.asarray(updates), np.asarray(sizes))

    selected = rng.choice(len(train), clients_per_round, replace=False)
    proposal_groups = tuple(train[int(index)] for index in selected)
    benign_updates = []
    sizes = []
    for group in proposal_groups:
        x, y = train_xy[group.cycle_id]
        benign_updates.append(
            _local_update(
                weights,
                x,
                y,
                learning_rate=0.08,
                epochs=3,
                positive_weight=positive_weight,
            )
        )
        sizes.append(len(y))
    benign_updates_array = np.asarray(benign_updates)
    sizes_array = np.asarray(sizes, dtype=float)
    score_update = _local_update(
        weights,
        score_x,
        score_y,
        learning_rate=0.04,
        epochs=1,
        positive_weight=positive_weight,
    )
    validator_groups = tuple(_stack((group,), mean, scale) for group in score[:20])

    before_artifact = LinearModelArtifact(
        tuple(float(value) for value in weights),
        tuple(float(value) for value in mean),
        tuple(float(value) for value in scale),
    )
    audit_probe_groups = _probe_groups(audit)
    policy = EvaluationPolicy("brier-decimal80-v1")
    before_accuracy = _balanced_accuracy(weights, audit_x, audit_y)
    methods = ("FedAvg", "CoordinateMedian", "Krum", "FLTrust", "FedVal", "FLShield")
    output = []
    for attack in ("none", "sign_flip", "model_replacement"):
        updates = benign_updates_array.copy()
        fault_count = (
            0
            if attack == "none"
            else max(1, int(math.floor(clients_per_round * byzantine_fraction)))
        )
        if fault_count:
            malicious = rng.choice(clients_per_round, fault_count, replace=False)
            if attack == "sign_flip":
                updates[malicious] = -6.0 * updates[malicious]
            else:
                updates[malicious] = -12.0 * weights - 8.0 * updates[malicious]
        for method in methods:
            if method == "FedAvg":
                aggregate = _weighted_mean(updates, sizes_array)
            elif method == "CoordinateMedian":
                aggregate = _coordinate_median(updates, sizes_array)
            elif method == "Krum":
                aggregate = _krum(updates, sizes_array, fault_count)
            elif method == "FLTrust":
                aggregate = _fltrust(updates, sizes_array, score_update)
            elif method == "FedVal":
                aggregate = _fedval(weights, updates, sizes_array, score_x, score_y)
            else:
                aggregate = _flshield_cluster(
                    weights,
                    updates,
                    sizes_array,
                    validator_groups,
                    seed=seed,
                )
            candidate_weights = weights + aggregate
            after_artifact = LinearModelArtifact(
                tuple(float(value) for value in candidate_weights),
                tuple(float(value) for value in mean),
                tuple(float(value) for value in scale),
            )
            protocol = _execute_protocol_trial(
                protocol_root / f"{attack}-{method.lower()}",
                trial_id=f"seed={seed};attack={attack};method={method}",
                before_artifact=before_artifact,
                after_artifact=after_artifact,
                proposal_groups=proposal_groups,
                proposal_updates=updates,
                proposal_sizes=sizes_array,
                score_groups=score,
                commit_groups=commit,
                policy=policy,
                exercise_append_race=attack == "none" and method == "FedAvg",
            )
            commit_delta = float(protocol["receipt_delta"])
            audit_delta = float(
                paired_model_loss_difference_exact(
                    before_artifact,
                    after_artifact,
                    audit_probe_groups,
                    policy,
                )
            )
            accepted = protocol["receipt_decision"] == "commit"
            if bool(protocol["installed_candidate"]) != accepted:
                raise RuntimeError("protocol decision and installed model disagree")
            candidate_accuracy = _balanced_accuracy(candidate_weights, audit_x, audit_y)
            output.append(
                {
                    "seed": seed,
                    "method": method,
                    "attack": attack,
                    "accepted": int(accepted),
                    "commit_delta": commit_delta,
                    "audit_delta": audit_delta,
                    "operational_harm": int(audit_delta >= OPERATIONAL_HARM),
                    "declared_harm": int(audit_delta >= EPSILON),
                    "harmful_escape": int(accepted and audit_delta >= OPERATIONAL_HARM),
                    "declared_harmful_escape": int(accepted and audit_delta >= EPSILON),
                    "beneficial": int(audit_delta < 0),
                    "false_rejection": int((not accepted) and audit_delta < 0),
                    "before_balanced_accuracy": before_accuracy,
                    "candidate_balanced_accuracy": candidate_accuracy,
                    "installed_balanced_accuracy": (
                        candidate_accuracy if accepted else before_accuracy
                    ),
                    **protocol,
                }
            )
    return output


def _summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_names = (
        "accepted",
        "operational_harm",
        "declared_harm",
        "harmful_escape",
        "declared_harmful_escape",
        "false_rejection",
        "candidate_balanced_accuracy",
        "installed_balanced_accuracy",
        "append_succeeded",
        "serving_bytes_verified",
        "risk_consumed",
        "probe_consumed",
        "competing_fixation_blocked",
        "witness_quorum_met",
        "event_chain_valid",
        "certificate_issue_ms",
        "protocol_e2e_ms",
    )
    for (method, attack), group in frame.groupby(["method", "attack"], sort=True):
        row: dict[str, object] = {
            "method": method,
            "attack": attack,
            "seeds": int(group["seed"].nunique()),
        }
        for metric in metric_names:
            mean, ci = _mean_ci(
                group[metric],
                binary=metric
                in {
                    "accepted",
                    "operational_harm",
                    "declared_harm",
                    "harmful_escape",
                    "declared_harmful_escape",
                    "false_rejection",
                    "append_succeeded",
                    "serving_bytes_verified",
                    "risk_consumed",
                    "probe_consumed",
                    "competing_fixation_blocked",
                    "witness_quorum_met",
                    "event_chain_valid",
                },
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95"] = ci
        beneficial = group[group["beneficial"] == 1]
        row["beneficial_cases"] = int(len(beneficial))
        row["conditional_false_rejection_rate"] = (
            float(beneficial["false_rejection"].mean()) if len(beneficial) else math.nan
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["attack", "method"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results_ur3"))
    parser.add_argument("--seeds", default=",".join(str(value) for value in range(20)))
    parser.add_argument("--pretrain-rounds", type=int, default=20)
    parser.add_argument("--clients-per-round", type=int, default=30)
    parser.add_argument("--byzantine-fraction", type=float, default=0.20)
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not seeds or args.pretrain_rounds <= 0 or args.clients_per_round <= 6:
        raise ValueError(
            "benchmark needs seeds, positive pretraining, and at least 7 clients"
        )
    if not 0 < args.byzantine_fraction < 0.5:
        raise ValueError("Byzantine fraction must lie in (0, 0.5)")
    groups = _load_groups(args.dataset)
    if args.clients_per_round > 120:
        raise ValueError("clients_per_round exceeds the fixed proposal split")
    records = []
    started = time.perf_counter()
    with TemporaryDirectory(prefix="fedmerit-ur3-protocol-") as directory:
        protocol_root = Path(directory)
        for seed in seeds:
            records.extend(
                _run_seed(
                    groups,
                    seed=seed,
                    pretrain_rounds=args.pretrain_rounds,
                    clients_per_round=args.clients_per_round,
                    byzantine_fraction=args.byzantine_fraction,
                    protocol_root=protocol_root / f"seed-{seed}",
                )
            )
    elapsed = time.perf_counter() - started
    frame = pd.DataFrame(records)
    protocol_checks = (
        "append_succeeded",
        "serving_bytes_verified",
        "risk_consumed",
        "probe_consumed",
        "competing_fixation_blocked",
        "witness_quorum_met",
        "event_chain_valid",
    )
    failed_checks = {
        name: int((frame[name] != 1).sum())
        for name in protocol_checks
        if bool((frame[name] != 1).any())
    }
    if failed_checks:
        raise RuntimeError(f"end-to-end protocol evidence failed: {failed_checks}")
    summary = _summarize(frame)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "raw_runs.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    metadata = {
        "schema": "fedmerit/ur3-noniid-benchmark/v2",
        "dataset": {
            "name": "UCI UR3 CobotOps",
            "doi": DATASET_DOI,
            "rows_after_complete_case_filter": int(
                sum(len(group.y) for group in groups)
            ),
            "operation_cycle_groups": len(groups),
        },
        "seeds": list(seeds),
        "split_groups_per_seed": {
            "proposal": 120,
            "score": 40,
            "commit_pool": 40,
            "commit_selected": COMMIT_GROUPS,
            "independent_audit": len(groups) - 200,
        },
        "attacks": ["none", "sign_flip", "model_replacement"],
        "byzantine_fraction": args.byzantine_fraction,
        "clients_per_round": args.clients_per_round,
        "pretrain_rounds": args.pretrain_rounds,
        "risk": {
            "alpha": ALPHA,
            "epsilon": EPSILON,
            "gamma": GAMMA,
            "operational_harm_threshold": OPERATIONAL_HARM,
            "group_count": COMMIT_GROUPS,
            "verified_bound": risk_bound(COMMIT_GROUPS, EPSILON, GAMMA),
        },
        "protocol_execution": {
            "stages": [
                "signed_sampling_frame",
                "risk_schedule_registration",
                "signed_beacon_checkpoint",
                "candidate_fixation_and_risk_consumption",
                "durable_beacon_fixation_reservation",
                "signed_successor_beacon_release",
                "one_use_probe_retirement",
                "2f_plus_1_witness_replay",
                "atomic_verify_and_append",
                "serving_state_readback",
            ],
            "trials": len(frame),
            "catalog_leaves_per_trial": 1,
            "witness_roster_size": 4,
            "byzantine_threshold_f": 1,
            "unavailable_witnesses_injected_per_trial": 1,
            "quorum_receipts": int(frame["witness_quorum_met"].sum()),
            "competing_fixations_blocked": int(
                frame["competing_fixation_blocked"].sum()
            ),
            "atomic_appends": int(frame["append_succeeded"].sum()),
            "append_attempts": int(frame["append_attempts"].sum()),
            "append_successes": int(frame["append_successes"].sum()),
            "concurrent_retry_trials": int((frame["append_attempts"] > 1).sum()),
            "serving_readbacks_verified": int(frame["serving_bytes_verified"].sum()),
            "event_chains_verified": int(frame["event_chain_valid"].sum()),
        },
        "records": len(frame),
        "elapsed_seconds": elapsed,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Produce the deterministic calculation and conformance evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from dataclasses import replace
from fractions import Fraction
from importlib.metadata import version
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

from fedmerit.cli import read_json_object, validate_manifest
from fedmerit.certificate import (
    AuditRegistry,
    CertificateAuthority,
    VerificationTrust,
    verify_receipt,
    verify_receipt_bytes,
)
from fedmerit.gate import (
    BeaconService,
    CommitProbeStore,
    RiskLedger,
    gate_decision,
    required_groups,
    risk_bound,
    risk_bound_interval,
    risk_is_satisfied,
    sign_beacon_round,
    sign_sampling_frame,
    verify_public_release,
    verify_release,
    verify_sampling_frame,
)
from fedmerit.model import (
    BeaconRound,
    RECEIPT_CORE_BYTES,
    Candidate,
    CommitProbe,
    ContributorLeaf,
    EvaluationPolicy,
    LinearModelArtifact,
    ProbeGroup,
    Receipt,
    RiskAllocation,
    RiskSchedule,
    SamplingFrame,
    SecurityProfile,
    SourcePartition,
    StateContext,
    ZERO_HASH,
    contributor_merkle_root,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "benchmark_protocol.json"
REFERENCE_VECTOR = ROOT / "artifacts" / "reference_receipt.json"
PROTOCOL_CONFIG = read_json_object(CONFIG)
validate_manifest(PROTOCOL_CONFIG)

_policy_fields = PROTOCOL_CONFIG["evaluation_policy"]
_security_fields = _policy_fields["security_profile"]
EVALUATION_POLICY = EvaluationPolicy(
    policy_id=_policy_fields["policy_id"],
    loss=_policy_fields["loss"],
    preprocessing=_policy_fields["preprocessing"],
    decimal_precision=_policy_fields["decimal_precision"],
    rounding=_policy_fields["rounding"],
    group_loss_quantum=_policy_fields["group_loss_quantum"],
    group_order=_policy_fields["group_order"],
    missing_value_rule=_policy_fields["missing_value_rule"],
    class_weights=tuple(_policy_fields["class_weights"]),
    group_reduction=_policy_fields["group_reduction"],
    security_profile=SecurityProfile(**_security_fields),
)
RISK_GRID = tuple(
    (item["alpha"], item["epsilon"], item["gamma"])
    for item in PROTOCOL_CONFIG["certificate_evaluation"]["risk_grid"]
)
QUORUM_FAULT_COUNTS = tuple(
    PROTOCOL_CONFIG["certificate_evaluation"]["quorum_fault_counts"]
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _risk_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for alpha, epsilon, gamma in RISK_GRID:
        count = required_groups(alpha, epsilon, gamma)
        lower, upper = risk_bound_interval(count, epsilon, gamma)
        interval_scale = 10**24
        lower_scaled = lower.numerator * interval_scale // lower.denominator
        upper_scaled = -(-(upper.numerator * interval_scale) // upper.denominator)
        achieved = risk_bound(count, epsilon, gamma)
        previous = risk_bound(count - 1, epsilon, gamma) if count > 1 else 1.0
        rows.append(
            {
                "kind": "risk_budget",
                "alpha": alpha,
                "epsilon": epsilon,
                "gamma": gamma,
                "required_groups": count,
                "bound_at_required": achieved,
                "bound_interval_scale": interval_scale,
                "bound_interval_lower_scaled": lower_scaled,
                "bound_interval_upper_scaled": upper_scaled,
                "bound_at_previous": previous,
                "interval_certifies_threshold": upper
                <= Fraction.from_float(float(alpha)),
                "minimality_verified": (
                    risk_is_satisfied(count, epsilon, gamma, alpha)
                    and not risk_is_satisfied(count - 1, epsilon, gamma, alpha)
                    if count > 1
                    else risk_is_satisfied(count, epsilon, gamma, alpha)
                ),
                "group_loss_evaluations": 2 * count,
            }
        )
    return rows


def _quorum_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for f in QUORUM_FAULT_COUNTS:
        witnesses = 3 * f + 1
        signatures = 2 * f + 1
        bitmap_bytes = (witnesses + 7) // 8
        encoded_bytes = RECEIPT_CORE_BYTES + bitmap_bytes + 64 * signatures
        rows.append(
            {
                "kind": "quorum_bytes",
                "f": f,
                "witnesses": witnesses,
                "required_signatures": signatures,
                "minimum_intersection": f + 1,
                "core_bytes": RECEIPT_CORE_BYTES,
                "bitmap_bytes": bitmap_bytes,
                "signature_bytes": 64 * signatures,
                "certificate_bytes": encoded_bytes,
            }
        )
    return rows


def _protocol_case(
    root: Path,
    *,
    name: str,
    after_bias: float,
    epsilon: float,
    gamma: float,
    alpha: float,
    group_count: int | None = None,
    probe_count: int = 1,
) -> dict[str, Any]:
    group_count = group_count or required_groups(alpha, epsilon, gamma)
    policy = EVALUATION_POLICY
    frame_private_key = Ed25519PrivateKey.from_private_bytes(
        hashlib.sha256(f"{name}:frame-authority-key".encode()).digest()
    )
    store_private_key = Ed25519PrivateKey.from_private_bytes(
        hashlib.sha256(f"{name}:probe-store-key".encode()).digest()
    )
    witness_private_keys = tuple(
        Ed25519PrivateKey.from_private_bytes(
            hashlib.sha256(f"{name}:witness-key:{index}".encode()).digest()
        )
        for index in range(4)
    )
    authority = CertificateAuthority.persistent(
        root / "witnesses", f=1, private_keys=witness_private_keys
    )
    trust = VerificationTrust.from_keys(
        authority.public_keys,
        f=1,
        store_public_key=store_private_key.public_key(),
        frame_public_key=frame_private_key.public_key(),
    )
    context = StateContext(
        f"twin-{name}",
        "domain-0",
        7,
        hashlib.sha256(f"{name}:schema".encode()).hexdigest(),
        policy.policy_hash,
        11,
        trust.authority_certificate_hash,
    )
    before = LinearModelArtifact((0.0, 0.0))
    after = LinearModelArtifact((0.0, after_bias))
    probes: list[CommitProbe] = []
    for probe_index in range(probe_count):
        groups = tuple(
            ProbeGroup(
                f"group-{probe_index:02d}-{index:04d}",
                hashlib.sha256(
                    f"{name}:probe:{probe_index}:group:{index}".encode()
                ).hexdigest(),
                ((0.0,),),
                (1,),
            )
            for index in range(group_count)
        )
        probes.append(
            CommitProbe(
                f"probe-{probe_index:02d}",
                context.context_hash,
                policy.policy_hash,
                groups,
                f"2026-01-{probe_index + 1:02d}T00:00:00Z",
                f"2026-01-{probe_index + 2:02d}T00:00:00Z",
                hashlib.sha256(
                    f"{name}:source-handle:{probe_index}".encode()
                ).hexdigest(),
                hashlib.sha256(
                    f"{name}:catalog-opening:{probe_index}".encode()
                ).digest(),
            )
        )
    beacon_private_key = Ed25519PrivateKey.from_private_bytes(
        hashlib.sha256(f"{name}:public-beacon-key".encode()).digest()
    )
    beacon_public_key_bytes = beacon_private_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    beacon_id = f"registered-beacon-{name}"
    beacon_round_number = 1_000 + int(
        hashlib.sha256(f"{name}:beacon-round".encode()).hexdigest()[:6], 16
    )
    signed_beacon_head = sign_beacon_round(
        BeaconRound(
            beacon_id,
            beacon_round_number - 1,
            hashlib.sha256(f"{name}:beacon-grandparent".encode()).hexdigest(),
            hashlib.sha256(f"{name}:authenticated-beacon-head".encode()).digest(),
        ),
        beacon_private_key,
    )
    contributor_root = contributor_merkle_root(
        tuple(
            ContributorLeaf(
                f"client-{index:03d}",
                hashlib.sha256(f"{name}:update:{index}".encode()).hexdigest(),
                0.25,
            )
            for index in range(4)
        )
    )
    score_commitment = hashlib.sha256(f"{name}:score".encode()).hexdigest()
    partition = SourcePartition(
        context.context_hash,
        contributor_root,
        score_commitment,
        policy.policy_hash,
        (hashlib.sha256(f"{name}:proposal-source".encode()).hexdigest(),),
        (hashlib.sha256(f"{name}:score-source".encode()).hexdigest(),),
    )
    frame = SamplingFrame(
        f"frame-{name}",
        context.context_hash,
        policy.policy_hash,
        tuple(
            sorted(
                (probe.frame_entry for probe in probes), key=lambda x: x.probe_id_hash
            )
        ),
        beacon_id,
        hashlib.sha256(beacon_public_key_bytes).hexdigest(),
        (partition.partition_hash,),
        beacon_checkpoint_round=signed_beacon_head.round.round_number,
        beacon_checkpoint_hash=signed_beacon_head.round.round_hash,
    )
    signed_frame = sign_sampling_frame(frame, frame_private_key)
    allocation = RiskAllocation(epsilon, gamma, alpha, group_count)
    schedule = RiskSchedule(
        f"schedule-{name}", context.context_hash, ZERO_HASH, alpha, (allocation,)
    )
    candidate = Candidate(
        context.context_hash,
        context,
        before,
        after,
        contributor_root,
        score_commitment,
        partition,
        policy,
        frame.frame_hash,
        frame.catalog_root,
        tuple(entry.probe_id_hash for entry in frame.entries),
        schedule.schedule_hash,
        0,
        allocation,
        signed_beacon_head.round.round_hash,
        beacon_round_number,
    )
    registry = AuditRegistry(
        root / "audit.sqlite3",
        genesis_model=before,
        initial_context=context,
        evaluation_policy=policy,
        verification_trust=trust,
    )
    registry.provision_lineage_risk_budget(0.9999)
    ledger = RiskLedger(root / "risk.sqlite3")
    ledger.register(schedule, audit_registry=registry)
    ledger.observe_beacon_head(
        signed_beacon_head,
        audit_registry=registry,
        beacon_public_key=beacon_private_key.public_key(),
        signed_frame=signed_frame,
        frame_public_key=frame_private_key.public_key(),
    )
    ledger.consume(
        candidate,
        schedule,
        audit_registry=registry,
        beacon_public_key=beacon_private_key.public_key(),
        signed_frame=signed_frame,
        frame_public_key=frame_private_key.public_key(),
    )
    beacon_service = BeaconService(
        root / "beacon.sqlite3",
        beacon_id=beacon_id,
        checkpoint=signed_beacon_head,
        private_key=beacon_private_key,
        entropy_seed=hashlib.sha256(f"{name}:beacon-entropy".encode()).digest(),
    )
    fixation_reservation = beacon_service.reserve_fixation(
        candidate,
        risk_ledger=ledger,
    )
    signed_beacon_round = beacon_service.finalize_successor(fixation_reservation)
    store = CommitProbeStore(
        probes,
        [partition],
        signed_frame,
        frame_private_key.public_key(),
        root / "probe.sqlite3",
        store_private_key=store_private_key,
    )
    return {
        "context": context,
        "policy": policy,
        "before": before,
        "after": after,
        "probes": probes,
        "frame": frame,
        "signed_frame": signed_frame,
        "frame_private_key": frame_private_key,
        "beacon_private_key": beacon_private_key,
        "beacon_service": beacon_service,
        "signed_beacon_round": signed_beacon_round,
        "signed_beacon_head": signed_beacon_head,
        "partition": partition,
        "allocation": allocation,
        "schedule": schedule,
        "candidate": candidate,
        "registry": registry,
        "ledger": ledger,
        "store": store,
        "authority": authority,
        "trust": trust,
    }


def _execute(case: dict[str, Any]) -> tuple[Any, Receipt, bool]:
    release = case["store"].release(
        case["candidate"],
        signed_beacon_round=case["signed_beacon_round"],
        beacon_public_key=case["beacon_private_key"].public_key(),
        schedule=case["schedule"],
        risk_ledger=case["ledger"],
        audit_registry=case["registry"],
    )
    receipt = case["authority"].issue(
        case["candidate"],
        release,
        store_public_key=case["store"].public_key,
        frame_public_key=case["frame_private_key"].public_key(),
        schedule=case["schedule"],
        risk_ledger=case["ledger"],
        audit_registry=case["registry"],
    )
    appended = case["registry"].verify_and_append(
        receipt,
        case["authority"].public_keys,
        f=1,
        release=release,
        candidate=case["candidate"],
        store_public_key=case["store"].public_key,
        frame_public_key=case["frame_private_key"].public_key(),
        schedule=case["schedule"],
        risk_ledger=case["ledger"],
    )
    return release, receipt, appended


def _reference_vector() -> dict[str, Any]:
    with TemporaryDirectory(prefix="fedmerit-reference-") as directory:
        case = _protocol_case(
            Path(directory),
            name="strict_improvement",
            after_bias=2.0,
            epsilon=0.10,
            gamma=0.05,
            alpha=0.10,
        )
        release, receipt, appended = _execute(case)
        core = receipt.core
        return {
            "schema": "fedmerit/reference-receipt/v1",
            "case": "strict_improvement",
            "fixation_hash": case["candidate"].fixation_hash,
            "release_hash": release.release_hash,
            "receipt_core_bytes": len(core.to_bytes()),
            "receipt_core_hex": core.to_bytes().hex(),
            "receipt_hash": core.receipt_hash,
            "source_group_count": core.source_group_count,
            "delta_hat_hex": float(core.delta_hat).hex(),
            "decision": core.decision,
            "append_accepted": appended,
            "installed_model_hash": case["registry"].installed_model_hash,
        }


def _gate_rows() -> list[dict[str, Any]]:
    cases = (
        ("strict_improvement", 2.0, 0.10, 0.05, 0.10),
        ("no_change", 0.0, 0.10, 0.05, 0.10),
        ("harmful_change", -2.0, 0.10, 0.05, 0.10),
        ("threshold_equality", 80.0, 0.10, 0.25, 0.10),
    )
    rows: list[dict[str, Any]] = []
    with TemporaryDirectory(prefix="fedmerit-gates-") as directory:
        root = Path(directory)
        for index, (name, bias, epsilon, gamma, alpha) in enumerate(cases):
            case = _protocol_case(
                root / str(index),
                name=name,
                after_bias=bias,
                epsilon=epsilon,
                gamma=gamma,
                alpha=alpha,
            )
            release, receipt, appended = _execute(case)
            exact, display, expected = gate_decision(case["candidate"], release.probe)
            installed_expected = (
                case["after"].artifact_hash
                if expected == "commit"
                else case["before"].artifact_hash
            )
            rows.append(
                {
                    "kind": "gate_case",
                    "case": name,
                    "after_bias": bias,
                    "gamma": gamma,
                    "threshold": -gamma,
                    "source_groups": case["allocation"].group_count,
                    "delta_hat": receipt.core.delta_hat,
                    "exact_delta_numerator": exact.numerator,
                    "exact_delta_denominator": exact.denominator,
                    "decision": receipt.core.decision,
                    "decision_rule_verified": (
                        receipt.core.delta_hat == display
                        and receipt.core.decision == expected
                    ),
                    "append_accepted": appended,
                    "installed_model_rule_verified": (
                        case["registry"].installed_model_hash == installed_expected
                    ),
                    "receipt_bytes": len(receipt.to_bytes()),
                }
            )
    return rows


def _handover_rows() -> list[dict[str, Any]]:
    positions = (
        "none",
        "before_release",
        "after_release",
        "after_issue",
        "after_append",
    )
    rows: list[dict[str, Any]] = []
    with TemporaryDirectory(prefix="fedmerit-fences-") as directory:
        root = Path(directory)
        for index, position in enumerate(positions):
            case = _protocol_case(
                root / str(index),
                name=f"fence-{position}",
                after_bias=2.0,
                epsilon=0.10,
                gamma=0.05,
                alpha=0.10,
            )
            registry = case["registry"]
            def handover_now() -> None:
                """Build the successor against the model installed at handover."""

                successor = StateContext(
                    case["context"].twin_id,
                    f"successor-{position}",
                    case["context"].state_version + 1,
                    hashlib.sha256(
                        f"successor-schema:{position}".encode()
                    ).hexdigest(),
                    case["policy"].policy_hash,
                    registry.installed_model_version,
                    case["trust"].authority_certificate_hash,
                )
                registry.handover(
                    state_context=successor,
                    evaluation_policy=case["policy"],
                    authorizer=case["authority"],
                )

            if position == "before_release":
                handover_now()
            release = case["store"].release(
                case["candidate"],
                signed_beacon_round=case["signed_beacon_round"],
                beacon_public_key=case["beacon_private_key"].public_key(),
                schedule=case["schedule"],
                risk_ledger=case["ledger"],
                audit_registry=case["registry"],
            )
            if position == "after_release":
                handover_now()
            receipt = None
            try:
                receipt = case["authority"].issue(
                    case["candidate"],
                    release,
                    store_public_key=case["store"].public_key,
                    frame_public_key=case["frame_private_key"].public_key(),
                    schedule=case["schedule"],
                    risk_ledger=case["ledger"],
                    audit_registry=registry,
                )
            except ValueError:
                pass
            receipt_expected = position not in ("before_release", "after_release")
            if position == "after_issue" and receipt is not None:
                handover_now()
            appended = False
            if receipt is not None:
                appended = registry.verify_and_append(
                    receipt,
                    case["authority"].public_keys,
                    f=1,
                    release=release,
                    candidate=case["candidate"],
                    store_public_key=case["store"].public_key,
                    frame_public_key=case["frame_private_key"].public_key(),
                    schedule=case["schedule"],
                    risk_ledger=case["ledger"],
                )
            if position == "after_append":
                handover_now()
            append_should_succeed = position in ("none", "after_append")
            expected_model = (
                case["after"].artifact_hash
                if append_should_succeed
                else case["before"].artifact_hash
            )
            expected_head = (
                receipt.receipt_hash
                if append_should_succeed and receipt is not None
                else ZERO_HASH
            )
            rows.append(
                {
                    "kind": "state_fence_schedule",
                    "handover_position": position,
                    "receipt_issued": receipt is not None,
                    "receipt_expected": receipt_expected,
                    "append_accepted": appended,
                    "append_expected": append_should_succeed,
                    "model_state_preserved": registry.installed_model_hash
                    == expected_model,
                    "receipt_head_preserved": registry.head == expected_head,
                    "schedule_passed": (
                        (receipt is not None) == receipt_expected
                        and appended == append_should_succeed
                        and registry.installed_model_hash == expected_model
                        and registry.head == expected_head
                    ),
                }
            )
    return rows


def _integrity_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with TemporaryDirectory(prefix="fedmerit-integrity-") as directory:
        case = _protocol_case(
            Path(directory),
            name="integrity",
            after_bias=2.0,
            epsilon=0.10,
            gamma=0.05,
            alpha=0.10,
            probe_count=2,
        )
        release = case["store"].release(
            case["candidate"],
            signed_beacon_round=case["signed_beacon_round"],
            beacon_public_key=case["beacon_private_key"].public_key(),
            schedule=case["schedule"],
            risk_ledger=case["ledger"],
            audit_registry=case["registry"],
        )
        recovered = case["store"].release(
            case["candidate"],
            signed_beacon_round=case["signed_beacon_round"],
            beacon_public_key=case["beacon_private_key"].public_key(),
            schedule=case["schedule"],
            risk_ledger=case["ledger"],
            audit_registry=case["registry"],
        )
        retry_identical = (
            release.release_hash == recovered.release_hash
            and release.signed_beacon_round == recovered.signed_beacon_round
            and release.beacon_public_key == recovered.beacon_public_key
            and release.eligible_probe_id_hashes == recovered.eligible_probe_id_hashes
            and release.signature == recovered.signature
        )
        rows.append(
            {
                "kind": "integrity_check",
                "check": "release_retry_idempotent",
                "passed": retry_identical,
            }
        )
        selected_id = release.public_release.selected_catalog_entry.probe_id_hash
        public_subset_rejected = not verify_public_release(
            replace(
                release.public_release,
                eligible_probe_id_hashes=(selected_id,),
            ),
            case["candidate"],
            case["store"].public_key,
            case["frame_private_key"].public_key(),
        )
        retirement_case = _protocol_case(
            Path(directory) / "concurrent-retirement",
            name="concurrent-retirement",
            after_bias=2.0,
            epsilon=0.10,
            gamma=0.05,
            alpha=0.10,
            probe_count=2,
        )
        with sqlite3.connect(retirement_case["store"].path) as db:
            db.execute(
                "UPDATE probes SET consumed=1 WHERE probe_id=?",
                (retirement_case["probes"][0].probe_id,),
            )
        concurrent_retirement_rejected = False
        try:
            retirement_case["store"].release(
                retirement_case["candidate"],
                signed_beacon_round=retirement_case["signed_beacon_round"],
                beacon_public_key=retirement_case["beacon_private_key"].public_key(),
                schedule=retirement_case["schedule"],
                risk_ledger=retirement_case["ledger"],
                audit_registry=retirement_case["registry"],
            )
        except ValueError:
            concurrent_retirement_rejected = True
        rows.append(
            {
                "kind": "integrity_check",
                "check": "post_beacon_eligible_subset_rejected",
                "public_subset_rejected": public_subset_rejected,
                "concurrent_retirement_rejected": concurrent_retirement_rejected,
                "passed": public_subset_rejected and concurrent_retirement_rejected,
            }
        )
        reassigned_probes = [
            replace(case["probes"][0], groups=case["probes"][1].groups),
            replace(case["probes"][1], groups=case["probes"][0].groups),
        ]
        catalog_reassignment_rejected = False
        try:
            CommitProbeStore(
                reassigned_probes,
                [case["partition"]],
                case["signed_frame"],
                case["frame_private_key"].public_key(),
                Path(directory) / "catalog-reassignment.sqlite3",
            )
        except ValueError:
            catalog_reassignment_rejected = True
        rows.append(
            {
                "kind": "integrity_check",
                "check": "catalog_reassignment_rejected",
                "passed": catalog_reassignment_rejected,
            }
        )
        original_entry = case["frame"].entries[0]
        tampered_entry = replace(
            original_entry,
            payload_commitment=hashlib.sha256(b"catalog-tamper").hexdigest(),
        )
        tampered_entries = tuple(
            tampered_entry if entry == original_entry else entry
            for entry in case["frame"].entries
        )
        tampered_catalog = replace(
            case["signed_frame"],
            frame=replace(case["frame"], entries=tampered_entries),
        )
        rows.append(
            {
                "kind": "integrity_check",
                "check": "catalog_tamper_rejected",
                "passed": not verify_sampling_frame(
                    tampered_catalog, case["frame_private_key"].public_key()
                ),
            }
        )
        mutation_case = _protocol_case(
            Path(directory) / "post-fixation-mutation",
            name="post-fixation-mutation",
            after_bias=2.0,
            epsilon=0.10,
            gamma=0.05,
            alpha=0.10,
            probe_count=2,
        )
        mutation_target = mutation_case["probes"][0]
        mutation_case["store"]._probes[mutation_target.probe_id] = replace(
            mutation_target, groups=mutation_case["probes"][1].groups
        )
        post_fixation_mutation_rejected = False
        try:
            mutation_case["store"].release(
                mutation_case["candidate"],
                signed_beacon_round=mutation_case["signed_beacon_round"],
                beacon_public_key=mutation_case["beacon_private_key"].public_key(),
                schedule=mutation_case["schedule"],
                risk_ledger=mutation_case["ledger"],
                audit_registry=mutation_case["registry"],
            )
        except ValueError:
            post_fixation_mutation_rejected = True
        rows.append(
            {
                "kind": "integrity_check",
                "check": "post_fixation_payload_mutation_rejected",
                "passed": post_fixation_mutation_rejected,
            }
        )
        late_fixation_rejected = False
        late_ledger = RiskLedger(Path(directory) / "late-risk.sqlite3")
        late_ledger.register(case["schedule"], audit_registry=case["registry"])
        try:
            late_ledger.observe_beacon_head(
                case["signed_beacon_round"],
                audit_registry=case["registry"],
                beacon_public_key=case["beacon_private_key"].public_key(),
                signed_frame=case["signed_frame"],
                frame_public_key=case["frame_private_key"].public_key(),
            )
            late_ledger.consume(
                case["candidate"],
                case["schedule"],
                audit_registry=case["registry"],
                beacon_public_key=case["beacon_private_key"].public_key(),
                signed_frame=case["signed_frame"],
                frame_public_key=case["frame_private_key"].public_key(),
            )
        except ValueError:
            late_fixation_rejected = True
        restarted_late_ledger = RiskLedger(Path(directory) / "late-risk.sqlite3")
        backdated_head_rejected = False
        try:
            restarted_late_ledger.observe_beacon_head(
                case["signed_beacon_head"],
                audit_registry=case["registry"],
                beacon_public_key=case["beacon_private_key"].public_key(),
                signed_frame=case["signed_frame"],
                frame_public_key=case["frame_private_key"].public_key(),
            )
        except ValueError:
            backdated_head_rejected = True
        tampered_randomness = (
            bytes([release.signed_beacon_round.round.randomness[0] ^ 1])
            + release.signed_beacon_round.round.randomness[1:]
        )
        tampered_beacon_release = replace(
            release,
            signed_beacon_round=replace(
                release.signed_beacon_round,
                round=replace(
                    release.signed_beacon_round.round,
                    randomness=tampered_randomness,
                ),
            ),
        )
        rows.extend(
            [
                {
                    "kind": "integrity_check",
                    "check": "sampling_frame_signature_verified",
                    "passed": verify_sampling_frame(
                        case["signed_frame"],
                        case["frame_private_key"].public_key(),
                    ),
                },
                {
                    "kind": "integrity_check",
                    "check": "post_fixation_beacon_draw_verified",
                    "passed": (
                        verify_release(
                            release,
                            case["candidate"],
                            case["store"].public_key,
                            case["frame_private_key"].public_key(),
                        )
                        and not verify_release(
                            tampered_beacon_release,
                            case["candidate"],
                            case["store"].public_key,
                            case["frame_private_key"].public_key(),
                        )
                    ),
                },
                {
                    "kind": "integrity_check",
                    "check": "pre_release_selection_contract_verified",
                    "future_round_head_rejected": late_fixation_rejected,
                    "backdated_head_after_restart_rejected": backdated_head_rejected,
                    "passed": late_fixation_rejected
                    and backdated_head_rejected
                    and case["candidate"].sealed_catalog_root
                    == case["frame"].catalog_root
                    and all(
                        hasattr(entry, "payload_commitment")
                        and not hasattr(entry, "groups")
                        and not hasattr(entry, "sealing_nonce")
                        for entry in case["frame"].entries
                    ),
                },
                {
                    "kind": "integrity_check",
                    "check": "score_commitment_separated_from_commit_probe",
                    "passed": (
                        case["candidate"].score_probe_commitment
                        != release.commit_probe_commitment
                    ),
                },
                {
                    "kind": "integrity_check",
                    "check": "signed_source_partition_inventory_verified",
                    "passed": (
                        case["frame"].source_partition_hashes
                        == (case["partition"].partition_hash,)
                        and not set(
                            case["partition"].source_manifest_hashes
                        ).intersection(
                            case["partition"].score_source_manifest_hashes
                        )
                    ),
                },
                {
                    "kind": "integrity_check",
                    "check": "public_release_uses_merkle_membership_path",
                    "passed": (
                        verify_public_release(
                            release.public_release,
                            case["candidate"],
                            case["store"].public_key,
                            case["frame_private_key"].public_key(),
                        )
                        and not hasattr(
                            release.public_release.signed_sampling_frame.commitment,
                            "entries",
                        )
                        and bool(release.public_release.catalog_membership_path)
                    ),
                },
                {
                    "kind": "integrity_check",
                    "check": "evaluation_policy_registered",
                    "passed": case["registry"].evaluation_policy_registered(
                        case["policy"]
                    ),
                },
            ]
        )

        wrong_context = StateContext(
            case["context"].twin_id,
            "wrong-domain",
            case["context"].state_version + 1,
            hashlib.sha256(b"wrong-schema").hexdigest(),
            case["policy"].policy_hash,
            case["context"].model_version,
            hashlib.sha256(b"wrong-authority").hexdigest(),
        )
        provisioning_mismatch_rejected = False
        try:
            AuditRegistry(
                Path(directory) / "audit.sqlite3",
                genesis_model=case["before"],
                initial_context=wrong_context,
                evaluation_policy=case["policy"],
            )
        except ValueError:
            provisioning_mismatch_rejected = True
        rows.append(
            {
                "kind": "integrity_check",
                "check": "initial_context_mismatch_rejected",
                "passed": provisioning_mismatch_rejected,
            }
        )

        tampered_signature = (
            bytes([case["signed_frame"].signature[0] ^ 1])
            + case["signed_frame"].signature[1:]
        )
        tampered_frame = replace(case["signed_frame"], signature=tampered_signature)
        tampered_release = replace(release, signed_sampling_frame=tampered_frame)
        rows.append(
            {
                "kind": "integrity_check",
                "check": "tampered_sampling_frame_rejected",
                "passed": not verify_release(
                    tampered_release,
                    case["candidate"],
                    case["store"].public_key,
                    case["frame_private_key"].public_key(),
                ),
            }
        )

        receipt = case["authority"].issue(
            case["candidate"],
            release,
            store_public_key=case["store"].public_key,
            frame_public_key=case["frame_private_key"].public_key(),
            schedule=case["schedule"],
            risk_ledger=case["ledger"],
            audit_registry=case["registry"],
        )
        short_receipt = Receipt(
            receipt.core, receipt.witness_count, receipt.signatures[:-1]
        )
        rows.append(
            {
                "kind": "integrity_check",
                "check": "subquorum_rejected",
                "passed": not verify_receipt(
                    short_receipt,
                    case["authority"].public_keys,
                    f=1,
                    release=release,
                    candidate=case["candidate"],
                    store_public_key=case["store"].public_key,
                    frame_public_key=case["frame_private_key"].public_key(),
                    schedule=case["schedule"],
                    risk_ledger=case["ledger"],
                    audit_registry=case["registry"],
                ),
            }
        )
        payload = bytearray(receipt.to_bytes())
        payload[40] ^= 0x01
        rows.append(
            {
                "kind": "integrity_check",
                "check": "tampered_core_rejected",
                "passed": not verify_receipt_bytes(
                    bytes(payload),
                    witness_count=receipt.witness_count,
                    public_keys=case["authority"].public_keys,
                    f=1,
                    release=release,
                    candidate=case["candidate"],
                    store_public_key=case["store"].public_key,
                    frame_public_key=case["frame_private_key"].public_key(),
                    schedule=case["schedule"],
                    risk_ledger=case["ledger"],
                    audit_registry=case["registry"],
                ),
            }
        )

        conflicting = replace(
            case["candidate"],
            after_model=LinearModelArtifact((0.0, 1.5)),
        )
        risk_reuse_blocked = False
        try:
            case["store"].release(
                conflicting,
                signed_beacon_round=case["signed_beacon_round"],
                beacon_public_key=case["beacon_private_key"].public_key(),
                schedule=case["schedule"],
                risk_ledger=case["ledger"],
                audit_registry=case["registry"],
            )
        except ValueError:
            risk_reuse_blocked = True
        rows.append(
            {
                "kind": "integrity_check",
                "check": "risk_index_reuse_blocked",
                "passed": risk_reuse_blocked,
            }
        )

        initial_append = case["registry"].verify_and_append(
            receipt,
            case["authority"].public_keys,
            f=1,
            release=release,
            candidate=case["candidate"],
            store_public_key=case["store"].public_key,
            frame_public_key=case["frame_private_key"].public_key(),
            schedule=case["schedule"],
            risk_ledger=case["ledger"],
        )
        appended_head = case["registry"].head
        appended_model = case["registry"].installed_model_hash
        successor = StateContext(
            case["context"].twin_id,
            "successor-integrity",
            case["context"].state_version + 1,
            hashlib.sha256(b"successor-integrity-schema").hexdigest(),
            case["policy"].policy_hash,
            case["context"].model_version + 1,
            case["trust"].authority_certificate_hash,
        )
        case["registry"].handover(
            state_context=successor,
            evaluation_policy=case["policy"],
            authorizer=case["authority"],
        )
        historical_replay = case["registry"].verify_and_append(
            receipt,
            case["authority"].public_keys,
            f=1,
            release=release,
            candidate=case["candidate"],
            store_public_key=case["store"].public_key,
            frame_public_key=case["frame_private_key"].public_key(),
            schedule=case["schedule"],
            risk_ledger=case["ledger"],
        )
        rows.append(
            {
                "kind": "integrity_check",
                "check": "historical_append_idempotent_after_handover",
                "passed": (
                    initial_append
                    and historical_replay
                    and case["registry"].head == appended_head
                    and case["registry"].installed_model_hash == appended_model
                    and case["registry"].context_head
                    == (successor.context_hash, successor.authority_certificate_hash)
                ),
            }
        )

        genesis_case = _protocol_case(
            Path(directory) / "genesis-byte-binding",
            name="genesis-byte-binding",
            after_bias=2.0,
            epsilon=0.10,
            gamma=0.05,
            alpha=0.10,
        )
        genesis_exact = genesis_case["registry"].serving_model_snapshot == (
            genesis_case["before"].artifact_hash,
            genesis_case["context"].model_version,
            genesis_case["before"].artifact_bytes,
        )
        with sqlite3.connect(genesis_case["registry"].path) as db:
            db.execute(
                "UPDATE serving_model SET artifact_blob=? WHERE id=1",
                (b"tampered-genesis",),
            )
        tampered_genesis_rejected = False
        try:
            AuditRegistry(
                genesis_case["registry"].path,
                genesis_model=genesis_case["before"],
                initial_context=genesis_case["context"],
                evaluation_policy=genesis_case["policy"],
                verification_trust=genesis_case["trust"],
            )
        except ValueError:
            tampered_genesis_rejected = True
        rows.append(
            {
                "kind": "integrity_check",
                "check": "exact_genesis_artifact_bound",
                "passed": genesis_exact and tampered_genesis_rejected,
            }
        )

        rotated_epoch = replace(case["trust"], roster_epoch=1)
        altered_authority_rejected = False
        try:
            case["registry"].provision_verification_trust(rotated_epoch)
        except ValueError:
            altered_authority_rejected = True
        rows.append(
            {
                "kind": "integrity_check",
                "check": "authority_roster_epoch_bound",
                "passed": (
                    case["context"].authority_certificate_hash
                    == case["trust"].authority_certificate_hash
                    and altered_authority_rejected
                ),
            }
        )

        lineage_model = LinearModelArtifact((0.0, 0.0))
        lineage_context = StateContext(
            "trace-lineage",
            "domain-0",
            0,
            hashlib.sha256(b"trace-lineage-schema-0").hexdigest(),
            EVALUATION_POLICY.policy_hash,
            0,
            hashlib.sha256(b"trace-lineage-authority-0").hexdigest(),
        )
        lineage_registry = AuditRegistry(
            Path(directory) / "lineage-budget.sqlite3",
            genesis_model=lineage_model,
            initial_context=lineage_context,
            evaluation_policy=EVALUATION_POLICY,
        )
        lineage_registry.provision_lineage_risk_budget(0.15)
        lineage_registry.register_risk_schedule(
            RiskSchedule(
                "trace-lineage-schedule-0",
                lineage_context.context_hash,
                ZERO_HASH,
                0.10,
                (RiskAllocation(0.25, 0.05, 0.10, 1),),
            )
        )
        lineage_successor = StateContext(
            lineage_context.twin_id,
            "domain-1",
            1,
            hashlib.sha256(b"trace-lineage-schema-1").hexdigest(),
            EVALUATION_POLICY.policy_hash,
            0,
            hashlib.sha256(b"trace-lineage-authority-1").hexdigest(),
        )
        lineage_registry.handover(
            state_context=lineage_successor,
            evaluation_policy=EVALUATION_POLICY,
        )
        cross_handover_budget_rejected = False
        try:
            lineage_registry.register_risk_schedule(
                RiskSchedule(
                    "trace-lineage-schedule-1",
                    lineage_successor.context_hash,
                    ZERO_HASH,
                    0.10,
                    (RiskAllocation(0.25, 0.05, 0.10, 1),),
                )
            )
        except ValueError:
            cross_handover_budget_rejected = True
        rows.append(
            {
                "kind": "integrity_check",
                "check": "cross_handover_budget_rejected",
                "passed": cross_handover_budget_rejected,
            }
        )
    return rows


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    risk = [row for row in rows if row["kind"] == "risk_budget"]
    quorum = [row for row in rows if row["kind"] == "quorum_bytes"]
    gates = [row for row in rows if row["kind"] == "gate_case"]
    fences = [row for row in rows if row["kind"] == "state_fence_schedule"]
    integrity = [row for row in rows if row["kind"] == "integrity_check"]
    return {
        "schema": "fedmerit/certificate-evidence/v3",
        "status": "deterministic_certificate_evidence",
        "paper_input": True,
        "evidence_class": "deterministic_calculation_and_executable_conformance",
        "risk_budget_grid": risk,
        "quorum_certificate_grid": quorum,
        "gate_operating_cases": gates,
        "state_fence_schedules": fences,
        "integrity_checks": integrity,
        "summary": {
            "raw_record_count": len(rows),
            "risk_grid_rows": len(risk),
            "risk_minimality_checks_passed": sum(
                bool(row["minimality_verified"]) for row in risk
            ),
            "quorum_grid_rows": len(quorum),
            "gate_cases": len(gates),
            "gate_cases_passed": sum(
                bool(row["decision_rule_verified"])
                and bool(row["installed_model_rule_verified"])
                and bool(row["append_accepted"])
                for row in gates
            ),
            "state_fence_schedules": len(fences),
            "state_fence_schedules_passed": sum(
                bool(row["schedule_passed"]) for row in fences
            ),
            "handover_before_append_cases": 3,
            "handover_before_append_blocked": sum(
                not bool(row["append_accepted"])
                for row in fences
                if row["handover_position"]
                in ("before_release", "after_release", "after_issue")
            ),
            "handover_before_quorum_cases": 2,
            "handover_before_quorum_blocked": sum(
                not bool(row["receipt_issued"])
                for row in fences
                if row["handover_position"] in ("before_release", "after_release")
            ),
            "integrity_checks": len(integrity),
            "integrity_checks_passed": sum(bool(row["passed"]) for row in integrity),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Produce deterministic FedMERIT evaluation evidence"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output",
        help="output directory",
    )
    args = parser.parse_args(argv)
    output = args.output.resolve()
    raw_trace = output / "raw_trace.jsonl"
    metrics_path = output / "metrics.json"
    reference_path = output / "reference_receipt.json"
    manifest_path = output / "manifest.json"
    output.mkdir(parents=True, exist_ok=True)
    rows = (
        _risk_rows()
        + _quorum_rows()
        + _gate_rows()
        + _handover_rows()
        + _integrity_rows()
    )
    metrics = _metrics(rows)
    summary = metrics["summary"]
    expected_passes = (
        summary["risk_minimality_checks_passed"] == summary["risk_grid_rows"]
        and summary["gate_cases_passed"] == summary["gate_cases"]
        and summary["state_fence_schedules_passed"] == summary["state_fence_schedules"]
        and summary["integrity_checks_passed"] == summary["integrity_checks"]
    )
    if not expected_passes:
        raise RuntimeError(
            "deterministic evaluation did not satisfy every declared check"
        )
    _write_jsonl(raw_trace, rows)
    _write_json(metrics_path, metrics)
    reference = _reference_vector()
    if REFERENCE_VECTOR.exists():
        expected_reference = read_json_object(REFERENCE_VECTOR)
        if reference != expected_reference:
            raise RuntimeError(
                "known-answer receipt differs from the checked-in vector"
            )
    _write_json(reference_path, reference)
    producer_paths = [
        Path(__file__).resolve(),
        CONFIG,
        REFERENCE_VECTOR,
        *sorted((ROOT / "fedmerit").glob("*.py")),
    ]
    producer_files = {
        str(path.relative_to(ROOT)): _sha256(path) for path in producer_paths
    }
    manifest = {
        "schema": "fedmerit/evidence-manifest/v4",
        "status": "deterministic_certificate_evidence",
        "paper_input": True,
        "evidence_class": "deterministic_calculation_and_executable_conformance",
        "claim_scope": (
            "Certified rational risk enclosures, exact encoding calculations, "
            "and deterministic pass/fail outcomes of the reference "
            "certification state machine."
        ),
        "evaluation_policy_hash": EVALUATION_POLICY.policy_hash,
        "config_path": "configs/benchmark_protocol.json",
        "config_sha256": _sha256(CONFIG),
        "raw_trace_path": "raw_trace.jsonl",
        "raw_trace_sha256": _sha256(raw_trace),
        "metrics_path": "metrics.json",
        "metrics_sha256": _sha256(metrics_path),
        "reference_vector_path": "reference_receipt.json",
        "reference_vector_sha256": _sha256(reference_path),
        "producer_path": "scripts/produce_evidence.py",
        "producer_sha256": _sha256(Path(__file__).resolve()),
        "producer_files_sha256": producer_files,
        "producer_bundle_sha256": hashlib.sha256(
            json.dumps(producer_files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "execution": {
            "command": (
                "PYTHONPATH=. python3 scripts/produce_evidence.py --output <directory>"
            ),
            "deterministic_outputs": True,
            "seeds": [],
            "runtime_observed": {
                "implementation": platform.python_implementation(),
                "python_version": platform.python_version(),
                "cryptography_version": version("cryptography"),
            },
        },
        "metric_bindings": [
            {"paper_object": "finite-risk grid", "metrics_key": "risk_budget_grid"},
            {
                "paper_object": "quorum/certificate grid",
                "metrics_key": "quorum_certificate_grid",
            },
            {"paper_object": "gate cases", "metrics_key": "gate_operating_cases"},
            {
                "paper_object": "state-fence schedules",
                "metrics_key": "state_fence_schedules",
            },
            {"paper_object": "integrity checks", "metrics_key": "integrity_checks"},
        ],
    }
    _write_json(manifest_path, manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

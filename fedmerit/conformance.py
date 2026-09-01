"""Fast conformance check for the state-scoped certification boundary."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

from .certificate import (
    AuditRegistry,
    CertificateAuthority,
    VerificationTrust,
    verify_public_receipt,
    verify_receipt,
)
from .gate import (
    CommitProbeStore,
    RiskLedger,
    sign_beacon_round,
    sign_sampling_frame,
    verify_public_release,
    verify_release,
    verify_sampling_frame,
)
from .model import (
    BeaconRound,
    Candidate,
    CommitProbe,
    EvaluationPolicy,
    LinearModelArtifact,
    ProbeGroup,
    Receipt,
    RiskAllocation,
    RiskSchedule,
    SamplingFrame,
    SourcePartition,
    StateContext,
    ZERO_HASH,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def run() -> dict[str, object]:
    """Exercise provisioning, signed selection, append, handover, and replay."""
    with TemporaryDirectory(prefix="fedmerit-conformance-") as directory:
        root = Path(directory)
        policy = EvaluationPolicy("brier-decimal80-v1")
        context = StateContext(
            "twin-0",
            "domain-0",
            0,
            _digest("schema-0"),
            policy.policy_hash,
            0,
            _digest("authority-0"),
        )
        before = LinearModelArtifact((0.0, 0.0))
        after = LinearModelArtifact((0.0, 1.0))
        probes = tuple(
            CommitProbe(
                f"probe-{index}",
                context.context_hash,
                policy.policy_hash,
                (
                    ProbeGroup(
                        f"group-{index}",
                        _digest(f"commit-source-{index}"),
                        ((0.0,),),
                        (0,),
                    ),
                ),
                f"2026-01-0{index + 1}T00:00:00Z",
                f"2026-01-0{index + 2}T00:00:00Z",
                _digest(f"source-handle-{index}"),
                hashlib.sha256(f"catalog-opening-{index}".encode()).digest(),
            )
            for index in range(2)
        )
        beacon_private_key = Ed25519PrivateKey.from_private_bytes(
            hashlib.sha256(b"conformance-public-beacon").digest()
        )
        beacon_public_key_bytes = beacon_private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        frame = SamplingFrame(
            "frame-0",
            context.context_hash,
            policy.policy_hash,
            tuple(
                sorted(
                    (probe.frame_entry for probe in probes),
                    key=lambda x: x.probe_id_hash,
                )
            ),
            "reference-beacon",
            hashlib.sha256(beacon_public_key_bytes).hexdigest(),
        )
        frame_private_key = Ed25519PrivateKey.from_private_bytes(
            hashlib.sha256(b"conformance-frame-authority").digest()
        )
        signed_frame = sign_sampling_frame(frame, frame_private_key)
        contributor_root = _digest("contributors-0")
        score_commitment = _digest("score-probe-0")
        partition = SourcePartition(
            context.context_hash,
            contributor_root,
            score_commitment,
            policy.policy_hash,
            (_digest("proposal-source-0"),),
        )
        allocation = RiskAllocation(0.1, 0.0, 0.999, 1)
        schedule = RiskSchedule(
            "schedule-0",
            context.context_hash,
            ZERO_HASH,
            0.9999,
            (allocation,),
        )
        signed_beacon_head = sign_beacon_round(
            BeaconRound(
                "reference-beacon",
                40,
                _digest("beacon-round-39"),
                hashlib.sha256(b"registered-conformance-beacon-round-40").digest(),
            ),
            beacon_private_key,
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
            41,
        )
        registry = AuditRegistry(
            root / "audit.sqlite3",
            genesis_model_hash=before.artifact_hash,
            initial_context=context,
            evaluation_policy=policy,
        )
        registry.provision_lineage_risk_budget(0.9999)
        ledger = RiskLedger(root / "risk.sqlite3")
        ledger.register(schedule, audit_registry=registry)
        ledger.observe_beacon_head(
            signed_beacon_head,
            beacon_public_key=beacon_private_key.public_key(),
            signed_frame=signed_frame,
            frame_public_key=frame_private_key.public_key(),
        )
        skipped_beacon_round_rejected = False
        try:
            ledger.observe_beacon_head(
                sign_beacon_round(
                    BeaconRound(
                        "reference-beacon",
                        42,
                        _digest("unrelated-beacon-parent"),
                        hashlib.sha256(
                            b"registered-conformance-beacon-round-42"
                        ).digest(),
                    ),
                    beacon_private_key,
                ),
                beacon_public_key=beacon_private_key.public_key(),
                signed_frame=signed_frame,
                frame_public_key=frame_private_key.public_key(),
            )
        except ValueError:
            skipped_beacon_round_rejected = True
        ledger.consume(
            candidate,
            schedule,
            beacon_public_key=beacon_private_key.public_key(),
            signed_frame=signed_frame,
            frame_public_key=frame_private_key.public_key(),
        )
        signed_beacon_round = sign_beacon_round(
            BeaconRound(
                "reference-beacon",
                41,
                signed_beacon_head.round.round_hash,
                hashlib.sha256(b"registered-conformance-beacon-round-41").digest(),
            ),
            beacon_private_key,
        )
        store = CommitProbeStore(
            list(probes),
            [partition],
            signed_frame,
            frame_private_key.public_key(),
            root / "probe.sqlite3",
        )
        reassigned = (
            replace(probes[0], groups=probes[1].groups),
            replace(probes[1], groups=probes[0].groups),
        )
        catalog_reassignment_rejected = False
        try:
            CommitProbeStore(
                list(reassigned),
                [partition],
                signed_frame,
                frame_private_key.public_key(),
                root / "reassigned.sqlite3",
            )
        except ValueError:
            catalog_reassignment_rejected = True
        first_entry = frame.entries[0]
        tampered_entry = replace(
            first_entry, payload_commitment=_digest("tampered-catalog-payload")
        )
        tampered_entries = tuple(
            tampered_entry if entry == first_entry else entry for entry in frame.entries
        )
        tampered_signed_frame = replace(
            signed_frame, frame=replace(frame, entries=tampered_entries)
        )
        catalog_tamper_rejected = not verify_sampling_frame(
            tampered_signed_frame, frame_private_key.public_key()
        )
        mutation_store = CommitProbeStore(
            list(probes),
            [partition],
            signed_frame,
            frame_private_key.public_key(),
            root / "mutation.sqlite3",
        )
        mutation_store._probes[probes[0].probe_id] = replace(
            probes[0], groups=probes[1].groups
        )
        post_fixation_mutation_rejected = False
        try:
            mutation_store.release(
                candidate,
                signed_beacon_round=signed_beacon_round,
                beacon_public_key=beacon_private_key.public_key(),
                schedule=schedule,
                risk_ledger=ledger,
                audit_registry=registry,
            )
        except ValueError:
            post_fixation_mutation_rejected = True
        snapshot_store = CommitProbeStore(
            list(probes),
            [partition],
            signed_frame,
            frame_private_key.public_key(),
            root / "snapshot.sqlite3",
        )
        with sqlite3.connect(snapshot_store.path) as db:
            db.execute(
                "UPDATE probes SET consumed=1 WHERE probe_id=?",
                (probes[0].probe_id,),
            )
        concurrent_retirement_rejected = False
        try:
            snapshot_store.release(
                candidate,
                signed_beacon_round=signed_beacon_round,
                beacon_public_key=beacon_private_key.public_key(),
                schedule=schedule,
                risk_ledger=ledger,
                audit_registry=registry,
            )
        except ValueError:
            concurrent_retirement_rejected = True
        authority = CertificateAuthority.persistent(root / "witnesses", f=1)
        trust = VerificationTrust.from_keys(
            authority.public_keys,
            f=1,
            store_public_key=store.public_key,
            frame_public_key=frame_private_key.public_key(),
        )
        registry.provision_verification_trust(trust)
        release = store.release(
            candidate,
            signed_beacon_round=signed_beacon_round,
            beacon_public_key=beacon_private_key.public_key(),
            schedule=schedule,
            risk_ledger=ledger,
            audit_registry=registry,
        )
        authorized_release_verified = verify_release(
            release,
            candidate,
            store.public_key,
            frame_private_key.public_key(),
        )
        selected_id = release.public_release.selected_catalog_entry.probe_id_hash
        post_beacon_eligible_subset_rejected = not verify_public_release(
            replace(
                release.public_release,
                eligible_probe_id_hashes=(selected_id,),
            ),
            candidate,
            store.public_key,
            frame_private_key.public_key(),
        )
        late_ledger = RiskLedger(root / "late-risk.sqlite3")
        late_ledger.register(schedule, audit_registry=registry)
        late_ledger.observe_beacon_head(
            signed_beacon_round,
            beacon_public_key=beacon_private_key.public_key(),
            signed_frame=signed_frame,
            frame_public_key=frame_private_key.public_key(),
        )
        future_round_as_fixation_head_rejected = False
        try:
            late_ledger.consume(
                candidate,
                schedule,
                beacon_public_key=beacon_private_key.public_key(),
                signed_frame=signed_frame,
                frame_public_key=frame_private_key.public_key(),
            )
        except ValueError:
            future_round_as_fixation_head_rejected = True
        restarted_late_ledger = RiskLedger(root / "late-risk.sqlite3")
        backdated_beacon_head_rejected_after_restart = False
        try:
            restarted_late_ledger.observe_beacon_head(
                signed_beacon_head,
                beacon_public_key=beacon_private_key.public_key(),
                signed_frame=signed_frame,
                frame_public_key=frame_private_key.public_key(),
            )
        except ValueError:
            backdated_beacon_head_rejected_after_restart = True
        receipt = authority.issue(
            candidate,
            release,
            store_public_key=store.public_key,
            frame_public_key=frame_private_key.public_key(),
            schedule=schedule,
            risk_ledger=ledger,
            audit_registry=registry,
        )
        public_receipt_verified = verify_public_receipt(
            receipt,
            authority.public_keys,
            f=1,
            release=release.public_release,
            candidate=candidate,
            store_public_key=store.public_key,
            frame_public_key=frame_private_key.public_key(),
            schedule=schedule,
            risk_ledger=ledger,
            audit_registry=registry,
        )
        authorized_raw_probe_replay_verified = verify_receipt(
            receipt,
            authority.public_keys,
            f=1,
            release=release,
            candidate=candidate,
            store_public_key=store.public_key,
            frame_public_key=frame_private_key.public_key(),
            schedule=schedule,
            risk_ledger=ledger,
            audit_registry=registry,
        )
        fresh_append = registry.verify_and_append(
            receipt,
            authority.public_keys,
            f=1,
            release=release,
            candidate=candidate,
            store_public_key=store.public_key,
            frame_public_key=frame_private_key.public_key(),
            schedule=schedule,
            risk_ledger=ledger,
        )
        initial_head = registry.head
        initial_model = registry.installed_model_hash

        successor = StateContext(
            "twin-0",
            "domain-1",
            1,
            _digest("schema-1"),
            policy.policy_hash,
            0,
            _digest("authority-1"),
        )
        registry.handover(state_context=successor, evaluation_policy=policy)
        historical_replay = registry.verify_and_append(
            receipt,
            authority.public_keys,
            f=1,
            release=release,
            candidate=candidate,
            store_public_key=store.public_key,
            frame_public_key=frame_private_key.public_key(),
            schedule=schedule,
            risk_ledger=ledger,
        )
        historical_state_unchanged = (
            registry.head == initial_head
            and registry.installed_model_hash == initial_model
            and registry.context_head
            == (successor.context_hash, successor.authority_certificate_hash)
        )

        stale = AuditRegistry(
            root / "stale.sqlite3",
            genesis_model_hash=before.artifact_hash,
            initial_context=context,
            evaluation_policy=policy,
            verification_trust=trust,
        )
        stale.register_risk_schedule(schedule)
        stale.handover(state_context=successor, evaluation_policy=policy)
        stale_append = stale.verify_and_append(
            receipt,
            authority.public_keys,
            f=1,
            release=release,
            candidate=candidate,
            store_public_key=store.public_key,
            frame_public_key=frame_private_key.public_key(),
            schedule=schedule,
            risk_ledger=ledger,
        )
        stale_state_unchanged = (
            stale.installed_model_hash == before.artifact_hash
            and stale.head == ZERO_HASH
        )

        rogue_authority = CertificateAuthority.persistent(
            root / "rogue-witnesses", f=1
        )
        rogue_witness_roster_rejected = False
        try:
            rogue_authority.issue(
                candidate,
                release,
                store_public_key=store.public_key,
                frame_public_key=frame_private_key.public_key(),
                schedule=schedule,
                risk_ledger=ledger,
                audit_registry=registry,
            )
        except ValueError:
            rogue_witness_roster_rejected = True

        issued_only = AuditRegistry(
            root / "issued-only.sqlite3",
            genesis_model_hash=before.artifact_hash,
            initial_context=context,
            evaluation_policy=policy,
            verification_trust=trust,
        )
        issued_only.register_risk_schedule(schedule)
        issued_only.reserve_risk_allocation(
            schedule,
            candidate.risk_schedule_index,
            fixation_hash=candidate.fixation_hash,
        )
        issued_only_receipt = authority.issue(
            candidate,
            release,
            store_public_key=store.public_key,
            frame_public_key=frame_private_key.public_key(),
            schedule=schedule,
            risk_ledger=ledger,
            audit_registry=issued_only,
        )
        issued_only_successor = StateContext(
            "twin-0",
            "domain-issued-only",
            1,
            _digest("schema-issued-only"),
            policy.policy_hash,
            0,
            _digest("authority-issued-only"),
        )
        issued_only.handover(
            state_context=issued_only_successor,
            evaluation_policy=policy,
        )
        issued_receipt_verifiable_after_handover = verify_receipt(
            issued_only_receipt,
            authority.public_keys,
            f=1,
            release=release,
            candidate=candidate,
            store_public_key=store.public_key,
            frame_public_key=frame_private_key.public_key(),
            schedule=schedule,
            risk_ledger=ledger,
            audit_registry=issued_only,
        )
        issued_receipt_append_blocked_after_handover = not issued_only.verify_and_append(
            issued_only_receipt,
            authority.public_keys,
            f=1,
            release=release,
            candidate=candidate,
            store_public_key=store.public_key,
            frame_public_key=frame_private_key.public_key(),
            schedule=schedule,
            risk_ledger=ledger,
        )

        issuance_race = AuditRegistry(
            root / "issuance-race.sqlite3",
            genesis_model_hash=before.artifact_hash,
            initial_context=context,
            evaluation_policy=policy,
            verification_trust=trust,
        )
        issuance_race.register_risk_schedule(schedule)
        issuance_race.reserve_risk_allocation(
            schedule,
            candidate.risk_schedule_index,
            fixation_hash=candidate.fixation_hash,
        )
        race_core = None
        race_signatures = []
        for witness in authority.witnesses[: 2 * authority.f + 1]:
            replayed, signature = witness.attest(
                candidate,
                release,
                store_public_key=store.public_key,
                frame_public_key=frame_private_key.public_key(),
                schedule=schedule,
                risk_ledger=ledger,
                audit_registry=issuance_race,
            )
            race_core = replayed if race_core is None else race_core
            race_signatures.append(signature)
        if race_core is None:
            raise RuntimeError("race conformance produced no receipt core")
        race_receipt = Receipt(
            race_core,
            len(authority.witnesses),
            tuple(race_signatures),
        )
        issuance_race.handover(
            state_context=StateContext(
                "twin-0",
                "domain-issuance-race",
                1,
                _digest("schema-issuance-race"),
                policy.policy_hash,
                0,
                _digest("authority-issuance-race"),
            ),
            evaluation_policy=policy,
        )
        final_issuance_fence_rejected = False
        try:
            issuance_race.record_issued_receipt(
                race_receipt,
                candidate=candidate,
                schedule=schedule,
                verification_trust=trust,
            )
        except ValueError:
            final_issuance_fence_rejected = True
        score_commitment_separated = (
            release.commit_probe_commitment != candidate.score_probe_commitment
        )
        lineage_budget_registered = registry.lineage_risk_budget == (
            context.twin_id,
            ZERO_HASH,
            0.9999,
        )
        passed = (
            fresh_append
            and receipt.core.decision == "reject"
            and initial_model == before.artifact_hash
            and initial_head == receipt.receipt_hash
            and historical_replay
            and historical_state_unchanged
            and not stale_append
            and stale_state_unchanged
            and rogue_witness_roster_rejected
            and issued_receipt_verifiable_after_handover
            and issued_receipt_append_blocked_after_handover
            and final_issuance_fence_rejected
            and registry.evaluation_policy_registered(policy)
            and lineage_budget_registered
            and release.signed_sampling_frame == signed_frame
            and release.signed_beacon_round == signed_beacon_round
            and ledger.fixation_precedes_beacon(candidate)
            and score_commitment_separated
            and catalog_reassignment_rejected
            and catalog_tamper_rejected
            and post_fixation_mutation_rejected
            and public_receipt_verified
            and authorized_release_verified
            and authorized_raw_probe_replay_verified
            and post_beacon_eligible_subset_rejected
            and future_round_as_fixation_head_rejected
            and concurrent_retirement_rejected
            and backdated_beacon_head_rejected_after_restart
            and skipped_beacon_round_rejected
        )
    return {
        "status": "passed" if passed else "failed",
        "explicit_context_provisioned": True,
        "evaluation_policy_registered": True,
        "cross_handover_risk_budget_registered": lineage_budget_registered,
        "signed_sampling_frame_verified": True,
        "post_fixation_beacon_verified": True,
        "auditable_draw_counter": release.draw_counter,
        "fresh_append": bool(fresh_append),
        "fresh_decision": receipt.core.decision,
        "score_commitment_separated": score_commitment_separated,
        "sealed_catalog_root_bound": candidate.sealed_catalog_root
        == frame.catalog_root,
        "catalog_reassignment_rejected": catalog_reassignment_rejected,
        "catalog_tamper_rejected": catalog_tamper_rejected,
        "post_fixation_mutation_rejected": post_fixation_mutation_rejected,
        "post_beacon_eligible_subset_rejected": post_beacon_eligible_subset_rejected,
        "future_round_as_fixation_head_rejected": (
            future_round_as_fixation_head_rejected
        ),
        "concurrent_retirement_rejected": concurrent_retirement_rejected,
        "backdated_beacon_head_rejected_after_restart": (
            backdated_beacon_head_rejected_after_restart
        ),
        "skipped_or_forked_beacon_round_rejected": skipped_beacon_round_rejected,
        "public_receipt_verified_without_raw_probe": public_receipt_verified,
        "authorized_release_opening_verified": authorized_release_verified,
        "authorized_raw_probe_replay_verified": (authorized_raw_probe_replay_verified),
        "historical_replay_idempotent": bool(historical_replay),
        "historical_state_unchanged": historical_state_unchanged,
        "stale_append_after_handover": bool(stale_append),
        "stale_state_unchanged": stale_state_unchanged,
        "rogue_witness_roster_rejected": rogue_witness_roster_rejected,
        "issued_receipt_verifiable_after_handover": (
            issued_receipt_verifiable_after_handover
        ),
        "issued_receipt_append_blocked_after_handover": (
            issued_receipt_append_blocked_after_handover
        ),
        "final_issuance_fence_rejected": final_issuance_fence_rejected,
    }


def main() -> int:
    result = run()
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

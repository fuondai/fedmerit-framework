"""Deterministic regression tests for the sealed-catalog boundary."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from fedmerit.canonical import canonical_bytes
from fedmerit.certificate import AuditRegistry, CertificateAuthority, Witness
from fedmerit.conformance import run
from fedmerit.gate import (
    CommitProbeStore,
    RiskLedger,
    required_groups,
    risk_bound_interval,
    risk_is_satisfied,
    sign_beacon_round,
    sign_sampling_frame,
)
from fedmerit.model import (
    BeaconRound,
    Candidate,
    CommitProbe,
    ContributorLeaf,
    EvaluationPolicy,
    LinearModelArtifact,
    ProbeGroup,
    Receipt,
    ReceiptCore,
    RiskAllocation,
    RiskSchedule,
    SamplingFrame,
    SecurityProfile,
    SourcePartition,
    StateContext,
    WitnessSignature,
    ZERO_HASH,
    contributor_merkle_root,
)
from scripts.produce_evidence import _execute, _handover_rows, _protocol_case


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


class ReceiptEncodingTests(unittest.TestCase):
    def test_contributor_root_binds_identity_update_and_weight(self) -> None:
        leaves = (
            ContributorLeaf("client-a", _digest("update-a"), 0.25),
            ContributorLeaf("client-b", _digest("update-b"), 0.75),
        )
        root = contributor_merkle_root(leaves)
        self.assertNotEqual(
            root,
            contributor_merkle_root(
                (leaves[0], replace(leaves[1], weight=0.5))
            ),
        )

    def test_certificate_byte_grid_matches_wire_encoder(self) -> None:
        core = ReceiptCore(
            *(_digest(f"receipt-field-{index}") for index in range(10)),
            1,
            0.25,
            0.05,
            0.01,
            -0.10,
            "commit",
        )
        for faults in range(9):
            witness_count = 3 * faults + 1
            signatures = tuple(
                WitnessSignature(index, bytes([index]) * 64)
                for index in range(2 * faults + 1)
            )
            receipt = Receipt(core, witness_count, signatures)
            expected = 357 + (witness_count + 7) // 8 + 64 * (2 * faults + 1)
            encoded = receipt.to_bytes()
            self.assertEqual(len(encoded), expected)
            self.assertEqual(
                Receipt.from_bytes(encoded, witness_count=witness_count).to_bytes(),
                encoded,
            )


class SealedCatalogConformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run()

    def test_complete_flow_passes(self) -> None:
        self.assertEqual(self.result["status"], "passed")

    def test_cross_handover_risk_budget_is_registered(self) -> None:
        self.assertTrue(self.result["cross_handover_risk_budget_registered"])

    def test_install_and_append_rolls_back_serving_bytes_on_audit_failure(self) -> None:
        with TemporaryDirectory(prefix="fedmerit-install-atomic-") as directory:
            case = _protocol_case(
                Path(directory),
                name="atomic-install",
                after_bias=2.0,
                epsilon=0.10,
                gamma=0.05,
                alpha=0.10,
            )
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
            self.assertEqual(receipt.core.decision, "commit")
            before = case["registry"].serving_model_snapshot
            with sqlite3.connect(case["registry"].path) as db:
                db.executescript("""
                    CREATE TRIGGER fail_audit_head_update
                    BEFORE UPDATE OF head ON audit_state
                    BEGIN
                        SELECT RAISE(ABORT, 'injected audit failure');
                    END;
                """)
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
            self.assertFalse(appended)
            self.assertEqual(case["registry"].head, ZERO_HASH)
            self.assertEqual(case["registry"].serving_model_snapshot, before)
            with sqlite3.connect(case["registry"].path) as db:
                db.execute("DROP TRIGGER fail_audit_head_update")
            self.assertTrue(
                case["registry"].verify_and_append(
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
            )
            installed_hash, installed_version, installed_blob = (
                case["registry"].serving_model_snapshot
            )
            self.assertEqual(installed_hash, case["candidate"].after_model_hash)
            self.assertEqual(
                installed_version, case["candidate"].state_context.model_version + 1
            )
            self.assertEqual(installed_blob, case["candidate"].after_model.artifact_bytes)

    def test_first_append_requires_exact_genesis_artifact_bytes(self) -> None:
        with TemporaryDirectory(prefix="fedmerit-genesis-bytes-") as directory:
            case = _protocol_case(
                Path(directory),
                name="genesis-bytes",
                after_bias=2.0,
                epsilon=0.10,
                gamma=0.05,
                alpha=0.10,
            )
            self.assertEqual(
                case["registry"].serving_model_snapshot,
                (
                    case["before"].artifact_hash,
                    case["context"].model_version,
                    case["before"].artifact_bytes,
                ),
            )
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
            with sqlite3.connect(case["registry"].path) as db:
                db.execute(
                    "UPDATE serving_model SET artifact_blob=? WHERE id=1",
                    (b"tampered-genesis",),
                )
            self.assertFalse(
                case["registry"].verify_and_append(
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
            )
            self.assertEqual(case["registry"].head, ZERO_HASH)

    def test_authority_certificate_opens_to_exact_roster_and_epoch(self) -> None:
        with TemporaryDirectory(prefix="fedmerit-authority-binding-") as directory:
            case = _protocol_case(
                Path(directory),
                name="authority-binding",
                after_bias=2.0,
                epsilon=0.10,
                gamma=0.05,
                alpha=0.10,
            )
            self.assertEqual(
                case["context"].authority_certificate_hash,
                case["trust"].authority_certificate_hash,
            )
            rotated_epoch = replace(case["trust"], roster_epoch=1)
            with self.assertRaisesRegex(ValueError, "authority certificate"):
                case["registry"].provision_verification_trust(rotated_epoch)

    def test_catalog_reassignment_is_rejected(self) -> None:
        self.assertTrue(self.result["catalog_reassignment_rejected"])

    def test_catalog_tamper_is_rejected(self) -> None:
        self.assertTrue(self.result["catalog_tamper_rejected"])

    def test_post_fixation_payload_mutation_is_rejected(self) -> None:
        self.assertTrue(self.result["post_fixation_mutation_rejected"])

    def test_post_beacon_eligible_subset_is_rejected(self) -> None:
        self.assertTrue(self.result["post_beacon_eligible_subset_rejected"])

    def test_future_round_cannot_serve_as_fixation_head(self) -> None:
        self.assertTrue(self.result["future_round_as_fixation_head_rejected"])

    def test_concurrent_retirement_invalidates_fixed_population(self) -> None:
        self.assertTrue(self.result["concurrent_retirement_rejected"])

    def test_beacon_head_cannot_rollback_after_restart(self) -> None:
        self.assertTrue(self.result["backdated_beacon_head_rejected_after_restart"])

    def test_beacon_head_cannot_skip_or_fork_the_parent_chain(self) -> None:
        self.assertTrue(self.result["skipped_or_forked_beacon_round_rejected"])

    def test_one_successor_round_cannot_fund_two_fixations(self) -> None:
        with TemporaryDirectory(prefix="fedmerit-beacon-reservation-") as directory:
            root = Path(directory)
            case = _protocol_case(
                root,
                name="exclusive-beacon-successor",
                after_bias=2.0,
                epsilon=0.10,
                gamma=0.05,
                alpha=0.10,
            )
            raw_key = case["beacon_private_key"].public_key().public_bytes(
                serialization.Encoding.Raw, serialization.PublicFormat.Raw
            )
            key_hash = hashlib.sha256(raw_key).hexdigest()
            with sqlite3.connect(case["ledger"].path) as db:
                reservation = db.execute(
                    "SELECT parent_round_hash, fixation_hash "
                    "FROM beacon_successor_reservations "
                    "WHERE beacon_public_key_hash=? AND round_number=?",
                    (key_hash, case["candidate"].beacon_round),
                ).fetchone()
                self.assertEqual(
                    reservation,
                    (
                        case["candidate"].beacon_parent_hash,
                        case["candidate"].fixation_hash,
                    ),
                )
                with self.assertRaises(sqlite3.IntegrityError):
                    db.execute(
                        "INSERT INTO beacon_successor_reservations VALUES(?,?,?,?)",
                        (
                            key_hash,
                            case["candidate"].beacon_round,
                            case["candidate"].beacon_parent_hash,
                            _digest("conflicting-fixation"),
                        ),
                    )

            # A second local risk ledger cannot bypass the authoritative
            # reservation by presenting another fixation for the same round.
            second_ledger = RiskLedger(root / "second-risk.sqlite3")
            second_ledger.register(
                case["schedule"], audit_registry=case["registry"]
            )
            second_ledger.observe_beacon_head(
                case["signed_beacon_head"],
                audit_registry=case["registry"],
                beacon_public_key=case["beacon_private_key"].public_key(),
                signed_frame=case["signed_frame"],
                frame_public_key=case["frame_private_key"].public_key(),
            )
            conflicting = replace(
                case["candidate"],
                after_model=LinearModelArtifact((0.0, 3.0)),
            )
            with self.assertRaisesRegex(
                ValueError, "already reserved by another fixation"
            ):
                second_ledger.consume(
                    conflicting,
                    case["schedule"],
                    audit_registry=case["registry"],
                    beacon_public_key=case["beacon_private_key"].public_key(),
                    signed_frame=case["signed_frame"],
                    frame_public_key=case["frame_private_key"].public_key(),
                )

    def test_post_reveal_fixation_fails_at_the_authoritative_head(self) -> None:
        with TemporaryDirectory(prefix="fedmerit-beacon-checkpoint-") as directory:
            root = Path(directory)
            case = _protocol_case(
                root,
                name="checkpoint-bootstrap",
                after_bias=2.0,
                epsilon=0.10,
                gamma=0.05,
                alpha=0.10,
            )
            fresh = RiskLedger(root / "fresh-risk.sqlite3")
            fresh.observe_beacon_head(
                case["signed_beacon_round"],
                audit_registry=case["registry"],
                beacon_public_key=case["beacon_private_key"].public_key(),
                signed_frame=case["signed_frame"],
                frame_public_key=case["frame_private_key"].public_key(),
            )
            with self.assertRaisesRegex(ValueError, "beacon head"):
                fresh.consume(
                    case["candidate"],
                    case["schedule"],
                    audit_registry=case["registry"],
                    beacon_public_key=case["beacon_private_key"].public_key(),
                    signed_frame=case["signed_frame"],
                    frame_public_key=case["frame_private_key"].public_key(),
                )

    def test_public_and_authorized_paths_are_distinct(self) -> None:
        self.assertTrue(self.result["public_receipt_verified_without_raw_probe"])
        self.assertTrue(self.result["authorized_raw_probe_replay_verified"])

    def test_quorum_skips_one_unavailable_witness(self) -> None:
        with TemporaryDirectory(prefix="fedmerit-quorum-fallback-") as directory:
            case = _protocol_case(
                Path(directory),
                name="quorum-fallback",
                after_bias=2.0,
                epsilon=0.10,
                gamma=0.05,
                alpha=0.10,
            )
            release = case["store"].release(
                case["candidate"],
                signed_beacon_round=case["signed_beacon_round"],
                beacon_public_key=case["beacon_private_key"].public_key(),
                schedule=case["schedule"],
                risk_ledger=case["ledger"],
                audit_registry=case["registry"],
            )
            original = Witness.attest

            def one_unavailable(witness: Witness, *args: object, **kwargs: object):
                if witness.witness_index == 0:
                    raise RuntimeError("simulated unavailable witness")
                return original(witness, *args, **kwargs)  # type: ignore[arg-type]

            with patch.object(Witness, "attest", new=one_unavailable):
                receipt = case["authority"].issue(
                    case["candidate"],
                    release,
                    store_public_key=case["store"].public_key,
                    frame_public_key=case["frame_private_key"].public_key(),
                    schedule=case["schedule"],
                    risk_ledger=case["ledger"],
                    audit_registry=case["registry"],
                )
            self.assertEqual(
                tuple(signature.witness_index for signature in receipt.signatures),
                (1, 2, 3),
            )

    def test_verifier_trust_roots_are_registry_bound(self) -> None:
        self.assertTrue(self.result["rogue_witness_roster_rejected"])

    def test_issued_receipt_remains_auditable_but_not_appendable(self) -> None:
        self.assertTrue(self.result["issued_receipt_verifiable_after_handover"])
        self.assertTrue(self.result["issued_receipt_append_blocked_after_handover"])

    def test_final_issuance_step_rechecks_the_live_head(self) -> None:
        self.assertTrue(self.result["final_issuance_fence_rejected"])

    def test_commitment_is_salted_by_the_secret_opening(self) -> None:
        group = ProbeGroup(
            "group-0",
            _digest("source-manifest"),
            ((0.0, 1.0),),
            (0,),
        )
        fields = dict(
            probe_id="probe-0",
            context_hash=_digest("context"),
            probe_policy_hash=_digest("policy"),
            groups=(group,),
            collection_window_start="2026-01-01T00:00:00Z",
            collection_window_end="2026-01-02T00:00:00Z",
            source_handle_hash=_digest("source-handle"),
        )
        first = CommitProbe(**fields, sealing_nonce=b"a" * 32)
        second = CommitProbe(**fields, sealing_nonce=b"b" * 32)
        self.assertNotEqual(first.probe_id_hash, second.probe_id_hash)
        self.assertNotEqual(first.commitment, second.commitment)

    def test_source_partition_rejects_proposal_score_overlap(self) -> None:
        shared = _digest("shared-source-manifest")
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            SourcePartition(
                _digest("partition-context"),
                _digest("partition-contributors"),
                _digest("partition-score"),
                _digest("partition-policy"),
                (shared,),
                (shared,),
            )

    def test_public_release_excludes_raw_probe_and_opening(self) -> None:
        with TemporaryDirectory(prefix="fedmerit-public-release-") as directory:
            case = _protocol_case(
                Path(directory),
                name="public-release-boundary",
                after_bias=0.0,
                epsilon=0.80,
                gamma=0.80,
                alpha=0.50,
                group_count=2,
                probe_count=2,
            )
            release, _, _ = _execute(case)
            public_bytes = canonical_bytes(release.public_release)
            selected_probe = release.probe

            self.assertNotIn(selected_probe.probe_id.encode("ascii"), public_bytes)
            self.assertNotIn(selected_probe.sealing_nonce.hex().encode("ascii"), public_bytes)
            self.assertNotIn(
                selected_probe.collection_window_start.encode("ascii"), public_bytes
            )
            self.assertNotIn(
                selected_probe.collection_window_end.encode("ascii"), public_bytes
            )
            self.assertNotIn(
                selected_probe.source_handle_hash.encode("ascii"), public_bytes
            )
            for group in selected_probe.groups:
                self.assertNotIn(group.group_id.encode("ascii"), public_bytes)
                self.assertNotIn(
                    group.source_manifest_hash.encode("ascii"), public_bytes
                )

            self.assertIn(
                selected_probe.probe_id_hash.encode("ascii"), public_bytes
            )
            self.assertIn(selected_probe.commitment.encode("ascii"), public_bytes)
            for probe in case["probes"]:
                if probe != selected_probe:
                    self.assertNotIn(probe.commitment.encode("ascii"), public_bytes)


class SourceManifestLedgerTests(unittest.TestCase):
    def test_reservation_is_idempotent_but_rejects_cross_release_reuse(self) -> None:
        with TemporaryDirectory(prefix="fedmerit-manifest-ledger-") as directory:
            path = Path(directory) / "risk.sqlite3"
            manifests = ("a" * 64, "b" * 64)
            ledger = RiskLedger(path)
            ledger.reserve_source_manifests(
                manifests,
                context_hash="context-a",
                fixation_hash="fixation-a",
                probe_id_hash="probe-a",
            )

            reopened = RiskLedger(path)
            reopened.reserve_source_manifests(
                manifests,
                context_hash="context-a",
                fixation_hash="fixation-a",
                probe_id_hash="probe-a",
            )
            with self.assertRaisesRegex(ValueError, "already been reserved"):
                reopened.reserve_source_manifests(
                    manifests,
                    context_hash="context-b",
                    fixation_hash="fixation-b",
                    probe_id_hash="probe-b",
                )


class RiskScheduleRegistryTests(unittest.TestCase):
    def test_schedule_registration_requires_lineage_envelope(self) -> None:
        with TemporaryDirectory(prefix="fedmerit-required-lineage-") as directory:
            policy = EvaluationPolicy("brier-decimal80-v1")
            context = StateContext(
                "required-lineage",
                "domain-0",
                0,
                _digest("required-lineage-schema"),
                policy.policy_hash,
                0,
                _digest("required-lineage-authority"),
            )
            registry = AuditRegistry(
                Path(directory) / "audit.sqlite3",
                genesis_model=LinearModelArtifact((0.0, 0.0)),
                initial_context=context,
                evaluation_policy=policy,
            )
            schedule = RiskSchedule(
                "required-lineage-schedule",
                context.context_hash,
                ZERO_HASH,
                0.10,
                (RiskAllocation(0.10, 0.05, 0.10, 205),),
            )
            with self.assertRaisesRegex(ValueError, "must be provisioned"):
                registry.register_risk_schedule(schedule)

    def test_fresh_ledger_cannot_reset_same_context_lifetime_budget(self) -> None:
        with TemporaryDirectory(prefix="fedmerit-schedule-reset-") as directory:
            root = Path(directory)
            case = _protocol_case(
                root / "first",
                name="schedule-reset",
                after_bias=0.0,
                epsilon=0.10,
                gamma=0.05,
                alpha=0.10,
            )
            _, receipt, appended = _execute(case)
            self.assertTrue(appended)
            self.assertEqual(receipt.core.decision, "reject")

            reset = RiskSchedule(
                "schedule-reset-second",
                case["context"].context_hash,
                receipt.receipt_hash,
                case["schedule"].lifetime_delta,
                case["schedule"].allocations,
            )
            fresh = RiskLedger(root / "fresh-risk.sqlite3")
            with self.assertRaisesRegex(ValueError, "different lifetime schedule"):
                fresh.register(reset, audit_registry=case["registry"])

            # Reopening the exact original schedule remains an idempotent recovery.
            exact_retry = RiskLedger(root / "exact-retry.sqlite3")
            exact_retry.register(
                case["schedule"], audit_registry=case["registry"]
            )
            case["registry"].reserve_risk_allocation(
                case["schedule"],
                0,
                fixation_hash=case["candidate"].fixation_hash,
            )
            with self.assertRaisesRegex(ValueError, "another fixation"):
                case["registry"].reserve_risk_allocation(
                    case["schedule"],
                    0,
                    fixation_hash=_digest("competing-fixation"),
                )


class LineageRiskBudgetTests(unittest.TestCase):
    def test_context_schedules_share_one_budget_across_handovers(self) -> None:
        with TemporaryDirectory(prefix="fedmerit-lineage-risk-") as directory:
            path = Path(directory) / "audit.sqlite3"
            policy = EvaluationPolicy("brier-decimal80-v1")
            model = LinearModelArtifact((0.0, 0.0))
            context = StateContext(
                "twin-lineage",
                "domain-0",
                0,
                _digest("lineage-schema-0"),
                policy.policy_hash,
                0,
                _digest("lineage-authority-0"),
            )
            registry = AuditRegistry(
                path,
                genesis_model=model,
                initial_context=context,
                evaluation_policy=policy,
            )
            registry.provision_lineage_risk_budget(0.25)
            registry.register_risk_schedule(
                RiskSchedule(
                    "lineage-schedule-0",
                    context.context_hash,
                    ZERO_HASH,
                    0.125,
                    (RiskAllocation(0.25, 0.05, 0.125, 1),),
                )
            )

            successor = StateContext(
                context.twin_id,
                "domain-1",
                1,
                _digest("lineage-schema-1"),
                policy.policy_hash,
                0,
                _digest("lineage-authority-1"),
            )
            registry.handover(
                state_context=successor, evaluation_policy=policy
            )
            registry.register_risk_schedule(
                RiskSchedule(
                    "lineage-schedule-1",
                    successor.context_hash,
                    ZERO_HASH,
                    0.0625,
                    (RiskAllocation(0.25, 0.05, 0.0625, 1),),
                )
            )

            reopened = AuditRegistry(
                path,
                genesis_model=model,
                initial_context=successor,
                evaluation_policy=policy,
            )
            self.assertEqual(
                reopened.lineage_risk_budget,
                (context.twin_id, ZERO_HASH, 0.25),
            )
            with self.assertRaisesRegex(ValueError, "already frozen"):
                reopened.provision_lineage_risk_budget(0.5)

            second_successor = StateContext(
                context.twin_id,
                "domain-2",
                2,
                _digest("lineage-schema-2"),
                policy.policy_hash,
                0,
                _digest("lineage-authority-2"),
            )
            reopened.handover(
                state_context=second_successor, evaluation_policy=policy
            )
            with self.assertRaisesRegex(ValueError, "cross-handover"):
                reopened.register_risk_schedule(
                    RiskSchedule(
                        "lineage-schedule-2",
                        second_successor.context_hash,
                        ZERO_HASH,
                        0.125,
                        (RiskAllocation(0.25, 0.05, 0.125, 1),),
                    )
                )

    def test_attempt_cap_is_cumulative_across_context_handovers(self) -> None:
        with TemporaryDirectory(prefix="fedmerit-lineage-attempt-cap-") as directory:
            profile = SecurityProfile(max_attempts=2)
            policy = EvaluationPolicy(
                "brier-decimal80-v1", security_profile=profile
            )
            model = LinearModelArtifact((0.0, 0.0))
            contexts = [
                StateContext(
                    "twin-attempt-cap",
                    f"domain-{index}",
                    index,
                    _digest(f"attempt-cap-schema-{index}"),
                    policy.policy_hash,
                    0,
                    _digest(f"attempt-cap-authority-{index}"),
                )
                for index in range(3)
            ]
            registry = AuditRegistry(
                Path(directory) / "audit.sqlite3",
                genesis_model=model,
                initial_context=contexts[0],
                evaluation_policy=policy,
            )
            registry.provision_lineage_risk_budget(0.20)
            for index in range(2):
                registry.register_risk_schedule(
                    RiskSchedule(
                        f"attempt-cap-schedule-{index}",
                        contexts[index].context_hash,
                        ZERO_HASH,
                        0.05,
                        (RiskAllocation(0.25, 0.05, 0.05, 67),),
                    )
                )
                registry.handover(
                    state_context=contexts[index + 1],
                    evaluation_policy=policy,
                )
            with self.assertRaisesRegex(ValueError, "lifetime attempt cap"):
                registry.register_risk_schedule(
                    RiskSchedule(
                        "attempt-cap-schedule-2",
                        contexts[2].context_hash,
                        ZERO_HASH,
                        0.05,
                        (RiskAllocation(0.25, 0.05, 0.05, 67),),
                    )
                )

    def test_protocol_event_journal_is_hash_linked_and_append_only(self) -> None:
        with TemporaryDirectory(prefix="fedmerit-event-journal-") as directory:
            case = _protocol_case(
                Path(directory),
                name="event-journal",
                after_bias=2.0,
                epsilon=0.10,
                gamma=0.05,
                alpha=0.10,
            )
            _, _, appended = _execute(case)
            self.assertTrue(appended)
            events = case["registry"].protocol_events
            self.assertTrue(case["registry"].protocol_event_chain_valid())
            self.assertTrue(
                {
                    "lineage-budget-provisioned",
                    "risk-schedule-registered",
                    "beacon-head-observed",
                    "beacon-successor-reserved",
                    "risk-allocation-spent",
                    "source-manifests-retired",
                    "receipt-issued",
                    "receipt-appended",
                }.issubset({str(event["event_type"]) for event in events})
            )
            with sqlite3.connect(case["registry"].path) as db:
                with self.assertRaisesRegex(sqlite3.IntegrityError, "append-only"):
                    db.execute(
                        "UPDATE protocol_events SET event_type='tampered' "
                        "WHERE sequence=1"
                    )


class RiskArithmeticTests(unittest.TestCase):
    def test_paper_risk_plan_uses_outward_intervals_and_minimum_group_counts(
        self,
    ) -> None:
        cases = (
            (0.10, 0.35, 0.00, 38),
            (0.05, 0.25, 0.05, 67),
            (0.01, 0.25, 0.05, 103),
            (0.001, 0.25, 0.05, 154),
            (0.01, 0.15, 0.05, 231),
            (0.01, 0.10, 0.05, 410),
            (0.0005, 0.25, 0.05, 169),
        )
        for alpha, epsilon, gamma, expected_count in cases:
            with self.subTest(alpha=alpha, epsilon=epsilon, gamma=gamma):
                count = required_groups(alpha, epsilon, gamma)
                self.assertEqual(count, expected_count)
                self.assertTrue(risk_is_satisfied(count, epsilon, gamma, alpha))
                self.assertFalse(
                    risk_is_satisfied(count - 1, epsilon, gamma, alpha)
                )

                lower, upper = risk_bound_interval(count, epsilon, gamma)
                total = Fraction.from_float(epsilon) + Fraction.from_float(gamma)
                exponent = Fraction(count, 2) * total * total
                with localcontext() as context:
                    context.prec = 160
                    exact = (
                        -Decimal(exponent.numerator) / Decimal(exponent.denominator)
                    ).exp()
                    lower_decimal = Decimal(lower.numerator) / Decimal(
                        lower.denominator
                    )
                    upper_decimal = Decimal(upper.numerator) / Decimal(
                        upper.denominator
                    )
                self.assertLessEqual(lower_decimal, exact)
                self.assertLessEqual(exact, upper_decimal)
                self.assertLessEqual(upper - lower, Fraction(1, 1 << 192))


class WireTypeValidationTests(unittest.TestCase):
    def test_risk_wire_scalars_are_canonical_binary64(self) -> None:
        allocation = RiskAllocation(1, 0, 0.1, 1)
        self.assertIs(type(allocation.epsilon), float)
        self.assertIs(type(allocation.gamma), float)
        self.assertIn(b'"epsilon":{"float64":', canonical_bytes(allocation))

        for field in ("epsilon", "gamma", "alpha"):
            values = {"epsilon": 0.1, "gamma": 0.0, "alpha": 0.1}
            values[field] = True
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "binary64 scalar"):
                    RiskAllocation(
                        values["epsilon"],
                        values["gamma"],
                        values["alpha"],
                        1,
                    )

        with self.assertRaisesRegex(ValueError, "binary64 scalar"):
            RiskSchedule(
                "invalid-boolean-budget",
                _digest("risk-context"),
                ZERO_HASH,
                True,
                (allocation,),
            )

    def test_receipt_core_normalizes_public_risk_scalars(self) -> None:
        core = ReceiptCore(
            *(_digest(f"wire-core-{index}") for index in range(10)),
            1,
            1,
            0,
            0.1,
            0,
            "reject",
        )
        self.assertTrue(
            all(
                type(getattr(core, name)) is float
                for name in ("epsilon", "gamma", "alpha", "delta_hat")
            )
        )
        with self.assertRaisesRegex(ValueError, "binary64 scalar"):
            replace(core, gamma=False)

    def test_evaluator_policy_binds_the_sigmoid_saturation_rule(self) -> None:
        policy = EvaluationPolicy("brier-decimal80-v1")

        self.assertEqual(policy.sigmoid_logit_clamp, 80)
        self.assertIn(b'"sigmoid_logit_clamp":80', canonical_bytes(policy))
        for invalid in (True, 79, 80.0):
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(ValueError, "sigmoid logit clamp"):
                    replace(
                        policy,
                        sigmoid_logit_clamp=invalid,  # type: ignore[arg-type]
                    )

    def test_digest_fields_reject_uppercase_aliases(self) -> None:
        with self.assertRaisesRegex(ValueError, "sensor_schema_hash"):
            StateContext(
                "twin",
                "domain",
                0,
                "A" * 64,
                _digest("policy"),
                0,
                _digest("authority"),
            )

    def test_uint32_fields_reject_bool_and_float_aliases(self) -> None:
        for invalid in (True, 1.0):
            with self.subTest(field="group_count", value=invalid):
                with self.assertRaisesRegex(ValueError, "group_count"):
                    RiskAllocation(0.1, 0.0, 0.05, invalid)  # type: ignore[arg-type]

            with self.subTest(field="source_group_count", value=invalid):
                fields = {
                    name: _digest(name) for name in ReceiptCore.DIGEST_FIELDS
                }
                with self.assertRaisesRegex(ValueError, "source_group_count"):
                    ReceiptCore(
                        **fields,
                        source_group_count=invalid,  # type: ignore[arg-type]
                        epsilon=0.1,
                        gamma=0.0,
                        alpha=0.05,
                        delta_hat=0.0,
                        decision="commit",
                    )

            with self.subTest(field="witness_count", value=invalid):
                with self.assertRaisesRegex(ValueError, "witness_count"):
                    Receipt.from_bytes(b"", witness_count=invalid)  # type: ignore[arg-type]

    def test_commitment_objects_snapshot_mutable_constructor_inputs(self) -> None:
        feature_rows = [[0.0, 1.0]]
        labels = [0]
        group = ProbeGroup(
            "group-immutable",
            _digest("immutable-source"),
            feature_rows,  # type: ignore[arg-type]
            labels,  # type: ignore[arg-type]
        )
        groups = [group]
        opening_key = bytearray(b"k" * 32)
        probe = CommitProbe(
            "probe-immutable",
            _digest("immutable-context"),
            _digest("immutable-policy"),
            groups,  # type: ignore[arg-type]
            "2026-01-01T00:00:00Z",
            "2026-01-02T00:00:00Z",
            _digest("immutable-source-handle"),
            opening_key,  # type: ignore[arg-type]
        )
        allocations = [RiskAllocation(0.8, 0.8, 0.5, 2)]
        schedule = RiskSchedule(
            "schedule-immutable",
            probe.context_hash,
            ZERO_HASH,
            0.5,
            allocations,  # type: ignore[arg-type]
        )
        entries = [probe.frame_entry]
        exclusions = [_digest("immutable-exclusion")]
        frame = SamplingFrame(
            "frame-immutable",
            probe.context_hash,
            probe.probe_policy_hash,
            entries,  # type: ignore[arg-type]
            "beacon-immutable",
            _digest("immutable-beacon-key"),
            (_digest("immutable-partition"),),
            1,
            _digest("immutable-beacon-checkpoint"),
            exclusions,  # type: ignore[arg-type]
        )
        before = (probe.commitment, schedule.schedule_hash, frame.frame_hash)

        feature_rows[0][0] = 99.0
        labels[0] = 1
        groups.clear()
        opening_key[0] = 0
        allocations.clear()
        entries.clear()
        exclusions.clear()

        self.assertEqual(group.features, ((0.0, 1.0),))
        self.assertEqual(group.labels, (0,))
        self.assertIsInstance(probe.sealing_nonce, bytes)
        self.assertEqual(
            (probe.commitment, schedule.schedule_hash, frame.frame_hash),
            before,
        )

    def test_commitment_inputs_reject_ambiguous_scalar_types(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer binary labels"):
            ProbeGroup(
                "group-bool-label",
                _digest("bool-label-source"),
                ((0.0,),),
                (True,),  # type: ignore[arg-type]
            )
        group = ProbeGroup(
            "group-valid",
            _digest("valid-source"),
            ((0.0,),),
            (0,),
        )
        with self.assertRaisesRegex(TypeError, "sealing_nonce must be bytes"):
            CommitProbe(
                "probe-invalid-key",
                _digest("valid-context"),
                _digest("valid-policy"),
                (group,),
                "2026-01-01T00:00:00Z",
                "2026-01-02T00:00:00Z",
                _digest("valid-source-handle"),
                "not-a-byte-key-of-the-right-size",  # type: ignore[arg-type]
            )

    def test_risk_schedule_budget_is_independent_of_decimal_context(self) -> None:
        fields = {
            "schedule_id": "schedule-exact-binary64-budget",
            "context_hash": _digest("risk-context"),
            "anchor_receipt_hash": ZERO_HASH,
            "lifetime_delta": 0.1,
        }
        with localcontext() as context:
            context.prec = 1
            accepted = RiskSchedule(
                **fields,
                allocations=(
                    RiskAllocation(0.1, 0.0, 0.05, 1),
                    RiskAllocation(0.1, 0.0, 0.05, 1),
                ),
            )
            self.assertEqual(len(accepted.allocations), 2)
            with self.assertRaisesRegex(ValueError, "lifetime budget"):
                RiskSchedule(
                    **fields,
                    allocations=(
                        RiskAllocation(0.1, 0.0, 0.06, 1),
                        RiskAllocation(0.1, 0.0, 0.06, 1),
                    ),
                )

    def test_certificate_authority_rejects_noninteger_fault_budget(self) -> None:
        # The constructor validates f before inspecting witness state; an empty
        # witness list is therefore sufficient for this type check.
        witnesses = []
        for invalid in (True, 1.0, -1):
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(ValueError, "f must be"):
                    CertificateAuthority(witnesses, invalid)  # type: ignore[arg-type]

    def test_persistent_authority_validates_fault_budget_before_opening_files(self) -> None:
        with TemporaryDirectory(prefix="fedmerit-invalid-f-") as directory:
            for invalid in (True, 1.0, -1):
                with self.subTest(value=invalid):
                    with self.assertRaisesRegex(ValueError, "f must be"):
                        CertificateAuthority.persistent(
                            directory, invalid  # type: ignore[arg-type]
                        )


class ContextHandoverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory(prefix="fedmerit-handover-")
        self.addCleanup(self.directory.cleanup)
        self.policy = EvaluationPolicy("brier-decimal80-v1")
        self.initial = StateContext(
            "twin-7",
            "domain-a",
            4,
            _digest("schema-a"),
            self.policy.policy_hash,
            7,
            _digest("authority-a"),
        )
        self.genesis_model = LinearModelArtifact((0.0, 0.0))
        self.path = Path(self.directory.name) / "audit.sqlite3"
        self.registry = AuditRegistry(
            self.path,
            genesis_model=self.genesis_model,
            initial_context=self.initial,
            evaluation_policy=self.policy,
        )

    def _context(
        self,
        *,
        twin_id: str = "twin-7",
        state_version: int = 5,
        model_version: int = 7,
    ) -> StateContext:
        return StateContext(
            twin_id,
            "domain-b",
            state_version,
            _digest("schema-b"),
            self.policy.policy_hash,
            model_version,
            _digest("authority-b"),
        )

    def test_accepts_immediate_successor_without_model_update(self) -> None:
        successor = self._context()
        self.registry.handover(
            state_context=successor,
            evaluation_policy=self.policy,
        )
        self.assertEqual(
            self.registry.context_head,
            (successor.context_hash, successor.authority_certificate_hash),
        )
        reopened = AuditRegistry(
            self.path,
            genesis_model=self.genesis_model,
            initial_context=successor,
            evaluation_policy=self.policy,
        )
        self.assertEqual(reopened.context_head, self.registry.context_head)
        # A domain/schema handover advances the authority state but does not
        # transport an older model or invent a new model version.
        self.assertEqual(reopened.installed_model_version, self.initial.model_version)

    def test_rejects_twin_substitution(self) -> None:
        with self.assertRaisesRegex(ValueError, "preserve the twin identity"):
            self.registry.handover(
                state_context=self._context(twin_id="twin-attacker"),
                evaluation_policy=self.policy,
            )

    def test_rejects_state_rollback_or_skip(self) -> None:
        for state_version in (3, 4, 6):
            with self.subTest(state_version=state_version):
                with self.assertRaisesRegex(ValueError, "immediate successor"):
                    self.registry.handover(
                        state_context=self._context(state_version=state_version),
                        evaluation_policy=self.policy,
                    )

    def test_rejects_model_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "installed model version"):
            self.registry.handover(
                state_context=self._context(model_version=6),
                evaluation_policy=self.policy,
            )

        with self.assertRaisesRegex(ValueError, "installed model version"):
            self.registry.handover(
                state_context=self._context(model_version=8),
                evaluation_policy=self.policy,
            )

    def test_bounded_producer_handles_commit_then_handover(self) -> None:
        rows = _handover_rows()
        by_position = {row["handover_position"]: row for row in rows}
        self.assertEqual(set(by_position), {
            "none",
            "before_release",
            "after_release",
            "after_issue",
            "after_append",
        })
        self.assertTrue(all(row["schedule_passed"] for row in rows))
        self.assertTrue(by_position["after_append"]["append_accepted"])


class ModelSuccessorTests(unittest.TestCase):
    def test_model_successor_preserves_state_identity(self) -> None:
        policy = EvaluationPolicy("brier-decimal80-v1")
        context = StateContext(
            "twin-7",
            "domain-a",
            4,
            _digest("schema-a"),
            policy.policy_hash,
            7,
            _digest("authority-a"),
        )
        successor = context.model_successor()
        self.assertEqual(successor.twin_id, context.twin_id)
        self.assertEqual(successor.domain_id, context.domain_id)
        self.assertEqual(successor.state_version, context.state_version)
        self.assertEqual(successor.sensor_schema_hash, context.sensor_schema_hash)
        self.assertEqual(successor.policy_hash, context.policy_hash)
        self.assertEqual(
            successor.authority_certificate_hash,
            context.authority_certificate_hash,
        )
        self.assertEqual(successor.model_version, context.model_version + 1)
        self.assertNotEqual(successor.context_hash, context.context_hash)

    def test_commit_activates_model_successor_and_accepts_next_candidate(self) -> None:
        with TemporaryDirectory(prefix="fedmerit-model-successor-") as directory:
            root = Path(directory)
            case = _protocol_case(
                root,
                name="model-successor",
                after_bias=2.0,
                epsilon=0.10,
                gamma=0.05,
                alpha=0.10,
            )
            release, receipt, appended = _execute(case)
            self.assertTrue(appended)
            self.assertEqual(receipt.core.decision, "commit")

            successor = case["context"].model_successor()
            registry = case["registry"]
            self.assertEqual(
                registry.context_head,
                (successor.context_hash, successor.authority_certificate_hash),
            )
            self.assertEqual(registry.installed_model_version, successor.model_version)

            reopened = AuditRegistry(
                root / "audit.sqlite3",
                genesis_model=case["before"],
                initial_context=successor,
                evaluation_policy=case["policy"],
            )
            self.assertEqual(reopened.context_head, registry.context_head)

            partition = SourcePartition(
                successor.context_hash,
                case["candidate"].contributor_root,
                case["candidate"].score_probe_commitment,
                case["policy"].policy_hash,
                case["candidate"].source_partition.source_manifest_hashes,
                case["candidate"].source_partition.score_source_manifest_hashes,
            )
            next_candidate = replace(
                case["candidate"],
                context_hash=successor.context_hash,
                state_context=successor,
                before_model=case["after"],
                after_model=case["after"],
                source_partition=partition,
                previous_receipt_hash=receipt.receipt_hash,
            )
            reopened.validate_candidate(next_candidate)

            head_before_retry = reopened.head
            context_before_retry = reopened.context_head
            model_version_before_retry = reopened.installed_model_version
            self.assertTrue(
                reopened.verify_and_append(
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
            )
            self.assertEqual(reopened.head, head_before_retry)
            self.assertEqual(reopened.context_head, context_before_retry)
            self.assertEqual(
                reopened.installed_model_version,
                model_version_before_retry,
            )

            handover = StateContext(
                successor.twin_id,
                "domain-b",
                successor.state_version + 1,
                _digest("schema-b"),
                successor.policy_hash,
                successor.model_version,
                case["trust"].authority_certificate_hash,
            )
            reopened.handover(
                state_context=handover,
                evaluation_policy=case["policy"],
            )
            self.assertEqual(reopened.installed_model_version, successor.model_version)
            self.assertEqual(reopened.context_head[0], handover.context_hash)

    def test_successor_catalog_completes_second_commit_with_inherited_roots(self) -> None:
        """A model successor gets a fresh inventory, not a fresh trust identity."""
        with TemporaryDirectory(prefix="fedmerit-two-commits-") as directory:
            root = Path(directory)
            first = _protocol_case(
                root / "first",
                name="two-commits",
                after_bias=2.0,
                epsilon=0.10,
                gamma=0.05,
                alpha=0.10,
            )
            first_release, first_receipt, first_appended = _execute(first)
            self.assertTrue(first_appended)

            successor = first["context"].model_successor()
            group_count = len(first["probes"][0].groups)
            groups = tuple(
                ProbeGroup(
                    f"successor-group-{index:04d}",
                    _digest(f"successor-group-{index}"),
                    ((0.0,),),
                    (0,),
                )
                for index in range(group_count)
            )
            successor_probe = CommitProbe(
                "successor-probe",
                successor.context_hash,
                first["policy"].policy_hash,
                groups,
                "2026-03-01T00:00:00Z",
                "2026-03-02T00:00:00Z",
                _digest("successor-source-handle"),
                b"c" * 32,
            )
            partition = SourcePartition(
                successor.context_hash,
                _digest("successor-contributors"),
                _digest("successor-score"),
                first["policy"].policy_hash,
                (_digest("successor-proposal-source"),),
                (_digest("successor-score-source"),),
            )
            successor_frame = SamplingFrame(
                "frame-two-commits-successor",
                successor.context_hash,
                first["policy"].policy_hash,
                (successor_probe.frame_entry,),
                first["frame"].beacon_id,
                first["frame"].beacon_public_key_hash,
                (partition.partition_hash,),
                beacon_checkpoint_round=first["signed_beacon_round"].round.round_number,
                beacon_checkpoint_hash=first["signed_beacon_round"].round.round_hash,
            )
            signed_successor_frame = sign_sampling_frame(
                successor_frame, first["frame_private_key"]
            )
            allocation = RiskAllocation(0.10, 0.05, 0.10, group_count)
            schedule = RiskSchedule(
                "schedule-two-commits-successor",
                successor.context_hash,
                first_receipt.receipt_hash,
                0.10,
                (allocation,),
            )
            ledger = first["ledger"]
            ledger.observe_beacon_head(
                first["signed_beacon_round"],
                audit_registry=first["registry"],
                beacon_public_key=first["beacon_private_key"].public_key(),
                signed_frame=first["signed_frame"],
                frame_public_key=first["frame_private_key"].public_key(),
            )
            ledger.register(schedule, audit_registry=first["registry"])
            parent_round_hash = first["signed_beacon_round"].round.round_hash
            future_round_number = first["signed_beacon_round"].round.round_number + 1
            future_round = sign_beacon_round(
                BeaconRound(
                    first["frame"].beacon_id,
                    future_round_number,
                    parent_round_hash,
                    _digest("successor-beacon")[:64].encode("ascii")[:32],
                ),
                first["beacon_private_key"],
            )
            candidate = Candidate(
                context_hash=successor.context_hash,
                state_context=successor,
                before_model=first["after"],
                after_model=LinearModelArtifact((0.0, 0.0)),
                contributor_root=partition.contributor_root,
                score_probe_commitment=partition.score_probe_commitment,
                source_partition=partition,
                evaluation_policy=first["policy"],
                sampling_frame_hash=successor_frame.frame_hash,
                sealed_catalog_root=successor_frame.catalog_root,
                eligible_probe_id_hashes=(successor_probe.probe_id_hash,),
                risk_schedule_hash=schedule.schedule_hash,
                risk_schedule_index=0,
                risk=allocation,
                beacon_parent_hash=parent_round_hash,
                beacon_round=future_round_number,
                previous_receipt_hash=first_receipt.receipt_hash,
            )
            ledger.consume(
                candidate,
                schedule,
                audit_registry=first["registry"],
                beacon_public_key=first["beacon_private_key"].public_key(),
                signed_frame=signed_successor_frame,
                frame_public_key=first["frame_private_key"].public_key(),
            )
            successor_store = CommitProbeStore.successor(
                first["store"],
                probes=[successor_probe],
                partitions=[partition],
                signed_frame=signed_successor_frame,
                path=root / "successor-probe.sqlite3",
            )
            raw_store_key = successor_store.public_key.public_bytes(
                serialization.Encoding.Raw, serialization.PublicFormat.Raw
            )
            first_store_key = first["store"].public_key.public_bytes(
                serialization.Encoding.Raw, serialization.PublicFormat.Raw
            )
            self.assertEqual(raw_store_key, first_store_key)
            with patch.object(
                ledger,
                "reserve_source_manifests",
                wraps=ledger.reserve_source_manifests,
            ) as reserve_manifests:
                release = successor_store.release(
                    candidate,
                    signed_beacon_round=future_round,
                    beacon_public_key=first["beacon_private_key"].public_key(),
                    schedule=schedule,
                    risk_ledger=ledger,
                    audit_registry=first["registry"],
                )
            reserve_manifests.assert_called_once_with(
                tuple(group.source_manifest_hash for group in groups),
                context_hash=successor.context_hash,
                fixation_hash=candidate.fixation_hash,
                probe_id_hash=successor_probe.probe_id_hash,
            )
            receipt = first["authority"].issue(
                candidate,
                release,
                store_public_key=successor_store.public_key,
                frame_public_key=first["frame_private_key"].public_key(),
                schedule=schedule,
                risk_ledger=ledger,
                audit_registry=first["registry"],
            )
            self.assertEqual(receipt.core.decision, "commit")
            self.assertTrue(
                first["registry"].verify_and_append(
                    receipt,
                    first["authority"].public_keys,
                    f=1,
                    release=release,
                    candidate=candidate,
                    store_public_key=successor_store.public_key,
                    frame_public_key=first["frame_private_key"].public_key(),
                    schedule=schedule,
                    risk_ledger=ledger,
                )
            )
            self.assertEqual(
                first["registry"].installed_model_version,
                successor.model_version + 1,
            )

            with self.assertRaisesRegex(ValueError, "different signing key"):
                CommitProbeStore(
                    [successor_probe],
                    [partition],
                    signed_successor_frame,
                    first["frame_private_key"].public_key(),
                    root / "successor-probe.sqlite3",
                    store_private_key=Ed25519PrivateKey.generate(),
                )

    def test_release_guard_rejects_reused_manifest_with_fresh_risk_ledger(self) -> None:
        """The canonical registry keeps freshness across a reset risk ledger."""
        with TemporaryDirectory(prefix="fedmerit-fresh-ledger-reuse-") as directory:
            root = Path(directory)
            first = _protocol_case(
                root / "first",
                name="fresh-ledger-reuse",
                after_bias=2.0,
                epsilon=0.10,
                gamma=0.05,
                alpha=0.10,
            )
            _, first_receipt, first_appended = _execute(first)
            self.assertTrue(first_appended)

            successor = first["context"].model_successor()
            group_count = len(first["probes"][0].groups)
            retired_manifest = first["probes"][0].groups[0].source_manifest_hash
            groups = tuple(
                ProbeGroup(
                    f"fresh-ledger-group-{index:04d}",
                    retired_manifest
                    if index == 0
                    else _digest(f"fresh-ledger-group-source-{index}"),
                    ((0.0,),),
                    (0,),
                )
                for index in range(group_count)
            )
            successor_probe = CommitProbe(
                "fresh-ledger-successor-probe",
                successor.context_hash,
                first["policy"].policy_hash,
                groups,
                "2026-04-01T00:00:00Z",
                "2026-04-02T00:00:00Z",
                _digest("fresh-ledger-successor-handle"),
                b"d" * 32,
            )
            partition = SourcePartition(
                successor.context_hash,
                _digest("fresh-ledger-successor-contributors"),
                _digest("fresh-ledger-successor-score"),
                first["policy"].policy_hash,
                (_digest("fresh-ledger-successor-proposal"),),
                (_digest("fresh-ledger-successor-score-source"),),
            )
            successor_frame = SamplingFrame(
                "fresh-ledger-successor-frame",
                successor.context_hash,
                first["policy"].policy_hash,
                (successor_probe.frame_entry,),
                first["frame"].beacon_id,
                first["frame"].beacon_public_key_hash,
                (partition.partition_hash,),
                beacon_checkpoint_round=first["signed_beacon_round"].round.round_number,
                beacon_checkpoint_hash=first["signed_beacon_round"].round.round_hash,
            )
            signed_successor_frame = sign_sampling_frame(
                successor_frame, first["frame_private_key"]
            )
            allocation = RiskAllocation(0.10, 0.05, 0.10, group_count)
            schedule = RiskSchedule(
                "fresh-ledger-successor-schedule",
                successor.context_hash,
                first_receipt.receipt_hash,
                0.10,
                (allocation,),
            )

            # A fresh risk database must not reset the append registry's
            # lineage-wide source-manifest retirement record.
            ledger = RiskLedger(root / "fresh-risk.sqlite3")
            ledger.observe_beacon_head(
                first["signed_beacon_round"],
                audit_registry=first["registry"],
                beacon_public_key=first["beacon_private_key"].public_key(),
                signed_frame=signed_successor_frame,
                frame_public_key=first["frame_private_key"].public_key(),
            )
            ledger.register(schedule, audit_registry=first["registry"])
            parent_round_hash = first["signed_beacon_round"].round.round_hash
            future_round_number = first["signed_beacon_round"].round.round_number + 1
            future_round = sign_beacon_round(
                BeaconRound(
                    first["frame"].beacon_id,
                    future_round_number,
                    parent_round_hash,
                    _digest("fresh-ledger-successor-beacon")[:64]
                    .encode("ascii")[:32],
                ),
                first["beacon_private_key"],
            )
            candidate = Candidate(
                context_hash=successor.context_hash,
                state_context=successor,
                before_model=first["after"],
                after_model=LinearModelArtifact((0.0, 0.0)),
                contributor_root=partition.contributor_root,
                score_probe_commitment=partition.score_probe_commitment,
                source_partition=partition,
                evaluation_policy=first["policy"],
                sampling_frame_hash=successor_frame.frame_hash,
                sealed_catalog_root=successor_frame.catalog_root,
                eligible_probe_id_hashes=(successor_probe.probe_id_hash,),
                risk_schedule_hash=schedule.schedule_hash,
                risk_schedule_index=0,
                risk=allocation,
                beacon_parent_hash=parent_round_hash,
                beacon_round=future_round_number,
                previous_receipt_hash=first_receipt.receipt_hash,
            )
            ledger.consume(
                candidate,
                schedule,
                audit_registry=first["registry"],
                beacon_public_key=first["beacon_private_key"].public_key(),
                signed_frame=signed_successor_frame,
                frame_public_key=first["frame_private_key"].public_key(),
            )
            successor_store = CommitProbeStore.successor(
                first["store"],
                probes=[successor_probe],
                partitions=[partition],
                signed_frame=signed_successor_frame,
                path=root / "fresh-ledger-successor.sqlite3",
            )
            head_before = first["registry"].head
            context_before = first["registry"].context_head
            version_before = first["registry"].installed_model_version
            with self.assertRaisesRegex(ValueError, "already been retired"):
                successor_store.release(
                    candidate,
                    signed_beacon_round=future_round,
                    beacon_public_key=first["beacon_private_key"].public_key(),
                    schedule=schedule,
                    risk_ledger=ledger,
                    audit_registry=first["registry"],
                )
            self.assertEqual(first["registry"].head, head_before)
            self.assertEqual(first["registry"].context_head, context_before)
            self.assertEqual(
                first["registry"].installed_model_version,
                version_before,
            )

    def test_release_reservation_survives_handover_and_fresh_risk_ledger(self) -> None:
        """A released source remains unavailable after a domain handover."""
        with TemporaryDirectory(prefix="fedmerit-release-handover-duplicate-") as directory:
            root = Path(directory)
            first = _protocol_case(
                root / "first",
                name="release-handover-duplicate",
                after_bias=2.0,
                epsilon=0.10,
                gamma=0.05,
                alpha=0.10,
            )
            first["store"].release(
                first["candidate"],
                signed_beacon_round=first["signed_beacon_round"],
                beacon_public_key=first["beacon_private_key"].public_key(),
                schedule=first["schedule"],
                risk_ledger=first["ledger"],
                audit_registry=first["registry"],
            )
            retired_manifest = first["probes"][0].groups[0].source_manifest_hash
            successor = StateContext(
                first["context"].twin_id,
                "domain-after-release",
                first["context"].state_version + 1,
                _digest("release-handover-schema"),
                first["policy"].policy_hash,
                first["context"].model_version,
                first["trust"].authority_certificate_hash,
            )
            first["registry"].handover(
                state_context=successor,
                evaluation_policy=first["policy"],
            )

            group_count = len(first["probes"][0].groups)
            groups = tuple(
                ProbeGroup(
                    f"release-handover-group-{index:04d}",
                    retired_manifest
                    if index == 0
                    else _digest(f"release-handover-source-{index}"),
                    ((0.0,),),
                    (0,),
                )
                for index in range(group_count)
            )
            successor_probe = CommitProbe(
                "release-handover-successor-probe",
                successor.context_hash,
                first["policy"].policy_hash,
                groups,
                "2026-05-01T00:00:00Z",
                "2026-05-02T00:00:00Z",
                _digest("release-handover-source-handle"),
                b"h" * 32,
            )
            partition = SourcePartition(
                successor.context_hash,
                _digest("release-handover-contributors"),
                _digest("release-handover-score"),
                first["policy"].policy_hash,
                (_digest("release-handover-proposal"),),
                (_digest("release-handover-score-source"),),
            )
            successor_frame = SamplingFrame(
                "release-handover-successor-frame",
                successor.context_hash,
                first["policy"].policy_hash,
                (successor_probe.frame_entry,),
                first["frame"].beacon_id,
                first["frame"].beacon_public_key_hash,
                (partition.partition_hash,),
                beacon_checkpoint_round=first["signed_beacon_round"].round.round_number,
                beacon_checkpoint_hash=first["signed_beacon_round"].round.round_hash,
            )
            signed_successor_frame = sign_sampling_frame(
                successor_frame, first["frame_private_key"]
            )
            allocation = RiskAllocation(0.10, 0.05, 0.10, group_count)
            schedule = RiskSchedule(
                "release-handover-successor-schedule",
                successor.context_hash,
                ZERO_HASH,
                0.10,
                (allocation,),
            )
            ledger = RiskLedger(root / "fresh-risk.sqlite3")
            ledger.register(schedule, audit_registry=first["registry"])
            ledger.observe_beacon_head(
                first["signed_beacon_round"],
                audit_registry=first["registry"],
                beacon_public_key=first["beacon_private_key"].public_key(),
                signed_frame=signed_successor_frame,
                frame_public_key=first["frame_private_key"].public_key(),
            )
            parent_round_hash = first["signed_beacon_round"].round.round_hash
            future_round_number = first["signed_beacon_round"].round.round_number + 1
            future_round = sign_beacon_round(
                BeaconRound(
                    first["frame"].beacon_id,
                    future_round_number,
                    parent_round_hash,
                    _digest("release-handover-successor-beacon")[:64]
                    .encode("ascii")[:32],
                ),
                first["beacon_private_key"],
            )
            candidate = Candidate(
                context_hash=successor.context_hash,
                state_context=successor,
                before_model=first["before"],
                after_model=first["after"],
                contributor_root=partition.contributor_root,
                score_probe_commitment=partition.score_probe_commitment,
                source_partition=partition,
                evaluation_policy=first["policy"],
                sampling_frame_hash=successor_frame.frame_hash,
                sealed_catalog_root=successor_frame.catalog_root,
                eligible_probe_id_hashes=(successor_probe.probe_id_hash,),
                risk_schedule_hash=schedule.schedule_hash,
                risk_schedule_index=0,
                risk=allocation,
                beacon_parent_hash=parent_round_hash,
                beacon_round=future_round_number,
                previous_receipt_hash=ZERO_HASH,
            )
            ledger.consume(
                candidate,
                schedule,
                audit_registry=first["registry"],
                beacon_public_key=first["beacon_private_key"].public_key(),
                signed_frame=signed_successor_frame,
                frame_public_key=first["frame_private_key"].public_key(),
            )
            successor_store = CommitProbeStore.successor(
                first["store"],
                probes=[successor_probe],
                partitions=[partition],
                signed_frame=signed_successor_frame,
                path=root / "successor-probe.sqlite3",
            )
            head_before = first["registry"].head
            context_before = first["registry"].context_head
            with self.assertRaisesRegex(ValueError, "already been retired"):
                successor_store.release(
                    candidate,
                    signed_beacon_round=future_round,
                    beacon_public_key=first["beacon_private_key"].public_key(),
                    schedule=schedule,
                    risk_ledger=ledger,
                    audit_registry=first["registry"],
                )
            self.assertEqual(first["registry"].head, head_before)
            self.assertEqual(first["registry"].context_head, context_before)
            with sqlite3.connect(successor_store.path) as db:
                consumed = db.execute(
                    "SELECT consumed FROM probes WHERE probe_id=?",
                    (successor_probe.probe_id,),
                ).fetchone()[0]
            self.assertEqual(consumed, 0)

    def test_reject_advances_only_the_audit_head(self) -> None:
        with TemporaryDirectory(prefix="fedmerit-model-reject-") as directory:
            case = _protocol_case(
                Path(directory),
                name="model-reject",
                after_bias=0.0,
                epsilon=0.10,
                gamma=0.05,
                alpha=0.10,
            )
            context_before = case["registry"].context_head
            version_before = case["registry"].installed_model_version
            _, receipt, appended = _execute(case)
            self.assertTrue(appended)
            self.assertEqual(receipt.core.decision, "reject")
            self.assertNotEqual(case["registry"].head, ZERO_HASH)
            self.assertEqual(case["registry"].context_head, context_before)
            self.assertEqual(
                case["registry"].installed_model_version,
                version_before,
            )


if __name__ == "__main__":
    unittest.main()

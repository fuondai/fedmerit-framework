"""Regression tests for quorum-authorized context handover."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from fedmerit.certificate import (
    AuditRegistry,
    CertificateAuthority,
    VerificationTrust,
    verify_handover_authorization,
)
from fedmerit.model import (
    EvaluationPolicy,
    LinearModelArtifact,
    SecurityProfile,
    StateContext,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _fixture(tmp_path, *, security_profile: SecurityProfile | None = None):
    policy = EvaluationPolicy(
        "handover-policy",
        security_profile=security_profile or SecurityProfile(),
    )
    model = LinearModelArtifact((0.0, 0.0))
    authority = CertificateAuthority.persistent(tmp_path / "witnesses", f=1)
    store = Ed25519PrivateKey.generate()
    frame = Ed25519PrivateKey.generate()
    trust = VerificationTrust.from_keys(
        authority.public_keys,
        f=1,
        store_public_key=store.public_key(),
        frame_public_key=frame.public_key(),
    )
    initial = StateContext(
        "twin-handover",
        "domain-0",
        0,
        _digest("schema-0"),
        policy.policy_hash,
        0,
        trust.authority_certificate_hash,
    )
    successor = StateContext(
        initial.twin_id,
        "domain-1",
        1,
        _digest("schema-1"),
        policy.policy_hash,
        0,
        trust.authority_certificate_hash,
    )
    registry = AuditRegistry(
        tmp_path / "audit.sqlite3",
        genesis_model=model,
        initial_context=initial,
        evaluation_policy=policy,
        verification_trust=trust,
    )
    return policy, model, authority, trust, initial, successor, registry


def test_trusted_context_requires_old_roster_quorum(tmp_path) -> None:
    policy, _, authority, _, _, successor, registry = _fixture(tmp_path)

    with pytest.raises(ValueError, match="old-roster"):
        registry.handover(state_context=successor, evaluation_policy=policy)

    registry.handover(
        state_context=successor,
        evaluation_policy=policy,
        authorizer=authority,
    )
    assert registry.context_head[0] == successor.context_hash


def test_authorization_binds_head_model_and_exact_successor(tmp_path) -> None:
    policy, model, authority, trust, initial, successor, registry = _fixture(tmp_path)
    authorization = authority.authorize_handover(
        previous_context_hash=initial.context_hash,
        successor_context=successor,
        previous_receipt_hash=registry.head,
        installed_model_hash=model.artifact_hash,
        installed_model_version=0,
    )
    assert verify_handover_authorization(authorization, trust)
    tampered = replace(
        authorization,
        installed_model_hash=_digest("different-installed-model"),
    )
    assert not verify_handover_authorization(tampered, trust)
    with pytest.raises(ValueError, match="live transition"):
        registry.handover(
            state_context=successor,
            evaluation_policy=policy,
            authorization=tampered,
        )


def test_honest_witnesses_refuse_conflicting_successors(tmp_path) -> None:
    _, model, authority, _, initial, successor, registry = _fixture(tmp_path)
    authority.authorize_handover(
        previous_context_hash=initial.context_hash,
        successor_context=successor,
        previous_receipt_hash=registry.head,
        installed_model_hash=model.artifact_hash,
        installed_model_version=0,
    )
    conflicting = replace(successor, domain_id="domain-conflict")
    with pytest.raises(ValueError, match=r"fewer than 2f\+1"):
        authority.authorize_handover(
            previous_context_hash=initial.context_hash,
            successor_context=conflicting,
            previous_receipt_hash=registry.head,
            installed_model_hash=model.artifact_hash,
            installed_model_version=0,
        )


def test_handover_count_is_lineage_scoped(tmp_path) -> None:
    policy, _, authority, _, _, successor, registry = _fixture(
        tmp_path,
        security_profile=SecurityProfile(max_handover_count=1),
    )
    registry.provision_lineage_risk_budget(0.5)
    registry.handover(
        state_context=successor,
        evaluation_policy=policy,
        authorizer=authority,
    )
    second = replace(successor, domain_id="domain-2", state_version=2)
    with pytest.raises(ValueError, match="lifetime cap"):
        registry.handover(
            state_context=second,
            evaluation_policy=policy,
            authorizer=authority,
        )


def test_verification_key_cap_is_cumulative_across_handovers(tmp_path) -> None:
    profile = SecurityProfile(max_verification_keys=6)
    policy, _, authority, _, _, successor, registry = _fixture(
        tmp_path,
        security_profile=profile,
    )
    registry.provision_lineage_risk_budget(0.5)
    next_authority = CertificateAuthority.persistent(tmp_path / "next-witnesses", f=1)
    next_trust = VerificationTrust.from_keys(
        next_authority.public_keys,
        f=1,
        store_public_key=Ed25519PrivateKey.generate().public_key(),
        frame_public_key=Ed25519PrivateKey.generate().public_key(),
    )
    successor = replace(
        successor,
        authority_certificate_hash=next_trust.authority_certificate_hash,
    )
    with pytest.raises(ValueError, match="lineage key cap"):
        registry.handover(
            state_context=successor,
            evaluation_policy=policy,
            verification_trust=next_trust,
            authorizer=authority,
        )


def test_lineage_budget_rejects_existing_roots_above_key_cap(tmp_path) -> None:
    _, _, _, _, _, _, registry = _fixture(
        tmp_path,
        security_profile=SecurityProfile(max_verification_keys=5),
    )
    with pytest.raises(ValueError, match="existing verification roots"):
        registry.provision_lineage_risk_budget(0.5)

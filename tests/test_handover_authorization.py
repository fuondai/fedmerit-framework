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
from fedmerit.model import EvaluationPolicy, LinearModelArtifact, StateContext


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _fixture(tmp_path):
    policy = EvaluationPolicy("handover-policy")
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

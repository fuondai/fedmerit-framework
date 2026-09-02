from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fedmerit import (
    EvaluationPolicy,
    LinearModelArtifact,
    ProbeGroup,
    paired_model_loss_difference_exact,
)
from scripts.run_ur3_benchmark import (
    COMMIT_GROUPS,
    CycleGroup,
    _execute_protocol_trial,
    _fltrust,
    _fedval,
    _foundationfl,
    _fault_configuration,
    _balanced_binary_rows,
    _mean_ci,
    _score_aware_updates,
    _split_groups,
)


def _groups(start: int, count: int, *, label: int) -> tuple[CycleGroup, ...]:
    return tuple(
        CycleGroup(
            cycle_id=start + index,
            raw_x=np.asarray([[float(index % 3), float((index + 1) % 5)]]),
            y=np.asarray([label], dtype=np.int64),
        )
        for index in range(count)
    )


@pytest.mark.parametrize(
    ("candidate_bias", "expected_decision", "expected_version", "append_race"),
    (
        (4.0, "commit", 1, True),
        (-4.0, "reject", 0, False),
    ),
)
def test_benchmark_trial_uses_end_to_end_protocol(
    tmp_path,
    candidate_bias: float,
    expected_decision: str,
    expected_version: int,
    append_race: bool,
) -> None:
    before = LinearModelArtifact((0.0, 0.0, 0.0), (0.0, 0.0), (1.0, 1.0))
    after = LinearModelArtifact((0.0, 0.0, candidate_bias), (0.0, 0.0), (1.0, 1.0))
    proposal = _groups(1, 3, label=1)
    score = _groups(100, 2, label=1)
    commit = _groups(200, COMMIT_GROUPS, label=1)
    result = _execute_protocol_trial(
        tmp_path / expected_decision,
        trial_id=f"focused-{expected_decision}",
        before_artifact=before,
        after_artifact=after,
        proposal_groups=proposal,
        proposal_updates=np.asarray(
            ((0.1, 0.0, 0.2), (0.0, 0.1, 0.2), (0.1, 0.1, 0.2))
        ),
        proposal_sizes=np.asarray((1.0, 1.0, 1.0)),
        score_groups=score,
        commit_groups=commit,
        policy=EvaluationPolicy("brier-decimal80-v1"),
        exercise_append_race=append_race,
        entropy_seed=b"e" * 32,
    )

    assert result["receipt_decision"] == expected_decision
    assert result["installed_candidate"] == int(expected_decision == "commit")
    assert result["installed_model_version"] == expected_version
    assert result["append_attempts"] == (2 if append_race else 1)
    assert result["append_successes"] == result["append_attempts"]
    assert result["witness_faults_injected"] == 1
    assert result["witness_signatures"] == result["witness_threshold"] == 3
    for check in (
        "append_succeeded",
        "serving_bytes_verified",
        "risk_consumed",
        "probe_consumed",
        "competing_fixation_blocked",
        "witness_quorum_met",
        "event_chain_valid",
    ):
        assert result[check] == 1
    assert result["protocol_event_count"] >= 8
    assert (result["receipt_delta"] < 0) == (expected_decision == "commit")


def test_two_leaf_catalog_selects_and_spends_only_one_leaf(tmp_path) -> None:
    before = LinearModelArtifact((0.0, 0.0, 0.0), (0.0, 0.0), (1.0, 1.0))
    after = LinearModelArtifact((0.0, 0.0, 4.0), (0.0, 0.0), (1.0, 1.0))
    selected = set()
    for seed in range(20):
        result = _execute_protocol_trial(
            tmp_path / str(seed),
            trial_id=f"two-leaf-conformance-{seed}",
            before_artifact=before,
            after_artifact=after,
            proposal_groups=_groups(1, 2, label=1),
            proposal_updates=np.asarray(((0.1, 0.0, 0.2), (0.0, 0.1, 0.2))),
            proposal_sizes=np.ones(2),
            score_groups=_groups(100, 2, label=1),
            commit_groups=_groups(200, COMMIT_GROUPS * 2, label=1),
            policy=EvaluationPolicy("brier-decimal80-v1"),
            catalog_leaf_count=2,
            entropy_seed=seed.to_bytes(32, "big"),
        )
        selected.add(result["selected_catalog_leaf"])
        assert result["catalog_leaves"] == 2
        assert result["catalog_leaves_consumed"] == 1
        assert result["eligible_subset_blocked"] == 1
        assert len(result["selected_cycle_ids"].split(";")) == COMMIT_GROUPS
        assert result["receipt_decision"] == "commit"
    assert selected == {0, 1}


@pytest.mark.parametrize("mode", ("random", "blocked"))
def test_cycle_partitions_are_disjoint_and_catalog_reproducible(mode) -> None:
    groups = _groups(1, 240, label=1)
    partitions = _split_groups(groups, seed=7, mode=mode)
    repeated = _split_groups(groups, seed=7, mode=mode)
    identifiers = [tuple(group.cycle_id for group in part) for part in partitions]
    assert identifiers == [tuple(group.cycle_id for group in part) for part in repeated]
    assert [len(part) for part in partitions] == [110, 30, 76, 20]
    flattened = [identity for part in identifiers for identity in part]
    assert len(flattened) == len(set(flattened))
    if mode == "blocked":
        assert max(identifiers[0]) < min(identifiers[1])
        assert max(identifiers[1]) < min(identifiers[2])
        assert max(identifiers[2]) < min(identifiers[3])


def test_split_rejects_short_or_duplicate_population() -> None:
    with pytest.raises(ValueError, match="at least 240"):
        _split_groups(_groups(1, 239, label=1), seed=0, mode="random")
    groups = _groups(1, 240, label=1)
    with pytest.raises(ValueError, match="unique"):
        _split_groups(groups[:-1] + groups[:1], seed=0, mode="random")


def test_wilson_intervals_are_asymmetric_and_inside_unit_interval() -> None:
    mean, lower, upper = _mean_ci(pd.Series([0] * 20), binary=True)
    assert mean == lower == 0
    assert 0.16 < upper < 0.162
    mean, lower, upper = _mean_ci(pd.Series([1] * 20), binary=True)
    assert mean == upper == 1
    assert 0.838 < lower < 0.84


def test_score_attack_uses_fixed_query_budget_and_preserves_honest_updates() -> None:
    from scripts.run_ur3_benchmark import ATTACK_QUERIES

    calls = []
    benign = np.asarray(((0.1, 0.2), (0.2, -0.1), (-0.1, 0.1), (0.05, 0.1)))

    def aggregate(updates):
        calls.append(updates.copy())
        return updates.mean(axis=0)

    attacked, selected, _ = _score_aware_updates(
        before=np.asarray((0.2, -0.1)),
        benign_updates=benign,
        sizes=np.ones(4),
        malicious=np.asarray([1]),
        own_x=np.asarray(((-1.0,), (2.0,))),
        own_y=np.asarray((0, 1)),
        score_x=np.asarray(((1.0,), (-2.0,))),
        score_y=np.asarray((1, 0)),
        aggregate=aggregate,
        rng=np.random.default_rng(0),
    )
    assert len(calls) == ATTACK_QUERIES
    assert 1 <= selected <= ATTACK_QUERIES
    np.testing.assert_array_equal(attacked, calls[selected - 1])
    for candidate in calls:
        np.testing.assert_array_equal(candidate[[0, 2, 3]], benign[[0, 2, 3]])


def test_fltrust_uses_root_norm_and_not_client_sample_sizes() -> None:
    updates = np.asarray(((2.0, 0.0), (0.0, 1.0)))
    root = np.asarray((1.0, 0.0))
    expected = np.asarray((1.0, 0.0))
    np.testing.assert_allclose(
        _fltrust(updates, np.asarray((1.0, 100.0)), root), expected
    )


def test_foundationfl_augments_interior_representative_then_trims() -> None:
    updates = np.asarray(
        ((0.0, 0.0), (0.1, 0.1), (0.2, 0.2), (10.0, 10.0)), dtype=float
    )
    # n=4, m=2 synthetic copies, c=1: 4 retained values per coordinate.
    result = _foundationfl(updates, np.ones(4), faults=1)
    assert result.shape == (2,)
    assert np.all(result >= 0.1)
    assert np.all(result <= 10.0)
    np.testing.assert_allclose(result, np.asarray((0.175, 0.175)))


def test_fedval_handles_zero_mad_without_infinite_scores() -> None:
    updates = np.asarray(((0.1, 0.0), (0.1, 0.0), (0.1, 0.0)))
    x = np.asarray(((1.0,), (-1.0,)))
    y = np.asarray((1, 0))
    result = _fedval(np.asarray((0.0, 0.0)), updates, np.ones(3), x, y)
    assert np.isfinite(result).all()


def test_balanced_validation_view_is_deterministic_and_source_ordered() -> None:
    x = np.arange(10, dtype=float).reshape(5, 2)
    y = np.asarray((0, 1, 1, 1, 0))
    balanced_x, balanced_y = _balanced_binary_rows(x, y)
    np.testing.assert_array_equal(balanced_y, np.asarray((0, 1, 1, 0)))
    np.testing.assert_array_equal(
        balanced_x, np.asarray(((0.0, 1.0), (2.0, 3.0), (4.0, 5.0), (8.0, 9.0)))
    )


def test_balanced_validation_view_rejects_single_class() -> None:
    with pytest.raises(ValueError, match="both labels"):
        _balanced_binary_rows(np.ones((3, 2)), np.ones(3, dtype=int))


def test_registered_fault_bound_does_not_leak_clean_attack_label() -> None:
    clean_bound, clean_actual = _fault_configuration(
        "none", clients_per_round=30, byzantine_fraction=0.20
    )
    attack_bound, attack_actual = _fault_configuration(
        "model_replacement", clients_per_round=30, byzantine_fraction=0.20
    )
    assert clean_bound == attack_bound == 6
    assert clean_actual == 0
    assert attack_actual == 6


def test_reused_score_can_accept_when_fresh_probe_rejects() -> None:
    before = LinearModelArtifact((0.0, 0.0, 0.0))
    score_conditioned = LinearModelArtifact((4.0, 4.0, 0.0))
    score = (
        ProbeGroup("score", "0" * 64, ((1.0, 0.0),), (1,)),
    )
    fresh = (
        ProbeGroup("fresh", "1" * 64, ((0.0, 1.0),), (0,)),
    )
    policy = EvaluationPolicy("brier-decimal80-v1")

    reused_delta = paired_model_loss_difference_exact(
        before, score_conditioned, score, policy
    )
    fresh_delta = paired_model_loss_difference_exact(
        before, score_conditioned, fresh, policy
    )

    assert reused_delta < 0
    assert fresh_delta > 0
    assert float(reused_delta) == pytest.approx(-0.249676496251, abs=1e-12)
    assert float(fresh_delta) == pytest.approx(0.714351083825, abs=1e-12)

from __future__ import annotations

import numpy as np
import pytest

from fedmerit import EvaluationPolicy, LinearModelArtifact
from scripts.run_ur3_benchmark import (
    COMMIT_GROUPS,
    CycleGroup,
    _execute_protocol_trial,
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

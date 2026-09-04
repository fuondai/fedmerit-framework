"""Regression tests for the controlled adaptive-reuse separation."""

from __future__ import annotations

import pytest

from scripts.run_adaptive_reuse_challenge import (
    required_groups,
    run_challenge,
    run_trial,
    summarize,
)


def test_registered_probe_size_matches_finite_sample_rule() -> None:
    assert required_groups(0.10, 0.10, 0.0) == 461


def test_candidate_is_accepted_on_reused_rows_and_rejected_on_fresh_rows() -> None:
    row = run_trial(0)
    assert row["score_delta"] == pytest.approx(-0.25)
    assert row["catalog_delta"] >= 0.10
    assert row["fresh_delta"] > 0.0
    assert row["reused_score_escape"] == 1
    assert row["fresh_probe_escape"] == 0


def test_reference_challenge_has_complete_separation() -> None:
    report = summarize(run_challenge())
    assert report["trials"] == 100
    assert report["harmful_candidates"] == 100
    assert report["reused_score_escapes"] == 100
    assert report["fresh_probe_escapes"] == 0


def test_challenge_rejects_invalid_population() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        run_trial(0, population_size=900)

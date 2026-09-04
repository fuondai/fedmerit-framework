#!/usr/bin/env python3
"""Controlled separation between reused-score and fresh-probe decisions.

The candidate memorizes the score partition and uses a deterministic prediction
on every unseen identifier. A trusted harness keeps the source partition and
probe RNG capabilities separate from the candidate view: the candidate receives
only score rows and its own prediction key. The construction isolates adaptive
reuse; it is not an FL workload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ALPHA = 0.10
EPSILON = 0.10
GAMMA = 0.0
POPULATION_SIZE = 2_000
TRIALS = 100


def required_groups(alpha: float, epsilon: float, gamma: float) -> int:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if epsilon <= 0.0 or gamma < 0.0:
        raise ValueError("epsilon must be positive and gamma nonnegative")
    return math.ceil(2.0 * math.log(1.0 / alpha) / (epsilon + gamma) ** 2)


def _unseen_prediction(candidate_key: int, identifier: int) -> float:
    encoded = f"fedmerit-reuse-challenge:{candidate_key}:{identifier}".encode()
    return float(hashlib.sha256(encoded).digest()[0] & 1)


@dataclass(frozen=True)
class MemorizingCandidate:
    candidate_key: int
    memorized: dict[int, int]

    @classmethod
    def from_score_view(cls, view: "CandidateScoreView") -> "MemorizingCandidate":
        return cls(
            candidate_key=view.candidate_key,
            memorized={
                int(identifier): int(label)
                for identifier, label in zip(
                    view.score_ids, view.score_labels, strict=True
                )
            },
        )

    def predict(self, identifiers: np.ndarray) -> np.ndarray:
        return np.fromiter(
            (
                self.memorized.get(
                    int(identifier),
                    int(_unseen_prediction(self.candidate_key, int(identifier))),
                )
                for identifier in identifiers
            ),
            dtype=float,
            count=len(identifiers),
        )


@dataclass(frozen=True)
class CandidateScoreView:
    """The complete interface available to the candidate before fixation."""

    candidate_key: int
    score_ids: np.ndarray
    score_labels: np.ndarray


@dataclass
class SourceOnlySampler:
    """Trusted source capability; this object is never passed to a candidate."""

    score_ids: np.ndarray
    _catalog_ids: np.ndarray
    _probe_rng: np.random.Generator

    @classmethod
    def from_streams(
        cls,
        *,
        population_size: int,
        probe_size: int,
        source_stream: np.random.SeedSequence,
        probe_stream: np.random.SeedSequence,
    ) -> "SourceOnlySampler":
        source_order = np.random.default_rng(source_stream).permutation(population_size)
        return cls(
            score_ids=source_order[:probe_size],
            _catalog_ids=source_order[probe_size:],
            _probe_rng=np.random.default_rng(probe_stream),
        )

    def candidate_view(
        self, labels: np.ndarray, candidate_key: int
    ) -> CandidateScoreView:
        return CandidateScoreView(
            candidate_key=candidate_key,
            score_ids=self.score_ids.copy(),
            score_labels=labels[self.score_ids].copy(),
        )

    def fresh_probe(self, probe_size: int) -> np.ndarray:
        return self._probe_rng.choice(
            self._catalog_ids, size=probe_size, replace=False
        )

    def catalog_for_evaluation(self) -> np.ndarray:
        """Return the sealed catalog to the trusted evaluator, never the candidate."""

        return self._catalog_ids.copy()


def _trial_interfaces(
    trial_id: int, *, population_size: int, probe_size: int
) -> tuple[np.ndarray, MemorizingCandidate, SourceOnlySampler]:
    """Create source and candidate interfaces from disjoint harness streams.

    ``trial_id`` indexes a retained result row. It is not accepted by the
    candidate API; only the harness uses it to make the construction reproducible.
    """

    data_stream, candidate_stream, source_stream, probe_stream = (
        np.random.SeedSequence(trial_id).spawn(4)
    )
    labels = np.random.default_rng(data_stream).integers(
        0, 2, size=population_size, dtype=np.int8
    )
    candidate_key = int(
        np.random.default_rng(candidate_stream).integers(
            0, np.iinfo(np.uint64).max, dtype=np.uint64
        )
    )
    source = SourceOnlySampler.from_streams(
        population_size=population_size,
        probe_size=probe_size,
        source_stream=source_stream,
        probe_stream=probe_stream,
    )
    candidate = MemorizingCandidate.from_score_view(
        source.candidate_view(labels, candidate_key)
    )
    return labels, candidate, source


def _paired_delta(
    candidate: MemorizingCandidate,
    identifiers: np.ndarray,
    labels: np.ndarray,
) -> float:
    candidate_loss = (candidate.predict(identifiers) - labels[identifiers]) ** 2
    baseline_loss = (0.5 - labels[identifiers]) ** 2
    return float(np.mean(candidate_loss - baseline_loss))


def run_trial(
    trial_id: int,
    *,
    population_size: int = POPULATION_SIZE,
    alpha: float = ALPHA,
    epsilon: float = EPSILON,
    gamma: float = GAMMA,
) -> dict[str, int | float]:
    probe_size = required_groups(alpha, epsilon, gamma)
    if population_size < 2 * probe_size:
        raise ValueError("population must hold disjoint score and fresh probes")

    labels, candidate, source = _trial_interfaces(
        trial_id, population_size=population_size, probe_size=probe_size
    )
    score_ids = source.score_ids
    catalog_ids = source.catalog_for_evaluation()
    fresh_ids = source.fresh_probe(probe_size)

    score_delta = _paired_delta(candidate, score_ids, labels)
    catalog_delta = _paired_delta(candidate, catalog_ids, labels)
    fresh_delta = _paired_delta(candidate, fresh_ids, labels)
    harmful = catalog_delta >= epsilon
    reused_accepted = score_delta <= -gamma
    fresh_accepted = fresh_delta <= -gamma
    return {
        "trial_id": trial_id,
        "population_size": population_size,
        "score_size": probe_size,
        "catalog_size": len(catalog_ids),
        "fresh_probe_size": probe_size,
        "alpha": alpha,
        "epsilon": epsilon,
        "gamma": gamma,
        "score_delta": score_delta,
        "catalog_delta": catalog_delta,
        "fresh_delta": fresh_delta,
        "harmful": int(harmful),
        "reused_score_accepted": int(reused_accepted),
        "fresh_probe_accepted": int(fresh_accepted),
        "reused_score_escape": int(harmful and reused_accepted),
        "fresh_probe_escape": int(harmful and fresh_accepted),
    }


def run_challenge(
    *,
    trials: int = TRIALS,
    population_size: int = POPULATION_SIZE,
    alpha: float = ALPHA,
    epsilon: float = EPSILON,
    gamma: float = GAMMA,
) -> pd.DataFrame:
    if trials <= 0:
        raise ValueError("trials must be positive")
    return pd.DataFrame(
        [
            run_trial(
                trial_id,
                population_size=population_size,
                alpha=alpha,
                epsilon=epsilon,
                gamma=gamma,
            )
            for trial_id in range(trials)
        ]
    )


def summarize(frame: pd.DataFrame) -> dict[str, int | float | str]:
    harmful = frame[frame["harmful"].astype(bool)]
    return {
        "design": "score-memorization with deterministic unseen predictions",
        "trials": int(len(frame)),
        "harmful_candidates": int(len(harmful)),
        "reused_score_escapes": int(harmful["reused_score_escape"].sum()),
        "fresh_probe_escapes": int(harmful["fresh_probe_escape"].sum()),
        "score_delta_min": float(frame["score_delta"].min()),
        "score_delta_max": float(frame["score_delta"].max()),
        "catalog_delta_min": float(frame["catalog_delta"].min()),
        "catalog_delta_max": float(frame["catalog_delta"].max()),
        "fresh_delta_min": float(frame["fresh_delta"].min()),
        "fresh_delta_max": float(frame["fresh_delta"].max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--population-size", type=int, default=POPULATION_SIZE)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    args = parser.parse_args()

    frame = run_challenge(
        trials=args.trials,
        population_size=args.population_size,
        alpha=args.alpha,
        epsilon=args.epsilon,
        gamma=args.gamma,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "records.csv", index=False)
    summary = summarize(frame)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

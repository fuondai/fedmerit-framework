#!/usr/bin/env python3
"""Controlled separation between reused-score and fresh-probe decisions.

The candidate memorizes the score partition and uses a deterministic prediction
on every unseen identifier. The source seals the disjoint catalog before the
candidate is fixed, and a source-only RNG samples the fresh probe after
fixation. The construction isolates adaptive reuse; it is not an FL workload.
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


def _unseen_prediction(candidate_seed: int, identifier: int) -> float:
    encoded = f"fedmerit-reuse-challenge:{candidate_seed}:{identifier}".encode()
    return float(hashlib.sha256(encoded).digest()[0] & 1)


@dataclass(frozen=True)
class MemorizingCandidate:
    candidate_seed: int
    memorized: dict[int, int]

    def predict(self, identifiers: np.ndarray) -> np.ndarray:
        return np.fromiter(
            (
                self.memorized.get(
                    int(identifier),
                    int(_unseen_prediction(self.candidate_seed, int(identifier))),
                )
                for identifier in identifiers
            ),
            dtype=float,
            count=len(identifiers),
        )


def _paired_delta(
    candidate: MemorizingCandidate,
    identifiers: np.ndarray,
    labels: np.ndarray,
) -> float:
    candidate_loss = (candidate.predict(identifiers) - labels[identifiers]) ** 2
    baseline_loss = (0.5 - labels[identifiers]) ** 2
    return float(np.mean(candidate_loss - baseline_loss))


def run_trial(
    seed: int,
    *,
    population_size: int = POPULATION_SIZE,
    alpha: float = ALPHA,
    epsilon: float = EPSILON,
    gamma: float = GAMMA,
) -> dict[str, int | float]:
    probe_size = required_groups(alpha, epsilon, gamma)
    if population_size < 2 * probe_size:
        raise ValueError("population must hold disjoint score and fresh probes")

    data_seed, source_seed, probe_seed = np.random.SeedSequence(seed).spawn(3)
    labels = np.random.default_rng(data_seed).integers(
        0, 2, size=population_size, dtype=np.int8
    )
    source_order = np.random.default_rng(source_seed).permutation(population_size)
    score_ids = source_order[:probe_size]
    catalog_ids = source_order[probe_size:]

    # Only score identifiers and labels enter the candidate. Catalog membership,
    # labels, and the later probe draw remain outside the candidate interface.
    candidate = MemorizingCandidate(
        candidate_seed=seed,
        memorized={int(i): int(labels[i]) for i in score_ids},
    )
    fresh_ids = np.random.default_rng(probe_seed).choice(
        catalog_ids, size=probe_size, replace=False
    )

    score_delta = _paired_delta(candidate, score_ids, labels)
    catalog_delta = _paired_delta(candidate, catalog_ids, labels)
    fresh_delta = _paired_delta(candidate, fresh_ids, labels)
    harmful = catalog_delta >= epsilon
    reused_accepted = score_delta <= -gamma
    fresh_accepted = fresh_delta <= -gamma
    return {
        "seed": seed,
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
                seed,
                population_size=population_size,
                alpha=alpha,
                epsilon=epsilon,
                gamma=gamma,
            )
            for seed in range(trials)
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

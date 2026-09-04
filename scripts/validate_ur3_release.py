#!/usr/bin/env python3
"""Validate a retained UR3 benchmark release from its primitive rows."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_ur3_benchmark import (
    CATALOG_LEAVES,
    COMMIT_GROUPS,
    COMMIT_POOL_GROUPS,
    EPSILON,
    GAMMA,
    OPERATIONAL_HARM,
    _summarize,
)


METHOD_ATTACKS = {
    "FedAvg": {"none", "sign_flip", "model_replacement", "score_aware"},
    "CoordinateMedian": {"none", "sign_flip", "model_replacement"},
    "Krum": {"none", "sign_flip", "model_replacement"},
    "FLTrust": {"none", "sign_flip", "model_replacement", "score_aware"},
    "FedVal": {"none", "sign_flip", "model_replacement"},
    "FLShield": {"none", "sign_flip", "model_replacement"},
    "FoundationFL": {"none", "sign_flip", "model_replacement", "score_aware"},
}
PROTOCOL_CHECKS = (
    "append_succeeded",
    "serving_bytes_verified",
    "risk_consumed",
    "probe_consumed",
    "competing_fixation_blocked",
    "eligible_subset_blocked",
    "witness_quorum_met",
    "event_chain_valid",
)
PARTITIONS = {
    "proposal_cycle_ids": 30,
    "score_cycle_ids": 30,
    "catalog_cycle_ids": 76,
    "audit_cycle_ids": 20,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _cycle_set(value: object, *, field: str) -> set[int]:
    _require(isinstance(value, str) and bool(value), f"{field} is empty")
    try:
        return {int(item) for item in value.split(";")}
    except ValueError as error:
        raise ValueError(f"{field} contains a non-integer identifier") from error


def _validate_partitions(frame: pd.DataFrame) -> None:
    unique_splits = frame[["seed", *PARTITIONS]].drop_duplicates()
    _require(len(unique_splits) == 20, "expected one partition tuple per seed")
    for row in unique_splits.itertuples(index=False):
        groups = {
            field: _cycle_set(getattr(row, field), field=field) for field in PARTITIONS
        }
        for field, expected_size in PARTITIONS.items():
            _require(
                len(groups[field]) == expected_size,
                f"seed {row.seed}: {field} has {len(groups[field])} cycles",
            )
        fields = tuple(PARTITIONS)
        for index, left in enumerate(fields):
            for right in fields[index + 1 :]:
                _require(
                    groups[left].isdisjoint(groups[right]),
                    f"seed {row.seed}: {left} overlaps {right}",
                )


def _validate_coverage(frame: pd.DataFrame, split: str) -> None:
    _require(set(frame["split"]) == {split}, f"release is not split={split!r}")
    _require(set(frame["seed"]) == set(range(20)), "seed coverage must be 0 through 19")
    _require(
        not frame.duplicated(["seed", "method", "attack"]).any(),
        "duplicate seed/method/attack row",
    )
    expected_pairs = {
        (method, attack) for method, attacks in METHOD_ATTACKS.items() for attack in attacks
    }
    observed_pairs = set(frame[["method", "attack"]].itertuples(index=False, name=None))
    _require(observed_pairs == expected_pairs, "method/attack coverage differs from contract")
    _require(len(frame) == 20 * len(expected_pairs), "unexpected transition count")


def _validate_fault_contract(frame: pd.DataFrame) -> None:
    _require(
        set(frame["configured_fault_bound"]) == {6},
        "every condition must register f=6",
    )
    clean = frame["attack"] == "none"
    _require(
        set(frame.loc[clean, "actual_byzantine_clients"]) == {0},
        "clean rows must inject zero Byzantine clients",
    )
    _require(
        set(frame.loc[~clean, "actual_byzantine_clients"]) == {6},
        "attack rows must inject six Byzantine clients",
    )


def _validate_decisions(frame: pd.DataFrame) -> None:
    accepted = frame["accepted"].astype(bool)
    score_accepted = frame["score_gate_accepted"].astype(bool)
    checks = {
        "installed_candidate": accepted,
        "population_escape": accepted & frame["population_harm"].astype(bool),
        "harmful_escape": accepted & frame["operational_harm"].astype(bool),
        "declared_harmful_escape": accepted & frame["declared_harm"].astype(bool),
        "score_gate_population_escape": score_accepted
        & frame["population_harm"].astype(bool),
        "score_gate_escape": score_accepted & frame["operational_harm"].astype(bool),
    }
    for column, expected in checks.items():
        _require(
            frame[column].astype(bool).equals(expected),
            f"{column} is inconsistent with primitive decision fields",
        )
    expected_receipt = accepted.map({True: "commit", False: "reject"})
    _require(
        frame["receipt_decision"].equals(expected_receipt),
        "receipt decision disagrees with accepted flag",
    )
    for column in PROTOCOL_CHECKS:
        _require(set(frame[column]) == {1}, f"protocol check failed: {column}")


def _validate_numeric_semantics(frame: pd.DataFrame) -> None:
    numeric = (
        "commit_delta",
        "audit_delta",
        "population_delta",
        "score_delta",
        "receipt_delta",
        "before_balanced_accuracy",
        "candidate_balanced_accuracy",
        "installed_balanced_accuracy",
        "paired_accuracy_gain",
        "paired_audit_risk_reduction",
    )
    _require(
        np.isfinite(frame[list(numeric)].to_numpy(dtype=float)).all(),
        "release contains a non-finite primitive metric",
    )
    accepted = frame["accepted"].astype(bool)
    score_accepted = frame["score_gate_accepted"].astype(bool)
    identities = {
        "accepted threshold": accepted == (frame["commit_delta"] <= -GAMMA),
        "reused-score threshold": score_accepted
        == (frame["score_delta"] <= -GAMMA),
        "catalog harm threshold": frame["population_harm"].astype(bool)
        == (frame["population_delta"] >= EPSILON),
        "audit diagnostic threshold": frame["operational_harm"].astype(bool)
        == (frame["audit_delta"] >= OPERATIONAL_HARM),
        "audit declared-harm threshold": frame["declared_harm"].astype(bool)
        == (frame["audit_delta"] >= EPSILON),
        "beneficial threshold": frame["beneficial"].astype(bool)
        == (frame["audit_delta"] < 0.0),
        "false rejection": frame["false_rejection"].astype(bool)
        == ((~accepted) & (frame["audit_delta"] < 0.0)),
    }
    for name, values in identities.items():
        _require(bool(values.all()), f"primitive identity failed: {name}")

    _require(
        np.allclose(frame["receipt_delta"], frame["commit_delta"], atol=1e-12),
        "receipt delta differs from the selected fresh-probe statistic",
    )
    expected_installed = np.where(
        accepted,
        frame["candidate_balanced_accuracy"],
        frame["before_balanced_accuracy"],
    )
    _require(
        np.allclose(frame["installed_balanced_accuracy"], expected_installed),
        "installed balanced accuracy disagrees with the receipt decision",
    )
    _require(
        np.allclose(
            frame["paired_accuracy_gain"],
            frame["installed_balanced_accuracy"]
            - frame["candidate_balanced_accuracy"],
        ),
        "paired accuracy recovery is inconsistent",
    )
    _require(
        np.allclose(
            frame["paired_audit_risk_reduction"],
            np.where(accepted, 0.0, frame["audit_delta"]),
        ),
        "paired audit-risk reduction is inconsistent",
    )

    _require(set(frame["catalog_leaves"]) == {CATALOG_LEAVES}, "catalog size mismatch")
    _require(
        set(frame["catalog_leaves_consumed"]) == {1},
        "every transition must consume exactly one catalog leaf",
    )
    _require(set(frame["witness_faults_injected"]) == {1}, "witness fault mismatch")
    _require(set(frame["witness_threshold"]) == {3}, "witness threshold mismatch")
    _require(set(frame["witness_signatures"]) == {3}, "signature count mismatch")
    _require(
        frame["receipt_hash"].map(
            lambda value: bool(re.fullmatch(r"[0-9a-f]{64}", str(value)))
        ).all(),
        "receipt hash is not canonical lowercase SHA-256",
    )
    for row in frame[["catalog_cycle_ids", "selected_cycle_ids"]].itertuples(
        index=False
    ):
        catalog = _cycle_set(row.catalog_cycle_ids, field="catalog_cycle_ids")
        selected = {
            int(item.removeprefix("cycle-"))
            for item in str(row.selected_cycle_ids).split(";")
        }
        _require(len(selected) == COMMIT_GROUPS, "selected leaf has wrong group count")
        _require(selected <= catalog, "selected leaf is outside the sealed catalog")


def _validate_metadata(metadata: dict[str, object], frame: pd.DataFrame, split: str) -> None:
    _require(metadata.get("records") == len(frame), "metadata record count mismatch")
    _require(metadata.get("split") == split, "metadata split mismatch")
    design = metadata.get("benchmark_design")
    _require(isinstance(design, dict), "benchmark_design metadata is missing")
    _require(design.get("registered_fault_bound") == 6, "metadata fault bound mismatch")
    _require(
        design.get("clean_actual_byzantine_clients") == 0,
        "metadata clean fault count mismatch",
    )
    _require(
        design.get("end_to_end_training_comparison") is False,
        "benchmark must identify itself as a candidate-transition test",
    )
    allocation = metadata.get("split_groups_per_seed")
    _require(isinstance(allocation, dict), "split allocation metadata is missing")
    _require(allocation.get("commit_pool") == COMMIT_POOL_GROUPS, "pool size mismatch")
    _require(
        allocation.get("commit_selected") == COMMIT_GROUPS * CATALOG_LEAVES,
        "sealed catalog allocation mismatch",
    )
    _require(
        allocation.get("commit_unused")
        == COMMIT_POOL_GROUPS - COMMIT_GROUPS * CATALOG_LEAVES,
        "unused pool allocation mismatch",
    )
    environment = metadata.get("environment")
    required_environment = {
        "python",
        "python_implementation",
        "numpy",
        "pandas",
        "scipy",
        "scikit_learn",
        "openpyxl",
        "cryptography",
        "platform",
        "machine",
        "logical_cpu_count",
    }
    _require(isinstance(environment, dict), "environment metadata is missing")
    _require(
        all(environment.get(field) not in {None, ""} for field in required_environment),
        "environment metadata is incomplete",
    )


def _validate_summary(frame: pd.DataFrame, retained: pd.DataFrame) -> None:
    recomputed = _summarize(frame).reset_index(drop=True)
    retained = retained.sort_values(["split", "attack", "method"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        retained,
        recomputed,
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def validate_release(root: Path, *, split: str) -> dict[str, int | str]:
    raw_path = root / "raw_runs.csv"
    summary_path = root / "summary.csv"
    metadata_path = root / "metadata.json"
    for path in (raw_path, summary_path, metadata_path):
        _require(path.is_file(), f"missing release file: {path.name}")
    frame = pd.read_csv(raw_path)
    summary = pd.read_csv(summary_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required_columns = {
        "seed",
        "split",
        "method",
        "attack",
        "configured_fault_bound",
        "actual_byzantine_clients",
        "accepted",
        "score_gate_accepted",
        "population_harm",
        "population_escape",
        "score_gate_population_escape",
        "operational_harm",
        "harmful_escape",
        "score_gate_escape",
        "declared_harm",
        "declared_harmful_escape",
        "installed_candidate",
        "receipt_decision",
        "commit_delta",
        "audit_delta",
        "population_delta",
        "score_delta",
        "receipt_delta",
        "beneficial",
        "false_rejection",
        "before_balanced_accuracy",
        "candidate_balanced_accuracy",
        "installed_balanced_accuracy",
        "paired_accuracy_gain",
        "paired_audit_risk_reduction",
        "receipt_hash",
        "catalog_leaves",
        "selected_cycle_ids",
        "catalog_leaves_consumed",
        "witness_faults_injected",
        "witness_signatures",
        "witness_threshold",
        *PROTOCOL_CHECKS,
        *PARTITIONS,
    }
    missing = sorted(required_columns - set(frame.columns))
    _require(not missing, f"raw release is missing columns: {missing}")
    _validate_coverage(frame, split)
    _validate_fault_contract(frame)
    _validate_decisions(frame)
    _validate_numeric_semantics(frame)
    _validate_partitions(frame)
    _validate_metadata(metadata, frame, split)
    _validate_summary(frame, summary)
    return {
        "split": split,
        "transitions": len(frame),
        "seeds": int(frame["seed"].nunique()),
        "catalog_harmful": int(frame["population_harm"].sum()),
        "catalog_escapes": int(frame["population_escape"].sum()),
        "audit_diagnostic_harmful": int(frame["operational_harm"].sum()),
        "audit_diagnostic_escapes": int(frame["harmful_escape"].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--split", choices=("random", "blocked"), required=True)
    args = parser.parse_args()
    print(json.dumps(validate_release(args.release, split=args.split), indent=2))


if __name__ == "__main__":
    main()

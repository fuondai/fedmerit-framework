"""Protocol-matched FL benchmark on the UCI UR3 CobotOps dataset.

The benchmark treats operation cycles as non-IID source groups. Proposal,
validation, commit-probe, and audit cycles are disjoint for every seed. It
compares several proposal aggregators before and after the same FedMERIT gate;
the protocol conformance suite remains the authority for cryptographic and
state-machine behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from fedmerit import (
    EvaluationPolicy,
    LinearModelArtifact,
    ProbeGroup,
    paired_model_loss_difference_exact,
    required_groups,
    risk_bound,
)


FEATURE_COLUMNS = (
    "Current_J0",
    "Temperature_T0",
    "Current_J1",
    "Temperature_J1",
    "Current_J2",
    "Temperature_J2",
    "Current_J3",
    "Temperature_J3",
    "Current_J4",
    "Temperature_J4",
    "Current_J5",
    "Temperature_J5",
    "Speed_J0",
    "Speed_J1",
    "Speed_J2",
    "Speed_J3",
    "Speed_J4",
    "Speed_J5",
    "Tool_current",
)
DATASET_DOI = "10.24432/C5J891"
ALPHA = 0.10
EPSILON = 0.35
GAMMA = 0.0
# This secondary threshold is an operational stress-test marker; the certified
# contract still uses EPSILON.  Keeping both metrics avoids conflating a useful
# workload-level diagnostic with the theorem's declared-harm event.
OPERATIONAL_HARM = 0.05
COMMIT_GROUPS = required_groups(ALPHA, EPSILON, GAMMA)


@dataclass(frozen=True)
class CycleGroup:
    cycle_id: int
    raw_x: np.ndarray
    y: np.ndarray


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits, -80.0, 80.0)
    return np.where(
        clipped >= 0,
        1.0 / (1.0 + np.exp(-clipped)),
        np.exp(clipped) / (1.0 + np.exp(clipped)),
    )


def _predict(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    return _sigmoid(x @ weights[:-1] + weights[-1])


def _balanced_accuracy(weights: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    predicted = _predict(weights, x) >= 0.5
    recalls = []
    for label in (0, 1):
        mask = y == label
        if np.any(mask):
            recalls.append(float(np.mean(predicted[mask] == bool(label))))
    return float(np.mean(recalls))


def _brier(weights: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((_predict(weights, x) - y) ** 2))


def _local_update(
    weights: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float,
    epochs: int,
    positive_weight: float,
) -> np.ndarray:
    result = weights.copy()
    sample_weights = np.where(y == 1, positive_weight, 1.0)
    normalizer = float(np.sum(sample_weights))
    for _ in range(epochs):
        error = (_predict(result, x) - y) * sample_weights
        gradient = np.empty_like(result)
        gradient[:-1] = x.T @ error / normalizer + 1e-4 * result[:-1]
        gradient[-1] = float(np.sum(error) / normalizer)
        result -= learning_rate * gradient
    return result - weights


def _weighted_mean(updates: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    return np.average(updates, axis=0, weights=sizes)


def _coordinate_median(updates: np.ndarray, _: np.ndarray) -> np.ndarray:
    return np.median(updates, axis=0)


def _krum(updates: np.ndarray, _: np.ndarray, faults: int) -> np.ndarray:
    count = len(updates)
    neighbours = count - faults - 2
    if neighbours <= 0:
        raise ValueError("Krum requires n > f + 2")
    distances = np.sum((updates[:, None, :] - updates[None, :, :]) ** 2, axis=2)
    scores = []
    for index in range(count):
        ordered = np.sort(np.delete(distances[index], index))
        scores.append(float(np.sum(ordered[:neighbours])))
    return updates[int(np.argmin(scores))]


def _fltrust(updates: np.ndarray, sizes: np.ndarray, root_update: np.ndarray) -> np.ndarray:
    root_norm = float(np.linalg.norm(root_update))
    if root_norm == 0:
        return np.zeros_like(root_update)
    norms = np.linalg.norm(updates, axis=1)
    cosine = (updates @ root_update) / np.maximum(norms * root_norm, 1e-15)
    trust = np.maximum(cosine, 0.0)
    clipped = updates * np.minimum(1.0, root_norm / np.maximum(norms, 1e-15))[:, None]
    weights = trust * sizes
    if float(np.sum(weights)) == 0:
        return np.zeros_like(root_update)
    return np.average(clipped, axis=0, weights=weights)


def _fedval(
    before: np.ndarray,
    updates: np.ndarray,
    sizes: np.ndarray,
    score_x: np.ndarray,
    score_y: np.ndarray,
) -> np.ndarray:
    losses = np.asarray([_brier(before + update, score_x, score_y) for update in updates])
    scale = max(float(np.std(losses)), 1e-6)
    validation_weights = np.exp(-(losses - float(np.min(losses))) / scale)
    return np.average(updates, axis=0, weights=validation_weights * sizes)


def _flshield_cluster(
    before: np.ndarray,
    updates: np.ndarray,
    sizes: np.ndarray,
    validator_groups: tuple[tuple[np.ndarray, np.ndarray], ...],
    *,
    seed: int,
) -> np.ndarray:
    """Tabular instantiation of FLShield's cluster representative path.

    Validators are honest in this experiment. K-Means plus silhouette selection
    creates representative models; classwise loss-impact scores rank them. The
    accepted clusters map back to individual updates before norm clipping and
    aggregation, matching the paper's five-stage path.
    """
    best_labels = None
    best_score = -math.inf
    max_clusters = min(6, len(updates) - 1)
    for clusters in range(2, max_clusters + 1):
        labels = KMeans(
            n_clusters=clusters,
            n_init=10,
            random_state=seed,
        ).fit_predict(updates)
        if len(set(labels.tolist())) < 2:
            continue
        score = float(silhouette_score(updates, labels))
        if score > best_score:
            best_labels = labels
            best_score = score
    if best_labels is None:
        return _coordinate_median(updates, sizes)

    cluster_ids = sorted(set(best_labels.tolist()))
    representative_scores = []
    for cluster_id in cluster_ids:
        member_mask = best_labels == cluster_id
        representative = np.average(
            updates[member_mask], axis=0, weights=sizes[member_mask]
        )
        impacts = []
        for x, y in validator_groups:
            per_class = []
            for label in (0, 1):
                mask = y == label
                if np.any(mask):
                    per_class.append(
                        _brier(before, x[mask], y[mask])
                        - _brier(before + representative, x[mask], y[mask])
                    )
            if per_class:
                impacts.append(min(per_class))
        representative_scores.append(float(np.median(impacts)))
    keep_count = max(1, math.ceil(len(cluster_ids) / 2))
    keep_clusters = {
        cluster_ids[index]
        for index in np.argsort(representative_scores)[-keep_count:]
    }
    accepted = np.asarray(
        [index for index, label in enumerate(best_labels) if label in keep_clusters]
    )
    accepted_updates = updates[accepted]
    accepted_sizes = sizes[accepted]
    norms = np.linalg.norm(accepted_updates, axis=1)
    clip = float(np.median(norms))
    clipped = accepted_updates * np.minimum(
        1.0, clip / np.maximum(norms, 1e-15)
    )[:, None]
    return np.average(clipped, axis=0, weights=accepted_sizes)


def _load_groups(path: Path) -> tuple[CycleGroup, ...]:
    frame = pd.read_excel(path, engine="openpyxl")
    required = set(FEATURE_COLUMNS) | {"cycle ", "Robot_ProtectiveStop", "grip_lost"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")
    selected = frame[list(FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    cycle = pd.to_numeric(frame["cycle "], errors="coerce")
    stop = pd.to_numeric(frame["Robot_ProtectiveStop"], errors="coerce")
    grip = pd.to_numeric(frame["grip_lost"], errors="coerce")
    valid = selected.notna().all(axis=1) & cycle.notna() & stop.notna() & grip.notna()
    selected = selected.loc[valid]
    labels = ((stop.loc[valid] > 0) | (grip.loc[valid] > 0)).astype(int)
    cycles = cycle.loc[valid].astype(int)
    groups = []
    for cycle_id in sorted(cycles.unique().tolist()):
        mask = cycles == cycle_id
        groups.append(
            CycleGroup(
                int(cycle_id),
                selected.loc[mask].to_numpy(dtype=np.float64),
                labels.loc[mask].to_numpy(dtype=np.int64),
            )
        )
    if len(groups) < 4 * 40:
        raise ValueError("benchmark requires at least 160 distinct operation cycles")
    return tuple(groups)


def _stack(
    groups: tuple[CycleGroup, ...], mean: np.ndarray, scale: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.vstack([group.raw_x for group in groups])
    labels = np.concatenate([group.y for group in groups])
    return (raw - mean) / scale, labels


def _probe_groups(groups: tuple[CycleGroup, ...]) -> tuple[ProbeGroup, ...]:
    return tuple(
        ProbeGroup(
            f"cycle-{group.cycle_id:04d}",
            hashlib.sha256(
                f"uci-ur3-cobotops:{DATASET_DOI}:cycle:{group.cycle_id}".encode()
            ).hexdigest(),
            tuple(tuple(float(value) for value in row) for row in group.raw_x),
            tuple(int(value) for value in group.y),
        )
        for group in sorted(groups, key=lambda item: item.cycle_id)
    )


def _mean_ci(values: pd.Series, *, binary: bool = False) -> tuple[float, float]:
    array = values.to_numpy(dtype=float)
    mean = float(np.mean(array))
    if binary:
        # Wilson's interval remains informative at 0 and 1, unlike the
        # normal approximation's zero-width interval at the boundaries.
        size = len(array)
        z = 1.96
        denominator = 1.0 + z * z / size
        center = (mean + z * z / (2.0 * size)) / denominator
        radius = (
            z
            * math.sqrt(
                mean * (1.0 - mean) / size + z * z / (4.0 * size * size)
            )
            / denominator
        )
        return mean, max(center + radius - mean, mean - (center - radius))
    half_width = (
        0.0
        if len(array) < 2
        else 1.96 * float(np.std(array, ddof=1)) / math.sqrt(len(array))
    )
    return mean, half_width


def _run_seed(
    groups: tuple[CycleGroup, ...],
    *,
    seed: int,
    pretrain_rounds: int,
    clients_per_round: int,
    byzantine_fraction: float,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    train = tuple(groups[index] for index in order[:120])
    score = tuple(groups[index] for index in order[120:160])
    commit_pool = tuple(groups[index] for index in order[160:200])
    audit = tuple(groups[index] for index in order[200:])
    commit = tuple(commit_pool[index] for index in sorted(rng.choice(40, COMMIT_GROUPS, replace=False)))

    train_raw = np.vstack([group.raw_x for group in train])
    mean = np.mean(train_raw, axis=0)
    scale = np.std(train_raw, axis=0)
    scale[scale < 1e-12] = 1.0
    train_xy = {group.cycle_id: _stack((group,), mean, scale) for group in train}
    score_x, score_y = _stack(score, mean, scale)
    audit_x, audit_y = _stack(audit, mean, scale)
    positives = int(sum(int(np.sum(group.y)) for group in train))
    rows = int(sum(len(group.y) for group in train))
    positive_weight = max(1.0, (rows - positives) / max(positives, 1))

    weights = np.zeros(len(FEATURE_COLUMNS) + 1, dtype=np.float64)
    for _ in range(pretrain_rounds):
        selected = rng.choice(len(train), clients_per_round, replace=False)
        updates = []
        sizes = []
        for index in selected:
            group = train[int(index)]
            x, y = train_xy[group.cycle_id]
            updates.append(
                _local_update(
                    weights,
                    x,
                    y,
                    learning_rate=0.08,
                    epochs=3,
                    positive_weight=positive_weight,
                )
            )
            sizes.append(len(y))
        weights += _weighted_mean(np.asarray(updates), np.asarray(sizes))

    selected = rng.choice(len(train), clients_per_round, replace=False)
    benign_updates = []
    sizes = []
    for index in selected:
        group = train[int(index)]
        x, y = train_xy[group.cycle_id]
        benign_updates.append(
            _local_update(
                weights,
                x,
                y,
                learning_rate=0.08,
                epochs=3,
                positive_weight=positive_weight,
            )
        )
        sizes.append(len(y))
    benign_updates_array = np.asarray(benign_updates)
    sizes_array = np.asarray(sizes, dtype=float)
    score_update = _local_update(
        weights,
        score_x,
        score_y,
        learning_rate=0.04,
        epochs=1,
        positive_weight=positive_weight,
    )
    validator_groups = tuple(_stack((group,), mean, scale) for group in score[:20])

    before_artifact = LinearModelArtifact(
        tuple(float(value) for value in weights),
        tuple(float(value) for value in mean),
        tuple(float(value) for value in scale),
    )
    commit_probe_groups = _probe_groups(commit)
    audit_probe_groups = _probe_groups(audit)
    policy = EvaluationPolicy("brier-decimal80-v1")
    before_accuracy = _balanced_accuracy(weights, audit_x, audit_y)
    methods = ("FedAvg", "CoordinateMedian", "Krum", "FLTrust", "FedVal", "FLShield")
    output = []
    for attack in ("none", "sign_flip", "model_replacement"):
        updates = benign_updates_array.copy()
        fault_count = 0 if attack == "none" else max(
            1, int(math.floor(clients_per_round * byzantine_fraction))
        )
        if fault_count:
            malicious = rng.choice(clients_per_round, fault_count, replace=False)
            if attack == "sign_flip":
                updates[malicious] = -6.0 * updates[malicious]
            else:
                updates[malicious] = -12.0 * weights - 8.0 * updates[malicious]
        for method in methods:
            if method == "FedAvg":
                aggregate = _weighted_mean(updates, sizes_array)
            elif method == "CoordinateMedian":
                aggregate = _coordinate_median(updates, sizes_array)
            elif method == "Krum":
                aggregate = _krum(updates, sizes_array, fault_count)
            elif method == "FLTrust":
                aggregate = _fltrust(updates, sizes_array, score_update)
            elif method == "FedVal":
                aggregate = _fedval(weights, updates, sizes_array, score_x, score_y)
            else:
                aggregate = _flshield_cluster(
                    weights,
                    updates,
                    sizes_array,
                    validator_groups,
                    seed=seed,
                )
            candidate_weights = weights + aggregate
            after_artifact = LinearModelArtifact(
                tuple(float(value) for value in candidate_weights),
                tuple(float(value) for value in mean),
                tuple(float(value) for value in scale),
            )
            started = time.perf_counter_ns()
            commit_delta = float(
                paired_model_loss_difference_exact(
                    before_artifact,
                    after_artifact,
                    commit_probe_groups,
                    policy,
                )
            )
            gate_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            audit_delta = float(
                paired_model_loss_difference_exact(
                    before_artifact,
                    after_artifact,
                    audit_probe_groups,
                    policy,
                )
            )
            accepted = commit_delta <= -GAMMA
            candidate_accuracy = _balanced_accuracy(candidate_weights, audit_x, audit_y)
            output.append(
                {
                    "seed": seed,
                    "method": method,
                    "attack": attack,
                    "accepted": int(accepted),
                    "commit_delta": commit_delta,
                    "audit_delta": audit_delta,
                    "operational_harm": int(audit_delta >= OPERATIONAL_HARM),
                    "declared_harm": int(audit_delta >= EPSILON),
                    "harmful_escape": int(
                        accepted and audit_delta >= OPERATIONAL_HARM
                    ),
                    "declared_harmful_escape": int(accepted and audit_delta >= EPSILON),
                    "beneficial": int(audit_delta < 0),
                    "false_rejection": int((not accepted) and audit_delta < 0),
                    "before_balanced_accuracy": before_accuracy,
                    "candidate_balanced_accuracy": candidate_accuracy,
                    "installed_balanced_accuracy": (
                        candidate_accuracy if accepted else before_accuracy
                    ),
                    "gate_replay_ms": gate_ms,
                }
            )
    return output


def _summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_names = (
        "accepted",
        "operational_harm",
        "declared_harm",
        "harmful_escape",
        "declared_harmful_escape",
        "false_rejection",
        "candidate_balanced_accuracy",
        "installed_balanced_accuracy",
        "gate_replay_ms",
    )
    for (method, attack), group in frame.groupby(["method", "attack"], sort=True):
        row: dict[str, object] = {
            "method": method,
            "attack": attack,
            "seeds": int(group["seed"].nunique()),
        }
        for metric in metric_names:
            mean, ci = _mean_ci(
                group[metric],
                binary=metric
                in {
                    "accepted",
                    "operational_harm",
                    "declared_harm",
                    "harmful_escape",
                    "declared_harmful_escape",
                    "false_rejection",
                },
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95"] = ci
        beneficial = group[group["beneficial"] == 1]
        row["beneficial_cases"] = int(len(beneficial))
        row["conditional_false_rejection_rate"] = (
            float(beneficial["false_rejection"].mean()) if len(beneficial) else math.nan
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["attack", "method"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results_ur3"))
    parser.add_argument("--seeds", default=",".join(str(value) for value in range(20)))
    parser.add_argument("--pretrain-rounds", type=int, default=20)
    parser.add_argument("--clients-per-round", type=int, default=30)
    parser.add_argument("--byzantine-fraction", type=float, default=0.20)
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not seeds or args.pretrain_rounds <= 0 or args.clients_per_round <= 6:
        raise ValueError("benchmark needs seeds, positive pretraining, and at least 7 clients")
    if not 0 < args.byzantine_fraction < 0.5:
        raise ValueError("Byzantine fraction must lie in (0, 0.5)")
    groups = _load_groups(args.dataset)
    if args.clients_per_round > 120:
        raise ValueError("clients_per_round exceeds the fixed proposal split")
    records = []
    started = time.perf_counter()
    for seed in seeds:
        records.extend(
            _run_seed(
                groups,
                seed=seed,
                pretrain_rounds=args.pretrain_rounds,
                clients_per_round=args.clients_per_round,
                byzantine_fraction=args.byzantine_fraction,
            )
        )
    elapsed = time.perf_counter() - started
    frame = pd.DataFrame(records)
    summary = _summarize(frame)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "raw_runs.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    metadata = {
        "schema": "fedmerit/ur3-noniid-benchmark/v1",
        "dataset": {
            "name": "UCI UR3 CobotOps",
            "doi": DATASET_DOI,
            "rows_after_complete_case_filter": int(sum(len(group.y) for group in groups)),
            "operation_cycle_groups": len(groups),
        },
        "seeds": list(seeds),
        "split_groups_per_seed": {
            "proposal": 120,
            "score": 40,
            "commit_pool": 40,
            "commit_selected": COMMIT_GROUPS,
            "independent_audit": len(groups) - 200,
        },
        "attacks": ["none", "sign_flip", "model_replacement"],
        "byzantine_fraction": args.byzantine_fraction,
        "clients_per_round": args.clients_per_round,
        "pretrain_rounds": args.pretrain_rounds,
        "risk": {
            "alpha": ALPHA,
            "epsilon": EPSILON,
            "gamma": GAMMA,
            "operational_harm_threshold": OPERATIONAL_HARM,
            "group_count": COMMIT_GROUPS,
            "verified_bound": risk_bound(COMMIT_GROUPS, EPSILON, GAMMA),
        },
        "records": len(frame),
        "elapsed_seconds": elapsed,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

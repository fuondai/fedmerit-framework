"""Deterministic command-line utilities for the reference artifact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .gate import required_groups, risk_bound, risk_is_satisfied
from .model import EvaluationPolicy, SecurityProfile


MAX_MANIFEST_BYTES = 1_048_576


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant: {value}")


def read_json_object(location: Path) -> dict[str, Any]:
    try:
        encoded = location.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read JSON document: {exc}") from exc
    if len(encoded) > MAX_MANIFEST_BYTES:
        raise ValueError(f"JSON document exceeds {MAX_MANIFEST_BYTES} bytes")
    try:
        value = json.loads(
            encoded,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, ValueError) as exc:
        raise ValueError(f"cannot read JSON document: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("JSON document root must be an object")
    return value


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown:
        raise ValueError(f"{name} has unknown fields: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"{name} is missing fields: {', '.join(missing)}")


def _sequence(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty array")
    return value


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _risk_fields(
    value: Any, name: str, *, require_count: bool
) -> dict[str, float | int]:
    item = _object(value, name)
    expected = {"alpha", "epsilon", "gamma"}
    if require_count:
        expected.add("group_count")
    _exact_keys(item, expected, name)
    alpha = _finite_number(item.get("alpha"), f"{name}.alpha")
    epsilon = _finite_number(item.get("epsilon"), f"{name}.epsilon")
    gamma = _finite_number(item.get("gamma"), f"{name}.gamma")
    if not 0 < alpha < 1 or epsilon <= 0 or gamma < 0:
        raise ValueError(f"{name} has an invalid risk allocation")
    fields: dict[str, float | int] = {
        "alpha": alpha,
        "epsilon": epsilon,
        "gamma": gamma,
    }
    if require_count:
        count = _positive_integer(item.get("group_count"), f"{name}.group_count")
        if not risk_is_satisfied(count, epsilon, gamma, alpha):
            raise ValueError(f"{name}.group_count does not satisfy its risk bound")
        fields["group_count"] = count
    return fields


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate the finite certificate-calculation contract."""
    required = {
        "schema_version",
        "analysis_unit",
        "model",
        "evaluation_policy",
        "sampling_frame_policy",
        "certificate_evaluation",
        "certified_schedule_example",
        "source_partition_policy",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(f"manifest is missing required fields: {', '.join(missing)}")
    _exact_keys(manifest, required, "manifest")
    if manifest.get("schema_version") != "fedmerit/certificate-calculation/v3":
        raise ValueError("unsupported schema_version")
    _nonempty_string(manifest.get("analysis_unit"), "analysis_unit")

    schedule = _object(
        manifest.get("certified_schedule_example"), "certified_schedule_example"
    )
    _exact_keys(
        schedule,
        {"schedule_id", "finite_horizon", "allocation_per_index"},
        "certified_schedule_example",
    )
    _nonempty_string(
        schedule.get("schedule_id"), "certified_schedule_example.schedule_id"
    )
    horizon = _positive_integer(
        schedule.get("finite_horizon"), "certified_schedule_example.finite_horizon"
    )
    allocation = _risk_fields(
        schedule.get("allocation_per_index"),
        "certified_schedule_example.allocation_per_index",
        require_count=True,
    )
    if horizon * float(allocation["alpha"]) >= 1:
        raise ValueError(
            "certified schedule must reserve a finite lifetime budget below one"
        )

    model = _object(manifest.get("model"), "model")
    _exact_keys(model, {"family", "activation", "output", "loss_quantum"}, "model")
    for field in ("family", "activation", "output"):
        _nonempty_string(model.get(field), f"model.{field}")
    if model["family"] != "linear_logistic" or model["output"] != "binary":
        raise ValueError("reference replay requires a binary linear-logistic model")
    if model["activation"] != "sigmoid_decimal80":
        raise ValueError("reference replay requires the decimal80 sigmoid profile")
    if _finite_number(model.get("loss_quantum"), "model.loss_quantum") != 1e-12:
        raise ValueError("reference replay loss_quantum must equal 1e-12")

    policy = _object(manifest.get("evaluation_policy"), "evaluation_policy")
    _exact_keys(
        policy,
        {
            "policy_id",
            "loss",
            "preprocessing",
            "decimal_precision",
            "rounding",
            "group_loss_quantum",
            "group_order",
            "missing_value_rule",
            "class_weights",
            "group_reduction",
            "security_profile",
        },
        "evaluation_policy",
    )
    security = _object(
        policy.get("security_profile"), "evaluation_policy.security_profile"
    )
    _exact_keys(
        security,
        {
            "security_parameter_bits",
            "max_attempts",
            "max_catalog_leaves",
            "max_verification_keys",
            "max_hash_queries",
            "max_collision_queries",
            "max_signature_queries",
            "max_beacon_queries",
        },
        "evaluation_policy.security_profile",
    )
    try:
        EvaluationPolicy(
            policy_id=_nonempty_string(
                policy.get("policy_id"), "evaluation_policy.policy_id"
            ),
            loss=policy.get("loss"),
            preprocessing=policy.get("preprocessing"),
            decimal_precision=policy.get("decimal_precision"),
            rounding=policy.get("rounding"),
            group_loss_quantum=policy.get("group_loss_quantum"),
            group_order=policy.get("group_order"),
            missing_value_rule=policy.get("missing_value_rule"),
            class_weights=tuple(policy.get("class_weights", ())),
            group_reduction=policy.get("group_reduction"),
            security_profile=SecurityProfile(**security),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid evaluation_policy: {exc}") from exc
    frame_policy = _object(
        manifest.get("sampling_frame_policy"), "sampling_frame_policy"
    )
    _exact_keys(
        frame_policy,
        {
            "descriptor",
            "frame_authority_signature",
            "beacon_authentication",
            "beacon_head",
            "beacon_randomness_contract",
            "watcher_completeness_required",
            "exclusive_successor_reservation",
            "selection",
            "catalog_leaf",
            "commitment_hiding_model",
            "catalog_root",
            "opening_nonce_bits",
            "catalog_signed_before_candidate_fixation",
            "eligible_set_bound_in_candidate",
            "opening_verified_before_scoring",
            "fixation_order",
        },
        "sampling_frame_policy",
    )
    if (
        frame_policy.get("descriptor") != "sealed-opaque-catalog-v1"
        or frame_policy.get("frame_authority_signature") != "Ed25519"
        or frame_policy.get("beacon_authentication") != "Ed25519"
        or frame_policy.get("beacon_head")
        != "durable-signed-monotonic-immediate-parent-v2"
        or frame_policy.get("beacon_randomness_contract")
        != "external-threshold-unpredictable-unbiasable-v1"
        or frame_policy.get("watcher_completeness_required") is not True
        or frame_policy.get("exclusive_successor_reservation") is not True
        or frame_policy.get("selection") != "beacon-sha256-rejection-sampling-v1"
        or frame_policy.get("catalog_leaf")
        != "opaque-id-plus-ro-hiding-payload-commitment-v1"
        or frame_policy.get("commitment_hiding_model")
        != "sha256-random-oracle-ind-hiding-v1"
        or frame_policy.get("catalog_root") != "sha256-merkle-v1"
        or frame_policy.get("opening_nonce_bits") != 256
        or frame_policy.get("catalog_signed_before_candidate_fixation") is not True
        or frame_policy.get("eligible_set_bound_in_candidate") is not True
        or frame_policy.get("opening_verified_before_scoring") is not True
        or frame_policy.get("fixation_order")
        != "signed-catalog-and-authenticated-beacon-parent-before-durable-candidate-before-future-beacon"
    ):
        raise ValueError("sampling_frame_policy does not match the reference contract")
    evaluation = _object(
        manifest.get("certificate_evaluation"), "certificate_evaluation"
    )
    _exact_keys(
        evaluation,
        {"risk_interval_precision_bits", "risk_grid", "quorum_fault_counts"},
        "certificate_evaluation",
    )
    if (
        _positive_integer(
            evaluation.get("risk_interval_precision_bits"),
            "certificate_evaluation.risk_interval_precision_bits",
        )
        < 128
    ):
        raise ValueError("risk interval precision must be at least 128 bits")
    risk_signatures: list[tuple[float, float, float]] = []
    for index, item in enumerate(
        _sequence(evaluation.get("risk_grid"), "certificate_evaluation.risk_grid")
    ):
        fields = _risk_fields(
            item, f"certificate_evaluation.risk_grid[{index}]", require_count=False
        )
        risk_signatures.append(
            (float(fields["alpha"]), float(fields["epsilon"]), float(fields["gamma"]))
        )
    if len(set(risk_signatures)) != len(risk_signatures):
        raise ValueError("certificate_evaluation.risk_grid contains duplicate cells")
    faults = _sequence(
        evaluation.get("quorum_fault_counts"),
        "certificate_evaluation.quorum_fault_counts",
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in faults
    ):
        raise ValueError("quorum fault counts must be non-negative integers")
    if faults != sorted(set(faults)):
        raise ValueError("quorum fault counts must be distinct and ascending")

    partition_policy = _object(
        manifest.get("source_partition_policy"), "source_partition_policy"
    )
    _exact_keys(
        partition_policy,
        {"proposal_and_commit_sources", "commit_probe"},
        "source_partition_policy",
    )
    for field in ("proposal_and_commit_sources", "commit_probe"):
        _nonempty_string(
            partition_policy.get(field), f"source_partition_policy.{field}"
        )

    return {
        "valid": True,
        "schema_version": manifest["schema_version"],
        "risk_grid_count": len(evaluation["risk_grid"]),
        "quorum_fault_count": len(faults),
        "risk_interval_precision_bits": evaluation["risk_interval_precision_bits"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="State-scoped certificate utilities")
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser(
        "risk-plan", help="compute the required source-group count"
    )
    plan.add_argument("--alpha", type=float, required=True)
    plan.add_argument("--epsilon", type=float, required=True)
    plan.add_argument("--gamma", type=float, required=True)

    check = commands.add_parser(
        "validate-manifest", help="validate a declarative benchmark contract"
    )
    check.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "risk-plan":
            count = required_groups(args.alpha, args.epsilon, args.gamma)
            payload = {
                "source_groups": count,
                "bound": risk_bound(count, args.epsilon, args.gamma),
            }
        else:
            payload = validate_manifest(read_json_object(args.config))
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

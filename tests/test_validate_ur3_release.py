from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.run_ur3_benchmark import _summarize
from scripts.validate_ur3_release import validate_release


ROOT = Path(__file__).resolve().parents[1]


def test_retained_release_validator_recomputes_all_contracts(tmp_path: Path) -> None:
    source = ROOT / "results" / "ur3_v3_random"
    frame = pd.read_csv(source / "raw_runs.csv")
    frame["configured_fault_bound"] = 6
    frame["actual_byzantine_clients"] = frame["attack"].map(
        lambda attack: 0 if attack == "none" else 6
    )
    frame.to_csv(tmp_path / "raw_runs.csv", index=False)
    _summarize(frame).to_csv(tmp_path / "summary.csv", index=False)

    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    metadata["benchmark_design"] = {
        "estimand": "one installed candidate transition from a shared checkpoint",
        "shared_checkpoint": "20 benign FedAvg rounds per seed",
        "candidate_rounds_per_method_attack_seed": 1,
        "end_to_end_training_comparison": False,
        "registered_fault_bound": 6,
        "clean_actual_byzantine_clients": 0,
        "run_label": "validator-regression-fixture",
    }
    metadata["environment"].update(
        {
            "python_implementation": "CPython",
            "scikit_learn": "fixture",
            "openpyxl": "fixture",
            "cryptography": "fixture",
            "machine": "fixture",
            "logical_cpu_count": 1,
        }
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    report = validate_release(tmp_path, split="random")

    assert report == {
        "split": "random",
        "transitions": 480,
        "seeds": 20,
        "catalog_harmful": 19,
        "catalog_escapes": 0,
        "audit_diagnostic_harmful": 54,
        "audit_diagnostic_escapes": 0,
    }

#!/usr/bin/env python3
"""Render the deterministic FedMERIT cost chart used by the manuscript.

The values in ``protocol_scaling.csv`` are the two plotted slices copied from
the verified deterministic certificate evidence.  They are not workload or
training measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fedmerit.gate import required_groups  # noqa: E402

try:
    import scienceplots  # noqa: F401  # registers the SciencePlots styles

    STYLE = ["science", "ieee"]
except ImportError:  # pragma: no cover - used only when the optional style is absent
    STYLE = ["seaborn-v0_8-whitegrid"]


# Publication parameters: the chart is rendered at IEEE single-column width.
TEXT_SIZE_PT = 7.5
LINE_WIDTH = 0.9
MARKER_SIZE = 3.2
WIDTH_IN = 3.45
HEIGHT_IN = 1.55
BLUE = "#0072B2"
ORANGE = "#E69F00"
GRID = "#D9D9D9"
DATA_FILE = Path(__file__).with_name("protocol_scaling.csv")
PDF_FILE = Path(__file__).with_name("fig_protocol_scaling.pdf")
PNG_FILE = Path(__file__).with_name("fig_protocol_scaling_preview.png")
DEFAULT_EVIDENCE_FILE = PROJECT_ROOT / "results_devready" / "metrics.json"
RISK_ALPHA = 0.01
RISK_GAMMA = 0.05


def read_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    with DATA_FILE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    unknown = {row["panel"] for row in rows} - {"risk", "quorum"}
    if unknown:
        raise ValueError(f"unknown panel values: {sorted(unknown)}")
    return (
        [row for row in rows if row["panel"] == "risk"],
        [row for row in rows if row["panel"] == "quorum"],
    )


def validate_rows(
    risk_rows: list[dict[str, str]],
    quorum_rows: list[dict[str, str]],
    evidence_file: Path | None,
) -> None:
    """Reject stale/malformed plot inputs before rendering a paper figure."""
    if len(risk_rows) != 3 or len(quorum_rows) != 9:
        raise ValueError("expected three risk rows and nine quorum rows")

    risk = []
    for row in risk_rows:
        alpha = float(row["alpha"])
        gamma = float(row["gamma"])
        epsilon = float(row["epsilon"])
        groups = int(row["required_groups"])
        if alpha != RISK_ALPHA or gamma != RISK_GAMMA:
            raise ValueError("risk slice must use alpha=0.01 and gamma=0.05")
        expected = required_groups(alpha, epsilon, gamma)
        if groups != expected:
            raise ValueError(f"risk row for epsilon={epsilon} fails the n_min formula")
        risk.append((epsilon, groups))
    if [epsilon for epsilon, _ in risk] != [0.10, 0.15, 0.25]:
        raise ValueError("risk rows must cover epsilon=0.10, 0.15, 0.25 in order")

    quorum = []
    for row in quorum_rows:
        faults = int(row["f"])
        certificate_bytes = int(row["certificate_bytes"])
        witnesses = 3 * faults + 1
        expected = 357 + (witnesses + 7) // 8 + 64 * (2 * faults + 1)
        if certificate_bytes != expected:
            raise ValueError(f"quorum row for f={faults} fails the byte formula")
        quorum.append((faults, certificate_bytes))
    if [faults for faults, _ in quorum] != list(range(9)):
        raise ValueError("quorum rows must cover f=0..8 in order")

    if evidence_file is None:
        return
    with evidence_file.open(encoding="utf-8") as handle:
        evidence = json.load(handle)
    evidence_risk = {
        (float(row["epsilon"]), float(row["gamma"]), float(row["alpha"])): int(
            row["required_groups"]
        )
        for row in evidence["risk_budget_grid"]
    }
    for row in risk_rows:
        key = (float(row["epsilon"]), float(row["gamma"]), float(row["alpha"]))
        if evidence_risk.get(key) != int(row["required_groups"]):
            raise ValueError(f"CSV risk row {key} disagrees with {evidence_file}")
    evidence_quorum = {
        int(row["f"]): int(row["certificate_bytes"])
        for row in evidence["quorum_certificate_grid"]
    }
    for row in quorum_rows:
        faults = int(row["f"])
        if evidence_quorum.get(faults) != int(row["certificate_bytes"]):
            raise ValueError(f"CSV quorum row f={faults} disagrees with {evidence_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE_FILE,
        help="optional metrics.json used to cross-check the plotted rows",
    )
    parser.add_argument(
        "--no-evidence-check",
        action="store_true",
        help="render from the checked-in CSV when metrics.json is unavailable",
    )
    return parser.parse_args()


def configure_axes(axis: plt.Axes) -> None:
    axis.tick_params(axis="both", labelsize=TEXT_SIZE_PT, width=0.45, length=2.5)
    axis.grid(True, which="major", color=GRID, linewidth=0.35, linestyle="--")
    axis.grid(False, which="minor")
    for spine in axis.spines.values():
        spine.set_linewidth(0.45)
    axis.set_axisbelow(True)


def main() -> None:
    args = parse_args()
    risk_rows, quorum_rows = read_rows()
    evidence_file = None if args.no_evidence_check else args.evidence
    if evidence_file is not None and not evidence_file.exists():
        raise FileNotFoundError(
            f"evidence file not found: {evidence_file}; run the evidence producer "
            "or pass --no-evidence-check"
        )
    validate_rows(risk_rows, quorum_rows, evidence_file)
    with plt.style.context(STYLE):
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                # Keep the figure self-contained; TeX is handled by the paper build.
                "text.usetex": False,
                "mathtext.fontset": "stix",
                "font.size": TEXT_SIZE_PT,
                "axes.labelsize": TEXT_SIZE_PT,
                "xtick.labelsize": TEXT_SIZE_PT,
                "ytick.labelsize": TEXT_SIZE_PT,
                "legend.fontsize": TEXT_SIZE_PT,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "savefig.bbox": None,
            }
        )
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(WIDTH_IN, HEIGHT_IN),
            sharex=False,
            constrained_layout=False,
        )
        figure.subplots_adjust(
            left=0.15, right=0.99, top=0.94, bottom=0.28, wspace=0.56
        )

        risk_x = [float(row["epsilon"]) for row in risk_rows]
        risk_y = [int(row["required_groups"]) for row in risk_rows]
        axes[0].plot(
            risk_x,
            risk_y,
            color=BLUE,
            marker="o",
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
        )
        for x_value, y_value in zip(risk_x, risk_y):
            axes[0].annotate(
                f"{y_value}",
                (x_value, y_value),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=TEXT_SIZE_PT,
            )
        axes[0].set_ylabel(r"Required groups $n_{\min}$")
        axes[0].set_xlabel(r"Detectable harm $\epsilon$")
        axes[0].set_xticks(risk_x)
        axes[0].set_ylim(0, 550)
        axes[0].set_yticks([0, 100, 200, 300, 400, 500])
        axes[0].text(
            0.96,
            0.90,
            r"$\alpha=.01,\ \gamma=.05$",
            transform=axes[0].transAxes,
            ha="right",
            va="top",
            fontsize=TEXT_SIZE_PT,
        )
        configure_axes(axes[0])
        axes[0].text(
            0.02,
            0.10,
            "(a)",
            transform=axes[0].transAxes,
            fontsize=TEXT_SIZE_PT,
            va="top",
        )

        quorum_x = [int(row["f"]) for row in quorum_rows]
        quorum_y = [int(row["certificate_bytes"]) for row in quorum_rows]
        axes[1].plot(
            quorum_x,
            quorum_y,
            color=ORANGE,
            marker="s",
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
        )
        axes[1].set_xlabel("Byzantine faults $f$")
        axes[1].set_ylabel("Certificate bytes")
        axes[1].set_xticks([0, 2, 4, 6, 8])
        axes[1].set_ylim(350, 1520)
        axes[1].set_yticks([400, 800, 1200, 1500])
        configure_axes(axes[1])
        axes[1].text(
            0.02,
            0.90,
            "(b)",
            transform=axes[1].transAxes,
            fontsize=TEXT_SIZE_PT,
            va="top",
        )

        figure.savefig(PDF_FILE)
        figure.savefig(PNG_FILE, dpi=300)
        plt.close(figure)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render the retained UR3 transition benchmark and reuse challenge.

The figure reports descriptive counts and observed latency quantiles. It does
not attach binomial intervals to rows that share seed-level data partitions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import scienceplots  # noqa: F401  # registers the SciencePlots styles

    STYLE = ["science"]
except ImportError:  # pragma: no cover - optional local style
    STYLE = ["seaborn-v0_8-whitegrid"]


METHODS = (
    "FedAvg",
    "CoordinateMedian",
    "Krum",
    "FLTrust",
    "FedVal",
    "FLShield",
    "FoundationFL",
)
METHOD_LABELS = {
    "FedAvg": "Avg",
    "CoordinateMedian": "Med",
    "Krum": "Krum",
    "FLTrust": "Trust",
    "FedVal": "Val",
    "FLShield": "Shield",
    "FoundationFL": "Found.",
}
ATTACKS = ("none", "sign_flip", "model_replacement", "score_aware")
ATTACK_LABELS = {
    "none": "Clean",
    "sign_flip": "Sign flip",
    "model_replacement": "Replacement",
    "score_aware": "Score-aware",
}
WONG_PALETTE = (
    "#7A7A7A",
    "#E69F00",
    "#0072B2",
    "#009E73",
)
TEXT_SIZE_PT = 8.0
MARKER_SIZE = 3.2
WIDTH_MM = 180.0
HEIGHT_MM = 50.0


def _configure(axis: plt.Axes, *, grid: bool = True) -> None:
    axis.tick_params(
        axis="both",
        which="both",
        labelsize=TEXT_SIZE_PT,
        width=0.45,
        length=2.2,
        top=False,
        right=False,
    )
    if grid:
        axis.grid(True, axis="y", color="#D9D9D9", linewidth=0.32, linestyle="--")
    else:
        axis.grid(False)
    axis.set_axisbelow(True)
    axis.minorticks_off()
    for side, spine in axis.spines.items():
        spine.set_linewidth(0.45)
        if side in {"top", "right"}:
            spine.set_visible(False)


def _escape_counts(
    ur3: pd.DataFrame, challenge: pd.DataFrame
) -> tuple[tuple[str, int, int, int], ...]:
    ur3_harmful = int(ur3["population_harm"].sum())
    challenge_harmful = int(challenge["harmful"].sum())
    if ur3_harmful == 0 or challenge_harmful == 0:
        raise ValueError("each safety evaluation must contain a harmful candidate")
    return (
        (
            "UR3 catalog\n$\\epsilon=0.35$",
            ur3_harmful,
            int(ur3["score_gate_population_escape"].sum()),
            int(ur3["population_escape"].sum()),
        ),
        (
            "Reuse challenge\n$\\epsilon=0.10$",
            challenge_harmful,
            int(challenge["reused_score_escape"].sum()),
            int(challenge["fresh_probe_escape"].sum()),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("fig_ur3_benchmark.pdf"))
    parser.add_argument("--split", choices=("random", "blocked"), default="random")
    args = parser.parse_args()

    frame = pd.read_csv(args.raw)
    challenge = pd.read_csv(args.challenge)
    required = {
        "split",
        "method",
        "attack",
        "seed",
        "accepted",
        "population_harm",
        "population_escape",
        "score_gate_population_escape",
        "protocol_e2e_ms",
    }
    challenge_required = {
        "harmful",
        "reused_score_escape",
        "fresh_probe_escape",
    }
    missing = sorted(required - set(frame.columns))
    challenge_missing = sorted(challenge_required - set(challenge.columns))
    if missing:
        raise ValueError(f"raw run file is missing columns: {missing}")
    if challenge_missing:
        raise ValueError(f"challenge file is missing columns: {challenge_missing}")
    frame = frame[
        (frame["split"] == args.split)
        & frame["method"].isin(METHODS)
        & frame["attack"].isin(ATTACKS)
    ].copy()
    if frame.empty:
        raise ValueError(f"raw run file has no {args.split!r} benchmark rows")

    with plt.style.context(STYLE):
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                "text.usetex": False,
                "mathtext.fontset": "stix",
                "font.size": TEXT_SIZE_PT,
                "font.weight": "normal",
                "axes.labelsize": TEXT_SIZE_PT,
                "axes.labelweight": "normal",
                "xtick.labelsize": TEXT_SIZE_PT,
                "ytick.labelsize": TEXT_SIZE_PT,
                "legend.fontsize": TEXT_SIZE_PT,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "svg.fonttype": "none",
                "savefig.bbox": None,
            }
        )
        figure, axes = plt.subplots(
            1,
            3,
            figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4),
            constrained_layout=False,
            gridspec_kw={"width_ratios": (1.15, 1.52, 1.05)},
        )
        figure.subplots_adjust(
            left=0.052, right=0.988, bottom=0.25, top=0.87, wspace=0.43
        )
        method_x = np.arange(len(METHODS))

        # (a) Safety evidence from two distinct evaluation designs.
        axis = axes[0]
        evidence = _escape_counts(frame, challenge)
        safety_x = np.arange(len(evidence))
        width = 0.235
        series = (
            (-width, 1, "No gate", WONG_PALETTE[0]),
            (0.0, 2, "Reuse", WONG_PALETTE[1]),
            (width, 3, "Fresh", WONG_PALETTE[2]),
        )
        for offset, metric_index, label, color in series:
            counts = [item[metric_index] for item in evidence]
            denominators = [item[1] for item in evidence]
            rates = [
                100.0 * count / denominator
                for count, denominator in zip(counts, denominators, strict=True)
            ]
            bars = axis.bar(
                safety_x + offset,
                rates,
                width,
                color=color,
                edgecolor="#333333",
                linewidth=0.4,
                label=label,
                zorder=2,
            )
            zero_x = [
                bar.get_x() + bar.get_width() / 2
                for bar, rate in zip(bars, rates, strict=True)
                if rate == 0.0
            ]
            if zero_x:
                axis.scatter(
                    zero_x,
                    np.zeros(len(zero_x)),
                    color=color,
                    edgecolor="#333333",
                    linewidth=0.4,
                    marker="o",
                    s=18,
                    zorder=3,
                )
        axis.set_ylim(0, 110)
        axis.set_ylabel("Harmful installs (%)")
        axis.set_xticks(safety_x, [item[0] for item in evidence])
        axis.legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.50, 1.17),
            ncol=3,
            handlelength=0.9,
            columnspacing=0.65,
            borderpad=0.0,
        )
        _configure(axis)

        # (b) Blank cells are out-of-contract combinations, not zero values.
        axis = axes[1]
        acceptance = np.full((len(ATTACKS), len(METHODS)), np.nan)
        for row_index, attack in enumerate(ATTACKS):
            for column_index, method in enumerate(METHODS):
                values = frame.loc[
                    (frame["method"] == method) & (frame["attack"] == attack),
                    "accepted",
                ]
                if len(values):
                    acceptance[row_index, column_index] = int(values.sum())
        color_map = plt.colormaps["viridis"].copy()
        color_map.set_bad("#EEEEEE")
        image = axis.imshow(acceptance, vmin=0, vmax=20, cmap=color_map, aspect="auto")
        for row_index in range(len(ATTACKS)):
            for column_index in range(len(METHODS)):
                value = acceptance[row_index, column_index]
                label = "n/a" if np.isnan(value) else str(int(value))
                text_color = "#333333" if np.isnan(value) or value >= 14 else "white"
                axis.text(
                    column_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=TEXT_SIZE_PT,
                )
        axis.set_xticks(
            method_x,
            [METHOD_LABELS[method] for method in METHODS],
            rotation=38,
            ha="right",
            rotation_mode="anchor",
        )
        axis.set_yticks(
            np.arange(len(ATTACKS)), [ATTACK_LABELS[item] for item in ATTACKS]
        )
        colorbar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.025)
        colorbar.set_label("Accepted / 20", fontsize=TEXT_SIZE_PT)
        colorbar.set_ticks([0, 10, 20])
        colorbar.ax.tick_params(labelsize=TEXT_SIZE_PT, width=0.45, length=2.0)
        _configure(axis, grid=False)

        # (c) Observed local protocol-time quantiles.
        axis = axes[2]
        common = frame[frame["attack"].isin(ATTACKS[:3])]
        medians, p95s = [], []
        for method in METHODS:
            values = common.loc[common["method"] == method, "protocol_e2e_ms"]
            medians.append(float(values.quantile(0.5)))
            p95s.append(float(values.quantile(0.95)))
        axis.vlines(
            method_x, medians, p95s, color="#555555", linewidth=0.9, zorder=1
        )
        axis.scatter(
            method_x,
            medians,
            color=WONG_PALETTE[3],
            edgecolor="#333333",
            linewidth=0.35,
            marker="s",
            s=(MARKER_SIZE + 0.8) ** 2,
            label="P50",
            zorder=3,
        )
        axis.scatter(
            method_x,
            p95s,
            color="#333333",
            marker="^",
            s=(MARKER_SIZE + 0.5) ** 2,
            label="P95",
            zorder=3,
        )
        axis.set_ylabel("Protocol time (ms)")
        axis.set_xticks(
            method_x,
            [METHOD_LABELS[method] for method in METHODS],
            rotation=38,
            ha="right",
            rotation_mode="anchor",
        )
        axis.legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.55, 1.10),
            ncol=2,
            handlelength=1.0,
            columnspacing=0.8,
            borderpad=0.0,
        )
        _configure(axis)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, transparent=True)
        figure.savefig(args.output.with_suffix(".svg"), transparent=True)
        figure.savefig(args.output.with_suffix(".png"), dpi=600, transparent=True)
        figure.savefig(args.output.with_suffix(".tiff"), dpi=600, transparent=True)
        plt.close(figure)


if __name__ == "__main__":
    main()

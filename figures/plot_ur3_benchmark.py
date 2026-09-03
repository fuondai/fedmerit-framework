#!/usr/bin/env python3
"""Render the retained UR3 candidate-transition benchmark.

Panel (a) exposes the primary safety comparison directly: harmful candidates,
reused-score installations, and fresh-probe installations. Panels (b) and (c)
then show the availability and local-cost consequences. Every quantity is
recomputed from primitive seed rows, so a stale summary cannot alter the plot.
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


# Single parameter block: all tunable visual values live here.
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
# Bang Wong order: orange, sky blue, blue green, pale violet, vermillion,
# yellow, blue, black. Series order is stable and never taste-reordered.
WONG_PALETTE = (
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#CC79A7",
    "#D55E00",
    "#F0E442",
    "#0072B2",
    "#000000",
)
ATTACK_COLORS = dict(zip(ATTACKS, WONG_PALETTE[: len(ATTACKS)], strict=True))
TEXT_SIZE_PT = 7.2
LINE_WIDTH = 0.55
MARKER_SIZE = 3.0
ERROR_CAPSIZE = 1.5
WIDTH_MM = 180.0
HEIGHT_MM = 53.0


def _wilson_interval(values: pd.Series) -> tuple[float, float, float]:
    array = values.to_numpy(dtype=float)
    if len(array) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(array))
    size = len(array)
    z = 1.96
    denominator = 1.0 + z * z / size
    center = (mean + z * z / (2.0 * size)) / denominator
    radius = (
        z
        * np.sqrt(mean * (1.0 - mean) / size + z * z / (4.0 * size * size))
        / denominator
    )
    return mean, max(0.0, center - radius), min(1.0, center + radius)


def _configure(axis: plt.Axes) -> None:
    axis.tick_params(
        axis="both",
        which="both",
        labelsize=TEXT_SIZE_PT,
        width=0.45,
        length=2.2,
        top=False,
        right=False,
    )
    axis.grid(True, axis="y", color="#D9D9D9", linewidth=0.32, linestyle="--")
    axis.set_axisbelow(True)
    axis.minorticks_off()
    for side, spine in axis.spines.items():
        spine.set_linewidth(0.45)
        if side in {"top", "right"}:
            spine.set_visible(False)


def _errorbar_arrays(
    values: list[float] | np.ndarray,
    lowers: list[float] | np.ndarray,
    uppers: list[float] | np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [np.asarray(values) - np.asarray(lowers), np.asarray(uppers) - np.asarray(values)]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("fig_ur3_benchmark.pdf"))
    parser.add_argument("--split", choices=("random", "blocked"), default="random")
    args = parser.parse_args()
    frame = pd.read_csv(args.raw)
    required = {
        "split",
        "method",
        "attack",
        "seed",
        "accepted",
        "population_harm",
        "population_escape",
        "score_gate_population_escape",
        "operational_harm",
        "harmful_escape",
        "score_gate_escape",
        "installed_balanced_accuracy",
        "protocol_e2e_ms",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"raw run file is missing columns: {missing}")
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
                "font.size": 7.2,
                "font.weight": "normal",
                "axes.labelsize": 7.2,
                "axes.labelweight": "normal",
                "xtick.labelsize": 7.2,
                "ytick.labelsize": 7.2,
                "legend.fontsize": 7.2,
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
            gridspec_kw={"width_ratios": (1.18, 1.48, 1.08)},
        )
        figure.subplots_adjust(
            left=0.050, right=0.995, bottom=0.24, top=0.88, wspace=0.40
        )
        method_x = np.arange(len(METHODS))

        # (a) Primary safety evidence. Catalog harm is the registered epsilon
        # event; audit harm at tau is diagnostic. The two gates are displayed
        # separately because only the fresh-probe gate has the theorem.
        axis = axes[0]
        evidence_sets = (
            (
                "Catalog\n$\\epsilon=0.35$",
                "population_harm",
                "score_gate_population_escape",
                "population_escape",
            ),
            (
                "Audit\n$\\tau=0.05$",
                "operational_harm",
                "score_gate_escape",
                "harmful_escape",
            ),
        )
        safety_x = np.arange(len(evidence_sets))
        width = 0.23
        safety_series = (
            (-width, 1, "Harmful candidate", "#8C8C8C", ""),
            (0.0, 2, "Reused-score install", WONG_PALETTE[0], "//"),
            (width, 3, "Fresh-probe install", WONG_PALETTE[1], ".."),
        )
        maxima = []
        for offset, metric_index, label, color, hatch in safety_series:
            counts = [int(frame[fields[metric_index]].sum()) for fields in evidence_sets]
            maxima.extend(counts)
            bars = axis.bar(
                safety_x + offset,
                counts,
                width,
                color=color,
                edgecolor="#333333",
                linewidth=0.4,
                hatch=hatch,
                label=label,
                zorder=2,
            )
            for bar, count in zip(bars, counts, strict=True):
                y = count + max(0.8, max(maxima, default=1) * 0.015)
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    y,
                    str(count),
                    ha="center",
                    va="bottom",
                    fontsize=TEXT_SIZE_PT,
                )
        axis.set_ylim(0, max(maxima) * 1.20)
        axis.set_ylabel("Transitions (count)")
        axis.set_xticks(safety_x, [item[0] for item in evidence_sets])
        axis.legend(
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(-0.02, 1.10),
            ncol=1,
            handlelength=1.1,
            labelspacing=0.15,
            borderpad=0.0,
        )
        axis.text(-0.16, 1.08, "(a)", transform=axis.transAxes, fontweight="bold")
        _configure(axis)

        # (b) Acceptance is shown for every attack; missing score-aware methods
        # remain absent rather than being imputed.
        axis = axes[1]
        offsets = np.linspace(-0.25, 0.25, len(ATTACKS))
        for attack, offset in zip(ATTACKS, offsets, strict=True):
            means, lowers, uppers = [], [], []
            for method in METHODS:
                values = frame[(frame["method"] == method) & (frame["attack"] == attack)][
                    "accepted"
                ]
                mean, lower, upper = _wilson_interval(values)
                means.append(mean)
                lowers.append(lower)
                uppers.append(upper)
            finite = np.isfinite(means)
            if not finite.any():
                continue
            axis.errorbar(
                method_x[finite] + offset,
                np.asarray(means)[finite],
                yerr=_errorbar_arrays(
                    np.asarray(means)[finite],
                    np.asarray(lowers)[finite],
                    np.asarray(uppers)[finite],
                ),
                fmt="o",
                markersize=MARKER_SIZE,
                capsize=ERROR_CAPSIZE,
                linewidth=LINE_WIDTH,
                color=ATTACK_COLORS[attack],
                label=ATTACK_LABELS[attack],
            )
        axis.set_ylim(-0.03, 1.08)
        axis.set_ylabel("Acceptance rate")
        axis.set_xticks(
            method_x,
            [METHOD_LABELS[m] for m in METHODS],
            rotation=38,
            ha="right",
            rotation_mode="anchor",
        )
        axis.text(-0.14, 1.08, "(b)", transform=axis.transAxes, fontweight="bold")
        _configure(axis)

        # (c) End-to-end cost: median and 95th percentile latency from primitive
        # rows. The p95 marker is an observed quantile, not a confidence interval.
        axis = axes[2]
        common = frame[frame["attack"].isin(ATTACKS[:3])]
        medians, p95s = [], []
        for method in METHODS:
            values = common.loc[common["method"] == method, "protocol_e2e_ms"]
            medians.append(float(values.quantile(0.5)))
            p95s.append(float(values.quantile(0.95)))
        axis.vlines(
            method_x,
            medians,
            p95s,
            color="#555555",
            linewidth=0.9,
            zorder=1,
        )
        axis.scatter(
            method_x,
            medians,
            color=WONG_PALETTE[2],
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
            [METHOD_LABELS[m] for m in METHODS],
            rotation=38,
            ha="right",
            rotation_mode="anchor",
        )
        axis.legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.55, 1.08),
            ncol=2,
            handlelength=1.0,
            columnspacing=0.8,
            borderpad=0.0,
        )
        axis.text(-0.14, 1.08, "(c)", transform=axis.transAxes, fontweight="bold")
        _configure(axis)

        # A shared attack legend keeps the middle panel free of collisions.
        handles, labels = axes[1].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.50, 0.005),
            ncol=len(ATTACKS),
            handlelength=1.0,
            columnspacing=1.0,
            borderpad=0.0,
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, transparent=True)
        figure.savefig(args.output.with_suffix(".svg"), transparent=True)
        figure.savefig(args.output.with_suffix(".png"), dpi=600, transparent=True)
        figure.savefig(args.output.with_suffix(".tiff"), dpi=600, transparent=True)
        plt.close(figure)


if __name__ == "__main__":
    main()

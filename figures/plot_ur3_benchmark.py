#!/usr/bin/env python3
"""Render the retained UR3 benchmark figure from primitive raw-run fields.

The figure is deliberately compact: safety, acceptance, and protocol latency
are the three quantities needed to interpret the RIVF results. Every interval
is recomputed from seed-level rows, so a stale summary cannot change a plotted
conclusion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

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
REPLACEMENT_HARM_THRESHOLD = 0.05


def _mean_ci(values: pd.Series, *, binary: bool = False) -> tuple[float, float, float]:
    array = values.to_numpy(dtype=float)
    if len(array) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(array))
    if binary:
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
    if len(array) < 2:
        return mean, mean, mean
    half_width = float(student_t.ppf(0.975, len(array) - 1)) * float(
        np.std(array, ddof=1)
    ) / np.sqrt(len(array))
    return mean, mean - half_width, mean + half_width


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
    parser.add_argument(
        "--harm-threshold",
        type=float,
        default=REPLACEMENT_HARM_THRESHOLD,
        help="Operational audit-delta threshold",
    )
    args = parser.parse_args()
    frame = pd.read_csv(args.raw)
    required = {
        "split",
        "method",
        "attack",
        "seed",
        "accepted",
        "audit_delta",
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
    frame["raw_harm"] = (frame["audit_delta"] >= args.harm_threshold).astype(float)
    frame["harmful_escape_derived"] = (
        frame["accepted"].astype(bool) & frame["raw_harm"].astype(bool)
    ).astype(float)
    if frame.empty:
        raise ValueError(f"raw run file has no {args.split!r} benchmark rows")

    with plt.style.context(STYLE):
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
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
                "savefig.bbox": None,
            }
        )
        figure, axes = plt.subplots(
            1,
            3,
            figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4),
            constrained_layout=False,
        )
        figure.subplots_adjust(
            left=0.045, right=0.995, bottom=0.24, top=0.90, wspace=0.38
        )
        x = np.arange(len(METHODS))

        # (a) Safety: the candidate rate is paired with the installed escape rate.
        axis = axes[0]
        replacement = frame[frame["attack"] == "model_replacement"]
        width = 0.33
        for offset, metric, label, color in (
            (-width / 2, "raw_harm", "Ungated", "#8C8C8C"),
            (width / 2, "harmful_escape_derived", "After gate", WONG_PALETTE[1]),
        ):
            means, lowers, uppers = [], [], []
            for method in METHODS:
                mean, lower, upper = _mean_ci(
                    replacement.loc[replacement["method"] == method, metric], binary=True
                )
                means.append(mean)
                lowers.append(lower)
                uppers.append(upper)
            axis.bar(
                x + offset,
                means,
                width,
                yerr=_errorbar_arrays(means, lowers, uppers),
                capsize=ERROR_CAPSIZE,
                color=color,
                edgecolor="#333333",
                linewidth=0.35,
                label=label,
            )
        axis.set_ylim(0, 1.12)
        axis.set_ylabel("Rate")
        axis.set_xticks(x, [METHOD_LABELS[m] for m in METHODS], rotation=38, ha="right")
        axis.legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.60, 1.08),
            ncol=2,
            handlelength=1.0,
            columnspacing=0.8,
            borderpad=0.0,
        )
        axis.text(-0.14, 1.08, "(a)", transform=axis.transAxes, fontweight="bold")
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
                mean, lower, upper = _mean_ci(values, binary=True)
                means.append(mean)
                lowers.append(lower)
                uppers.append(upper)
            finite = np.isfinite(means)
            if not finite.any():
                continue
            axis.errorbar(
                x[finite] + offset,
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
        axis.set_xticks(x, [METHOD_LABELS[m] for m in METHODS], rotation=38, ha="right")
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
        axis.bar(
            x,
            medians,
            color=WONG_PALETTE[2],
            edgecolor="#333333",
            linewidth=0.35,
            label="p50",
        )
        axis.vlines(x, medians, p95s, color="#222222", linewidth=LINE_WIDTH, label="p95")
        axis.scatter(x, p95s, color="#222222", s=MARKER_SIZE**2, zorder=3)
        axis.set_ylabel("Protocol time (ms)")
        axis.set_xticks(x, [METHOD_LABELS[m] for m in METHODS], rotation=38, ha="right")
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

        # A shared legend keeps the middle panel free of text collisions.
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
        figure.savefig(args.output.with_suffix(".png"), dpi=300, transparent=True)
        plt.close(figure)


if __name__ == "__main__":
    main()

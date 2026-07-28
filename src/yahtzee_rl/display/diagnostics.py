"""Matplotlib rendering for score diagnostics (data comes from
``yahtzee_rl.evaluation.diagnostics``)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from yahtzee_rl.config import Category
from yahtzee_rl.evaluation.diagnostics import (
    DiagnosticSummary,
    EpisodeReport,
    outcome_band,
)

_UPPER = set(Category.upper_categories())


def plot_diagnostic_dashboard(
    summary: DiagnosticSummary, title: str, show: bool = True
):
    """Render the 4-panel score-diagnostic dashboard. Returns the Figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title)
    ax_hist, ax_decomp, ax_mean, ax_dump = axes.ravel()

    # 1. Upper raw-sum histogram with 63 threshold + bonus-rate annotation
    if summary.upper_raw_values:
        ax_hist.hist(summary.upper_raw_values, bins=30, color="steelblue", alpha=0.8)
    ax_hist.axvline(63, color="red", linestyle="--", label="bonus @ 63")
    ax_hist.set_title(f"Upper raw sum (bonus rate {summary.bonus_rate:.0%})")
    ax_hist.set_xlabel("Upper raw sum")
    ax_hist.set_ylabel("Games")
    ax_hist.legend()

    # 2. Score decomposition stacked bar
    parts = [
        ("Upper base", summary.upper_base_mean, "steelblue"),
        ("+35 bonus", summary.upper_bonus_mean, "gold"),
        ("Lower base", summary.lower_base_mean, "seagreen"),
        ("+100 Yahtzee", summary.yahtzee_bonus_points_mean, "firebrick"),
    ]
    bottom = 0.0
    for label, value, color in parts:
        ax_decomp.bar("mean game", value, bottom=bottom, label=label, color=color)
        bottom += value
    ax_decomp.set_title(f"Score decomposition (mean total {summary.score_mean:.1f})")
    ax_decomp.set_ylabel("Points")
    ax_decomp.legend()

    # 3. Per-category mean score (upper vs lower colored)
    cats = list(Category)
    labels = [c.value for c in cats]
    means = [summary.per_category_mean_score[c] for c in cats]
    colors = ["steelblue" if c in _UPPER else "seagreen" for c in cats]
    ax_mean.bar(labels, means, color=colors)
    ax_mean.set_title("Mean score per category")
    ax_mean.tick_params(axis="x", rotation=45)

    # 4. Per-category dump rate
    dumps = [summary.per_category_dump_rate[c] for c in cats]
    ax_dump.bar(labels, dumps, color=colors)
    ax_dump.set_title("Dump rate per category (marked for 0)")
    ax_dump.set_ylim(0, 1)
    ax_dump.tick_params(axis="x", rotation=45)

    # Joker caption
    fig.text(
        0.5, 0.005,
        f"Joker unlocked: {summary.joker_unlock_rate:.0%}  |  "
        f"Joker bonuses cashed: mean {summary.yahtzee_bonus_count_mean:.2f} "
        f"(max {summary.yahtzee_bonus_count_max})",
        ha="center",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    if show:
        plt.show()
    return fig


_BANDS = [
    "Dumped (0)",
    "Low (1-10)",
    "Mid (11-20)",
    "High (21+)",
    "Yahtzee 50 (joker unlocked)",
]


def plot_score_sankey(reports: list[EpisodeReport], out_path: str | Path) -> Path:
    """Render a category -> outcome-band Sankey as standalone HTML.

    Ribbon width = game frequency (how often a category's fill landed in a
    band). YAHTZEE scored 50 routes to its own 'joker unlocked' endpoint.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cats = list(Category)
    cat_labels = [c.value for c in cats]
    node_labels = cat_labels + _BANDS
    cat_index = {c: i for i, c in enumerate(cats)}
    band_index = {b: len(cats) + j for j, b in enumerate(_BANDS)}
    node_colors = [
        "steelblue" if c in _UPPER else "seagreen" for c in cats
    ] + ["lightgray"] * len(_BANDS)

    flow: dict[tuple[int, int], int] = {}
    for r in reports:
        for c in cats:
            band = outcome_band(r.per_category[c])
            if band is None:
                continue
            key = (cat_index[c], band_index[band])
            flow[key] = flow.get(key, 0) + 1

    sources = [s for (s, _) in flow]
    targets = [t for (_, t) in flow]
    values = list(flow.values())
    link_colors = [
        "rgba(70,130,180,0.4)" if cats[s] in _UPPER else "rgba(46,139,87,0.4)"
        for s in sources
    ]

    fig = go.Figure(
        go.Sankey(
            node=dict(label=node_labels, color=node_colors, pad=12, thickness=14),
            link=dict(source=sources, target=targets, value=values, color=link_colors),
        )
    )
    fig.update_layout(title_text="Category -> outcome band", font_size=11)
    fig.write_html(str(out_path))
    return out_path

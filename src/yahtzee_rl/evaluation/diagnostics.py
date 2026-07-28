"""Pure score-diagnostic data derived from finished scorecards.

No plotting imports here — rendering lives in ``yahtzee_rl.display.diagnostics``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.stats import iqr

from yahtzee_rl.config import CATEGORIES, Category
from yahtzee_rl.scoring.scorecard import Scorecard


@dataclass
class CategoryOutcome:
    category: Category
    score: int
    marked: bool
    turn_filled: int | None
    was_dump: bool


@dataclass
class EpisodeReport:
    final_score: int
    upper_raw: int
    upper_bonus: int
    lower_raw: int
    yahtzee_bonus_count: int
    joker_unlocked: bool
    per_category: dict[Category, CategoryOutcome]


def report_from_scorecard(
    scorecard: Scorecard, turn_filled: dict[Category, int]
) -> EpisodeReport:
    """Derive an :class:`EpisodeReport` from a finished game's scorecard.

    ``turn_filled`` maps each scored category to the 1-based turn it was
    scored on (from the eval loop); categories absent from the map get None.
    """
    board = scorecard.score_board

    per_category: dict[Category, CategoryOutcome] = {}
    for c in Category:
        score = int(board[c]["score"])
        marked = bool(board[c]["marked"])
        per_category[c] = CategoryOutcome(
            category=c,
            score=score,
            marked=marked,
            turn_filled=turn_filled.get(c),
            was_dump=marked and score == 0,
        )

    upper_raw = sum(int(board[c]["score"]) for c in Category.upper_categories())
    lower_raw = sum(int(board[c]["score"]) for c in Category.lower_categories())
    upper_bonus = 35 if upper_raw >= 63 else 0
    yahtzee_bonus_count = max(
        0, int(board[Category.YAHTZEE]["num_times_achieved"]) - 1
    )

    return EpisodeReport(
        final_score=int(scorecard.compute_final_score()),
        upper_raw=upper_raw,
        upper_bonus=upper_bonus,
        lower_raw=lower_raw,
        yahtzee_bonus_count=yahtzee_bonus_count,
        joker_unlocked=scorecard.joker_eligible(),
        per_category=per_category,
    )


def outcome_band(outcome: CategoryOutcome) -> str | None:
    """Bucket a category outcome into a Sankey outcome band.

    Returns None for an unmarked category (should not occur at game end).
    YAHTZEE scored 50 is a distinct 'joker unlocked' endpoint.
    """
    if not outcome.marked:
        return None
    if outcome.category == Category.YAHTZEE and outcome.score == 50:
        return "Yahtzee 50 (joker unlocked)"
    if outcome.score == 0:
        return "Dumped (0)"
    if outcome.score <= 10:
        return "Low (1-10)"
    if outcome.score <= 20:
        return "Mid (11-20)"
    return "High (21+)"


@dataclass
class DiagnosticSummary:
    n_episodes: int
    score_mean: float
    score_median: float
    score_std: float
    score_iqr: float
    bonus_rate: float
    upper_raw_values: list[int]
    per_category_mean_score: dict[Category, float]
    per_category_dump_rate: dict[Category, float]
    per_category_mean_turn: dict[Category, float]
    upper_base_mean: float
    upper_bonus_mean: float
    lower_base_mean: float
    yahtzee_bonus_points_mean: float
    joker_unlock_rate: float
    yahtzee_bonus_count_mean: float
    yahtzee_bonus_count_max: int


def aggregate_reports(reports: list[EpisodeReport]) -> DiagnosticSummary:
    n = len(reports)
    if n == 0:
        empty = {c: 0.0 for c in Category}
        return DiagnosticSummary(
            n_episodes=0, score_mean=0.0, score_median=0.0, score_std=0.0,
            score_iqr=0.0, bonus_rate=0.0, upper_raw_values=[],
            per_category_mean_score=dict(empty),
            per_category_dump_rate=dict(empty),
            per_category_mean_turn=dict(empty),
            upper_base_mean=0.0, upper_bonus_mean=0.0, lower_base_mean=0.0,
            yahtzee_bonus_points_mean=0.0, joker_unlock_rate=0.0,
            yahtzee_bonus_count_mean=0.0, yahtzee_bonus_count_max=0,
        )

    scores = np.array([r.final_score for r in reports], dtype=float)

    per_category_mean_score = {
        c: float(np.mean([r.per_category[c].score for r in reports]))
        for c in Category
    }
    per_category_dump_rate = {
        c: float(np.mean([1.0 if r.per_category[c].was_dump else 0.0 for r in reports]))
        for c in Category
    }
    per_category_mean_turn: dict[Category, float] = {}
    for c in Category:
        turns = [r.per_category[c].turn_filled for r in reports
                 if r.per_category[c].turn_filled is not None]
        per_category_mean_turn[c] = float(np.mean(turns)) if turns else 0.0

    return DiagnosticSummary(
        n_episodes=n,
        score_mean=float(np.mean(scores)),
        score_median=float(np.median(scores)),
        score_std=float(np.std(scores)),
        score_iqr=float(iqr(scores)),
        bonus_rate=float(np.mean([1.0 if r.upper_bonus == 35 else 0.0 for r in reports])),
        upper_raw_values=[r.upper_raw for r in reports],
        per_category_mean_score=per_category_mean_score,
        per_category_dump_rate=per_category_dump_rate,
        per_category_mean_turn=per_category_mean_turn,
        upper_base_mean=float(np.mean([r.upper_raw for r in reports])),
        upper_bonus_mean=float(np.mean([r.upper_bonus for r in reports])),
        lower_base_mean=float(np.mean([r.lower_raw for r in reports])),
        yahtzee_bonus_points_mean=float(np.mean([r.yahtzee_bonus_count * 100 for r in reports])),
        joker_unlock_rate=float(np.mean([1.0 if r.joker_unlocked else 0.0 for r in reports])),
        yahtzee_bonus_count_mean=float(np.mean([r.yahtzee_bonus_count for r in reports])),
        yahtzee_bonus_count_max=int(max(r.yahtzee_bonus_count for r in reports)),
    )


def save_reports(reports: list[EpisodeReport], path: str | Path) -> Path:
    """Dump per-episode reports as JSON for ad-hoc analysis."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for r in reports:
        d = asdict(r)
        d["per_category"] = {c.value: asdict(o) for c, o in r.per_category.items()}
        payload.append(d)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path

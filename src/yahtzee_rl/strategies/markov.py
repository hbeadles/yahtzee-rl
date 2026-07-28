import random

import numpy as np

from yahtzee_rl.config import (
    ACTION_TO_CATEGORY,
    CATEGORY_SCORE_FUNC,
    UPPER_SECTION_MAP,
    Category,
)
from yahtzee_rl.envs.yahtzee_env import YahtzeeEnv
from yahtzee_rl.markov.lookaheads import determine_keep_positions
from yahtzee_rl.scoring.ops import combo_satisfied, is_yahtzee, joker_score
from yahtzee_rl.scoring.scorecard import Scorecard
from yahtzee_rl.strategies.base import Strategy


class MarkovStrategy(Strategy):

    def __init__(self, env: YahtzeeEnv | None = None):
        if env is not None:
            super().__init__(env)

    def strategy(self, obs: np.ndarray, scorecard: Scorecard) -> int:
        parsed = YahtzeeEnv.parse_observation(obs)
        if not parsed['time_to_score']:
            all_expected = {
                **parsed["upper_section_expected_scores"],
                **parsed["lower_section_expected_scores"],
            }
            all_expected = {
                cat: score for cat, score in all_expected.items()
                if not scorecard.is_category_marked(Category(cat))
                or cat == Category.YAHTZEE
            }

            best_cat = max(all_expected, key=all_expected.get)

            return determine_keep_positions(parsed['dice'], Category(best_cat))
        else:
            dice = parsed['dice']
            mask = self.env.action_masks()
            candidates = [ACTION_TO_CATEGORY[i] for i in range(13) if mask[i]]
            if not candidates:
                return random.choice(list(Category)).value

            if scorecard.joker_active(dice):
                return max(candidates, key=lambda c: joker_score(c, dice)).value

            total_moves = []
            for category in candidates:
                if combo_satisfied(dice, category):
                    score_func = scorecard.get_score_function(category)
                    score = score_func(dice)
                    total_moves.append((category, score))
            if total_moves:
                return max(total_moves, key=lambda x: x[1])[0].value
            return random.choice(candidates).value

    def strategy_dict(self, parsed: dict) -> str:
        """
        Markov strategy operating on a parsed observation dict
        (from YahtzeeEnv.parse_observation) instead of a Scorecard object.

        Best-effort Joker handling: ``parsed["scorecard"]`` only exposes
        marked-flags, not per-category scores, so we cannot distinguish a
        50-point YAHTZEE from a zeroed-out YAHTZEE. We treat Joker as active
        whenever YAHTZEE is marked AND the current dice are a Yahtzee. The
        false-positive case (zeroed YAHTZEE) is a known approximation and
        matches the existing parsed-observation contract.

        Args:
            parsed: Dict returned by YahtzeeEnv.parse_observation().

        Returns:
            Category name (str) when scoring, or keep-positions mask when rolling.
        """
        scorecard_dict = parsed["scorecard"]
        dice = parsed["dice"]

        if not parsed["time_to_score"]:
            all_expected = {
                **parsed["upper_section_expected_scores"],
                **parsed["lower_section_expected_scores"],
            }
            all_expected = {
                cat: score for cat, score in all_expected.items()
                if not scorecard_dict.get(cat, 0.0)
                or cat == Category.YAHTZEE
            }

            best_cat = max(all_expected, key=all_expected.get)
            return determine_keep_positions(dice, Category(best_cat))

        joker = bool(scorecard_dict.get(Category.YAHTZEE.value, 0.0)) and is_yahtzee(dice)
        candidates = self._candidates_under_joker(scorecard_dict, dice, joker)
        if not candidates:
            return random.choice(list(Category)).value

        if joker:
            return max(candidates, key=lambda c: joker_score(c, dice)).value

        total_moves = []
        for category in candidates:
            if combo_satisfied(dice, category):
                score_func = CATEGORY_SCORE_FUNC[category]
                score = score_func(dice)
                total_moves.append((category, score))
        if total_moves:
            return max(total_moves, key=lambda x: x[1])[0].value
        return random.choice(candidates).value

    @staticmethod
    def _candidates_under_joker(
        scorecard_dict: dict, dice: np.ndarray, joker: bool
    ) -> list[Category]:
        """Replicate :meth:`YahtzeeEnv.action_masks` scoring-phase priority for
        the parsed-dict code path (which has no env handle).
        """
        if joker:
            face = int(dice[0])
            matching_upper = next(
                c for c in Category.upper_categories() if UPPER_SECTION_MAP[c] == face
            )
            upper_unmarked = [
                c for c in Category.upper_categories()
                if not scorecard_dict.get(c.value, 0.0)
            ]
            lower_unmarked_non_yahtzee = [
                c for c in Category.lower_categories()
                if c != Category.YAHTZEE and not scorecard_dict.get(c.value, 0.0)
            ]
            if matching_upper in upper_unmarked:
                return [matching_upper]
            if lower_unmarked_non_yahtzee:
                return lower_unmarked_non_yahtzee
            return upper_unmarked

        return [c for c in Category if not scorecard_dict.get(c.value, 0.0)]


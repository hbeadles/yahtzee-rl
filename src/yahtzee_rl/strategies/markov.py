import random

import numpy as np

from yahtzee_rl.config import Category, CATEGORY_SCORE_FUNC
from yahtzee_rl.envs.yahtzee_env import YahtzeeEnv
from yahtzee_rl.markov.lookaheads import determine_keep_positions
from yahtzee_rl.scoring.ops import combo_satisfied
from yahtzee_rl.scoring.scorecard import Scorecard
from yahtzee_rl.strategies.base import Strategy


class MarkovStrategy(Strategy):

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
            total_moves = []
            available_categories = []
            for category in Category:
                if not scorecard.is_category_marked(category) or category == Category.YAHTZEE:
                    satisfied = combo_satisfied(dice, category)
                    if satisfied:
                        score_func = scorecard.get_score_function(category)
                        score = score_func(dice)
                        total_moves.append((category, score))
                    available_categories.append(category)
            if total_moves:
                return max(total_moves, key=lambda x: x[1])[0].value
            else:
                return random.choice(available_categories).value

    def strategy_dict(self, parsed: dict) -> str:
        """
        Markov strategy operating on a parsed observation dict
        (from YahtzeeEnv.parse_observation) instead of a Scorecard object.

        Args:
            parsed: Dict returned by YahtzeeEnv.parse_observation().

        Returns:
            Category name (str) when scoring, or keep-positions mask when rolling.
        """
        scorecard_dict = parsed["scorecard"]

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
            return determine_keep_positions(parsed["dice"], Category(best_cat))
        else:
            dice = parsed["dice"]
            total_moves = []
            available_categories = []
            for category in Category:
                is_marked = scorecard_dict.get(category.value, 0.0)
                if not is_marked or category == Category.YAHTZEE:
                    satisfied = combo_satisfied(dice, category)
                    if satisfied:
                        score_func = CATEGORY_SCORE_FUNC[category]
                        score = score_func(dice)
                        total_moves.append((category, score))
                    available_categories.append(category)
            if total_moves:
                return max(total_moves, key=lambda x: x[1])[0].value
            else:
                return random.choice(available_categories).value


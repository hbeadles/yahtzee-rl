import gymnasium as gym
from yahtzee_rl.envs.yahtzee_env import YahtzeeEnv
from yahtzee_rl.config import Category
from yahtzee_rl.scoring.scorecard import Scorecard
from yahtzee_rl.scoring.ops import combo_satisfied
from yahtzee_rl.strategies.base import Strategy
import numpy as np
import random

class RandomStrategy(Strategy):
    """
    A truly random strategy. 

    The choice of the reroll action is independent of the choice of scoring action. 
      P(A | B) = P(A)  — knowing the scoring choice doesn't affect the reroll choice
      P(B | A) = P(B)  — knowing the reroll choice doesn't affect the scoring choice

    Note that these events are mutually exclusive only in timing. That is, we must roll the 
    dice before we can score, twice. However, the choice of the action is independent

    """
    def strategy(self, obs: np.ndarray, scorecard: Scorecard) -> int:
        parsed = YahtzeeEnv.parse_observation(obs)
        if not parsed['time_to_score']:
            return random.randint(0, 31)
        else:
            categories_not_marked = [category for category in Category if not scorecard.is_category_marked(category)]
            return random.choice(categories_not_marked).value


class RandomStrategyWithScorecard(Strategy):
    """
    A random strategy that chooses random dice to keep, but then, when it decides to score,
    we choose the category that best satisfies the dice in hand. 
    In practice this roughly doubles the expected score of the random strategy, to approximately
    ~100 points on average.
    """
    def strategy(self, obs: np.ndarray, scorecard: Scorecard) -> int:
        # Take a random action
        parsed = YahtzeeEnv.parse_observation(obs)
        if not parsed['time_to_score']:
            # Rolling dice action
            return random.randint(0, 31)
        else:
            # Scoring action
            total_moves = []
            available_categories = []
            for category in Category:
                if not scorecard.is_category_marked(category):
                    satisfied = combo_satisfied(parsed['dice'], category)
                    if satisfied:
                        score_func = scorecard.get_score_function(category)
                        score = score_func(parsed['dice'])
                        total_moves.append((category, score))
                    available_categories.append(category)
            if total_moves:
                return max(total_moves, key=lambda x: x[1])[0].value
            else:
                return random.choice(available_categories).value
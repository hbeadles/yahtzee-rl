from yahtzee_rl.scoring.scorecard import Scorecard
from yahtzee_rl.config import Category
from typing import Tuple
import numpy as np


class YahtzeeGame:
    """
    Yahtzee Game logic class. Generates a scorecard
    and rolls a dice. Makes it easier to create an
    env wrapper around
    """
    def __init__(self, seed: int = None):
        self.scorecard = Scorecard(turn_number=0)
        self.dice = np.zeros(5, dtype=int)
        self.rolls_remaining = 3
        self.round = 0
        self._rng = np.random.default_rng(seed)

    def roll_dice(self, roll_mask: np.ndarray) -> np.ndarray:
        """
        Roll dice indicated by roll_mask (1 = reroll, 0 = keep).
        """
        for i in range(5):
            if roll_mask[i] == 1:
                self.dice[i] = self._rng.integers(1, 7)
        self.rolls_remaining -= 1
        return self.dice.copy()
    
    def score_category(self, category: Category) -> Tuple[int, int, bool]:
        """
        Score current dice in category.

        Computes the Joker flag from the current scorecard state and dice, and
        forwards it to :meth:`Scorecard.mark_score`. Under Joker the +100 bonus
        accrues automatically via the scorecard's ``num_times_achieved`` counter.

        The "already marked non-Yahtzee" guard remains as a defensive penalty
        path; :meth:`YahtzeeEnv.action_masks` should prevent it from being
        reached, even under Joker (Joker priority routes around marked boxes).
        """
        joker = self.scorecard.joker_active(self.dice)
        if self.scorecard.score_board[category]["marked"] and category != Category.YAHTZEE:
            self.round += 1
            self.rolls_remaining = 3
            return -10.0, -10.0, False
        score = self.scorecard.mark_score(category, self.dice, joker_active=joker)
        upper_score = self.scorecard.compute_upper_score()
        lower_score = self.scorecard.compute_lower_score()
        self.round += 1
        self.rolls_remaining = 3
        return score, upper_score, lower_score, True

    def is_game_over(self) -> bool:
        """
        Check if game is over
        """
        return self.round >= 13

    def get_final_score(self) -> int:
        """
        Get final score
        """
        return self.scorecard.compute_final_score()
    
    def reset_rolls(self) -> None:
        """
        Reset rolls remaining
        """
        self.rolls_remaining = 3
        self.roll_dice(np.ones(5))

    def reset(self, seed: int = None) -> None:
        """
        Reset game
        """
        self._rng = np.random.default_rng(seed)
        self.scorecard.reset()
        self.dice = np.zeros(5, dtype=int)
        self.rolls_remaining = 3
        self.round = 0
        self.roll_dice(np.ones(5))
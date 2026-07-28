from typing import Any, Dict, Union

from yahtzee_rl.config import CATEGORIES, Category, SCORE_TYPES, ScoreFunc
from yahtzee_rl.scoring.ops import is_yahtzee, joker_score
import numpy as np


class Scorecard:
    """
    Represents a Yahtzee scorecard that tracks scores across all categories.
    """

    def __init__(self, turn_number: int) -> None:
        self.turn_number: int = turn_number
        self.score_board: Dict[Category, Dict[str, Any]] = {}
        for category_t in SCORE_TYPES:
            self.score_board[category_t[0]] = {
                "marked": False,
                "score_func": category_t[1],
                "score": 0,
                "category": category_t[2],
                "num_times_achieved": 0
            }

    def is_category_marked(self, category: Category) -> bool:
        """
        Check if a category is marked
        Args:
            category: The category to check

        Returns:
            True if the category is marked, False otherwise
        """
        return self.score_board[category]["marked"]

    def joker_eligible(self) -> bool:
        """
        Determine whether the Joker rule is available at all this game.

        Broad/"available" signal, independent of the current dice: True iff
        the YAHTZEE category has been scored with an actual Yahtzee (50 pts).
        A zeroed-out Yahtzee (scored 0 because the roll wasn't a Yahtzee)
        permanently disables the Joker rule per Hasbro's official rules, even
        though the box is still "marked".

        Returns:
            True iff a future Yahtzee roll would trigger Joker handling.
        """
        y = self.score_board[Category.YAHTZEE]
        return bool(y["marked"]) and y["score"] == 50

    def joker_active(self, dice: np.ndarray) -> bool:
        """
        Determine whether the Hasbro Joker rule is triggered by this roll.

        Narrow/"this instant" check used for live routing and scoring
        decisions. Joker is active iff:
          - :meth:`joker_eligible` is True, AND
          - the current dice form a Yahtzee (all five equal).

        When active, the player must score elsewhere per Joker priority and
        receives a +100 bonus that accrues via ``num_times_achieved`` in
        :meth:`mark_score`.

        Args:
            dice: np.ndarray of 5 dice values (current hand)

        Returns:
            True iff the Joker rule applies to this hand.
        """
        return self.joker_eligible() and is_yahtzee(dice)

    def get_score_function(self, category: Category) -> ScoreFunc:
        """
        Get the score function for a given category
        Args:
            category: The category to get the score function for

        Returns:
            The score function for the category
        """
        return self.score_board[category]["score_func"]

    def reset(self) -> None:
        """Reset the scorecard for a new game."""
        for category, score_data in self.score_board.items():
            score_data["marked"] = False
            score_data["score"] = 0
            score_data["num_times_achieved"] = 0
        self.turn_number = 0

    def mark_score(
        self,
        category: Union[Category, str],
        dice_roll: np.ndarray,
        joker_active: bool = False,
    ) -> float:
        """
        Mark a score for a given category with the provided dice roll.

        When ``joker_active`` is True the Hasbro Joker rule applies:
          - the score for ``category`` uses :func:`joker_score` (so Full House
            scores 25, straights 30/40, sum-based and uppers score 5*face);
          - the YAHTZEE row's ``num_times_achieved`` is incremented so the
            +100 bonus path in :meth:`compute_lower_score` becomes live.
          - the caller (env action mask) is responsible for ensuring
            ``category`` is a legal joker target.

        Args:
            category: The scoring category
            dice_roll: Array of 5 dice values
            joker_active: True iff the Joker rule applies to this roll.
                Caller should compute this via :meth:`joker_active`.

        Returns:
            The score for the category if it was marked, -1.0 otherwise
        """
        cat = Category(category) if not isinstance(category, Category) else category
        if self.score_board[cat]["marked"] and cat != Category.YAHTZEE:
            return -1.0
        if joker_active:
            score = joker_score(cat, dice_roll)   # was: self.score_board[cat]["score_func"](dice_roll)            if not self.score_board[cat]["marked"]:
            self.score_board[cat]["marked"] = True
            self.score_board[cat]["score"] = score
            self.score_board[cat]["num_times_achieved"] += 1
            self.score_board[Category.YAHTZEE]["num_times_achieved"] += 1
            return float(score)
        if self.score_board[cat]["marked"]:
            return -1.0
        self.score_board[cat]["marked"] = True
        self.score_board[cat]["score"] = self.score_board[cat]["score_func"](dice_roll)
        self.score_board[cat]["num_times_achieved"] += 1
        return self.score_board[cat]["score"]

    def compute_upper_score(self) -> int:
        """
        Calculate the upper section score (aces through sixes).
        
        Includes the 35-point bonus if the upper section total is >= 63.
        
        Returns:
            Total upper section score including bonus if applicable
        """
        upper_bonus = 35
        total_upper = 0

        for category, score_data in self.score_board.items():
            if score_data["marked"] and score_data["category"] == CATEGORIES.UPPER:
                total_upper += score_data["score"]

        if total_upper >= 63:
            total_upper += upper_bonus

        return total_upper

    def compute_lower_score(self) -> int:
        """
        Calculate the lower section score (three of a kind through chance).
        
        Includes 100-point bonus for each additional yahtzee beyond the first.
        
        Returns:
            Total lower section score including yahtzee bonuses if applicable
        """
        extra_yahtzee_bonus = 100
        total_lower = 0

        for category, score_data in self.score_board.items():
            if score_data["marked"] and score_data["category"] == CATEGORIES.LOWER:
                total_lower += score_data["score"]

        if self.score_board["yahtzee"]["num_times_achieved"] >= 2:
            total_lower += (extra_yahtzee_bonus * (self.score_board["yahtzee"]["num_times_achieved"] - 1))

        return total_lower

    def compute_final_score(self) -> int:
        """
        Calculate the total final score combining upper and lower sections.
        
        Returns:
            Total game score including all bonuses
        """
        return self.compute_upper_score() + self.compute_lower_score()

    def get_category_score(self, category: Category) -> int:
        """
        Get the score for a specific category.
        
        Args:
            category: The scoring category
            
        Returns:
            The score for the category (0 if not yet marked)
        """
        return self.score_board[category]["score"]

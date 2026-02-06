from typing import Any, Callable, Dict, List, Tuple

from yahtzee_rl import CATEGORY_NAMES, CATEGORIES, Category
from yahtzee_rl.scoring.ops import aces, twos, threes, fours, fives, sixes, \
    three_of_a_kind, four_of_a_kind, yahtzee, small_straight, full_house, \
    large_straight, chance
import numpy as np


# Type alias for score function
ScoreFunc = Callable[[np.ndarray], int]


class Scorecard:
    """
    Represents a Yahtzee scorecard that tracks scores across all categories.
    """

    SCORE_TYPES: List[Tuple[Category, ScoreFunc, CATEGORIES]] = [
        (Category.ACES, aces, CATEGORIES.UPPER),
        (Category.TWOS, twos, CATEGORIES.UPPER),
        (Category.THREES, threes, CATEGORIES.UPPER),
        (Category.FOURS, fours, CATEGORIES.UPPER),
        (Category.FIVES, fives, CATEGORIES.UPPER),
        (Category.SIXES, sixes, CATEGORIES.UPPER),
        (Category.THREE_OF_A_KIND, three_of_a_kind, CATEGORIES.LOWER),
        (Category.FOUR_OF_A_KIND, four_of_a_kind, CATEGORIES.LOWER),
        (Category.FULL_HOUSE, full_house, CATEGORIES.LOWER),
        (Category.SMALL_STRAIGHT, small_straight, CATEGORIES.LOWER),
        (Category.LARGE_STRAIGHT, large_straight, CATEGORIES.LOWER),
        (Category.YAHTZEE, yahtzee, CATEGORIES.LOWER),
        (Category.CHANCE, chance, CATEGORIES.LOWER),
    ]

    def __init__(self, turn_number: int) -> None:
        self.turn_number: int = turn_number
        self.score_board: Dict[Category, Dict[str, Any]] = {}
        for category_t in self.SCORE_TYPES:
            self.score_board[category_t[0]] = {
                "marked": False,
                "score_func": category_t[1],
                "score": 0,
                "category": category_t[2],
                "num_times_achieved": 0
            }

    def reset(self) -> None:
        """Reset the scorecard for a new game."""
        for category, score_data in self.score_board.items():
            score_data["marked"] = False
            score_data["score"] = 0
            score_data["num_times_achieved"] = 0
        self.turn_number = 0

    def mark_score(self, category: Category, dice_roll: np.ndarray) -> bool:
        """
        Mark a score for a given category with the provided dice roll.
        
        Args:
            category: The scoring category
            dice_roll: Array of 5 dice values
            
        Returns:
            True if score was marked, False if category was already marked
        """
        if self.score_board[category]["marked"]:
            return False
        else:
            self.score_board[category]["marked"] = True
            self.score_board[category]["score"] = self.score_board[category]["score_func"](dice_roll)
            self.score_board[category]["num_times_achieved"] += 1
            return True

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

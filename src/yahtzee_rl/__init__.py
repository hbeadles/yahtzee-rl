"""
Yahtzee RL - A reinforcement learning approach to Yahtzee.
"""
import random
import math
from enum import Enum, StrEnum
from typing import Callable, List, Dict, Tuple

import numpy


class CATEGORIES(Enum):
    """Enum representing the two main sections of the scorecard."""
    UPPER = 0
    LOWER = 1


class Category(StrEnum):
    """
    Enum of all valid Yahtzee scoring categories.
    
    Categories are divided into upper section (aces through sixes)
    and lower section (three_of_a_kind through chance).
    """
    # Upper section
    ACES = "aces"
    TWOS = "twos"
    THREES = "threes"
    FOURS = "fours"
    FIVES = "fives"
    SIXES = "sixes"
    
    # Lower section
    THREE_OF_A_KIND = "three_of_a_kind"
    FOUR_OF_A_KIND = "four_of_a_kind"
    FULL_HOUSE = "full_house"
    SMALL_STRAIGHT = "small_straight"
    LARGE_STRAIGHT = "large_straight"
    YAHTZEE = "yahtzee"
    CHANCE = "chance"
    
    @classmethod
    def upper_categories(cls) -> List["Category"]:
        """Return list of upper section categories."""
        return [cls.ACES, cls.TWOS, cls.THREES, cls.FOURS, cls.FIVES, cls.SIXES]
    
    @classmethod
    def lower_categories(cls) -> List["Category"]:
        """Return list of lower section categories."""
        return [
            cls.THREE_OF_A_KIND, cls.FOUR_OF_A_KIND, cls.FULL_HOUSE,
            cls.SMALL_STRAIGHT, cls.LARGE_STRAIGHT, cls.YAHTZEE, cls.CHANCE
        ]
    
    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if a string is a valid category name."""
        return value in cls._value2member_map_


# Legacy lists for backwards compatibility
UPPER_CATEGORY_NAMES: List[str] = [c.value for c in Category.upper_categories()]
LOWER_CATEGORY_NAMES: List[str] = [c.value for c in Category.lower_categories()]
CATEGORY_NAMES: List[str] = UPPER_CATEGORY_NAMES + LOWER_CATEGORY_NAMES
UPPER_SECTION_MAP: Dict[Category, int] = {
    Category.ACES: 1,
    Category.TWOS: 2,
    Category.THREES: 3,
    Category.FOURS: 4,
    Category.FIVES: 5,
    Category.SIXES: 6,
}

# Mapping from action index (0-12) to Category
ACTION_TO_CATEGORY: Dict[int, Category] = {
    i: category for i, category in enumerate(Category.upper_categories() + Category.lower_categories())
}
CATEGORY_TO_ACTION: Dict[Category, int] = {
    category: i for i, category in enumerate(Category.upper_categories() + Category.lower_categories())
}

ScoreFunc = Callable[[numpy.ndarray], int]

from yahtzee_rl.scoring.ops import (
    aces, twos, threes, fours, fives, sixes,
    three_of_a_kind, four_of_a_kind, yahtzee, small_straight,
    full_house, large_straight, chance,
)

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

CATEGORY_SCORE_FUNC: Dict[Category, ScoreFunc] = {
    cat: fn for cat, fn, _ in SCORE_TYPES
}


def dice_roll(num_dice: int = 5) -> numpy.ndarray:
    """
    Roll a specified number of dice.
    
    Args:
        num_dice: Number of dice to roll (default 5).
    
    Returns:
        Array of dice values (integers 1-6).
    """
    return numpy.ceil(numpy.random.uniform(0, 1, num_dice) * 6).astype(int)




def _build_cli():
    """Construct the root Typer CLI lazily to avoid circular imports at package import time."""
    import typer
    from yahtzee_rl.train.train_cli import app as train_app

    cli = typer.Typer(help="Yahtzee RL command-line interface.")
    cli.add_typer(train_app, name="train", help="Train RL agents on the Yahtzee environment.")

    # Future: evaluation CLI
    # from yahtzee_rl.evaluate.evaluate_cli import app as evaluate_app
    # cli.add_typer(evaluate_app, name="evaluate", help="Evaluate trained agents.")

    return cli


def main() -> None:
    """Entry point for the ``yahtzee-rl`` console script."""
    _build_cli()()

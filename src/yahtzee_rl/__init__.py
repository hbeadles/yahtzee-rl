"""
Yahtzee RL - A reinforcement learning approach to Yahtzee.
"""
import random
import math
from enum import Enum, StrEnum
from typing import List

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


def dice_roll(num_dice: int = 5) -> numpy.ndarray:
    """
    Roll a specified number of dice.
    
    Args:
        num_dice: Number of dice to roll (default 5).
    
    Returns:
        Array of dice values (integers 1-6).
    """
    return numpy.ceil(numpy.random.uniform(0, 1, num_dice) * 6).astype(int)


def main() -> None:
    """Entry point for the package."""
    print("Hello from yahtzee-rl!")

"""
Lookahead utilities for Yahtzee dice-keeping decisions.

This module provides functions to determine which dice to keep
based on the target scoring category (move).
"""
import numpy as np
import numpy.ma as ma
from functools import reduce
from typing import Dict, Tuple, Union

from yahtzee_rl import Category
from yahtzee_rl.scoring.ops import dice_count


# Mapping for upper section categories to their target dice values
UPPER_SECTION_MAP: Dict[Category, int] = {
    Category.ACES: 1,
    Category.TWOS: 2,
    Category.THREES: 3,
    Category.FOURS: 4,
    Category.FIVES: 5,
    Category.SIXES: 6,
}


def determine_keep_positions(
    dice: np.ndarray,
    withheld: np.ndarray,
    move: Union[Category, str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Determine which dice to keep based on the target scoring category.
    
    Given a set of rolled dice, previously withheld dice, and a target move,
    this function decides which dice from the current roll should be kept
    to maximize the chance of achieving the target category.
    
    Args:
        dice: Array of dice values from the current roll (values 1-6).
        withheld: Array of dice values already set aside from previous rolls.
        move: The target scoring category. Can be a Category enum value or
            a string matching a valid category name.
    
    Returns:
        A tuple of (final_withheld, remaining) where:
            - final_withheld: Combined array of previously withheld dice plus
              newly kept dice from this roll.
            - remaining: Array of dice from this roll that were not kept
              (to be re-rolled).
    
    Raises:
        ValueError: If move is not a valid Category.
    
    Examples:
        >>> dice = np.array([1, 2, 3, 1, 5])
        >>> withheld = np.array([])
        >>> final_withheld, remaining = determine_keep_positions(dice, withheld, Category.ACES)
        >>> # final_withheld contains [1, 1], remaining contains [2, 3, 5]
    """
    # Validate and convert move to Category enum
    if isinstance(move, str):
        if not Category.is_valid(move):
            raise ValueError(
                f"Invalid category '{move}'. Valid categories are: "
                f"{[c.value for c in Category]}"
            )
        move = Category(move)
    
    combined = np.concatenate((dice, withheld))
    mask = np.ones(len(dice))
    
    # Handle upper section categories (aces through sixes)
    if move in UPPER_SECTION_MAP:
        target = UPPER_SECTION_MAP[move]
        for i, d in enumerate(dice):
            if d == target:
                mask[i] = 0
    elif move == Category.THREE_OF_A_KIND:
        dice_combined = [(k, v) for k, v in dice_count(combined).items()]
        max_dice = reduce(lambda x, y: x if x[1] > y[1] else y, dice_combined)
        max_val = max_dice[0]
        for i, d in enumerate(dice):
            if d == max_val:
                mask[i] = 0
    elif move == Category.FOUR_OF_A_KIND:
        dice_combined = [(k, v) for k, v in dice_count(combined).items()]
        max_dice = reduce(lambda x, y: x if x[1] > y[1] else y, dice_combined)
        max_val = max_dice[0]
        for i, d in enumerate(dice):
            if d == max_val:
                mask[i] = 0
    elif move == Category.FULL_HOUSE:
        d_combined = dice_count(combined)
        dice_combined = [(k, v) for k, v in d_combined.items()]
        max_dice = reduce(lambda x, y: x if x[1] > y[1] else y, dice_combined)
        max_val = max_dice[0]
        
        # Guard: if only one unique value, just keep those dice
        dice_combined_second = [(k, v) for k, v in d_combined.items() if k != max_val]
        if dice_combined_second:
            max_dice_second = reduce(lambda x, y: x if x[1] > y[1] else y, dice_combined_second)
            max_val_second = max_dice_second[0]
        else:
            max_val_second = None  # No second value exists
        
        for i, d in enumerate(dice):
            if d == max_val or (max_val_second is not None and d == max_val_second):
                mask[i] = 0
    elif move == Category.SMALL_STRAIGHT:
        first = np.array([1,2,3,4])
        first_in = np.unique(combined[np.isin(combined, first)])
        first_not_in = first[~np.isin(first, withheld)]
        second = np.array([2,3,4,5])
        third = np.array([3,4,5,6])
        second_in = np.unique(combined[np.isin(combined, second)])
        second_not_in = second[~np.isin(second, withheld)]
        third_in = np.unique(combined[np.isin(combined, third)])
        third_not_in = third[~np.isin(third, withheld)]
        l_first = first_in.size
        l_second = second_in.size
        l_third = third_in.size
        closest = np.max([l_first, l_second, l_third])
        if closest == l_first:
            for i, d in enumerate(dice):
                if d in first_not_in:
                    mask[i] = 0
                    first_not_in = first_not_in[~np.isin(first_not_in, np.array([d]))]
        elif closest == l_second:
            for i, d in enumerate(dice):
                if d in second_not_in:
                    mask[i] = 0
                    second_not_in = second_not_in[~np.isin(second_not_in, np.array([d]))]
        elif closest == l_third:
            for i, d in enumerate(dice):
                if d in third_not_in:
                    mask[i] = 0
                    third_not_in = third_not_in[~np.isin(third_not_in, np.array([d]))]

    elif move == Category.LARGE_STRAIGHT:
        first = np.array([1,2,3,4,5])
        first_in = np.unique(combined[np.isin(combined, first)])
        first_not_in = first[~np.isin(first, withheld)]
        second = np.array([2,3,4,5,6])
        second_in = np.unique(combined[np.isin(combined, second)])
        second_not_in = second[~np.isin(second, withheld)]
        l_first = first_in.size
        l_second = second_in.size
        closest = np.max([l_first, l_second])
        if closest == l_first:
            for i, d in enumerate(dice):
                if d in first_not_in:
                    mask[i] = 0
                    first_not_in = first_not_in[~np.isin(first_not_in, np.array([d]))]
        elif closest == l_second:
            for i, d in enumerate(dice):
                if d in second_not_in:
                    mask[i] = 0
                    second_not_in = second_not_in[~np.isin(second_not_in, np.array([d]))]
    elif move == Category.YAHTZEE:
        dice_combined = [(k, v) for k, v in dice_count(combined).items()]
        max_dice = reduce(lambda x, y: x if x[1] > y[1] else y, dice_combined)
        max_val = max_dice[0]
        for i, d in enumerate(dice):
            if d == max_val:
                mask[i] = 0
    elif move == Category.CHANCE:
        max_d = np.max(combined)
        for i, d in enumerate(dice):
            if d >= max_d:
                mask[i] = 0

    mask_d = ma.array(dice, mask=mask)
    mask_d_remain = ma.array(dice, mask=np.logical_not(mask))
    kept = mask_d.compressed()
    remaining = mask_d_remain.compressed()
    final_withheld = np.concatenate((withheld, kept))

    return final_withheld, remaining
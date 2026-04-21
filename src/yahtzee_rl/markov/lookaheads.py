"""
Lookahead utilities for Yahtzee dice-keeping decisions.

This module provides functions to determine which dice to keep
based on the target scoring category (move).
"""
import numpy as np
from functools import reduce
from typing import List, Union
from yahtzee_rl import Category, UPPER_SECTION_MAP
from yahtzee_rl.scoring.ops import dice_count



def _best_run(dice: np.ndarray, runs: List[np.ndarray]) -> np.ndarray:
    """Return the run from *runs* that has the most unique matches in *dice*."""
    best, best_count = runs[0], 0
    for run in runs:
        count = np.unique(dice[np.isin(dice, run)]).size
        if count > best_count:
            best, best_count = run, count
    return best


def _keep_for_run(dice: np.ndarray, mask: np.ndarray, run: np.ndarray) -> None:
    """Mark mask positions as keep (0) for one die per unique value in *run*."""
    needed = list(run)
    for i, d in enumerate(dice):
        if int(d) in needed:
            mask[i] = 0
            needed.remove(int(d))


def determine_keep_positions(
    dice: np.ndarray,
    move: Union[Category, str]
) -> int:
    """
    Determine which dice to keep based on the target scoring category.

    Given a 5-die array and a target category, returns an action integer
    (0-31) whose 5-bit representation is the reroll mask compatible with
    ``YahtzeeGame.roll_dice`` / ``YahtzeeEnv.step``.

    Bit semantics (little-endian): bit *i* = 1 means **reroll** die *i*,
    bit *i* = 0 means **keep** die *i*.

    Args:
        dice: Array of 5 dice values (values 1-6).
        move: The target scoring category. Can be a Category enum value or
            a string matching a valid category name.

    Returns:
        An integer action in [0, 31] encoding the reroll mask.

    Raises:
        ValueError: If move is not a valid Category.

    Examples:
        >>> dice = np.array([1, 2, 3, 1, 5])
        >>> action = determine_keep_positions(dice, Category.ACES)
        >>> # action encodes: keep positions 0 and 3 (the 1s), reroll the rest
    """
    if isinstance(move, str):
        if not Category.is_valid(move):
            raise ValueError(
                f"Invalid category '{move}'. Valid categories are: "
                f"{[c.value for c in Category]}"
            )
        move = Category(move)

    mask = np.ones(5, dtype=np.uint8)

    # Handle upper section categories (aces through sixes)
    if move in UPPER_SECTION_MAP:
        target = UPPER_SECTION_MAP[move]
        for i, d in enumerate(dice):
            if d == target:
                mask[i] = 0
    elif move == Category.THREE_OF_A_KIND:
        dice_combined = [(k, v) for k, v in dice_count(dice).items()]
        max_dice = reduce(lambda x, y: x if x[1] > y[1] else y, dice_combined)
        max_val = max_dice[0]
        for i, d in enumerate(dice):
            if d == max_val:
                mask[i] = 0
    elif move == Category.FOUR_OF_A_KIND:
        dice_combined = [(k, v) for k, v in dice_count(dice).items()]
        max_dice = reduce(lambda x, y: x if x[1] > y[1] else y, dice_combined)
        max_val = max_dice[0]
        for i, d in enumerate(dice):
            if d == max_val:
                mask[i] = 0
    elif move == Category.FULL_HOUSE:
        d_combined = dice_count(dice)
        dice_combined = [(k, v) for k, v in d_combined.items()]
        max_dice = reduce(lambda x, y: x if x[1] > y[1] else y, dice_combined)
        max_val = max_dice[0]

        # Guard: if only one unique value, just keep those dice
        dice_combined_second = [(k, v) for k, v in d_combined.items() if k != max_val]
        if dice_combined_second:
            max_dice_second = reduce(lambda x, y: x if x[1] > y[1] else y, dice_combined_second)
            max_val_second = max_dice_second[0]
        else:
            max_val_second = None

        for i, d in enumerate(dice):
            if d == max_val or (max_val_second is not None and d == max_val_second):
                mask[i] = 0
    elif move == Category.SMALL_STRAIGHT:
        runs = [np.array([1,2,3,4]), np.array([2,3,4,5]), np.array([3,4,5,6])]
        best = _best_run(dice, runs)
        _keep_for_run(dice, mask, best)
    elif move == Category.LARGE_STRAIGHT:
        runs = [np.array([1,2,3,4,5]), np.array([2,3,4,5,6])]
        best = _best_run(dice, runs)
        _keep_for_run(dice, mask, best)
    elif move == Category.YAHTZEE:
        dice_combined = [(k, v) for k, v in dice_count(dice).items()]
        max_dice = reduce(lambda x, y: x if x[1] > y[1] else y, dice_combined)
        max_val = max_dice[0]
        for i, d in enumerate(dice):
            if d == max_val:
                mask[i] = 0
    elif move == Category.CHANCE:
        max_d = np.max(dice)
        for i, d in enumerate(dice):
            if d >= max_d:
                mask[i] = 0

    return int(np.packbits(mask, bitorder='little')[0])
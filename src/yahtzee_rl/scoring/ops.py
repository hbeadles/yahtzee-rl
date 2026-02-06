import numpy as np
from collections import Counter
from yahtzee_rl import Category
from typing import Union

def max_val_sums(num: int) -> int:
    """
    Calculate max sum of a potential number
    :param num: Given a number, multiply it by 5 to get the
    total max value for (aces through 6s)
    :return: num * 5
    """
    return num * 5

def dice_sum_if_equal(dice: np.ndarray, num_target: int) -> int:
    """
    Sum the dice values that equal a particular target value.
    :param dice: np.ndarray of dice
    :param num_target: Dice value we're looking for
    :return: sum of dice values that equal num_target
    """
    return np.sum(dice[dice == num_target])

def dice_count(dice: np.ndarray) -> Counter:
    """
    Get a counter object of dice values
    :param dice: np.ndarray of dice
    :return: Counter object
    """
    return Counter(dice)


def combo_satisfied(dice: np.ndarray, move: Union[Category, str]) -> bool:
    """
    Check if a particular dice combination is satisfied
    :param dice: np.ndarray of dice
    :param move: string of the move
    :return: True if satisfied, False otherwise
    """
    satisfied = False
    if move == Category.THREE_OF_A_KIND:
        d = dice_count(dice)
        satisfied = False
        for k, v in d.items():
            if v >= 3:
                satisfied = True
                break
    elif move == Category.FOUR_OF_A_KIND:
        d = dice_count(dice)
        satisfied = False
        for k, v in d.items():
            if v >= 4:
                satisfied = True
                break
    elif move == Category.FULL_HOUSE:
        d = dice_count(dice).most_common(2)
        satisfied = True
        for (k, v) in d:
            if v == 3 or v == 2:
                continue
            else:
                satisfied = False
                break
    elif move == Category.SMALL_STRAIGHT:
        first = np.array([1, 2, 3, 4])
        second = np.array([2, 3, 4, 5])
        third = np.array([3, 4, 5, 6])
        first_in = np.unique(dice[np.isin(dice, first)])
        second_in = np.unique(dice[np.isin(dice, second)])
        third_in = np.unique(dice[np.isin(dice, third)])
        if first_in.size == 4 or second_in.size == 4 or third_in.size == 4:
            satisfied = True
    elif move == Category.LARGE_STRAIGHT:
        first = np.array([1, 2, 3, 4, 5])
        second = np.array([2, 3, 4, 5, 6])
        first_in = np.unique(dice[np.isin(dice, first)])
        second_in = np.unique(dice[np.isin(dice, second)])
        if first_in.size == 5 or second_in.size == 5:
            satisfied = True
    elif move == Category.YAHTZEE:
        d = dice_count(dice)
        satisfied = False
        for k, v in d.items():
            if v == 5:
                satisfied = True
                break
    elif move == Category.CHANCE:
        satisfied = True

    return satisfied

def aces(dice_roll: np.ndarray) -> int:
    """
    Helper to calculate number of aces given a dice roll
    -- Total of Aces only
    :param dice_roll: input dice roll
    :return: total score (int)
    """
    total = dice_sum_if_equal(dice_roll, 1)
    return total

def twos(dice_roll: np.ndarray) -> int:
    """
    Helper to calculate number of twos given a dice roll
    -- Total of Twos only
    :param dice_roll: input dice roll
    :return: total_score (int)
    """
    total = dice_sum_if_equal(dice_roll, 2)
    return total


def threes(dice_roll: np.ndarray) -> int:
    """
    Helper to calculate number of threes given a dice roll
    -- Total of Threes only
    :param dice_roll: input dice roll
    :return: total_score (int)
    """
    total = dice_sum_if_equal(dice_roll, 3)
    return total


def fours(dice_roll: np.ndarray) -> int:
    """
    Helper to calculate number of fours given a dice roll
    -- Total of Fours only
    :param dice_roll: input dice roll
    :return: total_score (int)
    """
    total = dice_sum_if_equal(dice_roll, 4)
    return total


def fives(dice_roll: np.ndarray) -> int:
    """
    Helper to calculate number of fives given a dice roll
    -- Total of Fives only
    :param dice_roll: input dice roll
    :return: total_score (int)
    """
    total = dice_sum_if_equal(dice_roll, 5)
    return total


def sixes(dice_roll: np.ndarray) -> int:
    """
    Helper to calculate number of sixes given a dice roll
    -- Total of Sixes only
    :param dice_roll: input dice roll
    :return: total_score (int)
    """
    total = dice_sum_if_equal(dice_roll, 6)
    return total

def three_of_a_kind(dice_roll: np.ndarray) -> int:
    """
    If combo of three of a kind is satisfied,
    return the total of all 5 dice
    :param dice_roll: input dice roll
    :return: total score (int)
    """
    satisfied = combo_satisfied(dice_roll, Category.THREE_OF_A_KIND)
    if satisfied:
        return np.sum(dice_roll)
    else:
        return 0


def four_of_a_kind(dice_roll: np.ndarray) -> int:
    """
    If combo of four of a kind is satisfied, ,
    return the total of ol 5 dice
    :param dice_roll: input dice roll
    :return: total score (int)
    """
    satisfied = combo_satisfied(dice_roll, Category.FOUR_OF_A_KIND)
    if satisfied:
        return np.sum(dice_roll)
    else:
        return 0

def full_house(dice_roll: np.ndarray) -> int:
    """
    If combo of 3 of one number, and two of another number is satisfied,
    return the total of 5 dice
    :param dice_roll: input dice roll
    :return: total score (int)
    """
    satisfied = combo_satisfied(dice_roll, Category.FULL_HOUSE)
    if satisfied:
        return 25
    else:
        return 0


def small_straight(dice_roll: np.ndarray) -> int:
    """
    If the dice show any sequence of four numbers,
    return 30 points
    :param dice_roll: input dice roll
    :return: total score (int)
    """
    if combo_satisfied(dice_roll, Category.SMALL_STRAIGHT):
        return 30
    else:
        return 0


def large_straight(dice_roll: np.ndarray) -> int:
    """
    If the dice show any sequence of five numbers,
    return 40 points
    :param dice_roll: input dice roll
    :return: total score (int)
    """
    if combo_satisfied(dice_roll, Category.LARGE_STRAIGHT):
        return 40
    else:
        return 0


def yahtzee(dice_roll: np.ndarray) -> int:
    """
    If all dice are the same value, return 50 points
    :param dice_roll: input dice roll
    :return: total score (int)
    """
    if combo_satisfied(dice_roll, Category.YAHTZEE):
        return 50
    else:
        return 0

def chance(dice_roll: np.ndarray) -> int:
    """
    Chance sums up all the dice in the current hand
    :param dice_roll: input dice roll
    :return: total score (int)
    """
    return np.sum(dice_roll)

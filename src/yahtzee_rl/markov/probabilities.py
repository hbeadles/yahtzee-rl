import numpy as np
from functools import lru_cache
from typing import Tuple

from yahtzee_rl import Category
from yahtzee_rl.scoring.ops import dice_count
from yahtzee_rl.scoring.scorecard import Scorecard
from yahtzee_rl.scoring.ops import dice_sum_if_equal

faces: dict[Category, int] = {
    Category.ACES: 1,
    Category.TWOS: 2,
    Category.THREES: 3,
    Category.FOURS: 4,
    Category.FIVES: 5,
    Category.SIXES: 6,
}
### Lower section payoff utilities ###

LOWER_FIXED_SCORES: dict[Category, int] = {
    Category.FULL_HOUSE: 25,
    Category.SMALL_STRAIGHT: 30,
    Category.LARGE_STRAIGHT: 40,
    Category.YAHTZEE: 50,
}

YAHTZEE_BONUS = 100
# Max possible sum-of-dice score (five 6s)
MAX_DICE_SUM = 30
# Expected sum of 5 fair dice (3.5 × 5) — conservative baseline for sum-based categories
MEAN_DICE_SUM = 17.5


@lru_cache(maxsize=4)
def _upper_count_t_powered(remaining_rolls: int) -> np.ndarray:
    """Cached matrix power of upper_count_t_matrix for a given number of remaining rolls."""
    return np.linalg.matrix_power(upper_count_t_matrix(), remaining_rolls)

@lru_cache(maxsize=4)
def _runs_t_powered(remaining_rolls: int) -> np.ndarray:
    """Cached matrix power of runs_t_matrix for a given number of remaining rolls."""
    return np.linalg.matrix_power(runs_t_matrix(), remaining_rolls)

@lru_cache(maxsize=4)
def _straight_t_powered(remaining_rolls: int) -> np.ndarray:
    """Cached matrix power of straight_t_matrix for a given number of remaining rolls."""
    return np.linalg.matrix_power(straight_t_matrix(), remaining_rolls)

@lru_cache(maxsize=4)
def _large_straight_t_powered(remaining_rolls: int) -> np.ndarray:
    """Cached matrix power of large_straight_t_matrix for a given number of remaining rolls."""
    return np.linalg.matrix_power(large_straight_t_matrix(), remaining_rolls)

"""
Fix reaching x to sum probabilities above target state (>= survival function)
"""

def lower_section_probabilities(dice: np.ndarray, score_card: Scorecard, 
                                move: Category, remaining_rolls: int) -> float:
    """
    Determine final probabilities for reaching lower section moves. 
    We also consider whether a move has already been marked. Unless it is a yahtzee.
    Args:
        dice: array of dice values (values 1-6)
        score_card: the scorecard
        move: the move to consider
        remaining_rolls: the number of remaining rolls

    Returns:
        the probability of reaching the move
    """
    def _is_marked(score_card: Scorecard, move: Category) -> bool:
        return score_card.is_category_marked(move)
    if move == Category.THREE_OF_A_KIND and not _is_marked(score_card, move):
        counts = dice_count(dice)
        max_face = max(counts, key=counts.get)
        return simple_three_of_a_kind(dice, max_face, remaining_rolls)
    elif move == Category.FOUR_OF_A_KIND and not _is_marked(score_card, move):
        counts = dice_count(dice)
        max_face = max(counts, key=counts.get)
        return simple_four_of_a_kind(dice, max_face, remaining_rolls)
    elif move == Category.FULL_HOUSE and not _is_marked(score_card, move):
        return full_house(dice, remaining_rolls)
    elif move == Category.SMALL_STRAIGHT:
        return determine_small_straight_probability(dice, remaining_rolls)
    elif move == Category.LARGE_STRAIGHT and not _is_marked(score_card, move):
        return determine_large_straight_probability(dice, remaining_rolls)
    elif move == Category.YAHTZEE and not _is_marked(score_card, move):
        return yahtzee(dice, remaining_rolls)
    elif move == Category.CHANCE and not _is_marked(score_card, move):
        return 1.0

    return 0.0



def upper_section_probability(dice: np.ndarray, score_card: Scorecard,
                               move: Category, remaining_rolls: int) -> float:
    """
    P(getting >= 3 matching dice for this upper section category).

    Threshold of 3 is strategic: 3 x face_value is the per-category average
    needed to hit the 63-point upper bonus. Same semantics as lower section
    probabilities (a true probability in [0, 1]).

    Args:
        dice: array of dice values (values 1-6)
        score_card: the scorecard
        move: the upper section category
        remaining_rolls: the number of remaining rolls

    Returns:
        P(count >= 3) for the category, or 0.0 if marked/invalid
    """
    if move not in faces or score_card.is_category_marked(move):
        return 0.0
    _, dist = upper_section_markov(dice, move, remaining_rolls)
    return float(np.sum(dist[3:]))


def upper_section_expected_score(dice: np.ndarray, score_card: Scorecard,
                                  move: Category, remaining_rolls: int) -> float:
    """
    Expected score for a single upper section category.
    Returns face_value * expected_count (raw expected points).

    Args:
        dice: array of dice values (values 1-6)
        score_card: the scorecard
        move: the upper section category
        remaining_rolls: the number of remaining rolls

    Returns:
        Expected score (0.0 to face_value * 5), or 0.0 if marked/invalid
    """
    remaining = 375.0 - score_card.compute_final_score()
    denominator = remaining if remaining > 0 else 375.0
    upper_score_max = 63.0
    upper_score_current = score_card.compute_upper_score()
    top_remaining = upper_score_max - upper_score_current
    denom = .2 * (top_remaining / upper_score_max) 
    if move not in faces or score_card.is_category_marked(move):
        return 0.0
    exp_count, _ = upper_section_markov(dice, move, remaining_rolls)
    return (faces[move] * exp_count * (1 + denom)) / denominator



def lower_section_expected_score(dice: np.ndarray, score_card: Scorecard,
                                  move: Category, remaining_rolls: int) -> float:
    """
    Expected score for a single lower section category.

    Fixed-score categories (full house, straights, yahtzee):
        P(combo) × fixed score.
    Sum-based categories (three/four of a kind, chance):
        P(combo) × E[dice sum].
    Yahtzee bonus: if yahtzee was previously scored, P(yahtzee) × 100
        is still available even though the category is marked.

    Args:
        dice: array of dice values (values 1-6)
        score_card: the scorecard
        move: the lower section category
        remaining_rolls: the number of remaining rolls

    Returns:
        Expected score (float), or 0.0 if marked/invalid
    """
    remaining = 375.0 - score_card.compute_final_score()
    denominator = remaining if remaining > 0 else 375.0
    if move not in Category.lower_categories():
        return 0.0
    # Yahtzee requires special handling for the bonus
    if move == Category.YAHTZEE:
        yahtzee_achieved = score_card.score_board[Category.YAHTZEE]["num_times_achieved"]
        raw_prob = yahtzee(dice, remaining_rolls)
        if score_card.is_category_marked(Category.YAHTZEE):
            # Category filled, but each additional yahtzee earns +100
            if yahtzee_achieved >= 1:
                return (raw_prob * YAHTZEE_BONUS) / denominator
            return 0.0
        else:
            # First yahtzee: base 50 pts
            return (raw_prob * LOWER_FIXED_SCORES[Category.YAHTZEE]) / denominator

    prob = lower_section_probabilities(dice, score_card, move, remaining_rolls)

    # Fixed-score categories
    if move in LOWER_FIXED_SCORES:
        return (prob * LOWER_FIXED_SCORES[move]) / denominator

    # Sum-based categories: score = sum of all 5 dice when combo is achieved
    if move in (Category.THREE_OF_A_KIND, Category.FOUR_OF_A_KIND):
        return (prob * (np.sum(dice))) / denominator

    # Chance: always achievable (prob=1.0), score = sum of dice
    # With remaining rolls we'd keep high dice; current sum is a lower bound
    if move == Category.CHANCE:
        return (float(np.sum(dice))) / denominator

    return 0.0



def upper_section_prob_vector(dice: np.ndarray, score_card: Scorecard,
                               remaining_rolls: int) -> np.ndarray:
    """
    Observation vector of P(>=3) probabilities for all 6 upper section categories.
    Marked categories get 0.0. All values in [0, 1].

    Args:
        dice: array of dice values (values 1-6)
        score_card: the scorecard
        remaining_rolls: the number of remaining rolls

    Returns:
        np.ndarray of shape (6,) with P(>=3) per upper category
    """
    obs = np.zeros(6, dtype=np.float32)
    for i, category in enumerate(Category.upper_categories()):
        obs[i] = upper_section_probability(dice, score_card, category, remaining_rolls)
    return obs


def upper_section_expected_score_vector(dice: np.ndarray, score_card: Scorecard,
                                         remaining_rolls: int) -> np.ndarray:
    """
    Observation vector of normalized expected scores for all 6 upper section categories.
    Each value is expected_score / max_possible_score = expected_count / 5, in [0, 1].
    Marked categories get 0.0.

    Args:
        dice: array of dice values (values 1-6)
        score_card: the scorecard
        remaining_rolls: the number of remaining rolls

    Returns:
        np.ndarray of shape (6,) with normalized expected scores per upper category
    """
    obs = np.zeros(6, dtype=np.float32)
    for i, category in enumerate(Category.upper_categories()):
        if score_card.is_category_marked(category):
            continue
        exp_count = upper_section_expected_score(dice, score_card, category, remaining_rolls)
        obs[i] = exp_count  
    return obs


def lower_section_prob_vector(dice: np.ndarray, score_card: Scorecard,
                               remaining_rolls: int) -> np.ndarray:
    """
    Observation vector of probabilities for all 7 lower section categories.
    Marked categories get 0.0 (except yahtzee bonus). All values in [0, 1].

    Args:
        dice: array of dice values (values 1-6)
        score_card: the scorecard
        remaining_rolls: the number of remaining rolls

    Returns:
        np.ndarray of shape (7,) with P(combo) per lower category
    """
    obs = np.zeros(7, dtype=np.float32)
    for i, category in enumerate(Category.lower_categories()):
        obs[i] = lower_section_probabilities(dice, score_card, category, remaining_rolls)
    return obs


def lower_section_expected_score_vector(dice: np.ndarray, score_card: Scorecard,
                                         remaining_rolls: int) -> np.ndarray:
    """
    Observation vector of normalized expected scores for all 7 lower section categories.
    Each value is expected_score / max_possible_score, in [0, 1].
    Marked categories get 0.0 (except yahtzee bonus).

    Args:
        dice: array of dice values (values 1-6)
        score_card: the scorecard
        remaining_rolls: the number of remaining rolls

    Returns:
        np.ndarray of shape (7,) with normalized expected scores per lower category
    """
    obs = np.zeros(7, dtype=np.float32)
    for i, category in enumerate(Category.lower_categories()):
        expected = lower_section_expected_score(dice, score_card, category, remaining_rolls)
        obs[i] = expected 
    return obs


def upper_section_markov(dice: np.ndarray, category: Category, 
                         remaining_rolls: int) -> Tuple[float, np.ndarray]:
    """
    Core Markov chain computation for upper section categories.
    Returns both the expected count and the full probability distribution
    over final counts (0-5 matching dice).

    Args:
        dice: array of dice values (values 1-6)
        category: the upper section category to consider
        remaining_rolls: the number of remaining rolls

    Returns:
        Tuple of (expected_count, distribution) where:
            - expected_count: E[matching dice] after remaining rolls (0.0 to 5.0)
            - distribution: np.ndarray of shape (6,) with P(count=i) for i in 0..5
    """
    count = dice_count(dice)[faces[category]]
    state = np.zeros(6, dtype=float)
    state[count] = 1
    dist = _upper_count_t_powered(remaining_rolls) @ state
    expected = float(np.dot(np.arange(6), dist))
    return expected, dist.astype(np.float32)

def reaching_x(dice: np.ndarray, dice_number: int, target_state: int, remaining_rolls: int) -> float:
    """
    Probability of reaching a state with a given number of dice of a certain number
    
    Args:
        dice: array of dice values (values 1-6)
        dice_number: the number of the dice to reach
        target_state: the state to reach (0-4). 0 is 1 matching dice, 2
        is a pair, 3 is 3 of a kind, 4 is 4 of a kind, 5 is yahtzee
        remaining_rolls: the number of remaining rolls

    Returns:
        probability of reaching the state
    """
    count = dice_count(dice)[dice_number]
    initial_state = np.zeros(5)
    initial_state[count-1] = 1
    state_vec = _runs_t_powered(remaining_rolls) @ initial_state
    return np.sum(state_vec[target_state:])

def determine_straight_state(dice: np.ndarray, straight: np.ndarray) -> int:
    """
    Determine the state of a straight
    Args:
        dice: array of dice values (values 1-6)
        straight: array of straight values (values 1-6)

    Returns:
        the state of the straight (0-4)
    """
    intersect = np.intersect1d(dice, straight)
    return intersect.size

def determine_small_straight_probability(dice: np.ndarray, remaining_rolls: int) -> float:
    """
    Probability of achieving a small straight (4 consecutive numbers in a row).
    Args:
        dice: array of dice values (values 1-6)
        remaining_rolls: the number of remaining rolls
    """
    possibles = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])
    for possible in possibles:
        l_state = determine_straight_state(dice, possible)
        if l_state >= 1:
            if l_state == 4:
                return 1.0
            else:
                initial_state = np.zeros(4)
                initial_state[l_state-1] = 1
                probability_v = _straight_t_powered(remaining_rolls) @ initial_state
                chosen_prob = probability_v[3]
                return chosen_prob
    return 0.0

def determine_large_straight_probability(dice: np.ndarray, remaining_rolls: int) -> float:
    """
    Probability of achieving a large straight (5 consecutive numbers in a row).
    Args:
        dice: array of dice values (values 1-6)
        remaining_rolls: the number of remaining rolls
    """
    pass
    possibles = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
    for possible in possibles:
        l_state = determine_straight_state(dice, possible)
        if l_state >= 1:
            if l_state == 5:
                return 1.0
            else:
                initial_state = np.zeros(5)
                initial_state[l_state-1] = 1
                probability_v = _large_straight_t_powered(remaining_rolls) @ initial_state
                chosen_prob = probability_v[4]
                return chosen_prob
    return 0.0

def simple_three_of_a_kind(dice: np.ndarray, dice_number: int, remaining_rolls: int) -> float:
    """
    Probability of reaching a state with a given number of dice of a certain number
    Args:
        dice: array of dice values (values 1-6)
        dice_number: the number of the dice to reach
        remaining_rolls: the number of remaining rolls
    """
    count = dice_count(dice)[dice_number]
    if count >= 3:
        return 1.0
    else:
        return reaching_x(dice, dice_number, 2, remaining_rolls)


def simple_four_of_a_kind(dice: np.ndarray, dice_number: int, remaining_rolls: int) -> float:
    """
    Probability of reaching a state with a given number of dice of a certain number
    Args:
        dice: array of dice values (values 1-6)
        dice_number: the number of the dice to reach
        remaining_rolls: the number of remaining rolls
    """
    count = dice_count(dice)[dice_number]
    if count >= 4:
        return 1.0
    else:
        return reaching_x(dice, dice_number, 3, remaining_rolls)


def full_house(dice: np.ndarray, remaining_rolls: int) -> float:
    """
    Probability of achieving a full house (3 of one number + 2 of another).
    
    Args:
        dice: array of dice values (values 1-6)
        remaining_rolls: the number of remaining rolls
    
    Returns:
        probability of reaching a full house
    """
    counts = dice_count(dice)
    sorted_counts = sorted(counts.values(), reverse=True)
    while len(sorted_counts) < 2:
        sorted_counts.append(0)
    # Case 1: Already full house
    if sorted_counts[0] >= 3 and sorted_counts[1] >= 2:
        return 1.0
    
    # Case 2: Triple + singles
    if sorted_counts[0] >= 3:
        return 1 - (5/6) ** remaining_rolls
    
    # Case 3: Two pairs
    if sorted_counts[0] >= 2 and sorted_counts[1] >= 2:
        return 1 - (4/6) ** remaining_rolls
    
    # Case 4: Pair + singles
    if sorted_counts[0] >= 2:
        pair_num = max(counts, key=counts.get)
        p_triple = reaching_x(dice, pair_num, 2, remaining_rolls)
        p_pair_forms = 1 - (120/216) ** remaining_rolls
        return p_triple * p_pair_forms
    
    # Case 5: All singles - approximate
    return 0.05 * remaining_rolls

def yahtzee(dice: np.ndarray, remaining_rolls: int) -> float:
    """
    Probability of achieving a yahtzee (all five dice the same).
    Args:
        dice: array of dice values (values 1-6)
        remaining_rolls: the number of remaining rolls
    """
    counts = dice_count(dice)
    max_face = max(counts, key=counts.get)
    return reaching_x(dice, max_face, 4, remaining_rolls)

def runs_t_matrix():
    """
    A lower triangular form of the Transition matrix for a run. In this case
    it assumes that it won't go back and release a die it has already gained.
    This can be viewed as a "greedy" algorithm.

    :return: transition_matrix
    """
    return np.array([[120 / 1296, 0, 0, 0, 0],
                     [900 / 1296, 120 / 216, 0, 0, 0],
                     [250 / 1296, 80 / 216, 25 / 36, 0, 0],
                     [25 / 1296, 15 / 216, 10 / 36, 5 / 6, 0],
                     [1 / 1296, 1 / 216, 1 / 36, 1 / 6, 1]])


def straight_t_matrix():
    """
    Small straight computation, for four consecutive numbers in a row

    :return:
    """
    return np.array([[108 / 1296, 0, 0, 0],
                     [525 / 1296, 64 / 216, 0, 0],
                     [582 / 1296, 122 / 216, 25 / 36, 0],
                     [108 / 1296, 30 / 216, 11 / 36, 1]])


def large_straight_t_matrix():
    """
    Large Straight computation, five consecutive numbers in a row
    :return:
    """
    return np.array([[16 / 1296, 0, 0, 0, 0],
                     [260 / 1296, 27 / 216, 0, 0, 0],
                     [600 / 1296, 111 / 216, 16 / 36, 0, 0],
                     [336 / 1296, 72 / 216, 18 / 36, 5 / 6, 0],
                     [24 / 1296, 6 / 216, 2 / 36, 1 / 6, 1]])

def upper_count_t_matrix():
    """
    Transition matrix for upper section counts
    """
    return np.array([[3125/7776,0,0,0,0,0],
                     [3125/7776, 625/1296, 0, 0, 0, 0],
                     [ 625/3888, 125/324, 125/216, 0, 0, 0],
                     [ 125/3888, 25/216, 25/72, 25/36, 0, 0],
                     [  25/7776, 5/324, 5/72, 5/18, 5/6, 0],
                     [   1/7776, 1/1296, 1/216, 1/36, 1/6, 1]])
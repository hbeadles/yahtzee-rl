import numpy as np
from functools import lru_cache
from typing import Tuple

from yahtzee_rl.config import Category, UPPER_SECTION_MAP
from yahtzee_rl.scoring.ops import JOKER_LOWER_FIXED, dice_count
from yahtzee_rl.scoring.scorecard import Scorecard

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


def _joker_eligible(score_card: Scorecard) -> bool:
    """True iff a future Yahtzee would trigger the Hasbro Joker rule.

    Delegates to :meth:`Scorecard.joker_eligible` — requires the YAHTZEE
    category to be filled with 50; a zeroed-out Yahtzee permanently disables
    joker per Hasbro rules.
    """
    return score_card.joker_eligible()


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


def lower_section_probabilities(dice: np.ndarray,
                                move: Category, remaining_rolls: int) -> float:
    """
    Determine final probabilities for reaching lower section moves. 
    We also consider whether a move has already been marked. Unless it is a yahtzee.
    Args:
        dice: array of dice values (values 1-6)
        move: the move to consider
        remaining_rolls: the number of remaining rolls

    Returns:
        the probability of reaching the move
    """
    if move == Category.THREE_OF_A_KIND:
        counts = dice_count(dice)
        final_probs = [simple_three_of_a_kind(dice, c, remaining_rolls) for c in counts.keys()]
        max_prob = max(final_probs)
        return max_prob
    elif move == Category.FOUR_OF_A_KIND:
        counts = dice_count(dice)
        final_probs = [simple_four_of_a_kind(dice, c, remaining_rolls) for c in counts.keys()]
        max_prob = max(final_probs)
        return max_prob
    elif move == Category.FULL_HOUSE:
        return full_house(dice, remaining_rolls)
    elif move == Category.SMALL_STRAIGHT:
        return determine_small_straight_probability(dice, remaining_rolls)
    elif move == Category.LARGE_STRAIGHT:
        return determine_large_straight_probability(dice, remaining_rolls)
    elif move == Category.YAHTZEE:
        return yahtzee(dice, remaining_rolls)
    elif move == Category.CHANCE:
        return 0.1

    return 0.0



def upper_section_probability(dice: np.ndarray,
                               move: Category, remaining_rolls: int) -> float:
    """
    P(getting >= 3 matching dice for this upper section category).

    Threshold of 3 is strategic: 3 x face_value is the per-category average
    needed to hit the 63-point upper bonus. Same semantics as lower section
    probabilities (a true probability in [0, 1]).

    Args:
        dice: array of dice values (values 1-6)
        move: the upper section category
        remaining_rolls: the number of remaining rolls

    Returns:
        P(count >= 3) for the category, or 0.0 if marked/invalid
    """
    _, p_three_plus, _ = upper_section_markov(dice, move, remaining_rolls)
    return p_three_plus


def upper_section_expected_score(dice: np.ndarray,
                                 score_card: Scorecard,
                                 move: Category,
                                 remaining_rolls: int,
                                 lambda_v: float = 0.075) -> float:
    """
    Expected score for a single upper section category.
    Returns face_value * expected_count (raw expected points).

    Args:
        dice: array of dice values (values 1-6)
        score_card: the scorecard
        move: the upper section category
        remaining_rolls: the number of remaining rolls
        lambda_v: Lambda value multiplied against final expected point probability;
        this value reduces the overall effect of the top scores weight

    Returns:
        Expected score (0.0 to face_value * 5), or 0.0 if marked/invalid
         lambda_v:
    """
    remaining = 500.0 - score_card.compute_final_score()
    denominator = remaining if remaining > 0 else 1.0
    upper_score_max = 63.0
    upper_score_current = score_card.compute_upper_score()
    top_remaining = upper_score_max - upper_score_current
    if move not in UPPER_SECTION_MAP or score_card.is_category_marked(move):
        return 0.0
    exp_score, _, dist = upper_section_markov(dice, move, remaining_rolls)
    ev = (exp_score / denominator) + (lambda_v * (top_remaining / upper_score_max))

    # Joker forced-upper path: when YAHTZEE is filled with 50 and the player
    # rolls a yahtzee of this face, joker priority forces routing to this
    # upper category for face*5 + 100. The face*5 portion is already inside
    # `exp_score` via dist[5]; add only the +100 bonus contribution here.
    # See ../strategies/markov.py and ../envs/yahtzee_env.py for the runtime
    # joker handling. Caveat: each open upper's row gets this contribution
    # as if the yahtzee would route to it, matching the per-category-
    # independence approximation used by the strict EV.
    if _joker_eligible(score_card):
        p_yahtzee_face = float(dist[5])
        ev += p_yahtzee_face * YAHTZEE_BONUS / denominator
    return ev



def lower_section_expected_score(dice: np.ndarray,
                                 score_card: Scorecard,
                                 move: Category,
                                 remaining_rolls: int,
                                 lambda_v: float = 0.05) -> float:
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
        lambda_v: Lambda value multiplied against yahtzee expectation, to bring up its
        lower probability

    Returns:
        Expected score (float), or 0.0 if marked/invalid
    """
    remaining = 375.0 - score_card.compute_final_score()
    denominator = remaining if remaining > 0 else 1.0
    if move not in Category.lower_categories():
        return 0.0
    # Yahtzee requires special handling for the bonus.
    # Note: under the action mask, the agent never picks YAHTZEE in the
    # scoring phase when joker is alive, so this row primarily informs the
    # rolling-phase strategy. The +100 contribution here is a strict
    # undercount of the true value of another yahtzee (which would also pay
    # the routed lower/upper joker payoff) — but the joker contributions on
    # the open upper/lower rows below already surface that signal. Adding
    # the routed payoff here too would double-count.
    if move == Category.YAHTZEE:
        yahtzee_achieved = score_card.score_board[Category.YAHTZEE]["num_times_achieved"]
        raw_prob = yahtzee(dice, remaining_rolls)
        if score_card.is_category_marked(Category.YAHTZEE):
            # Category filled, but each additional yahtzee earns +100
            if yahtzee_achieved >= 1:
                payoff = ((YAHTZEE_BONUS) / denominator) + lambda_v
                return (raw_prob * payoff)
            return 0.0
        else:
            # First yahtzee: base 50 pts
            payoff = (LOWER_FIXED_SCORES[Category.YAHTZEE] / denominator) + lambda_v
            return (raw_prob * payoff)

    prob = lower_section_probabilities(dice, move, remaining_rolls)

    # Strict EV (natural combo route).
    if move in LOWER_FIXED_SCORES:
        ev = prob * float(LOWER_FIXED_SCORES[move]) / denominator
    elif move == Category.THREE_OF_A_KIND:
        d_counts = dice_count(dice)
        max_face = max(d_counts, key=d_counts.get)
        l_dice = [max_face] * 3
        upper_dice = l_dice + [6, 6]
        lower_dice = l_dice + [1, 1]
        mean_dice = (np.sum(upper_dice) + np.sum(lower_dice)) / 2
        ev = prob * mean_dice / denominator
    elif move == Category.FOUR_OF_A_KIND:
        d_counts = dice_count(dice)
        max_face = max(d_counts, key=d_counts.get)
        l_dice = [max_face] * 4
        upper_dice = l_dice + [6]
        lower_dice = l_dice + [1]
        mean_dice = (np.sum(upper_dice) + np.sum(lower_dice)) / 2
        ev = prob * mean_dice / denominator
    elif move == Category.CHANCE:
        # Always achievable (prob=1.0); current sum is a lower bound on the
        # final sum we'd keep across remaining rolls.
        ev = prob * float(np.sum(dice)) / denominator
    else:
        return 0.0

    # Joker free-lower path: when YAHTZEE is filled with 50, this category is
    # still open, and the player rolls a yahtzee whose matching upper is also
    # already filled, joker priority routes the dice here for
    # joker_payoff(move) + 100. Caveat: same per-category-independence
    # approximation as above — each open lower's row gets this contribution
    # as if it'd be the chosen joker route. The agent's argmax over the EV
    # vector then picks the best joker route, which is the desired behavior.
    if _joker_eligible(score_card) and not score_card.is_category_marked(move):
        face_dist = yahtzee_face_distribution(dice, remaining_rolls)
        p_y = float(face_dist.sum())
        if move in JOKER_LOWER_FIXED:
            joker_payoff = JOKER_LOWER_FIXED[move]
            joker_contrib = p_y * (joker_payoff + YAHTZEE_BONUS) / denominator
        elif move in (Category.THREE_OF_A_KIND, Category.FOUR_OF_A_KIND, Category.CHANCE):
            # joker payoff = 5 * face; weight by per-face yahtzee probability
            face_values = np.arange(1, 7)
            expected_payoff_sum = float(np.dot(face_dist, 5 * face_values))
            joker_contrib = (expected_payoff_sum + p_y * YAHTZEE_BONUS) / denominator
        else:
            joker_contrib = 0.0
        ev += joker_contrib

    return ev



def upper_section_prob_vector(dice: np.ndarray,
                               remaining_rolls: int) -> np.ndarray:
    """
    Observation vector of P(>=3) probabilities for all 6 upper section categories.
    Marked categories get 0.0. All values in [0, 1].

    Args:
        dice: array of dice values (values 1-6)
        remaining_rolls: the number of remaining rolls

    Returns:
        np.ndarray of shape (6,) with P(>=3) per upper category
    """
    obs = np.zeros(6, dtype=np.float32)
    for i, category in enumerate(Category.upper_categories()):
        obs[i] = upper_section_probability(dice, category, remaining_rolls)
    return obs


def upper_section_expected_score_vector(dice: np.ndarray,
                                        score_card: Scorecard,
                                        remaining_rolls: int,
                                        lambda_v: float = 0.05) -> np.ndarray:
    """
    Observation vector of normalized expected scores for all 6 upper section categories.
    Each value is expected_score / max_possible_score = expected_count / 5, in [0, 1].
    Marked categories get 0.0.

    Args:
        dice: array of dice values (values 1-6)
        score_card: the scorecard
        remaining_rolls: the number of remaining rolls
        lambda_v: Lambda value passed into upper section probability

    Returns:
        np.ndarray of shape (6,) with normalized expected scores per upper category
    """
    obs = np.zeros(6, dtype=np.float32)
    for i, category in enumerate(Category.upper_categories()):
        exp_count = upper_section_expected_score(dice, score_card, category, remaining_rolls, lambda_v=lambda_v,)
        obs[i] = exp_count  
    return obs


def lower_section_prob_vector(dice: np.ndarray,
                               remaining_rolls: int) -> np.ndarray:
    """
    Observation vector of probabilities for all 7 lower section categories.
    Marked categories get 0.0 (except yahtzee bonus). All values in [0, 1].

    Args:
        dice: array of dice values (values 1-6)
        remaining_rolls: the number of remaining rolls

    Returns:
        np.ndarray of shape (7,) with P(combo) per lower category
    """
    obs = np.zeros(7, dtype=np.float32)
    for i, category in enumerate(Category.lower_categories()):
        obs[i] = lower_section_probabilities(dice, category, remaining_rolls)
    return obs


def lower_section_expected_score_vector(dice: np.ndarray,
                                        score_card: Scorecard,
                                        remaining_rolls: int,
                                        lambda_yahtzee: float = 0.4) -> np.ndarray:
    """
    Observation vector of normalized expected scores for all 7 lower section categories.
    Each value is expected_score / max_possible_score, in [0, 1].
    Marked categories get 0.0 (except yahtzee bonus).

    Args:
        dice: array of dice values (values 1-6)
        score_card: the scorecard
        remaining_rolls: the number of remaining rolls
        lambda_yahtzee: Lambda value passed into lower section probability

    Returns:
        np.ndarray of shape (7,) with normalized expected scores per lower category
    """
    obs = np.zeros(7, dtype=np.float32)
    for i, category in enumerate(Category.lower_categories()):
        expected = lower_section_expected_score(dice, score_card,
                                                category, remaining_rolls, lambda_v=lambda_yahtzee)
        obs[i] = expected 
    return obs


def upper_section_markov(dice: np.ndarray, category: Category,
                         remaining_rolls: int) -> tuple[float, float, np.ndarray]:
    """
    Core Markov chain computation for upper section categories.

    Args:
        dice: array of dice values (values 1-6)
        category: the upper section category to consider
        remaining_rolls: the number of remaining rolls

    Returns:
        Tuple of (expected_score, p_three_plus, dist) where:
            - expected_score: face_value * E[matching dice] after remaining rolls
            - p_three_plus: P(matching count >= 3) — the upper-bonus threshold
            - dist: np.ndarray of shape (6,) with P(count=i) for i in 0..5.
              ``dist[5]`` is P(yahtzee of this face).
    """
    face_value = UPPER_SECTION_MAP[category]

    count = dice_count(dice)[UPPER_SECTION_MAP[category]]
    state = np.zeros(6, dtype=float)
    state[count] = 1
    dist = _upper_count_t_powered(remaining_rolls) @ state
    expected_count = float(np.dot(np.arange(6), dist))
    expected_score = face_value * expected_count
    p_three_plus = float(dist[3:].sum())
    return expected_score, p_three_plus, dist


def yahtzee_face_distribution(dice: np.ndarray, remaining_rolls: int) -> np.ndarray:
    """Per-face yahtzee probability vector.

    Returns a shape-(6,) array where ``out[k-1]`` is the probability of
    finishing with a yahtzee of face k after ``remaining_rolls`` more rolls,
    *under the strategy that targets face k* (keep matching, re-roll the rest).

    Each entry is the per-face Markov upper-count distribution evaluated at
    ``dist[5]``. Because the per-face values condition on different keep
    strategies, ``out.sum()`` is an upper bound on the true P(any yahtzee
    under a single fixed strategy), not an equality. In practice the
    overestimate is dominated by the best-case face and is small.
    """
    out = np.zeros(6, dtype=float)
    for i, cat in enumerate(Category.upper_categories()):
        _, _, dist = upper_section_markov(dice, cat, remaining_rolls)
        out[i] = float(dist[5])
    return out

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
    if count == 0: 
        f_count = 0
    else:
        f_count = count - 1
    initial_state[f_count] = 1
    state_vec = _runs_t_powered(remaining_rolls) @ initial_state
    return np.sum(state_vec[target_state:])
    #return state_vec[target_state]

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
    prob_matches = []
    for possible in possibles:
        l_state = determine_straight_state(dice, possible)
        if l_state >= 1:
            if l_state == 4:
                return 1.0
            else:
                initial_state = np.zeros(4)
                if l_state == 0:
                    initial_state[0] = 1
                else:
                    initial_state[l_state-1] = 1
                probability_v = _straight_t_powered(remaining_rolls) @ initial_state
                chosen_prob = probability_v[3]
                prob_matches.append(chosen_prob)
    if len(prob_matches) > 0:
        return max(prob_matches)
    else:
        return 0.0
 
 
def determine_large_straight_probability(dice: np.ndarray, remaining_rolls: int) -> float:
    """
    Probability of achieving a large straight (5 consecutive numbers in a row).
    Args:
        dice: array of dice values (values 1-6)
        remaining_rolls: the number of remaining rolls
    """
    possibles = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
    prob_matches = []
    for possible in possibles:
        l_state = determine_straight_state(dice, possible)
        if l_state >= 1:
            if l_state == 5:
                return 1.0
            else:
                initial_state = np.zeros(5)
                if l_state == 0:
                    initial_state[0] = 1
                else:
                    initial_state[l_state-1] = 1
                probability_v = _large_straight_t_powered(remaining_rolls) @ initial_state
                chosen_prob = probability_v[4]
                prob_matches.append(chosen_prob)
    if len(prob_matches) > 0:
        return max(prob_matches)
    else:
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
    # if combo_satisfied(dice, Category.FULL_HOUSE):
    #     return 1.0
    # else:
    #     return 0.1

def yahtzee(dice: np.ndarray, remaining_rolls: int) -> float:
    """
    Probability of achieving a yahtzee (all five dice the same).
    Args:
        dice: array of dice values (values 1-6)
        remaining_rolls: the number of remaining rolls
    """
    counts = dice_count(dice)
    max_face = max(counts, key=counts.get)
    if counts[max_face] == 5:
        return 1.0
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
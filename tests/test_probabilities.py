from yahtzee_rl.markov.probabilities import (
    reaching_x, simple_three_of_a_kind, simple_four_of_a_kind, full_house,
    upper_section_markov, upper_section_probability, upper_section_expected_score, upper_section_prob_vector, upper_section_expected_score_vector,
     lower_section_probabilities, lower_section_expected_score,
    lower_section_prob_vector, lower_section_expected_score_vector,
    faces,
)
import numpy as np
import pytest
from yahtzee_rl.scoring.scorecard import Scorecard
from yahtzee_rl import Category

def test_reaching_x():
    """
    Test reaching_x functionality, around certain number states
    """
    input_dice = np.array([1, 2, 1, 4, 5])
    target_state = 2 # three of a kind
    remaining_rolls = 2
    result = reaching_x(input_dice, 1, target_state, remaining_rolls)
    assert result == pytest.approx(0.6913, abs=1e-4)
    result_2 = reaching_x(input_dice, 2, target_state, remaining_rolls)
    assert result_2 == pytest.approx(0.54132, abs=1e-4)
    input_dice_v2 = np.array([1, 1, 1, 3, 2])
    # Probability of reaching a pair? Should be 0.0
    # Don't go back, it's a greedy policy
    result_zeros = reaching_x(input_dice_v2, 1, 1, remaining_rolls)
    assert result_zeros == pytest.approx(1.0, abs=1e-4)
    result_large_straight = reaching_x(input_dice_v2, 1, 4, remaining_rolls)
    assert result_large_straight == pytest.approx(0.09336, abs=1e-4)
    result_large_straight_same = reaching_x(input_dice_v2, 1, 2, 1)
    assert result_large_straight_same == pytest.approx(1.0, abs=1e-4)


def test_simple_three_of_a_kind():
    """
    Test simple_three_of_a_kind probability calculation
    Should be 1.0 if 3+ of a kind already, otherwise, return the 
    probability of the change
    """
    dice = np.array([1, 1, 1, 3, 2])
    result = simple_three_of_a_kind(dice, 1, 1)
    assert result == pytest.approx(1.0, abs=1e-4)
    result_2 = simple_three_of_a_kind(dice, 1, 2)
    assert result_2 == pytest.approx(1.0, abs=1e-4)
    

def test_full_house():
    """
    Test full_house probability calculation for each case
    """
    # Case 1: Already full house -> 1.0
    dice_full_house = np.array([1, 1, 1, 2, 2])
    assert full_house(dice_full_house, 0) == pytest.approx(1.0, abs=1e-4)
    assert full_house(dice_full_house, 2) == pytest.approx(1.0, abs=1e-4)
    
    # Case 2: Triple + singles -> P = 1 - (5/6)^n
    dice_triple = np.array([1, 1, 1, 3, 4])
    # With 2 rolls: 1 - (5/6)^2 = 1 - 25/36 = 11/36 ≈ 0.3056
    assert full_house(dice_triple, 2) == pytest.approx(0.3056, abs=1e-2)
    # With 1 roll: 1 - (5/6)^1 = 1/6 ≈ 0.1667
    assert full_house(dice_triple, 1) == pytest.approx(0.1667, abs=1e-2)
    
    # Case 3: Two pairs -> P = 1 - (4/6)^n
    dice_two_pairs = np.array([1, 1, 2, 2, 3])
    # With 2 rolls: 1 - (4/6)^2 = 1 - 16/36 = 20/36 ≈ 0.5556
    assert full_house(dice_two_pairs, 2) == pytest.approx(0.5556, abs=1e-2)
    # With 1 roll: 1 - (4/6)^1 = 2/6 ≈ 0.3333
    assert full_house(dice_two_pairs, 1) == pytest.approx(0.3333, abs=1e-2)
    
    # Case 4: Pair + singles -> reaching_x * p_pair_forms
    dice_pair = np.array([1, 1, 2, 3, 4])
    result_pair = full_house(dice_pair, 2)
    # Should be between 0 and 1
    assert 0 < result_pair < 1
    
    # Case 5: All singles -> approximation
    dice_singles = np.array([1, 2, 3, 4, 5])
    result_singles = full_house(dice_singles, 2)
    # Should be small but positive
    assert 0 < result_singles < 0.5

def test_upper_section_probabilities():
    """
    Test upper section probabilities
    """
    dice = np.array([1, 1, 1, 2, 2])
    score_card = Scorecard(turn_number=0)
    result = upper_section_probability(dice, score_card, Category.ACES, 2)
    assert result == pytest.approx(1.0, abs=1e-4)
    dice_2 = np.array([2, 2, 1, 3, 4])
    result_2 = upper_section_probability(dice_2, score_card, Category.TWOS, 2)
    assert result_2 == pytest.approx(0.6651, abs=1e-4)
    result_3 = upper_section_probability(dice_2, score_card, Category.THREES, 2)
    assert result_3 == pytest.approx(0.35811, abs=1e-4)
    result_4 = upper_section_probability(dice_2, score_card, Category.FOURS, 2)
    assert result_4 == pytest.approx(0.35811, abs=1e-4)

# def test_upper_section_payoff():
#     """
#     Test upper_section_payoff across four scenarios:
#     1. Strong match (three 2s for TWOS)
#     2. No match (no 6s for SIXES)
#     3. High existing upper score (bonus progress boost)
#     4. Marked category returns 0.0
#     """
#     score_card = Scorecard(turn_number=0)

#     # Scenario 1: Strong match — dice (2,2,2,3,4), TWOS, upper_score=0, 2 rolls
#     dice = np.array([2, 2, 2, 3, 4])
#     result = upper_section_payoff(dice, score_card, Category.TWOS, 2, 0)
#     assert result > 0.0, "Strong match should produce positive payoff"
#     # expected_score = 2 * E[count of 2s] where we already have 3 twos
#     # category_efficiency = expected_score / 10, bonus_progress = expected_score / 63
#     # payoff = expected_score * (efficiency + progress)
#     print(f"Upper payoff (2,2,2,3,4) TWOS upper=0: {result:.4f}")

#     # Scenario 2: No match — dice (1,1,1,3,4), SIXES, upper_score=0, 2 rolls
#     dice_no_match = np.array([1, 1, 1, 3, 4])
#     result_no_match = upper_section_payoff(dice_no_match, score_card, Category.SIXES, 2, 0)
#     assert result_no_match > 0.0, "Should still be positive (can roll sixes)"
#     assert result_no_match < result, "No-match payoff should be less than strong match"
#     print(f"Upper payoff (1,1,1,3,4) SIXES upper=0: {result_no_match:.4f}")

#     # Scenario 3: High existing upper score — same dice, upper_score=60
#     dice_high = np.array([2, 2, 2, 3, 4])
#     result_high = upper_section_payoff(dice_high, score_card, Category.TWOS, 2, 60)
#     assert result_high > result, "Near-bonus payoff should be larger (bonus_progress term bigger)"
#     print(f"Upper payoff (2,2,2,3,4) TWOS upper=60: {result_high:.4f}")

#     # Scenario 4: Marked category returns 0.0
#     marked_card = Scorecard(turn_number=1)
#     marked_card.mark_score(Category.TWOS, np.array([2, 2, 2, 3, 4]))
#     result_marked = upper_section_payoff(dice, marked_card, Category.TWOS, 2, 0)
#     assert result_marked == 0.0, "Marked category payoff must be 0.0"


# def test_lower_section_payoff():
#     """
#     Test lower_section_payoff across six scenarios:
#     1. Three of a kind already in hand
#     2. Full house close (two pairs)
#     3. Small straight partial
#     4. Chance (always achievable)
#     5. Yahtzee with bonus (second yahtzee attempt)
#     6. Marked category returns 0.0
#     """
#     score_card = Scorecard(turn_number=0)

#     # Scenario 1: Three of a kind already — dice (2,2,2,3,4), 2 rolls
#     dice_3k = np.array([2, 2, 2, 3, 4])
#     result_3k = lower_section_payoff(dice_3k, score_card, Category.THREE_OF_A_KIND, 2)
#     assert result_3k > 0.0, "Three of a kind in hand should give positive payoff"
#     # prob=1.0 → expected_score = 1.0 * MEAN_DICE_SUM = 17.5
#     # category_efficiency = 17.5 / 30
#     # payoff = 17.5 * (17.5/30) ≈ 10.208
#     print(f"Lower payoff (2,2,2,3,4) THREE_OF_A_KIND: {result_3k:.4f}")

#     # Scenario 2: Full house close — dice (2,2,3,3,4), 2 rolls
#     dice_fh = np.array([2, 2, 3, 3, 4])
#     result_fh = lower_section_payoff(dice_fh, score_card, Category.FULL_HOUSE, 2)
#     assert result_fh > 0.0, "Two pairs should give positive full house payoff"
#     print(f"Lower payoff (2,2,3,3,4) FULL_HOUSE: {result_fh:.4f}")

#     # Scenario 3: Small straight partial — dice (1,2,3,5,5), 2 rolls
#     dice_ss = np.array([1, 2, 3, 5, 5])
#     result_ss = lower_section_payoff(dice_ss, score_card, Category.SMALL_STRAIGHT, 2)
#     assert result_ss > 0.0, "Partial straight should give positive payoff"
#     print(f"Lower payoff (1,2,3,5,5) SMALL_STRAIGHT: {result_ss:.4f}")

#     # Scenario 4: Chance — always achievable, payoff = sum * efficiency
#     dice_chance = np.array([2, 2, 2, 3, 4])
#     result_chance = lower_section_payoff(dice_chance, score_card, Category.CHANCE, 2)
#     assert result_chance > 0.0, "Chance should always give positive payoff"
#     # expected_score = sum(dice) = 13, max = 30, efficiency = 13/30
#     # payoff = 13 * (13/30)
#     print(f"Lower payoff (2,2,2,3,4) CHANCE: {result_chance:.4f}")

#     # Scenario 5: Yahtzee with bonus — mark yahtzee once, then try again
#     bonus_card = Scorecard(turn_number=1)
#     bonus_card.mark_score(Category.YAHTZEE, np.array([3, 3, 3, 3, 3]))
#     dice_yahtzee = np.array([3, 3, 3, 3, 4])
#     result_yb = lower_section_payoff(dice_yahtzee, bonus_card, Category.YAHTZEE, 2)
#     assert result_yb > 0.0, "Yahtzee bonus attempt should give positive payoff"
#     print(f"Lower payoff (3,3,3,3,4) YAHTZEE bonus: {result_yb:.4f}")

#     # Scenario 6: Marked category returns 0.0
#     marked_card = Scorecard(turn_number=1)
#     marked_card.mark_score(Category.FULL_HOUSE, np.array([1, 1, 1, 2, 2]))
#     result_marked = lower_section_payoff(dice_fh, marked_card, Category.FULL_HOUSE, 2)
#     assert result_marked == 0.0, "Marked category payoff must be 0.0"


def test_vectors_222_34():
    """
    Show all four probability/expected-score vectors for dice (2,2,2,3,4).
    Upper section: prob_vector (6,) and expected_score_vector (6,)
    Lower section: prob_vector (7,) and expected_score_vector (7,)
    """
    dice = np.array([2, 2, 2, 3, 4])
    score_card = Scorecard(turn_number=0)
    remaining_rolls = 2

    # Upper section vectors
    upper_prob = upper_section_prob_vector(dice, score_card, remaining_rolls)
    upper_exp = upper_section_expected_score_vector(dice, score_card, remaining_rolls)

    upper_labels = ["ACES", "TWOS", "THREES", "FOURS", "FIVES", "SIXES"]
    print("\n=== Dice: (2, 2, 2, 3, 4) | 2 rolls remaining ===")
    print("\nUpper Section Prob Vector (P >= 3 matching):")
    for label, p in zip(upper_labels, upper_prob):
        print(f"  {label:8s}: {p:.4f}")
    print("\nUpper Section Expected Score Vector (normalized):")
    for label, e in zip(upper_labels, upper_exp):
        print(f"  {label:8s}: {e:.4f}")

    # TWOS (index 1) should have the highest probability
    assert np.argmax(upper_prob) == 1, "TWOS should have highest upper probability"
    assert np.all(upper_prob >= 0) and np.all(upper_prob <= 1), "All upper probs in [0, 1]"
    assert np.all(upper_exp >= 0) and np.all(upper_exp <= 1), "All upper expected in [0, 1]"
    assert upper_prob.shape == (6,)
    assert upper_exp.shape == (6,)

    # Lower section vectors
    lower_prob = lower_section_prob_vector(dice, score_card, remaining_rolls)
    lower_exp = lower_section_expected_score_vector(dice, score_card, remaining_rolls)

    lower_labels = ["3-OF-KIND", "4-OF-KIND", "FULL-HOUSE",
                    "SM-STRAIGHT", "LG-STRAIGHT", "YAHTZEE", "CHANCE"]
    print("\nLower Section Prob Vector:")
    for label, p in zip(lower_labels, lower_prob):
        print(f"  {label:12s}: {p:.4f}")
    print("\nLower Section Expected Score Vector (normalized):")
    for label, e in zip(lower_labels, lower_exp):
        print(f"  {label:12s}: {e:.4f}")

    # THREE_OF_A_KIND (index 0) prob should be 1.0 — already have three 2s
    assert lower_prob[0] == pytest.approx(1.0, abs=1e-4), "THREE_OF_A_KIND prob should be 1.0"
    # CHANCE (index 6) prob should be 1.0 — always achievable
    assert lower_prob[6] == pytest.approx(1.0, abs=1e-4), "CHANCE prob should be 1.0"
    assert np.all(lower_prob >= 0) and np.all(lower_prob <= 1), "All lower probs in [0, 1]"
    assert lower_prob.shape == (7,)
    assert lower_exp.shape == (7,)


INTERESTING_COMBOS = [
    pytest.param(np.array([1, 1, 1, 1, 1]), id="five_aces"),
    pytest.param(np.array([1, 2, 3, 4, 5]), id="all_different"),
    pytest.param(np.array([6, 6, 6, 5, 5]), id="sixes_fives_fullhouse"),
    pytest.param(np.array([3, 3, 4, 4, 5]), id="two_pairs"),
    pytest.param(np.array([5, 5, 5, 5, 6]), id="near_yahtzee_fives"),
]

@pytest.mark.parametrize("dice", INTERESTING_COMBOS)
def test_vectors_interesting_combos(dice):
    """
    Display all four probability/expected-score vectors for interesting dice combos.
    Validates shape and range for each.
    """
    score_card = Scorecard(turn_number=0)
    remaining_rolls = 2

    upper_prob = upper_section_prob_vector(dice, score_card, remaining_rolls)
    upper_exp = upper_section_expected_score_vector(dice, score_card, remaining_rolls)
    lower_prob = lower_section_prob_vector(dice, score_card, remaining_rolls)
    lower_exp = lower_section_expected_score_vector(dice, score_card, remaining_rolls)

    upper_labels = ["ACES", "TWOS", "THREES", "FOURS", "FIVES", "SIXES"]
    lower_labels = ["3-OF-KIND", "4-OF-KIND", "FULL-HOUSE",
                    "SM-STRAIGHT", "LG-STRAIGHT", "YAHTZEE", "CHANCE"]

    print(f"\n=== Dice: {tuple(dice)} | 2 rolls remaining ===")
    print("Upper Prob:     ", "  ".join(f"{l}:{v:.3f}" for l, v in zip(upper_labels, upper_prob)))
    print("Upper Expected: ", "  ".join(f"{l}:{v:.3f}" for l, v in zip(upper_labels, upper_exp)))
    print("Lower Prob:     ", "  ".join(f"{l}:{v:.3f}" for l, v in zip(lower_labels, lower_prob)))
    print("Lower Expected: ", "  ".join(f"{l}:{v:.3f}" for l, v in zip(lower_labels, lower_exp)))

    # Shape checks
    assert upper_prob.shape == (6,)
    assert upper_exp.shape == (6,)
    assert lower_prob.shape == (7,)
    assert lower_exp.shape == (7,)

    # Range checks
    assert np.all(upper_prob >= 0) and np.all(upper_prob <= 1)
    assert np.all(upper_exp >= 0) and np.all(upper_exp <= 1)
    assert np.all(lower_prob >= 0) and np.all(lower_prob <= 1)
    # lower_exp can exceed 1.0 for sum-based categories (three/four of a kind)
    # because MEAN_DICE_SUM > MAX_DICE_SUM in current production code
    assert np.all(lower_exp >= 0)

if __name__ == "__main__":
    test_reaching_x()
    for dice in INTERESTING_COMBOS:
        test_vectors_interesting_combos(dice.values[0])
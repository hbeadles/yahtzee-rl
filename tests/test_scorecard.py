import numpy as np
import pytest
from yahtzee_rl.scoring.scorecard import Scorecard


# ============================================================
# Upper Section Tests
# ============================================================

def test_upper_score_calculation():
    """Test that compute_upper_score correctly sums upper section categories."""
    scorecard = Scorecard(turn_number=0)
    
    # Mark some upper section categories
    scorecard.mark_score("aces", np.array([1, 1, 1, 2, 3]))  # 3 aces = 3 points
    scorecard.mark_score("twos", np.array([2, 2, 3, 4, 5]))  # 2 twos = 4 points
    scorecard.mark_score("threes", np.array([3, 3, 3, 1, 2]))  # 3 threes = 9 points
    
    # Upper score should be 3 + 4 + 9 = 16 (no bonus since < 63)
    assert scorecard.compute_upper_score() == 16


def test_upper_bonus():
    """Test that the 35-point upper bonus is applied when upper section >= 63."""
    scorecard = Scorecard(turn_number=0)
    
    # Score to reach exactly 63 in upper section:
    # aces: 3x1 = 3, twos: 3x2 = 6, threes: 3x3 = 9
    # fours: 3x4 = 12, fives: 3x5 = 15, sixes: 3x6 = 18
    # Total: 3 + 6 + 9 + 12 + 15 + 18 = 63
    scorecard.mark_score("aces", np.array([1, 1, 1, 2, 3]))     # 3 points
    scorecard.mark_score("twos", np.array([2, 2, 2, 4, 5]))     # 6 points (3 twos * 2)
    scorecard.mark_score("threes", np.array([3, 3, 3, 1, 2]))   # 9 points (3 threes * 3)
    scorecard.mark_score("fours", np.array([4, 4, 4, 1, 2]))    # 12 points (3 fours * 4)
    scorecard.mark_score("fives", np.array([5, 5, 5, 1, 2]))    # 15 points (3 fives * 5)
    scorecard.mark_score("sixes", np.array([6, 6, 6, 1, 2]))    # 18 points (3 sixes * 6)
    
    # Upper score should be 63 + 35 (bonus) = 98
    assert scorecard.compute_upper_score() == 98
    
    # Verify final score includes the bonus
    assert scorecard.compute_final_score() == 98


def test_upper_bonus_not_applied_below_threshold():
    """Test that upper bonus is NOT applied when upper section < 63."""
    scorecard = Scorecard(turn_number=0)
    
    # Score less than 63 in upper section
    scorecard.mark_score("aces", np.array([1, 1, 2, 3, 4]))     # 2 points
    scorecard.mark_score("twos", np.array([2, 2, 3, 4, 5]))     # 4 points
    scorecard.mark_score("threes", np.array([3, 3, 4, 5, 6]))   # 6 points
    
    # Total = 12, no bonus
    assert scorecard.compute_upper_score() == 12


# ============================================================
# Lower Section Tests
# ============================================================

def test_lower_score_calculation():
    """Test that compute_lower_score correctly sums lower section categories."""
    scorecard = Scorecard(turn_number=0)
    
    # Mark some lower section categories
    scorecard.mark_score("three_of_a_kind", np.array([3, 3, 3, 4, 5]))  # Sum = 18
    scorecard.mark_score("full_house", np.array([2, 2, 2, 5, 5]))       # 25 points
    
    # Lower score should be 18 + 25 = 43
    assert scorecard.compute_lower_score() == 43


def test_chance_scoring():
    """Test chance category correctly sums all dice."""
    scorecard = Scorecard(turn_number=0)
    
    # Chance always sums all dice
    scorecard.mark_score("chance", np.array([1, 2, 3, 4, 5]))  # Sum = 15
    
    assert scorecard.get_category_score("chance") == 15
    assert scorecard.compute_lower_score() == 15
    
    # Test with different roll
    scorecard2 = Scorecard(turn_number=0)
    scorecard2.mark_score("chance", np.array([6, 6, 6, 6, 6]))  # Sum = 30
    
    assert scorecard2.get_category_score("chance") == 30
    assert scorecard2.compute_lower_score() == 30


def test_yahtzee_scoring():
    """Test basic yahtzee scoring (50 points)."""
    scorecard = Scorecard(turn_number=0)
    
    # Score a yahtzee (all same dice)
    scorecard.mark_score("yahtzee", np.array([5, 5, 5, 5, 5]))
    
    assert scorecard.get_category_score("yahtzee") == 50
    assert scorecard.compute_lower_score() == 50


def test_yahtzee_bonus():
    """Test that yahtzee bonus (100 points per extra) is applied correctly."""
    scorecard = Scorecard(turn_number=0)
    
    # Score first yahtzee
    scorecard.mark_score("yahtzee", np.array([5, 5, 5, 5, 5]))
    
    # Simulate achieving additional yahtzees by incrementing num_times_achieved
    # In a real game, this would happen through joker rules
    scorecard.score_board["yahtzee"]["num_times_achieved"] = 2  # Second yahtzee
    
    # Lower score should be 50 (yahtzee) + 100 (one bonus) = 150
    assert scorecard.compute_lower_score() == 150
    
    # Test with 3 yahtzees
    scorecard.score_board["yahtzee"]["num_times_achieved"] = 3
    
    # Lower score should be 50 + 200 (two bonuses) = 250
    assert scorecard.compute_lower_score() == 250


# ============================================================
# Full Game Tests
# ============================================================

def test_basic_game():
    """Test a basic game flow with multiple categories marked."""
    scorecard = Scorecard(turn_number=0)
    
    # Upper section
    scorecard.mark_score("aces", np.array([1, 1, 1, 1, 2]))     # 4 points
    scorecard.mark_score("twos", np.array([2, 2, 2, 3, 4]))     # 6 points
    scorecard.mark_score("threes", np.array([3, 3, 3, 1, 1]))   # 9 points
    scorecard.mark_score("fours", np.array([4, 4, 4, 4, 1]))    # 16 points
    scorecard.mark_score("fives", np.array([5, 5, 5, 1, 2]))    # 15 points
    scorecard.mark_score("sixes", np.array([6, 6, 6, 1, 2]))    # 18 points
    
    # Upper total = 4 + 6 + 9 + 16 + 15 + 18 = 68 (>= 63, gets bonus)
    expected_upper = 68 + 35  # 103 with bonus
    
    # Lower section
    scorecard.mark_score("three_of_a_kind", np.array([4, 4, 4, 2, 3]))  # Sum = 17
    scorecard.mark_score("four_of_a_kind", np.array([5, 5, 5, 5, 2]))   # Sum = 22
    scorecard.mark_score("full_house", np.array([3, 3, 3, 6, 6]))       # 25 points
    scorecard.mark_score("small_straight", np.array([1, 2, 3, 4, 6]))   # 30 points
    scorecard.mark_score("large_straight", np.array([2, 3, 4, 5, 6]))   # 40 points
    scorecard.mark_score("yahtzee", np.array([6, 6, 6, 6, 6]))          # 50 points
    scorecard.mark_score("chance", np.array([5, 5, 4, 3, 2]))           # Sum = 19
    
    expected_lower = 17 + 22 + 25 + 30 + 40 + 50 + 19  # 203
    
    assert scorecard.compute_upper_score() == expected_upper
    assert scorecard.compute_lower_score() == expected_lower
    assert scorecard.compute_final_score() == expected_upper + expected_lower  # 306


def test_basic_game_without_bonus():
    """Test a game where upper bonus is not achieved."""
    scorecard = Scorecard(turn_number=0)
    
    # Upper section - scoring low to miss the bonus
    scorecard.mark_score("aces", np.array([1, 2, 3, 4, 5]))     # 1 point
    scorecard.mark_score("twos", np.array([2, 3, 4, 5, 6]))     # 2 points
    scorecard.mark_score("threes", np.array([3, 4, 5, 6, 1]))   # 3 points
    scorecard.mark_score("fours", np.array([4, 5, 6, 1, 2]))    # 4 points
    scorecard.mark_score("fives", np.array([5, 6, 1, 2, 3]))    # 5 points
    scorecard.mark_score("sixes", np.array([6, 1, 2, 3, 4]))    # 6 points
    
    # Upper total = 1 + 2 + 3 + 4 + 5 + 6 = 21 (< 63, no bonus)
    expected_upper = 21
    
    # Lower section
    scorecard.mark_score("chance", np.array([1, 1, 1, 1, 1]))   # Sum = 5
    
    expected_lower = 5
    
    assert scorecard.compute_upper_score() == expected_upper
    assert scorecard.compute_lower_score() == expected_lower
    assert scorecard.compute_final_score() == expected_upper + expected_lower  # 26


def test_game_with_yahtzee_bonus():
    """Test a full game scenario with multiple yahtzees."""
    scorecard = Scorecard(turn_number=0)
    
    # Score a yahtzee
    scorecard.mark_score("yahtzee", np.array([4, 4, 4, 4, 4]))  # 50 points
    
    # Score some other categories
    scorecard.mark_score("fours", np.array([4, 4, 4, 4, 4]))    # 20 points (upper)
    scorecard.mark_score("chance", np.array([6, 6, 6, 6, 6]))   # 30 points
    
    # Simulate two additional yahtzees achieved during the game
    scorecard.score_board["yahtzee"]["num_times_achieved"] = 3
    
    # Upper = 20 (no bonus, < 63)
    # Lower = 50 (yahtzee) + 30 (chance) + 200 (2 yahtzee bonuses) = 280
    expected_upper = 20
    expected_lower = 50 + 30 + 200
    
    assert scorecard.compute_upper_score() == expected_upper
    assert scorecard.compute_lower_score() == expected_lower
    assert scorecard.compute_final_score() == expected_upper + expected_lower  # 300


# ============================================================
# Scorecard State Tests
# ============================================================

def test_mark_score_returns_false_when_already_marked():
    """Test that marking an already-marked category returns False."""
    scorecard = Scorecard(turn_number=0)
    
    # First mark should succeed
    assert scorecard.mark_score("aces", np.array([1, 1, 1, 2, 3])) is True
    
    # Second mark on same category should fail
    assert scorecard.mark_score("aces", np.array([1, 1, 1, 1, 1])) is False
    
    # Score should remain from first mark
    assert scorecard.get_category_score("aces") == 3


def test_reset_clears_all_scores():
    """Test that reset() properly clears all scorecard state."""
    scorecard = Scorecard(turn_number=5)
    
    # Mark some categories
    scorecard.mark_score("aces", np.array([1, 1, 1, 2, 3]))
    scorecard.mark_score("yahtzee", np.array([5, 5, 5, 5, 5]))
    scorecard.score_board["yahtzee"]["num_times_achieved"] = 3
    
    # Reset the scorecard
    scorecard.reset()
    
    # Verify all state is cleared
    assert scorecard.turn_number == 0
    assert scorecard.get_category_score("aces") == 0
    assert scorecard.get_category_score("yahtzee") == 0
    assert scorecard.score_board["yahtzee"]["num_times_achieved"] == 0
    assert scorecard.score_board["aces"]["marked"] is False
    
    # Should be able to mark categories again
    assert scorecard.mark_score("aces", np.array([1, 1, 2, 3, 4])) is True


def test_get_category_score():
    """Test get_category_score returns correct values."""
    scorecard = Scorecard(turn_number=0)
    
    # Unmarked category should return 0
    assert scorecard.get_category_score("aces") == 0
    
    # After marking, should return the score
    scorecard.mark_score("aces", np.array([1, 1, 1, 1, 1]))
    assert scorecard.get_category_score("aces") == 5


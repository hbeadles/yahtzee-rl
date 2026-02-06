import numpy as np
from yahtzee_rl.scoring.ops import (
    aces, twos, threes, fours, fives, sixes,
    three_of_a_kind, four_of_a_kind, full_house,
    small_straight, large_straight, yahtzee, chance
)


# ============================================================
# Upper Section Tests
# ============================================================

def test_aces():
    """Test aces scoring: counts dice showing 1."""
    # Satisfying cases
    assert aces(np.array([1, 1, 1, 2, 3])) == 3  # Three 1s
    assert aces(np.array([1, 1, 4, 5, 6])) == 2  # Two 1s
    # Non-satisfying case
    assert aces(np.array([2, 3, 4, 5, 6])) == 0  # No 1s


def test_twos():
    """Test twos scoring: sums dice showing 2."""
    # Satisfying cases
    assert twos(np.array([2, 2, 3, 4, 5])) == 4  # Two 2s = 4 points
    assert twos(np.array([2, 2, 2, 2, 1])) == 8  # Four 2s = 8 points
    # Non-satisfying case
    assert twos(np.array([1, 3, 4, 5, 6])) == 0  # No 2s


def test_threes():
    """Test threes scoring: sums dice showing 3."""
    # Satisfying cases
    assert threes(np.array([3, 3, 3, 1, 2])) == 9  # Three 3s = 9 points
    assert threes(np.array([3, 3, 4, 5, 6])) == 6  # Two 3s = 6 points
    # Non-satisfying case
    assert threes(np.array([1, 2, 4, 5, 6])) == 0  # No 3s


def test_fours():
    """Test fours scoring: sums dice showing 4."""
    # Satisfying cases
    assert fours(np.array([4, 4, 1, 2, 3])) == 8  # Two 4s = 8 points
    assert fours(np.array([4, 4, 4, 4, 1])) == 16  # Four 4s = 16 points
    # Non-satisfying case
    assert fours(np.array([1, 2, 3, 5, 6])) == 0  # No 4s


def test_fives():
    """Test fives scoring: sums dice showing 5."""
    # Satisfying cases
    assert fives(np.array([5, 5, 5, 5, 1])) == 20  # Four 5s = 20 points
    assert fives(np.array([5, 5, 1, 2, 3])) == 10  # Two 5s = 10 points
    # Non-satisfying case
    assert fives(np.array([1, 2, 3, 4, 6])) == 0  # No 5s


def test_sixes():
    """Test sixes scoring: sums dice showing 6."""
    # Satisfying cases
    assert sixes(np.array([6, 6, 6, 1, 2])) == 18  # Three 6s = 18 points
    assert sixes(np.array([6, 6, 3, 4, 5])) == 12  # Two 6s = 12 points
    # Non-satisfying case
    assert sixes(np.array([1, 2, 3, 4, 5])) == 0  # No 6s


# ============================================================
# Lower Section Tests
# ============================================================

def test_three_of_a_kind():
    """Test three of a kind: returns sum of all dice if 3+ dice match, else 0."""
    # Satisfying cases - returns sum of all dice
    assert three_of_a_kind(np.array([3, 3, 3, 4, 5])) == 18  # Three 3s, sum=18
    assert three_of_a_kind(np.array([6, 6, 6, 1, 2])) == 21  # Three 6s, sum=21
    # Non-satisfying case - returns 0
    assert three_of_a_kind(np.array([1, 2, 3, 4, 5])) == 0  # No matching triplet


def test_four_of_a_kind():
    """Test four of a kind: returns sum of all dice if 4+ dice match, else 0."""
    # Satisfying cases - returns sum of all dice
    assert four_of_a_kind(np.array([4, 4, 4, 4, 2])) == 18  # Four 4s, sum=18
    assert four_of_a_kind(np.array([5, 5, 5, 5, 1])) == 21  # Four 5s, sum=21
    # Non-satisfying case - returns 0 (three of a kind doesn't count)
    assert four_of_a_kind(np.array([3, 3, 3, 4, 5])) == 0  # Only three 3s


def test_full_house():
    """Test full house: returns 25 if 3 of one value + 2 of another, else 0."""
    # Satisfying cases - returns 25 points
    assert full_house(np.array([2, 2, 2, 5, 5])) == 25  # Three 2s + two 5s
    assert full_house(np.array([3, 3, 6, 6, 6])) == 25  # Two 3s + three 6s
    # Non-satisfying case - returns 0
    assert full_house(np.array([1, 2, 3, 4, 5])) == 0  # All different
    assert full_house(np.array([3, 3, 3, 3, 5])) == 0  # Four of a kind, not full house


def test_small_straight():
    """Test small straight: returns 30 if 4 consecutive values present, else 0."""
    # Satisfying cases - returns 30 points
    assert small_straight(np.array([1, 2, 3, 4, 6])) == 30  # Sequence 1-2-3-4
    assert small_straight(np.array([2, 3, 4, 5, 5])) == 30  # Sequence 2-3-4-5
    assert small_straight(np.array([1, 3, 4, 5, 6])) == 30  # Sequence 3-4-5-6
    # Non-satisfying case - returns 0
    assert small_straight(np.array([1, 2, 3, 5, 6])) == 0  # Gap at 4, no valid sequence


def test_large_straight():
    """Test large straight: returns 40 if 5 consecutive values present, else 0."""
    # Satisfying cases - returns 40 points
    assert large_straight(np.array([1, 2, 3, 4, 5])) == 40  # Low straight 1-2-3-4-5
    assert large_straight(np.array([2, 3, 4, 5, 6])) == 40  # High straight 2-3-4-5-6
    # Non-satisfying case - returns 0
    assert large_straight(np.array([1, 2, 3, 4, 6])) == 0  # Missing 5, only 4 consecutive


def test_yahtzee():
    """Test yahtzee: returns 50 if all 5 dice show same value, else 0."""
    # Satisfying cases - returns 50 points
    assert yahtzee(np.array([5, 5, 5, 5, 5])) == 50  # Five 5s
    assert yahtzee(np.array([1, 1, 1, 1, 1])) == 50  # Five 1s
    # Non-satisfying case - returns 0
    assert yahtzee(np.array([5, 5, 5, 5, 4])) == 0  # Only four 5s


def test_chance():
    """Test chance: always returns sum of all dice regardless of combination."""
    # Always returns sum of all dice
    assert chance(np.array([1, 2, 3, 4, 5])) == 15  # Sum of 1+2+3+4+5
    assert chance(np.array([6, 6, 6, 6, 6])) == 30  # Sum of five 6s
    assert chance(np.array([1, 1, 1, 1, 1])) == 5   # Sum of five 1s

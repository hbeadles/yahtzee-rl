"""Tests for yahtzee_rl.markov.lookaheads module."""
import pytest
import numpy as np
from yahtzee_rl.markov.lookaheads import determine_keep_positions


def _unpack(action: int) -> np.ndarray:
    """Unpack an action integer to a 5-bit reroll mask (1=reroll, 0=keep)."""
    return np.unpackbits(np.array([action], dtype=np.uint8), count=5, bitorder='little')


def _kept_values(dice: np.ndarray, action: int):
    """Return sorted list of dice values kept (not rerolled) by this action."""
    mask = _unpack(action)
    return sorted(dice[mask == 0])


def _rerolled_values(dice: np.ndarray, action: int):
    """Return sorted list of dice values rerolled by this action."""
    mask = _unpack(action)
    return sorted(dice[mask == 1])


class TestDetermineKeepPositions:
    """Test suite for determine_keep_positions function."""

    def test_aces_keeps_ones(self):
        """Test that aces move keeps all 1s."""
        dice = np.array([1, 2, 3, 1, 5])

        action = determine_keep_positions(dice, 'aces')

        assert _kept_values(dice, action) == [1, 1]
        assert _rerolled_values(dice, action) == [2, 3, 5]

    def test_threes_keeps_threes(self):
        """Test that threes move keeps all 3s."""
        dice = np.array([3, 3, 2, 4, 3])

        action = determine_keep_positions(dice, 'threes')

        assert _kept_values(dice, action) == [3, 3, 3]
        assert _rerolled_values(dice, action) == [2, 4]

    def test_sixes_keeps_sixes(self):
        """Test that sixes move keeps all 6s."""
        dice = np.array([6, 1, 6, 2, 3])

        action = determine_keep_positions(dice, 'sixes')

        assert _kept_values(dice, action) == [6, 6]
        assert _rerolled_values(dice, action) == [1, 2, 3]

    def test_small_straight_picks_best_run(self):
        """Test that small_straight picks dice toward the best run."""
        dice = np.array([3, 4, 5, 6, 1])

        action = determine_keep_positions(dice, 'small_straight')

        # [3,4,5,6] run has 4 matches -- best of the three candidates
        assert _kept_values(dice, action) == [3, 4, 5, 6]
        assert _rerolled_values(dice, action) == [1]

    def test_large_straight_keeps_sequence_dice(self):
        """Test that large_straight keeps dice toward a 5-sequence."""
        dice = np.array([1, 2, 3, 4, 6])

        action = determine_keep_positions(dice, 'large_straight')

        # [1,2,3,4,5] has 4 matches vs [2,3,4,5,6] also 4 -- picks first
        assert _kept_values(dice, action) == [1, 2, 3, 4]
        assert _rerolled_values(dice, action) == [6]

    def test_full_house_keeps_two_most_common(self):
        """Test that full_house keeps the two most common dice values."""
        dice = np.array([2, 2, 5, 5, 3])

        action = determine_keep_positions(dice, 'full_house')

        assert _kept_values(dice, action) == [2, 2, 5, 5]
        assert _rerolled_values(dice, action) == [3]

    def test_full_house_all_same_no_crash(self):
        """Test that full_house doesn't crash when all dice are the same value.

        This was a bug where reduce() on empty dice_combined_second would crash.
        """
        dice = np.array([5, 5, 5, 5, 5])

        action = determine_keep_positions(dice, 'full_house')

        # All 5s kept, action 0 means keep everything
        assert action == 0
        assert _kept_values(dice, action) == [5, 5, 5, 5, 5]

    def test_yahtzee_keeps_most_common(self):
        """Test that yahtzee keeps the most common dice value."""
        dice = np.array([4, 4, 2, 4, 1])

        action = determine_keep_positions(dice, 'yahtzee')

        assert _kept_values(dice, action) == [4, 4, 4]
        assert _rerolled_values(dice, action) == [1, 2]

    def test_chance_keeps_max_value(self):
        """Test that chance keeps dice equal to the max value."""
        dice = np.array([6, 5, 4, 3, 2])

        action = determine_keep_positions(dice, 'chance')

        assert _kept_values(dice, action) == [6]
        assert _rerolled_values(dice, action) == [2, 3, 4, 5]

    def test_three_of_a_kind_keeps_most_common(self):
        """Test that three_of_a_kind keeps the most common dice value."""
        dice = np.array([3, 3, 3, 2, 5])

        action = determine_keep_positions(dice, 'three_of_a_kind')

        assert _kept_values(dice, action) == [3, 3, 3]
        assert _rerolled_values(dice, action) == [2, 5]

    def test_four_of_a_kind_keeps_most_common(self):
        """Test that four_of_a_kind keeps the most common dice value."""
        dice = np.array([2, 2, 2, 2, 6])

        action = determine_keep_positions(dice, 'four_of_a_kind')

        assert _kept_values(dice, action) == [2, 2, 2, 2]
        assert _rerolled_values(dice, action) == [6]

    def test_returns_int_in_valid_range(self):
        """Test that the return value is an int in [0, 31]."""
        dice = np.array([1, 2, 3, 4, 5])

        action = determine_keep_positions(dice, 'aces')

        assert isinstance(action, int)
        assert 0 <= action <= 31

    def test_keep_all_returns_zero(self):
        """Test that keeping every die returns action 0."""
        dice = np.array([3, 3, 3, 3, 3])

        action = determine_keep_positions(dice, 'threes')

        assert action == 0

    def test_reroll_all_returns_31(self):
        """Test that rerolling every die returns action 31."""
        dice = np.array([2, 3, 4, 5, 6])

        action = determine_keep_positions(dice, 'aces')

        # No 1s to keep → reroll everything
        assert action == 31

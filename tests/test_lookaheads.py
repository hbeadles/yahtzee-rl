"""Tests for yahtzee_rl.markov.lookaheads module."""
import pytest
import numpy as np
from yahtzee_rl.markov.lookaheads import determine_keep_positions


class TestDetermineKeepPositions:
    """Test suite for determine_keep_positions function."""

    def test_aces_keeps_ones(self):
        """Test that aces move keeps all 1s."""
        dice = np.array([1, 2, 3, 1, 5])
        withheld = np.array([])
        
        final_withheld, remaining = determine_keep_positions(dice, withheld, 'aces')
        
        # Should keep both 1s
        assert sorted(final_withheld) == [1, 1]
        assert sorted(remaining) == [2, 3, 5]

    def test_threes_keeps_threes(self):
        """Test that threes move keeps all 3s."""
        dice = np.array([3, 3, 2, 4, 3])
        withheld = np.array([])
        
        final_withheld, remaining = determine_keep_positions(dice, withheld, 'threes')
        
        # Should keep all three 3s
        assert sorted(final_withheld) == [3, 3, 3]
        assert sorted(remaining) == [2, 4]

    def test_sixes_keeps_sixes(self):
        """Test that sixes move keeps all 6s."""
        dice = np.array([6, 1, 6, 2, 3])
        withheld = np.array([])
        
        final_withheld, remaining = determine_keep_positions(dice, withheld, 'sixes')
        
        # Should keep both 6s
        assert sorted(final_withheld) == [6, 6]
        assert sorted(remaining) == [1, 2, 3]

    def test_small_straight_picks_best_run(self):
        """Test that small_straight picks dice toward the best run."""
        dice = np.array([1, 2, 4, 5, 6])
        withheld = np.array([3])
        
        final_withheld, remaining = determine_keep_positions(dice, withheld, 'small_straight')
        
        # With 3 already withheld, should pick from [2,3,4,5] or [3,4,5,6] run
        # The function should keep dice that contribute to a straight
        assert 3 in final_withheld  # Original withheld should be preserved
        # Should have kept some dice toward a run
        assert len(final_withheld) > 1

    def test_large_straight_keeps_sequence_dice(self):
        """Test that large_straight keeps dice toward a 5-sequence."""
        dice = np.array([1, 2, 3, 4, 6])
        withheld = np.array([])
        
        final_withheld, remaining = determine_keep_positions(dice, withheld, 'large_straight')
        
        # Should keep 1,2,3,4 (toward [1,2,3,4,5]) since that has 4 matches
        assert sorted(final_withheld) == [1, 2, 3, 4]
        assert sorted(remaining) == [6]

    def test_full_house_keeps_two_most_common(self):
        """Test that full_house keeps the two most common dice values."""
        dice = np.array([2, 2, 5, 5, 3])
        withheld = np.array([])
        
        final_withheld, remaining = determine_keep_positions(dice, withheld, 'full_house')
        
        # Should keep both 2s and both 5s
        assert sorted(final_withheld) == [2, 2, 5, 5]
        assert sorted(remaining) == [3]

    def test_full_house_all_same_no_crash(self):
        """Test that full_house doesn't crash when all dice are the same value.
        
        This was a bug where reduce() on empty dice_combined_second would crash.
        """
        dice = np.array([5, 5, 5])
        withheld = np.array([5, 5])
        
        # Should NOT raise TypeError
        final_withheld, remaining = determine_keep_positions(dice, withheld, 'full_house')
        
        # All 5s should be kept
        assert sorted(final_withheld) == [5, 5, 5, 5, 5]
        assert len(remaining) == 0

    def test_yahtzee_keeps_most_common(self):
        """Test that yahtzee keeps the most common dice value."""
        dice = np.array([4, 4, 2, 4, 1])
        withheld = np.array([])
        
        final_withheld, remaining = determine_keep_positions(dice, withheld, 'yahtzee')
        
        # Should keep all three 4s
        assert sorted(final_withheld) == [4, 4, 4]
        assert sorted(remaining) == [1, 2]

    def test_chance_keeps_max_value(self):
        """Test that chance keeps dice equal to the max value."""
        dice = np.array([6, 5, 4, 3, 2])
        withheld = np.array([])
        
        final_withheld, remaining = determine_keep_positions(dice, withheld, 'chance')
        
        # Should keep the 6 (max value)
        assert sorted(final_withheld) == [6]
        assert sorted(remaining) == [2, 3, 4, 5]

    def test_withheld_preserved(self):
        """Test that previously withheld dice are preserved in output."""
        dice = np.array([1, 2, 3])
        withheld = np.array([1, 1])
        
        final_withheld, remaining = determine_keep_positions(dice, withheld, 'aces')
        
        # Should have original withheld [1, 1] plus the new 1 from dice
        assert sorted(final_withheld) == [1, 1, 1]
        assert sorted(remaining) == [2, 3]

    def test_three_of_a_kind_keeps_most_common(self):
        """Test that three_of_a_kind keeps the most common dice value."""
        dice = np.array([3, 3, 3, 2, 5])
        withheld = np.array([])
        
        final_withheld, remaining = determine_keep_positions(dice, withheld, 'three_of_a_kind')
        
        # Should keep all three 3s
        assert sorted(final_withheld) == [3, 3, 3]
        assert sorted(remaining) == [2, 5]

    def test_four_of_a_kind_keeps_most_common(self):
        """Test that four_of_a_kind keeps the most common dice value."""
        dice = np.array([2, 2, 2, 2, 6])
        withheld = np.array([])
        
        final_withheld, remaining = determine_keep_positions(dice, withheld, 'four_of_a_kind')
        
        # Should keep all four 2s
        assert sorted(final_withheld) == [2, 2, 2, 2]
        assert sorted(remaining) == [6]

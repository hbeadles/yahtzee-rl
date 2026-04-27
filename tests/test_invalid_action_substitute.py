"""Tests for YahtzeeEnv's invalid_action_substitute behavior."""
from __future__ import annotations

import numpy as np
import pytest

from yahtzee_rl.config import ACTION_TO_CATEGORY, Category
from yahtzee_rl.envs.yahtzee_env import YahtzeeEnv


def _advance_to_scoring_phase(env: YahtzeeEnv) -> np.ndarray:
    """Step through the two remaining rolls so rolls_remaining == 0."""
    obs = None
    while env.game.rolls_remaining > 0:
        obs, _reward, _done, _trunc, _info = env.step(0)
    assert env.game.rolls_remaining == 0
    return obs


# ---------------------------------------------------------------------------
# Helper-level tests (target _substitute_invalid_action directly)
# ---------------------------------------------------------------------------


def test_helper_returns_valid_category():
    env = YahtzeeEnv(invalid_action_substitute=True)
    env.reset(seed=0)
    mask = np.zeros(32, dtype=bool)
    # Only a handful of categories are valid.
    mask[[2, 5, 9]] = True

    for _ in range(25):
        new_action, _ = env._substitute_invalid_action(action=25, mask=mask)
        assert new_action in (2, 5, 9)


def test_helper_populates_info_patch():
    env = YahtzeeEnv(invalid_action_substitute=True)
    env.reset(seed=0)
    mask = np.zeros(32, dtype=bool)
    mask[:13] = True
    # Mark one category as invalid.
    mask[3] = False

    new_action, info_patch = env._substitute_invalid_action(action=3, mask=mask)

    assert info_patch["invalid_action"] is True
    assert info_patch["invalid_action_original"] == 3
    assert info_patch["invalid_action_substituted"] == new_action
    assert isinstance(info_patch["invalid_action_original"], int)
    assert isinstance(info_patch["invalid_action_substituted"], int)
    assert new_action != 3
    assert new_action in range(13)


# ---------------------------------------------------------------------------
# End-to-end step() tests
# ---------------------------------------------------------------------------


def test_default_behavior_unchanged():
    """Regression guard: env with no new args runs a full episode without
    producing invalid actions or substitution metadata."""
    env = YahtzeeEnv()
    assert env.invalid_action_substitute is False
    assert env.invalid_action_penalty == -20.0

    obs, _ = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape

    total_reward = 0.0
    step_count = 0
    done = False
    while not done:
        mask = env.action_masks()
        valid_actions = np.where(mask)[0]
        # Use the first valid action deterministically.
        action = int(valid_actions[0])
        _obs, reward, done, _trunc, info = env.step(action)
        total_reward += reward
        step_count += 1
        # No substitution should ever happen under defaults.
        assert "invalid_action" not in info

    assert env.game.is_game_over()
    # 13 rounds x 3 actions per round = 39 total steps.
    assert step_count == 39


def test_substitute_mode_replaces_invalid_scoring_action():
    env = YahtzeeEnv(
        invalid_action_substitute=True,
        invalid_action_penalty=-20.0,
    )
    env.reset(seed=7)

    # Pre-mark category 0 (ACES) so action=0 is invalid in scoring phase.
    env.game.scorecard.score_board[Category.ACES]["marked"] = True

    _advance_to_scoring_phase(env)

    _obs, reward, _done, _trunc, info = env.step(0)

    assert info["invalid_action"] is True
    assert info["invalid_action_original"] == 0
    substituted = info["invalid_action_substituted"]
    assert substituted in range(13)
    assert substituted != 0
    assert "game_reward" in info
    # Reward equals game_reward plus the penalty.
    assert reward == pytest.approx(info["game_reward"] + env.invalid_action_penalty)


def test_substitute_mode_handles_out_of_range_action():
    """action=25 would KeyError on ACTION_TO_CATEGORY without substitution."""
    env = YahtzeeEnv(invalid_action_substitute=True)
    env.reset(seed=1)
    _advance_to_scoring_phase(env)

    # No pre-marking: category 0-12 are all valid. Action 25 is out of range
    # and must be substituted before the ACTION_TO_CATEGORY lookup.
    _obs, _reward, _done, _trunc, info = env.step(25)

    assert info["invalid_action"] is True
    assert info["invalid_action_original"] == 25
    assert info["invalid_action_substituted"] in range(13)


def test_substitute_mode_is_deterministic_with_seed():
    """Seeding np_random via reset() makes substitution reproducible."""
    seen = []
    for _ in range(2):
        env = YahtzeeEnv(invalid_action_substitute=True)
        env.reset(seed=123)
        env.game.scorecard.score_board[Category.ACES]["marked"] = True
        _advance_to_scoring_phase(env)
        _, _, _, _, info = env.step(0)
        seen.append(info["invalid_action_substituted"])

    assert seen[0] == seen[1]


def test_off_mode_raises_on_out_of_range_action():
    """With substitute disabled, action=25 in scoring phase raises (today's behavior)."""
    env = YahtzeeEnv()  # substitute=False by default
    env.reset(seed=0)
    _advance_to_scoring_phase(env)
    with pytest.raises(KeyError):
        env.step(25)


def test_rolling_phase_info_contains_game_reward():
    """Rolling phase returns 0 reward and includes game_reward for consistency."""
    env = YahtzeeEnv()
    env.reset(seed=0)
    _obs, reward, done, _trunc, info = env.step(0)
    assert reward == 0.0
    assert done is False
    assert info["game_reward"] == 0.0

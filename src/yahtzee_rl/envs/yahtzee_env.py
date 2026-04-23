from yahtzee_rl.markov.probabilities import upper_section_probability, \
    upper_section_expected_score_vector, upper_section_prob_vector, lower_section_prob_vector, lower_section_expected_score_vector
from yahtzee_rl.scoring.scorecard import Scorecard
from yahtzee_rl.scoring.ops import combo_satisfied
from yahtzee_rl.envs.yahtzee_game import YahtzeeGame
from typing import Optional
from yahtzee_rl import Category, ACTION_TO_CATEGORY
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class YahtzeeEnv(gym.Env):
    """
    Yahtzee Environment.
    This environment has the following features:
    1. We force the player to roll the dice, and then choose a category
    2. Observation state is defined by _OBS_FIELDS, contains
        1. Dice values (5)
        2. Roll number (1)
        3. Round Number (1)
        4. Time to Score flag (1)
        5. Score Card (13)
        6. Upper Score (1)
        7. Lower Score (1)
        8. Upper Section Probabilities (6)  — optional via ``use_probabilities``
        9. Lower Section Probabilities (7)  — optional via ``use_probabilities``
        10. Upper Section Expected Scores (6) — optional via ``use_expecteds``
        11. Lower Section Expected Scores (7) — optional via ``use_expecteds``
    3. Action space is 32
       1. 2^5 = 32 possible when a roll action
       2. 13 possible when a score action
    """
    _OBS_FIELDS = [
        ("dice",                          5),
        ("rolls_remaining",               1),
        ("round",                         1),
        ("time_to_score",                 1),
        ("scorecard",                    13),
        ("upper_score",                   1),
        ("lower_score",                   1),
        ("upper_section_probabilities",   6),
        ("lower_section_probabilities",   7),
        ("upper_section_expected_scores", 6),
        ("lower_section_expected_scores", 7),
    ]
    OBS_DIM = sum(size for _, size in _OBS_FIELDS)

    def __init__(self,
                 render_mode: Optional[str] = None,
                 lambda_upper: Optional[float] = 0.05,
                 lambda_yahtzee: Optional[float] = 0.1,
                 use_probabilities: bool = True,
                 use_expecteds: bool = True,
                 invalid_action_substitute: bool = False,
                 invalid_action_penalty: float = -20.0):
        super().__init__()
        self.render_mode = render_mode
        self.game = YahtzeeGame()
        self.use_probabilities = use_probabilities
        self.use_expecteds = use_expecteds
        self.lambda_upper = lambda_upper
        self.lambda_yahtzee = lambda_yahtzee
        self.invalid_action_substitute = invalid_action_substitute
        self.invalid_action_penalty = invalid_action_penalty
        obs_dim = self.OBS_DIM
        if not use_probabilities:
            obs_dim -= 13
        if not use_expecteds:
            obs_dim -= 13
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.game.reset(seed=seed)
        return self._build_observation(), {}
    
    def step(self, action: int):
        if self.game.rolls_remaining > 0:
            roll_bits = np.unpackbits(np.array([int(action)], dtype=np.uint8), count=5, bitorder='little')
            self.game.roll_dice(roll_bits)
            return self._build_observation(), 0.0, False, False, {"game_reward": 0.0}

        info: dict = {}
        if self.invalid_action_substitute:
            mask = self.action_masks()
            if not mask[action]:
                action, info = self._substitute_invalid_action(action, mask)

        category = ACTION_TO_CATEGORY[action]
        upper_score, lower_score, valid = self.game.score_category(category)
        reward = upper_score + lower_score
        info["game_reward"] = reward
        if info.get("invalid_action"):
            reward += self.invalid_action_penalty
        done = self.game.is_game_over()
        if not done:
            self.game.reset_rolls()
        return self._build_observation(), reward, done, False, info

    def _substitute_invalid_action(self, action: int, mask: np.ndarray) -> tuple:
        """
        Pick a random valid scoring category to replace an invalid action.

        Called only in the scoring phase, only when ``invalid_action_substitute``
        is True and ``mask[action]`` is False. Returns ``(new_action, info_patch)``
        where ``info_patch`` should be merged into the ``step()`` info dict. The
        caller is responsible for applying ``invalid_action_penalty`` to the reward.
        """
        valid_categories = np.where(mask[:13])[0]
        new_action = int(self.np_random.choice(valid_categories))
        info_patch = {
            "invalid_action": True,
            "invalid_action_original": int(action),
            "invalid_action_substituted": new_action,
        }
        return new_action, info_patch

    def action_masks(self):
        """
        Generate action masks for RL models
        :return:
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        category_mask = []
        if self.game.rolls_remaining > 0:
            mask[:] = True
        else:
            for category in Category:
                if category == Category.YAHTZEE:
                    category_mask.append(not self.game.scorecard.is_category_marked(category) or combo_satisfied(self.game.dice, category))
                else:
                    category_mask.append(not self.game.scorecard.is_category_marked(category))
            mask[:13] = np.array(category_mask, dtype=bool)

        return mask


    def _build_dice_observation(self) -> np.ndarray:
        """
        Buid dice observation
        """
        return np.array(self.game.dice, dtype=np.float32)

    def _build_roll_observation(self) -> np.ndarray:
        """
        Build roll observation
        """
        return np.array([self.game.rolls_remaining], dtype=np.float32)

    def _build_round_observation(self) -> np.ndarray:
        """
        Build round observation
        """
        return np.array([self.game.round], dtype=np.float32)

    def _build_time_to_score_observation(self) -> np.ndarray:
        """
        Build time-to-score flag: 1.0 when rolls_remaining == 0, else 0.0.
        """
        return np.array(
            [1.0 if self.game.rolls_remaining == 0 else 0.0],
            dtype=np.float32,
        )

    def _build_scorecard_observation(self) -> np.ndarray:
        """
        Build scorecard observation
        """
        return np.array([self.game.scorecard.score_board[category]["marked"] for category in Category], dtype=np.float32)

    def _build_upper_score_observation(self) -> np.ndarray:
        return np.array([self.game.scorecard.compute_upper_score()], dtype=np.float32)

    def _build_lower_score_observation(self) -> np.ndarray:
        return np.array([self.game.scorecard.compute_lower_score()], dtype=np.float32)

    def _build_upper_section_probabilities_observation(self) -> np.ndarray:
        return np.array(upper_section_prob_vector(self.game.dice, self.game.rolls_remaining), dtype=np.float32)

    def _build_lower_section_probabilities_observation(self) -> np.ndarray:
        return np.array(lower_section_prob_vector(self.game.dice, self.game.rolls_remaining), dtype=np.float32)

    def _build_upper_section_expected_score_observation(self) -> np.ndarray:
        return np.array(upper_section_expected_score_vector(self.game.dice, self.game.scorecard, self.game.rolls_remaining, lambda_v=self.lambda_upper), dtype=np.float32)

    def _build_lower_section_expected_score_observation(self) -> np.ndarray:
        return np.array(lower_section_expected_score_vector(self.game.dice, self.game.scorecard, self.game.rolls_remaining, lambda_yahtzee=self.lambda_yahtzee), dtype=np.float32)

    def _build_observation(self) -> np.ndarray:
        obs_parts = [
            self._build_dice_observation(),
            self._build_roll_observation(),
            self._build_round_observation(),
            self._build_time_to_score_observation(),
            self._build_scorecard_observation(),
            self._build_upper_score_observation(),
            self._build_lower_score_observation(),
        ]
        if self.use_probabilities:
            obs_parts.extend([
                self._build_upper_section_probabilities_observation(),
                self._build_lower_section_probabilities_observation(),
            ])
        if self.use_expecteds:
            obs_parts.extend([
                self._build_upper_section_expected_score_observation(),
                self._build_lower_section_expected_score_observation(),
            ])
        return np.concatenate(obs_parts)

    @classmethod
    def parse_observation(cls, obs: np.ndarray,
                          use_probabilities: bool = True,
                          use_expecteds: bool = True) -> dict:
        """
        Parse a flat observation vector back into a labelled dictionary.

        The observation layout is defined by ``_OBS_FIELDS``.

        Args:
            obs: np.ndarray flat observation from _build_observation.
            use_probabilities: Whether the observation includes probability fields.
            use_expecteds: Whether the observation includes expected score fields.

        Returns:
            dict with human-readable keys and sub-arrays / scalars.
        """
        _PROB_FIELDS = {"upper_section_probabilities", "lower_section_probabilities"}
        _EXPECTED_FIELDS = {"upper_section_expected_scores", "lower_section_expected_scores"}

        excluded = set()
        if not use_probabilities:
            excluded |= _PROB_FIELDS
        if not use_expecteds:
            excluded |= _EXPECTED_FIELDS

        fields_to_parse = [(n, s) for n, s in cls._OBS_FIELDS if n not in excluded]

        result, idx = {}, 0
        for name, size in fields_to_parse:
            result[name] = float(obs[idx]) if size == 1 else obs[idx:idx + size]
            idx += size

        upper_cats = [c.value for c in Category.upper_categories()]
        lower_cats = [c.value for c in Category.lower_categories()]

        keys_to_convert = [
            ("scorecard", upper_cats + lower_cats),
        ]

        if use_probabilities:
            keys_to_convert.extend([
                ("upper_section_probabilities", upper_cats),
                ("lower_section_probabilities", lower_cats),
            ])

        if use_expecteds:
            keys_to_convert.extend([
                ("upper_section_expected_scores", upper_cats),
                ("lower_section_expected_scores", lower_cats),
            ])

        for key, cats in keys_to_convert:
            result[key] = dict(zip(cats, result[key].tolist()))

        result["time_to_score"] = bool(result["time_to_score"])
        return result

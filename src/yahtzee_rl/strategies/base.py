import numpy as np
from yahtzee_rl.scoring.scorecard import Scorecard
from yahtzee_rl.envs.yahtzee_env import YahtzeeEnv

class Strategy:
    def __init__(self, env: YahtzeeEnv):
        """
        Base strategy class initialization
        Args:
            env: The environment to use
        """
        self.env = env

    def strategy(self, obs: np.ndarray, scorecard: Scorecard) -> int:
        """
        Base strategy class strategy method
        Args:
            obs: The observation to use
            scorecard: The scorecard to use
        Returns:
            The action to take
        """
        raise NotImplementedError

from typing import Dict, List, Optional, Tuple, Type, Union
import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib.common.maskable.evaluation import evaluate_policy as evaluate_policy_maskable
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy, window_func
from datetime import datetime
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO
from enum import Enum
from sb3_contrib.common.wrappers import ActionMasker
import os

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class ProbabilityAnnealingCallback(BaseCallback):
    """Gradually reduce probability features in observations during training."""

    def __init__(self, start_step: int = 500_000, end_step: int = 2_000_000, verbose: int = 0):
        super().__init__(verbose)
        self.start_step = start_step
        self.end_step = end_step

    def _on_step(self) -> bool:
        if self.num_timesteps < self.start_step:
            scale = 1.0
        elif self.num_timesteps >= self.end_step:
            scale = 0.0
        else:
            progress = (self.num_timesteps - self.start_step) / (self.end_step - self.start_step)
            scale = 1.0 - progress

        # Set the scale on the environment
        env = self.model.get_env()
        if hasattr(env, 'envs'):
            for e in env.envs:
                if hasattr(e, 'env'):
                    e.env.prob_scale = scale
                elif hasattr(e, 'prob_scale'):
                    e.prob_scale = scale

        if self.verbose > 0 and self.num_timesteps % 50000 == 0:
            print(f"Step {self.num_timesteps}: prob_scale = {scale:.3f}")

        return True


class GAELambdaScheduleCallback(BaseCallback):
    """
    Callback to schedule gae_lambda during training.
    Starts with initial_lambda and increases to final_lambda over training.
    """

    def __init__(
            self,
            initial_lambda: float = 0.8,
            final_lambda: float = 0.95,
            schedule: str = "linear",  # "linear" or "exponential"
            verbose: int = 0
    ):
        super().__init__(verbose)
        self.initial_lambda = initial_lambda
        self.final_lambda = final_lambda
        self.schedule = schedule

    def _on_step(self) -> bool:
        # Calculate progress (0 to 1)
        progress = self.num_timesteps / self.model._total_timesteps

        if self.schedule == "linear":
            new_lambda = self.initial_lambda + progress * (self.final_lambda - self.initial_lambda)
        elif self.schedule == "exponential":
            # Exponential interpolation
            new_lambda = self.initial_lambda * (self.final_lambda / self.initial_lambda) ** progress
        else:
            new_lambda = self.initial_lambda

        # Update the model's gae_lambda
        self.model.gae_lambda = new_lambda

        if self.verbose > 0 and self.num_timesteps % 10000 == 0:
            print(f"Step {self.num_timesteps}: gae_lambda = {new_lambda:.4f}")

        return True


class ModelType(Enum):
    PPO = "PPO"
    MASKABLE_PPO = "MASKABLE_PPO"
    DQN = "DQN"
    A2C = "A2C"
    SAC = "SAC"


class TrainerBaselines:

    def __init__(self,
                 model_type: ModelType,
                 env: gym.Env,
                 exp_name: str,
                 verbose: int = 1,
                 save_dir: str = "experiments",
                 vec_normalize: bool = False,
                 gae_lambda: Union[float, Tuple[float, float]] = 0.95,
                 **kwargs):
        self.model_type = model_type
        self.env = env
        self.save_dir = os.path.join(save_dir, exp_name, datetime.now().strftime("%Y-%m-%d"))

        if isinstance(gae_lambda, tuple):
            initial, final = gae_lambda
            self.gae_callback = GAELambdaScheduleCallback(
                initial_lambda=initial,
                final_lambda=final,
                schedule="linear",
                verbose=1,
            )
            if model_type in (ModelType.PPO, ModelType.MASKABLE_PPO, ModelType.A2C):
                kwargs.setdefault("gae_lambda", initial)
        else:
            self.gae_callback = None
            if model_type in (ModelType.PPO, ModelType.MASKABLE_PPO, ModelType.A2C):
                kwargs.setdefault("gae_lambda", gae_lambda)

        os.makedirs(self.save_dir, exist_ok=True)
        if model_type == ModelType.PPO:
            self.env = Monitor(self.env, self.save_dir)
            self.model = PPO("MlpPolicy", self.env, verbose=verbose, **kwargs)
        elif model_type == ModelType.MASKABLE_PPO:
            def mask_fn(env):
                return env.action_masks()

            self.env = ActionMasker(self.env, mask_fn)
            self.env = Monitor(self.env, self.save_dir)

            self.model = MaskablePPO(MaskableActorCriticPolicy, self.env, verbose=verbose, **kwargs)

        elif model_type == ModelType.DQN:
            self.env = Monitor(self.env, self.save_dir)

            self.model = DQN("MlpPolicy", self.env, verbose=verbose, **kwargs)
        elif model_type == ModelType.A2C:
            self.env = Monitor(self.env, self.save_dir)

            self.model = A2C("MlpPolicy", self.env, verbose=verbose, **kwargs)
        elif model_type == ModelType.SAC:
            self.env = Monitor(self.env, self.save_dir)

            self.model = SAC("MlpPolicy", self.env, verbose=verbose, **kwargs)
        if vec_normalize:
            self.env = DummyVecEnv([lambda: self.env])
            self.env = VecNormalize(self.env, norm_obs=True, norm_reward=False, clip_obs=10.0)
            self.model.set_env(self.env)
        self.exp_name = exp_name

    def train(self,
              max_timesteps,
              save_freq: Optional[int] = None
              ):
        callbacks = []
        if save_freq is not None:
            checkpoint_callback = CheckpointCallback(
                save_freq=save_freq,
                save_path=self.save_dir,
                name_prefix="checkpoint",
                save_replay_buffer=False,
                save_vecnormalize=True
            )
            callbacks.append(checkpoint_callback)
        if self.gae_callback is not None:
            callbacks.append(self.gae_callback)
        self.model.learn(
            total_timesteps=max_timesteps,
            progress_bar=True,
            callback=callbacks if callbacks else None
        )
        self.model.save(os.path.join(self.save_dir, "model"))
        if isinstance(self.env, VecNormalize):
            self.env.save(os.path.join(self.save_dir, "vecnormalize.pkl"))

    def load(self, model_path: Optional[str] = None, vecnormalize_path: Optional[str] = None):
        path = model_path if model_path is not None else os.path.join(self.save_dir, "model")

        # Load VecNormalize if path provided
        if vecnormalize_path is not None:
            # Wrap env in DummyVecEnv if not already vectorized
            if not isinstance(self.env, VecEnv):
                print("Wrapping the env in a DummyVecEnv.")
                self.env = DummyVecEnv([lambda: self.env])
            # If already a VecNormalize, get the underlying venv for loading
            elif isinstance(self.env, VecNormalize):
                self.env = self.env.venv
            self.env = VecNormalize.load(vecnormalize_path, self.env)
            self.env.training = False
            self.env.norm_reward = False

        # Load returns a new model instance - must assign it back and pass env for continued training
        self.model = self.model.load(path, env=self.env)
        return self.model

    def evaluate(self, num_episodes: int):
        if isinstance(self.model, MaskablePPO):
            mean_reward, std_reward = evaluate_policy_maskable(
                self.model, self.env, n_eval_episodes=num_episodes
            )
        else:
            mean_reward, std_reward = evaluate_policy(
                self.model, self.env, n_eval_episodes=num_episodes
            )
        return mean_reward, std_reward

    def plot_results(self, max_timesteps):
        df = load_results(self.save_dir)
        x, y = ts2xy(df, "timesteps")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(x, y, s=2, alpha=0.6, color="blue")
        ax.set_xlabel("Number of Timesteps")
        ax.set_ylabel("Reward")
        ax.set_title("Learning Curve")
        if len(x) > 50:
            x_smooth, y_smooth = window_func(x, y, window=50, func=np.mean)
            ax.plot(x_smooth, y_smooth, linewidth=2, color="red")

        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "learning_curve.png"))
        plt.show()


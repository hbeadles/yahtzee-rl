import copy
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import gymnasium as gym
from stable_baselines3 import PPO, A2C, SAC
from yahtzee_rl.train.models.dqn import DQNAgent, transition
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib.common.maskable.evaluation import evaluate_policy as evaluate_policy_maskable
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy, window_func
from datetime import datetime
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO
from tqdm import tqdm
from enum import Enum
from sb3_contrib.common.wrappers import ActionMasker
import os
import json
from yahtzee_rl.paths import artifact_dir
from yahtzee_rl.config import Category
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import glob
import torch as th
import torch.nn as nn
import pandas as pd
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch

class YahtzeeTrunk(BaseFeaturesExtractor):

    def __init__(self,
                 observation_space,
                 features_dim: int = 600,
                 n_layers: int = 3, 
                 dropout: float = 0.2):
        super().__init__(observation_space, features_dim)
        in_dim = int(observation_space.shape[0])
        layers, d = [], in_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(d, features_dim),
                nn.SiLU(),
                nn.LayerNorm(features_dim),
            ]
            d = features_dim
        self.net = nn.Sequential(*layers)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)


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

from stable_baselines3.common.callbacks import BaseCallback

class TotalScoreLoggerCallback(BaseCallback):
    def _on_step(self) -> bool:
        # Monitor stores episode info in 'infos' when episode ends
        for info in self.locals.get("infos", []):
            if "episode" in info and "total_score" in info:
                self.logger.record_mean("rollout/ep_total_score_mean", info["total_score"])
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
                 save_dir: Union[str, Path, None] = None,
                 vec_normalize: bool = False,
                 gae_lambda: Union[float, Tuple[float, float]] = 0.95,
                 env_factory: Optional[Callable[[], gym.Env]] = None,
                 **kwargs):
        self.model_type = model_type
        self.env = env
        self.custom = False
        self.verbose = verbose
        self.vec_normalize = vec_normalize
        self._env_factory = env_factory
        self._raw_env = env
        base_dir = Path(save_dir) if save_dir is not None else artifact_dir()
        self.save_dir = str(base_dir / exp_name / datetime.now().strftime("%Y-%m-%d"))

        if isinstance(gae_lambda, tuple):
            initial, final = gae_lambda
            self.gae_callback = GAELambdaScheduleCallback(
                initial_lambda=initial,
                final_lambda=final,
                schedule="exponential",
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
            self.env = self._wrap_monitor(env)
            self.model = PPO("MlpPolicy", self.env, verbose=verbose, **kwargs)
            self.custom = False
        elif model_type == ModelType.MASKABLE_PPO:
            def mask_fn(env):
                return env.action_masks()
            self.env = ActionMasker(self.env, mask_fn)
            self.env = self._wrap_monitor(self.env)
            self.model = MaskablePPO(MaskableActorCriticPolicy, self.env, verbose=verbose, **kwargs)
            self.custom = False
        elif model_type == ModelType.DQN:
            dqn_agent = DQNAgent(
                observation_state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n,
                hidden_dim=kwargs.get("hidden_dim", 128),
                learning_rate=kwargs.get("learning_rate", 1e-3),
                gamma=kwargs.get("gamma", 0.99),
                epsilon_start=kwargs.get("epsilon_start", 1.0),
                epsilon_end=kwargs.get("epsilon_end", 0.01),
                epsilon_decay=kwargs.get("epsilon_decay", 2500),
                target_update_freq=kwargs.get("target_update_freq", 100),
                buffer_size=kwargs.get("buffer_size", 10000),
                batch_size=kwargs.get("batch_size", 64),
                aux_lambda=kwargs.get("aux_lambda", 0.5),
            )
            self.env = self._wrap_monitor(env)
            self.model = dqn_agent
            self._tau = kwargs.get("tau", 1.0)
            self.custom = True
        elif model_type == ModelType.A2C:
            self.env = self._wrap_monitor(env)
            self.model = A2C("MlpPolicy", self.env, verbose=verbose, **kwargs)
            self.custom = False
        elif model_type == ModelType.SAC:
            self.env = self._wrap_monitor(env)
            self.model = SAC("MlpPolicy", self.env, verbose=verbose, **kwargs)
            self.custom = False
        if vec_normalize:
            self.env = DummyVecEnv([lambda: self.env])
            self.env = VecNormalize(self.env, norm_obs=True, norm_reward=False, clip_obs=10.0)
            self.model.set_env(self.env)
        self.exp_name = exp_name

    def _build_eval_env(self):
        """Construct a fresh eval env that mirrors the training wrapping chain.

        - Uses ``self._env_factory`` if provided; otherwise falls back to a
          ``deepcopy`` of the pre-wrap env captured in ``__init__``.
        - For ``MASKABLE_PPO`` wraps in ``ActionMasker`` so masks survive into
          ``MaskableEvalCallback`` rollouts.
        - Always wraps in ``Monitor`` so episode rewards / lengths propagate.
        - If the trainer was constructed with ``vec_normalize=True``, wraps in
          ``DummyVecEnv`` + ``VecNormalize(training=False, norm_reward=False)``;
          SB3's ``EvalCallback`` will ``sync_envs_normalization`` from the
          training-side stats before each eval pass.
        """
        if self._env_factory is not None:
            base_env = self._env_factory()
        else:
            base_env = copy.deepcopy(self._raw_env)

        if self.model_type == ModelType.MASKABLE_PPO:
            base_env = ActionMasker(base_env, lambda e: e.action_masks())

        base_env = self._wrap_monitor(base_env)

        if self.vec_normalize:
            vec_eval = DummyVecEnv([lambda: base_env])
            vec_eval = VecNormalize(
                vec_eval, norm_obs=True, norm_reward=False, clip_obs=10.0
            )
            vec_eval.training = False
            return vec_eval

        return base_env

    def _wrap_monitor(self, env):
        return Monitor(env, self.save_dir, info_keywords=("total_score",))

    def train(self,
              max_timesteps,
              save_freq: Optional[int] = None,
              eval_freq: Optional[int] = None,
              n_eval_episodes: int = 5,
              ):
        if self.custom:
            raise ValueError("Custom training loop is required for DQN. Use `train_custom` instead.")
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
        if eval_freq is not None:
            eval_env = self._build_eval_env()
            EvalCallbackCls = (
                MaskableEvalCallback
                if self.model_type == ModelType.MASKABLE_PPO
                else EvalCallback
            )
            eval_callback = EvalCallbackCls(
                eval_env,
                best_model_save_path=self.save_dir,
                log_path=self.save_dir,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
            )
            callbacks.append(eval_callback)

        callbacks.append(TotalScoreLoggerCallback())
        self.model.learn(
            total_timesteps=max_timesteps,
            progress_bar=True,
            callback=callbacks if callbacks else None
        )
        self.model.save(os.path.join(self.save_dir, "model"))
        if isinstance(self.env, VecNormalize):
            self.env.save(os.path.join(self.save_dir, "vecnormalize.pkl"))
        
        self.cleanup_checkpoints()


    def train_custom(self,
              max_timesteps: int,
              eval_interval: int = 10000,
              num_eval_episodes: int = 10,
              save_best: bool = True,
              update_timestep: int = 4000) -> Dict:
        """Train the agent using timesteps.
        
        Args:
            max_timesteps: Maximum number of timesteps to train for
            eval_interval: Number of timesteps between evaluations
            num_eval_episodes: Number of episodes for evaluation
            save_best: Whether to save the best model
            action_std_decay_rate: Rate at which to decay action standard deviation
            min_action_std: Minimum action standard deviation
            action_std_decay_freq: Frequency of action std decay in timesteps
            update_timestep: Number of timesteps between policy updates
            
        Returns:
            Dictionary containing training history
        """
        if not self.custom:
            raise ValueError("Custom training loop is only for DQN. Use `train` for other algorithms.")
        self.model.train_mode()
        history = {
            'timesteps': [],
            'train_rewards': [],
            'eval_rewards': [],
            'episode_lengths': [],
            'best_eval_reward': float('-inf')
        }

        timestep = 0
        episode_num = 0
        last_eval_t = 0
        with tqdm(total=max_timesteps) as pbar:
            while timestep < max_timesteps:
                state, _ = self.env.reset()
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.model.device)
                episode_reward = 0
                episode_length = 0
                done = False
                truncated = False

                episode_start_idx = self.model.replay_memory.write_ptr
                episode_transitions = 0
                while not (done or truncated):
                    current_mask = torch.tensor(self.env.unwrapped.action_masks(), dtype=torch.bool).unsqueeze(0).to(self.model.device)
                    action = self.model.select_action(state, current_mask, timestep)
                    bonus_achieved = self.model.predict_bonus(state)
                    next_state, reward, done, truncated, _ = self.env.step(action)
                    next_mask = torch.tensor(self.env.unwrapped.action_masks(), dtype=torch.bool).unsqueeze(0).to(self.model.device)
                    next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.model.device) if not (done or truncated) else None
                    reward_tensor = torch.tensor([reward + bonus_achieved], dtype=torch.float32).to(self.model.device)
                    self.model.push_to_memory(
                        transition(state, torch.tensor([action]), next_state_tensor, reward_tensor, current_mask, next_mask, None)
                    )
                    episode_transitions += 1

                    if (timestep % update_timestep == 0) and (timestep > 0):
                        self.model.update()
                        target_net_state_dict = self.model.target_network.state_dict()
                        policy_net_state_dict = self.model.network.state_dict()
                        for key in policy_net_state_dict:
                            target_net_state_dict[key] = policy_net_state_dict[key] * self._tau + target_net_state_dict[key] * (1 - self._tau)
                        self.model.target_network.load_state_dict(target_net_state_dict)

                    episode_reward += reward
                    episode_length += 1
                    state = next_state_tensor
                    timestep += 1
                    pbar.update(1)
                    if timestep >= max_timesteps:
                        break

                raw_upper = sum(
                    self.env.unwrapped.game.scorecard.score_board[c]["score"]
                    for c in Category.upper_categories()
                )
                self.model.replay_memory.label_episode(
                    start_idx=episode_start_idx,
                    n_transitions=episode_transitions,
                    achieved=raw_upper >= 63,
                )
                episode_num += 1
                history['timesteps'].append(timestep)
                if done or truncated:
                    history['train_rewards'].append(episode_reward)
                    history['episode_lengths'].append(episode_length)

                if timestep >= last_eval_t + eval_interval:
                    eval_reward, eval_std = self._evaluate_dqn(num_eval_episodes)
                    history['eval_rewards'].append(eval_reward)
                    last_eval_t = timestep
                    last_fifty_train_rewards = history['train_rewards'][-50:]
                    print(f"Timestep {timestep}/{max_timesteps}")
                    print(f"Episode {episode_num}")
                    print(f"Training reward (last 50 episodes): {np.mean(last_fifty_train_rewards):.2f}")
                    print(f"Evaluation reward: {eval_reward:.2f} ± {eval_std:.2f}")
                    print(f"Epsilon: {self.model.epsilon:.4f}")

                    if save_best and eval_reward > history['best_eval_reward']:
                        history['best_eval_reward'] = eval_reward
                        self.save_agent('best_model.pt')

        self.save_agent('final_model.pt')
        self.save_history(history)
        self._plot_results_dqn(history)

        return history
    
    def _evaluate_dqn(self, num_episodes: int, capture_metrics: bool = False) -> float:
        self.model.eval_mode()
        rewards = []

        for _ in range(num_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.model.device)
            done = False
            truncated = False

            while not (done or truncated):
                mask = torch.tensor(self.env.unwrapped.action_masks(), dtype=torch.bool).unsqueeze(0).to(self.model.device)
                action = self.model.evaluate(state, mask)
                next_state, _, done, truncated, _ = self.env.step(action)
                state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.model.device)

            rewards.append(self.env.unwrapped.game.get_final_score())

        if capture_metrics:
            self.plot_evaluation_metrics(num_episodes, rewards)

        self.model.train_mode()
        return np.mean(rewards), np.std(rewards)
    
    def save_history(self, history: Dict) -> None:
        """Save the training history.
        
        Args:
            history: Dictionary containing training history
        """
        # Convert numpy arrays to lists for JSON serialization
        history_json = {
            k: v if isinstance(v, (list, float, int)) else v.tolist()
            for k, v in history.items()
        }
        
        path = os.path.join(self.save_dir, 'history.json')
        with open(path, 'w') as f:
            json.dump(history_json, f, indent=4)

    def plot_evaluation_metrics(self, num_episodes: int, reward: list[float]):
        """Plot evaluation metrics.
        
        Args:
            num_episodes: Number of episodes to evaluate
            reward: List of rewards
        """

        fig, ax = plt.subplots(1, 1, figsize=(5,5))
        ax.plot(range(num_episodes), reward)
        ax.axhline(y=np.mean(reward), color='r', linestyle='--', label=f'$\mu$ = {np.mean(reward):.2f}')   
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Evaluation Reward')
        ax.grid(alpha=0.2)
        ax.legend()
        plt.savefig(os.path.join(self.save_dir, 'evaluation_reward.png'))
        plt.close()

    def _plot_results_dqn(self, history: Dict) -> None:
        """Plot training results.

        Args:
            history: Dictionary containing training history
        """
        window_len = 200  # Rolling window length for smoothing
        
        # Plot rewards
        plt.figure(figsize=(6, 6))
        
        # Training and evaluation rewards
        plt.subplot(2, 1, 1)
        train_rewards = history['train_rewards']
        timesteps = history['timesteps'][:len(train_rewards)]  # Ensure matching lengths
        
        # Plot raw rewards with low alpha
        plt.plot(timesteps, train_rewards, alpha=0.2, color='blue', label='Raw Training Rewards')
        # Plot smoothed rewards
        smoothed_rewards = pd.Series(train_rewards).rolling(window=window_len, min_periods=1).mean()
        plt.plot(timesteps, smoothed_rewards, color='blue', linewidth=1.5, 
                label=f'Smoothed Training Rewards (window={window_len})')
        
        # Plot evaluation rewards if they exist
        if history['eval_rewards']:
            eval_timesteps = timesteps[::len(timesteps)//len(history['eval_rewards'])][:len(history['eval_rewards'])]
            plt.plot(eval_timesteps, history['eval_rewards'], color='red', linewidth=1.5, label='Evaluation Rewards')
        
        plt.xlabel('Timesteps')
        plt.ylabel('Reward')
        plt.title('Training Progress')
        plt.grid(alpha=0.2)
        plt.legend()
        
        # Plot episode lengths
        plt.subplot(2, 1, 2)
        lengths = history['episode_lengths']
        timesteps_lengths = history['timesteps'][:len(lengths)]  # Ensure matching lengths
        
        # Plot raw lengths with low alpha
        plt.plot(timesteps_lengths, lengths, alpha=0.2, color='red', label='Raw Episode Lengths')
        # Plot smoothed lengths
        smoothed_lengths = pd.Series(lengths).rolling(window=window_len, min_periods=1).mean()
        plt.plot(timesteps_lengths, smoothed_lengths, color='red', linewidth=1.5, 
                label=f'Smoothed Lengths (window={window_len})')
        plt.xlabel('Timesteps')
        plt.ylabel('Episode Length')
        plt.title('Episode Lengths')
        plt.grid(alpha=0.2)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'rewards.png'))
        plt.close()


    def cleanup_checkpoints(self):
        if os.path.exists(os.path.join(self.save_dir, "model.zip")):
            # Get all checkpoint files in the save dir
            checkpoint_files = glob.glob(os.path.join(self.save_dir, "checkpoint_*.zip"))
            if checkpoint_files:
                for checkpoint_file in checkpoint_files:
                    os.remove(checkpoint_file)
                print(f"Removed {len(checkpoint_files)} checkpoint files.")
            else:
                print("No model checkpoint files found.")
        
        if os.path.exists(os.path.join(self.save_dir, "vecnormalize.pkl")):
            vecnormalize_files = glob.glob(os.path.join(self.save_dir, "checkpoint_vecnormalize_*.pkl"))
            if vecnormalize_files:
                for vecnormalize_file in vecnormalize_files:
                    os.remove(vecnormalize_file)
                print(f"Removed {len(vecnormalize_files)} vecnormalize files.")
            else:
                print("No vecnormalize checkpointfiles found.")

    def load(self, model_path: Optional[str] = None, vecnormalize_path: Optional[str] = None,
             should_train: bool = False):
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
            self.env.training = should_train
            self.env.norm_reward = False

        # Load returns a new model instance - must assign it back and pass env for continued training
        self.model = self.model.load(path, env=self.env)
        # SB3 restores ``verbose`` from the saved pickle via ``model.__dict__.update(data)``,
        # which can silently demote a verbose=1 trainer to verbose=0 if the checkpoint was
        # saved with verbose=0 (e.g. anything chained from ``bc_pretrain.py``). Re-pin to
        # the trainer's configured verbose so SB3's logger keeps writing to stdout.
        self.model.verbose = self.verbose
        # SB3 may wrap env (e.g. DummyVecEnv); keep trainer.env aligned with model.get_env().
        self.env = self.model.get_env()
        return self.model
    
    def load_custom_dqn(self, model_path: Optional[str] = None):
        if self.model_type is not ModelType.DQN:
            raise ValueError("load_custom_dqn can only be used with DQN models.")
        path = model_path if model_path is not None else os.path.join(self.save_dir, "best_model.pt")
        
        self.model.load(path, env=self.env)
        return self.model

    def save_agent(self, filename: str):
        self.model.save(os.path.join(self.save_dir, filename))

    def evaluate(self, num_episodes: int):
        if self.custom:
            mean, std = self._evaluate_dqn(num_episodes)
            return mean, std
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


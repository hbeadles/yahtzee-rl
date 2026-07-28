"""Behavioral cloning pretraining from MarkovStrategy demonstrations."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import numpy as np
import torch as th
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from torch.utils.data import DataLoader, TensorDataset

from yahtzee_rl.config import CATEGORY_TO_ACTION, Category
from yahtzee_rl.envs.yahtzee_env import YahtzeeEnv
from yahtzee_rl.run_config import EnvConfig
from yahtzee_rl.strategies.markov import MarkovStrategy
from yahtzee_rl.train.train_baselines import ModelType, TrainerBaselines


def make_yahtzee_env(env_config: EnvConfig) -> YahtzeeEnv:
    return YahtzeeEnv(
        lambda_upper=env_config.lambda_upper,
        lambda_yahtzee=env_config.lambda_yahtzee,
        use_expecteds=env_config.use_expecteds,
        use_probabilities=env_config.use_probabilities,
        invalid_action_substitute=env_config.invalid_action_substitute,
        invalid_action_penalty=env_config.invalid_action_penalty,
        s_ref=env_config.s_ref,
        reward_exponent=env_config.reward_exponent,
    )


def _normalize_markov_action(action: Union[int, str, Category]) -> int:
    if isinstance(action, Category):
        return CATEGORY_TO_ACTION[action]
    if isinstance(action, str):
        return CATEGORY_TO_ACTION[Category(action)]
    return int(action)


def collect_markov_demos(
    env_config: EnvConfig,
    num_episodes: int,
    output_path: Union[str, Path],
    seed: int | None = None
) -> Path:
    """Roll out MarkovStrategy and save (obs, action, mask) transitions."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = make_yahtzee_env(env_config)
    strategy = MarkovStrategy(env)

    obs_buf: list[np.ndarray] = []
    act_buf: list[int] = []
    mask_buf: list[np.ndarray] = []
    rew_buf: list[float] = []
    start_buf: list[bool] = []
    final_scores: list[int] = []

    for ep in range(num_episodes):
        ep_seed = None if seed is None else seed + ep
        obs, _ = env.reset(seed=ep_seed)
        done = False
        is_first_step = True
        while not done:
            action_masks = env.action_masks().copy()
            raw_action = strategy.strategy(obs, env.game.scorecard)
            action = _normalize_markov_action(raw_action)

            obs_buf.append(obs.copy())
            act_buf.append(action)
            mask_buf.append(action_masks)
            start_buf.append(is_first_step)
            is_first_step = False

            obs, reward, done, _, _ = env.step(action)
            rew_buf.append(float(reward))

        final_scores.append(env.game.get_final_score())

    np.savez(
        output_path,
        obs=np.array(obs_buf, dtype=np.float32),
        acts=np.array(act_buf, dtype=np.int64),
        masks=np.array(mask_buf, dtype=bool),
        rewards=np.array(rew_buf, dtype=np.float32),
        episode_starts=np.array(start_buf, dtype=bool),
    )

    mean_score = float(np.mean(final_scores)) if final_scores else 0.0
    print(
        f"Collected {len(act_buf)} transitions from {num_episodes} episodes "
        f"(mean Markov final score: {mean_score:.1f})"
    )
    print(f"Saved demos to {output_path}")
    return output_path


def _build_vec_env(env_config: EnvConfig, save_dir: str, vec_normalize: bool) -> VecNormalize | DummyVecEnv:
    def make_env():
        env = make_yahtzee_env(env_config)
        env = ActionMasker(env, lambda e: e.action_masks())
        return Monitor(env, save_dir)

    vec_env = DummyVecEnv([make_env])
    if vec_normalize:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return vec_env


def _compute_returns_to_go(
    rewards: np.ndarray,
    episode_starts: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Compute discounted returns-to-go aligned with the flat (obs, reward) buffers.

    ``rewards[i]`` is the reward received after the action taken at step ``i``.
    ``episode_starts[i]`` is True iff step ``i`` is the first step of its episode.
    Discounting resets at every episode boundary.
    """
    n = rewards.shape[0]
    returns = np.zeros(n, dtype=np.float32)
    running = 0.0
    for i in range(n - 1, -1, -1):
        if i + 1 < n and episode_starts[i + 1]:
            running = 0.0
        running = float(rewards[i]) + gamma * running
        returns[i] = running
    return returns


def train_bc(
    demos_path: Union[str, Path],
    env_config: EnvConfig,
    save_dir: Union[str, Path],
    policy_net_arch: dict,
    n_epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    vec_normalize: bool = True,
    gamma: float = 0.99,
    value_epochs: int = 10,
    value_learning_rate: float = 1e-3,
) -> Path:
    """Train MaskablePPO policy weights via behavioral cloning on saved demos.

    Runs two sequential fits:
      1. Policy fit: max-likelihood on (obs, action) with mask-aware log-probs.
      2. Value fit: MSE between ``policy.predict_values(obs)`` and Monte-Carlo
         returns-to-go computed from the demos. Skipped if the demos file does
         not contain ``rewards`` / ``episode_starts`` (older datasets).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    vec_env = _build_vec_env(env_config, str(save_dir), vec_normalize)
    policy_kwargs = dict(net_arch=policy_net_arch)
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        verbose=0,
    )

    data = np.load(demos_path)
    obs_raw = data["obs"]
    acts = data["acts"]
    masks = data["masks"]
    rewards = data["rewards"] if "rewards" in data.files else None
    episode_starts = data["episode_starts"] if "episode_starts" in data.files else None

    if vec_normalize:
        for obs in obs_raw:
            vec_env.obs_rms.update(obs[None, :])
        obs = vec_env.normalize_obs(obs_raw)
    else:
        obs = obs_raw

    obs_t = th.from_numpy(obs).float()
    acts_t = th.from_numpy(acts).long()
    masks_t = th.from_numpy(masks).bool()

    dataset = TensorDataset(obs_t, acts_t, masks_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optim = th.optim.Adam(model.policy.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_samples = 0
        for obs_b, acts_b, masks_b in loader:
            _, log_prob, _ = model.policy.evaluate_actions(
                obs_b, acts_b, action_masks=masks_b
            )
            loss = -log_prob.mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            batch_n = obs_b.shape[0]
            epoch_loss += loss.item() * batch_n
            n_samples += batch_n
        print(f"BC epoch {epoch + 1}/{n_epochs} loss={epoch_loss / n_samples:.4f}")

    if rewards is None or episode_starts is None:
        print(
            "Demos file missing 'rewards' / 'episode_starts' — skipping value-head fit. "
            "Re-run collect-markov to enable VF pretraining."
        )
    elif value_epochs > 0:
        returns = _compute_returns_to_go(rewards, episode_starts, gamma)
        returns_t = th.from_numpy(returns).float()
        vf_dataset = TensorDataset(obs_t, returns_t)
        vf_loader = DataLoader(vf_dataset, batch_size=batch_size, shuffle=True)
        vf_optim = th.optim.Adam(model.policy.parameters(), lr=value_learning_rate)

        for epoch in range(value_epochs):
            epoch_loss = 0.0
            n_samples = 0
            for obs_b, ret_b in vf_loader:
                v = model.policy.predict_values(obs_b).squeeze(-1)
                vf_loss = ((v - ret_b) ** 2).mean()
                vf_optim.zero_grad()
                vf_loss.backward()
                vf_optim.step()
                batch_n = obs_b.shape[0]
                epoch_loss += vf_loss.item() * batch_n
                n_samples += batch_n
            print(f"VF epoch {epoch + 1}/{value_epochs} mse={epoch_loss / n_samples:.4f}")

    model_path = save_dir / "model"
    model.save(str(model_path))
    if vec_normalize:
        vec_env.save(str(save_dir / "vecnormalize.pkl"))

    print(f"Saved BC model to {model_path}.zip")
    return save_dir


def evaluate_bc_checkpoint(
    model_path: Union[str, Path],
    vecnormalize_path: Union[str, Path] | None,
    env_config: EnvConfig,
    policy_net_arch: dict,
    num_episodes: int = 10,
    vec_normalize: bool = True,
) -> float:
    """Evaluate a BC checkpoint; returns mean episode score (sum of delta rewards)."""
    env = make_yahtzee_env(env_config)
    trainer = TrainerBaselines(
        ModelType.MASKABLE_PPO,
        env,
        exp_name="bc_eval",
        policy_kwargs=dict(net_arch=policy_net_arch),
        vec_normalize=vec_normalize,
        verbose=0,
    )

    vec_path = str(vecnormalize_path) if vecnormalize_path is not None else None
    trainer.load(model_path=str(model_path), vecnormalize_path=vec_path)

    scores: list[float] = []
    for _ in range(num_episodes):
        obs = trainer.env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action_masks = trainer.env.get_attr("action_masks")[0]()
            action, _ = trainer.model.predict(
                obs, action_masks=action_masks, deterministic=True
            )
            obs, reward, done, _ = trainer.env.step(action)
            episode_reward += float(reward[0])
        scores.append(episode_reward)

    mean_score = float(np.mean(scores))
    print(f"BC eval mean score over {num_episodes} episodes: {mean_score:.1f}")
    return mean_score

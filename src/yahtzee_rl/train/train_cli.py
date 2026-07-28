from datetime import datetime
from pathlib import Path

import typer
from rich import print
from typing import Annotated, Optional
from yahtzee_rl.train.train_baselines import TrainerBaselines, ModelType, YahtzeeTrunk
from yahtzee_rl.envs.yahtzee_env import YahtzeeEnv
from yahtzee_rl.paths import artifact_dir
from yahtzee_rl.train.bc_pretrain import (
    collect_markov_demos,
    evaluate_bc_checkpoint,
    train_bc,
)
from yahtzee_rl.run_config import (
    A2CRunConfig,
    BCRunConfig,
    CollectMarkovRunConfig,
    DQNRunConfig,
    EnvConfig,
    PPORunConfig,
    save_run_config,
)

app = typer.Typer()


def _default_policy_net_arch() -> dict:
    return {"pi": [256, 256], "vf": [256, 256]}


@app.command("collect-markov")
def collect_markov(
    experiment_name: Annotated[str, typer.Argument(help="Name for this demo dataset")],
    num_episodes: Annotated[int, typer.Option(help="Number of Markov episodes to collect")] = 20_000,
    output: Annotated[Optional[Path], typer.Option(help="Output .npz path")] = None,
    seed: Annotated[Optional[int], typer.Option(help="Base RNG seed")] = None,
    use_probabilities: Annotated[bool, typer.Option(help="Whether to use probabilities")] = True,
):
    """Collect MarkovStrategy demonstrations for behavioral cloning.

    Note: ``use_expecteds`` is intentionally not exposed here. ``MarkovStrategy``
    reads category expected scores out of the observation vector to pick a
    target during the roll phase, so disabling them would make the expert
    unable to act.
    """
    env_config = EnvConfig(
        lambda_upper=0.05,
        lambda_yahtzee=0.2,
        use_expecteds=True,
        use_probabilities=use_probabilities,
    )
    dataset_dir = artifact_dir() / experiment_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_path = output if output is not None else dataset_dir / "demos.npz"

    run_config = CollectMarkovRunConfig(
        experiment_name=experiment_name,
        created_at=datetime.now().isoformat(timespec="seconds"),
        num_episodes=num_episodes,
        output_path=str(output_path),
        seed=seed,
        env=env_config,
    )
    save_run_config(run_config, dataset_dir)
    print(run_config.model_dump())

    collect_markov_demos(
        env_config=env_config,
        num_episodes=num_episodes,
        output_path=output_path,
        seed=seed
    )


@app.command()
def bc(
    experiment_name: Annotated[str, typer.Argument(help="The name of the BC experiment")],
    demos: Annotated[Path, typer.Option(help="Path to demos.npz from collect-markov")],
    n_epochs: Annotated[int, typer.Option(help="BC training epochs")] = 10,
    batch_size: Annotated[int, typer.Option(help="BC minibatch size")] = 256,
    learning_rate: Annotated[float, typer.Option(help="BC learning rate")] = 1e-3,
    eval_episodes: Annotated[int, typer.Option(help="Episodes for post-BC sanity eval")] = 10,
    use_expecteds: Annotated[bool, typer.Option(help="Whether to use expecteds")] = True,
    use_probabilities: Annotated[bool, typer.Option(help="Whether to use probabilities")] = True,
    vec_normalize: Annotated[bool, typer.Option(help="Whether to use vector normalization")] = True,
    gamma: Annotated[float, typer.Option(help="Discount factor for value-head returns-to-go targets")] = 0.99,
    value_epochs: Annotated[int, typer.Option(help="Epochs of value-head MSE fit after BC (0 = skip)")] = 10,
    value_learning_rate: Annotated[float, typer.Option(help="Learning rate for the value-head fit")] = 1e-3,
):
    """Train a MaskablePPO policy via behavioral cloning on Markov demos."""
    env_config = EnvConfig(
        lambda_upper=0.05,
        lambda_yahtzee=0.2,
        use_expecteds=use_expecteds,
        use_probabilities=use_probabilities,
    )
    policy_net_arch = _default_policy_net_arch()
    save_dir = artifact_dir() / experiment_name / datetime.now().strftime("%Y-%m-%d")

    run_config = BCRunConfig(
        experiment_name=experiment_name,
        created_at=datetime.now().isoformat(timespec="seconds"),
        demos_path=str(demos),
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        vec_normalize=vec_normalize,
        policy_net_arch=policy_net_arch,
        env=env_config,
        gamma=gamma,
        value_epochs=value_epochs,
        value_learning_rate=value_learning_rate,
    )
    save_run_config(run_config, save_dir)
    print(run_config.model_dump())

    train_bc(
        demos_path=demos,
        env_config=env_config,
        save_dir=save_dir,
        policy_net_arch=policy_net_arch,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        vec_normalize=vec_normalize,
        gamma=gamma,
        value_epochs=value_epochs,
        value_learning_rate=value_learning_rate,
    )

    vec_path = save_dir / "vecnormalize.pkl" if vec_normalize else None
    evaluate_bc_checkpoint(
        model_path=save_dir / "model.zip",
        vecnormalize_path=vec_path,
        env_config=env_config,
        policy_net_arch=policy_net_arch,
        num_episodes=eval_episodes,
        vec_normalize=vec_normalize,
    )


@app.command()
def ppo(experiment_name: Annotated[str, typer.Argument(help="The name of the experiment")],
              max_timesteps: Annotated[float, typer.Argument(help="The maximum number of timesteps")] = 30e6,
              save_freq: Annotated[int, typer.Option(help="The frequency of saving the model (in timesteps)")] = 100000,
              eval_freq: Annotated[int, typer.Option(help="Mid-training eval cadence in env steps; 0 to disable. Defaults to save_freq.")] = -1,
              n_eval_episodes: Annotated[int, typer.Option(help="Episodes per mid-training eval pass")] = 5,
              resume_from: Annotated[Optional[Path], typer.Option(help="BC or checkpoint dir with model.zip")] = None,
              use_expecteds: Annotated[bool, typer.Option(help="Whether to use expecteds")] = True,
              use_probabilities: Annotated[bool, typer.Option(help="Whether to use probabilities")] = True,
              batch_size: Annotated[int, typer.Option(help="The batch size")] = 96,
              n_steps: Annotated[int, typer.Option(help="The number of steps")] = 512,
              gamma: Annotated[float, typer.Option(help="The discount factor")] = 0.99,
              n_epochs: Annotated[int, typer.Option(help="The number of epochs")] = 5,
              ent_coef: Annotated[float, typer.Option(help="The entropy coefficient")] = 0.02,
              vec_normalize: Annotated[bool, typer.Option(help="Whether to use vector normalization")] = True,
              clip_range: Annotated[float, typer.Option(help="The clip range")] = 0.1,
              gae_lambda_initial: Annotated[float, typer.Option(help="The initial GAE lambda")] = 0.2,
              gae_lambda_final: Annotated[float, typer.Option(help="The GAE lambda")] = 0.95,
              normalize_advantage: Annotated[bool, typer.Option(help="Whether to normalize the advantage")] = False,
              target_kl: Annotated[Optional[float], typer.Option(help="Early-stop PPO epoch loop when approx KL exceeds this. Recommended ~0.02 for BC-resume runs; leave unset for from-scratch.")] = None,
              s_ref: Annotated[float, typer.Option(help="Reference score for reward normalization")] = 150.0,
              reward_exponent: Annotated[float, typer.Option(help="Exponent applied to normalized score in the reward shaping function")] = 6.0):
    """Train a Maskable PPO agent on the Yahtzee environment.

    Args:
        experiment_name: Name used as the artifact subdirectory under ``artifacts/``.
        max_timesteps: Total environment steps to train for.
        save_freq: Checkpoint save frequency, in environment steps.
        use_expecteds: If True, include per-category expected-score features in the observation.
        use_probabilities: If True, include per-category probability features in the observation.
        batch_size: Minibatch size used for each PPO update.
        n_steps: Number of environment steps collected per rollout before an update.
        gamma: Discount factor applied to future rewards.
        n_epochs: Number of optimization epochs per rollout.
        ent_coef: Entropy regularization coefficient on the policy loss.
        vec_normalize: If True, wrap the environment in ``VecNormalize`` to normalize observations.
        clip_range: PPO surrogate-objective clipping range.
        gae_lambda_initial: Starting value for the linearly-scheduled GAE lambda.
        gae_lambda_final: Ending value for the linearly-scheduled GAE lambda.
        normalize_advantage: If True, normalize advantages within each minibatch.
        s_ref: Reference score used to normalize the reward signal.
        reward_exponent: Exponent applied to the normalized score in the reward shaping function.

    Side Effects:
        Writes checkpoints, the final model, VecNormalize stats (if enabled), and
        a learning-curve plot under ``artifacts/<experiment_name>/<date>/``. Prints
        the mean and standard deviation of episode rewards over a 10-episode
        evaluation pass.
    """

    print("Training PPO with the following parameters")
    env_config = EnvConfig(
        lambda_upper=0.05,
        lambda_yahtzee=0.2,
        use_expecteds=use_expecteds,
        use_probabilities=use_probabilities,
        s_ref=s_ref,
        reward_exponent=reward_exponent,
    )
    resolved_eval_freq: Optional[int] = save_freq if eval_freq < 0 else (eval_freq or None)
    run_config = PPORunConfig(

        features_extractor_class=YahtzeeTrunk,
        features_extractor_kwargs=dict(features_dim=356, n_layers=3, dropout=0.1),
        experiment_name=experiment_name,
        created_at=datetime.now().isoformat(timespec="seconds"),
        max_timesteps=max_timesteps,
        save_freq=save_freq,
        eval_freq=resolved_eval_freq,
        n_eval_episodes=n_eval_episodes,
        policy_net_arch={"pi": [256, 256], "vf": [256, 256]},
        env=env_config,
        batch_size=batch_size,
        n_steps=n_steps,
        gamma=gamma,
        n_epochs=n_epochs,
        ent_coef=ent_coef,
        vec_normalize=vec_normalize,
        clip_range=clip_range,
        gae_lambda_initial=gae_lambda_initial,
        gae_lambda_final=gae_lambda_final,
        normalize_advantage=normalize_advantage,
        target_kl=target_kl,
    )
    print(run_config.model_dump())
    def env_factory() -> YahtzeeEnv:
        return YahtzeeEnv(
            lambda_upper=env_config.lambda_upper,
            lambda_yahtzee=env_config.lambda_yahtzee,
            use_expecteds=env_config.use_expecteds,
            use_probabilities=env_config.use_probabilities,
            s_ref=env_config.s_ref,
            reward_exponent=env_config.reward_exponent,
        )
    env = env_factory()
    policy_kwargs = dict(net_arch=run_config.policy_net_arch,
                                    #features_extractor_class=run_config.features_extractor_class, 
                                    #features_extractor_kwargs=run_config.features_extractor_kwargs,
                                    share_features_extractor=True)
    trainer = TrainerBaselines(ModelType.MASKABLE_PPO,
                               env, experiment_name, batch_size=batch_size, n_steps=n_steps,
                               gamma=gamma, n_epochs=n_epochs, policy_kwargs=policy_kwargs, ent_coef=ent_coef,
                               vec_normalize=vec_normalize, clip_range=clip_range, gae_lambda=(gae_lambda_initial, gae_lambda_final), normalize_advantage=normalize_advantage,
                               target_kl=target_kl, env_factory=env_factory)
    save_run_config(run_config, trainer.save_dir)
    if resume_from is not None:
        resume_path = Path(resume_from)
        vec_path = resume_path / "vecnormalize.pkl"
        trainer.load(
            model_path=str(resume_path / "model.zip"),
            vecnormalize_path=str(vec_path) if vec_normalize and vec_path.exists() else None,
            should_train=True,
        )
        print(f"Resumed from {resume_path}")
    trainer.train(
        max_timesteps=max_timesteps,
        save_freq=save_freq,
        eval_freq=resolved_eval_freq,
        n_eval_episodes=n_eval_episodes,
    )
    mean_reward, std_reward = trainer.evaluate(num_episodes=10)
    print(f"Mean reward: {mean_reward:.2f}, Std reward: {std_reward:.2f}")
    trainer.plot_results(max_timesteps=max_timesteps)


@app.command()
def dqn(experiment_name: Annotated[str, typer.Argument(help="The name of the experiment")],
              max_timesteps: Annotated[float, typer.Argument(help="The maximum number of timesteps")] = 30e6,
              save_freq: Annotated[int, typer.Option(help="The frequency of saving the model (in timesteps)")] = 100000,
              eval_freq: Annotated[int, typer.Option(help="Mid-training eval cadence in env steps; 0 to disable. Defaults to save_freq.")] = -1,
              n_eval_episodes: Annotated[int, typer.Option(help="Episodes per mid-training eval pass")] = 30,
              use_expecteds: Annotated[bool, typer.Option(help="Whether to use expecteds")] = True,
              use_probabilities: Annotated[bool, typer.Option(help="Whether to use probabilities")] = True,
              invalid_action_substitute: Annotated[bool, typer.Option(help="Whether to substitute a valid action when the agent picks an invalid one")] = True,
              invalid_action_penalty: Annotated[float, typer.Option(help="Reward penalty applied when an invalid action is substituted")] = -20.0,
              hidden_dim: Annotated[int, typer.Option(help="Hidden layer size for the DQN network")] = 128,
              learning_rate: Annotated[float, typer.Option(help="Learning rate")] = 1e-3,
              buffer_size: Annotated[int, typer.Option(help="Replay buffer size")] = 500_000,
              batch_size: Annotated[int, typer.Option(help="The batch size")] = 64,
              gamma: Annotated[float, typer.Option(help="The discount factor")] = 0.99,
              epsilon_start: Annotated[float, typer.Option(help="Starting epsilon for epsilon-greedy exploration")] = 1.0,
              epsilon_end: Annotated[float, typer.Option(help="Final epsilon for epsilon-greedy exploration")] = 0.01,
              exploration_fraction: Annotated[float, typer.Option(help="Fraction of training over which epsilon is annealed")] = 0.2,
              target_update_freq: Annotated[int, typer.Option(help="Steps between target network hard updates")] = 100,
              update_timestep: Annotated[int, typer.Option(help="Perform a training update every N environment steps")] = 8,
              aux_lambda: Annotated[float, typer.Option(help="Auxiliary loss weight for the value head")] = 0.2,
              tau: Annotated[float, typer.Option(help="Target network soft-update coefficient (1.0 = hard update)")] = 1.0,
              s_ref: Annotated[float, typer.Option(help="Reference score for reward normalization")] = 150.0,
              reward_exponent: Annotated[float, typer.Option(help="Exponent applied to normalized score in the reward shaping function")] = 6.0):
    """Train a DQN agent on the Yahtzee environment.

    Args:
        experiment_name: Name used as the artifact subdirectory under ``artifacts/``.
        max_timesteps: Total environment steps to train for.
        save_freq: Checkpoint save frequency, in environment steps.
        use_expecteds: If True, include per-category expected-score features in the observation.
        use_probabilities: If True, include per-category probability features in the observation.
        invalid_action_substitute: If True, the environment substitutes a valid action
            when the agent selects an invalid one (DQN has no action masking).
        invalid_action_penalty: Reward penalty applied when an invalid action is substituted.
        hidden_dim: Hidden layer width for the Q-network.
        learning_rate: Adam learning rate.
        buffer_size: Replay buffer capacity.
        batch_size: Minibatch size sampled from the replay buffer.
        gamma: Discount factor applied to future rewards.
        epsilon_start: Initial epsilon for epsilon-greedy exploration.
        epsilon_end: Final epsilon for epsilon-greedy exploration.
        exploration_fraction: Fraction of training over which epsilon is annealed (computes epsilon_decay).
        target_update_freq: Steps between target-network hard-copy updates.
        update_timestep: Perform a gradient update every N environment steps.
        tau: Soft-update coefficient for the target network (1.0 = hard copy).
        s_ref: Reference score used to normalize the reward signal.
        reward_exponent: Exponent applied to the normalized score in the reward shaping function.

    Side Effects:
        Writes checkpoints, the final model, and a learning-curve plot under
        ``artifacts/<experiment_name>/<date>/``. Prints the mean and standard
        deviation of episode rewards over a 10-episode evaluation pass.
    """

    print("Training DQN with the following parameters")
    env_config = EnvConfig(
        lambda_upper=0.05,
        lambda_yahtzee=0.2,
        use_expecteds=use_expecteds,
        use_probabilities=use_probabilities,
        invalid_action_substitute=invalid_action_substitute,
        invalid_action_penalty=invalid_action_penalty,
        s_ref=s_ref,
        reward_exponent=reward_exponent,
    )
    resolved_eval_freq: Optional[int] = save_freq if eval_freq < 0 else (eval_freq or None)
    epsilon_decay = int(max_timesteps * exploration_fraction)
    run_config = DQNRunConfig(
        experiment_name=experiment_name,
        created_at=datetime.now().isoformat(timespec="seconds"),
        max_timesteps=max_timesteps,
        save_freq=save_freq,
        eval_freq=resolved_eval_freq,
        n_eval_episodes=n_eval_episodes,
        policy_net_arch={"q": [hidden_dim, hidden_dim]},
        env=env_config,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        target_update_freq=target_update_freq,
        tau=tau,
        update_timestep=update_timestep,
        aux_lambda=aux_lambda
    )
    print(run_config.model_dump())
    def env_factory() -> YahtzeeEnv:
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
    env = env_factory()
    trainer = TrainerBaselines(
        ModelType.DQN,
        env, experiment_name,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        target_update_freq=target_update_freq,
        tau=tau,
        env_factory=env_factory,
    )
    save_run_config(run_config, trainer.save_dir)
    trainer.train_custom(
        max_timesteps=int(max_timesteps),
        eval_interval=resolved_eval_freq if resolved_eval_freq is not None else int(max_timesteps) + 1,
        num_eval_episodes=n_eval_episodes,
        update_timestep=update_timestep,
    )
    mean_reward, std_reward = trainer.evaluate(num_episodes=10)
    print(f"Mean reward: {mean_reward:.2f}, Std reward: {std_reward:.2f}")
    trainer.plot_results(max_timesteps=max_timesteps)


@app.command()
def a2c(experiment_name: Annotated[str, typer.Argument(help="The name of the experiment")],
              max_timesteps: Annotated[float, typer.Argument(help="The maximum number of timesteps")] = 30e6,
              save_freq: Annotated[int, typer.Option(help="The frequency of saving the model (in timesteps)")] = 100000,
              eval_freq: Annotated[int, typer.Option(help="Mid-training eval cadence in env steps; 0 to disable. Defaults to save_freq.")] = -1,
              n_eval_episodes: Annotated[int, typer.Option(help="Episodes per mid-training eval pass")] = 5,
              resume_from: Annotated[Optional[Path], typer.Option(help="BC or checkpoint dir with model.zip")] = None,
              use_expecteds: Annotated[bool, typer.Option(help="Whether to use expecteds")] = False,
              use_probabilities: Annotated[bool, typer.Option(help="Whether to use probabilities")] = True,
              invalid_action_penalty: Annotated[float, typer.Option(help="Reward penalty applied when an invalid action is substituted")] = -20.0,
              n_steps: Annotated[int, typer.Option(help="The number of steps")] = 2512,
              gamma: Annotated[float, typer.Option(help="The discount factor")] = 0.99,
              ent_coef: Annotated[float, typer.Option(help="The entropy coefficient")] = 0.02,
              vec_normalize: Annotated[bool, typer.Option(help="Whether to use vector normalization")] = True,
              gae_lambda_initial: Annotated[float, typer.Option(help="The initial GAE lambda")] = 0.3,
              gae_lambda_final: Annotated[float, typer.Option(help="The GAE lambda")] = 0.95,
              normalize_advantage: Annotated[bool, typer.Option(help="Whether to normalize the advantage")] = False,
              s_ref: Annotated[float, typer.Option(help="Reference score for reward normalization")] = 150.0,
              reward_exponent: Annotated[float, typer.Option(help="Exponent applied to normalized score in the reward shaping function")] = 6.0):
    """Train an A2C agent on the Yahtzee environment.

    Args:
        experiment_name: Name used as the artifact subdirectory under ``artifacts/``.
        max_timesteps: Total environment steps to train for.
        save_freq: Checkpoint save frequency, in environment steps.
        use_expecteds: If True, include per-category expected-score features in the observation.
        use_probabilities: If True, include per-category probability features in the observation.
        invalid_action_penalty: Reward penalty applied when an invalid action is substituted.
        n_steps: Number of environment steps collected per rollout before an update.
        gamma: Discount factor applied to future rewards.
        ent_coef: Entropy regularization coefficient on the policy loss.
        vec_normalize: If True, wrap the environment in ``VecNormalize`` to normalize observations.
        gae_lambda_initial: Starting value for the linearly-scheduled GAE lambda.
        gae_lambda_final: Ending value for the linearly-scheduled GAE lambda.
        normalize_advantage: If True, normalize advantages within each minibatch.
        s_ref: Reference score used to normalize the reward signal.
        reward_exponent: Exponent applied to the normalized score in the reward shaping function.

    Side Effects:
        Writes checkpoints, the final model, VecNormalize stats (if enabled), and
        a learning-curve plot under ``artifacts/<experiment_name>/<date>/``. Prints
        the mean and standard deviation of episode rewards over a 10-episode
        evaluation pass.
    """

    print("Training A2C with the following parameters")
    env_config = EnvConfig(
        lambda_upper=0.05,
        lambda_yahtzee=0.2,
        use_expecteds=use_expecteds,
        use_probabilities=use_probabilities,
        invalid_action_substitute=True,
        invalid_action_penalty=invalid_action_penalty,
        s_ref=s_ref,
        reward_exponent=reward_exponent,
    )
    resolved_eval_freq: Optional[int] = save_freq if eval_freq < 0 else (eval_freq or None)
    run_config = A2CRunConfig(
        experiment_name=experiment_name,
        created_at=datetime.now().isoformat(timespec="seconds"),
        max_timesteps=max_timesteps,
        save_freq=save_freq,
        eval_freq=resolved_eval_freq,
        n_eval_episodes=n_eval_episodes,
        policy_net_arch={"pi": [128, 128], "vf": [128, 128]},
        env=env_config,
        n_steps=n_steps,
        gamma=gamma,
        ent_coef=ent_coef,
        vec_normalize=vec_normalize,
        gae_lambda_initial=gae_lambda_initial,
        gae_lambda_final=gae_lambda_final,
        normalize_advantage=normalize_advantage,
    )
    print(run_config.model_dump())
    def env_factory() -> YahtzeeEnv:
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
    env = env_factory()
    policy_kwargs = dict(net_arch=run_config.policy_net_arch)

    trainer = TrainerBaselines(
        ModelType.A2C,
        env, experiment_name,
        n_steps=n_steps,
        gamma=gamma,
        policy_kwargs=policy_kwargs,
        ent_coef=ent_coef,
        vec_normalize=vec_normalize,
        gae_lambda=(gae_lambda_initial, gae_lambda_final),
        normalize_advantage=normalize_advantage,
        env_factory=env_factory,
    )
    save_run_config(run_config, trainer.save_dir)
    if resume_from is not None:
        resume_path = Path(resume_from)
        vec_path = resume_path / "vecnormalize.pkl"
        trainer.load(
            model_path=str(resume_path / "model.zip"),
            vecnormalize_path=str(vec_path) if vec_normalize and vec_path.exists() else None,
            should_train=True,
        )
        print(f"Resumed from {resume_path}")
    trainer.train(
        max_timesteps=max_timesteps,
        save_freq=save_freq,
        eval_freq=resolved_eval_freq,
        n_eval_episodes=n_eval_episodes,
    )
    mean_reward, std_reward = trainer.evaluate(num_episodes=10)
    print(f"Mean reward: {mean_reward:.2f}, Std reward: {std_reward:.2f}")
    trainer.plot_results(max_timesteps=max_timesteps)


if __name__ == "__main__":
    app()

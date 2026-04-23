import typer
from rich import print
from typing import Annotated
from yahtzee_rl.train.train_baselines import TrainerBaselines, ModelType
from yahtzee_rl.envs.yahtzee_env import YahtzeeEnv

app = typer.Typer()


@app.command()
def ppo(experiment_name: Annotated[str, typer.Argument(help="The name of the experiment")],
              max_timesteps: Annotated[float, typer.Argument(help="The maximum number of timesteps")] = 30e6,
              save_freq: Annotated[int, typer.Argument(help="The frequency of saving the model (in timesteps)")] = 100000,
              use_expecteds: Annotated[bool, typer.Option(help="Whether to use expecteds")] = True,
              use_probabilities: Annotated[bool, typer.Option(help="Whether to use probabilities")] = True,
              batch_size: Annotated[int, typer.Argument(help="The batch size")] = 128,
              n_steps: Annotated[int, typer.Argument(help="The number of steps")] = 2512,
              gamma: Annotated[float, typer.Argument(help="The discount factor")] = 0.99,
              n_epochs: Annotated[int, typer.Argument(help="The number of epochs")] = 8,
              ent_coef: Annotated[float, typer.Argument(help="The entropy coefficient")] = 0.02,
              vec_normalize: Annotated[bool, typer.Option(help="Whether to use vector normalization")] = True,
              clip_range: Annotated[float, typer.Argument(help="The clip range")] = 0.1,
              gae_lambda_initial: Annotated[float, typer.Argument(help="The initial GAE lambda")] = 0.2,
              gae_lambda_final: Annotated[float, typer.Argument(help="The GAE lambda")] = 0.95,
              normalize_advantage: Annotated[bool, typer.Option(help="Whether to normalize the advantage")] = False):
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

    Side Effects:
        Writes checkpoints, the final model, VecNormalize stats (if enabled), and
        a learning-curve plot under ``artifacts/<experiment_name>/<date>/``. Prints
        the mean and standard deviation of episode rewards over a 10-episode
        evaluation pass.
    """

    print("Training PPO with the following parameters")
    output_params = {
        "experiment_name": experiment_name,
        "use_expecteds": use_expecteds,
        "max_timesteps": max_timesteps,
        "save_freq": save_freq,
        "use_probabilities": use_probabilities,
        "batch_size": batch_size,
        "n_steps": n_steps,
        "gamma": gamma,
        "n_epochs": n_epochs,
        "ent_coef": ent_coef,
        "vec_normalize": vec_normalize,
        "gae_lambda": (gae_lambda_initial, gae_lambda_final),
        "normalize_advantage": normalize_advantage,
        "clip_range": clip_range,
    }
    print(output_params)
    env = YahtzeeEnv(lambda_upper=0.05, lambda_yahtzee=0.2, use_expecteds=use_expecteds, use_probabilities=use_probabilities)
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    trainer = TrainerBaselines(ModelType.MASKABLE_PPO,
                               env, experiment_name, batch_size=batch_size, n_steps=n_steps,
                               gamma=gamma, n_epochs=n_epochs, policy_kwargs=policy_kwargs, ent_coef=ent_coef,
                               vec_normalize=vec_normalize, clip_range=clip_range, gae_lambda=(gae_lambda_initial, gae_lambda_final), normalize_advantage=normalize_advantage)
    trainer.train(max_timesteps=max_timesteps, save_freq=save_freq)
    mean_reward, std_reward = trainer.evaluate(num_episodes=10)
    print(f"Mean reward: {mean_reward:.2f}, Std reward: {std_reward:.2f}")
    trainer.plot_results(max_timesteps=max_timesteps)


@app.command()
def dqn(experiment_name: Annotated[str, typer.Argument(help="The name of the experiment")],
              max_timesteps: Annotated[float, typer.Argument(help="The maximum number of timesteps")] = 30e6,
              save_freq: Annotated[int, typer.Argument(help="The frequency of saving the model (in timesteps)")] = 100000,
              use_expecteds: Annotated[bool, typer.Option(help="Whether to use expecteds")] = True,
              use_probabilities: Annotated[bool, typer.Option(help="Whether to use probabilities")] = True,
              invalid_action_substitute: Annotated[bool, typer.Option(help="Whether to substitute a valid action when the agent picks an invalid one")] = True,
              invalid_action_penalty: Annotated[float, typer.Argument(help="Reward penalty applied when an invalid action is substituted")] = -20.0,
              buffer_size: Annotated[int, typer.Argument(help="Replay buffer size")] = 1_000_000,
              learning_starts: Annotated[int, typer.Argument(help="Steps collected before learning begins")] = 10_000,
              batch_size: Annotated[int, typer.Argument(help="The batch size")] = 40,
              gamma: Annotated[float, typer.Argument(help="The discount factor")] = 0.99,
              train_freq: Annotated[int, typer.Argument(help="Train the model every N environment steps")] = 4,
              gradient_steps: Annotated[int, typer.Argument(help="Gradient steps per training update")] = 1,
              exploration_fraction: Annotated[float, typer.Argument(help="Fraction of training over which epsilon is annealed")] = 0.2,
              tau: Annotated[float, typer.Argument(help="Target network soft-update coefficient (1.0 = hard update)")] = 1.0):
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
        buffer_size: Replay buffer capacity.
        learning_starts: Steps collected before the first gradient update.
        batch_size: Minibatch size sampled from the replay buffer.
        gamma: Discount factor applied to future rewards.
        train_freq: Perform a training update every ``train_freq`` environment steps.
        gradient_steps: Number of gradient steps per training update.
        exploration_fraction: Fraction of training over which the epsilon-greedy rate is annealed.
        tau: Soft-update coefficient for the target network (1.0 = hard copy).

    Side Effects:
        Writes checkpoints, the final model, and a learning-curve plot under
        ``artifacts/<experiment_name>/<date>/``. Prints the mean and standard
        deviation of episode rewards over a 10-episode evaluation pass.
    """

    print("Training DQN with the following parameters")
    output_params = {
        "experiment_name": experiment_name,
        "max_timesteps": max_timesteps,
        "save_freq": save_freq,
        "use_expecteds": use_expecteds,
        "use_probabilities": use_probabilities,
        "invalid_action_substitute": invalid_action_substitute,
        "invalid_action_penalty": invalid_action_penalty,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "batch_size": batch_size,
        "gamma": gamma,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "exploration_fraction": exploration_fraction,
        "tau": tau,
    }
    print(output_params)
    env = YahtzeeEnv(
        lambda_upper=0.05, lambda_yahtzee=0.2,
        use_expecteds=use_expecteds, use_probabilities=use_probabilities,
        invalid_action_substitute=invalid_action_substitute,
        invalid_action_penalty=invalid_action_penalty,
    )
    policy_kwargs = dict(net_arch=[128, 128])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    trainer = TrainerBaselines(
        ModelType.DQN,
        env, experiment_name,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        exploration_fraction=exploration_fraction,
        tau=tau,
        policy_kwargs=policy_kwargs,
    )
    trainer.train(max_timesteps=max_timesteps, save_freq=save_freq)
    mean_reward, std_reward = trainer.evaluate(num_episodes=10)
    print(f"Mean reward: {mean_reward:.2f}, Std reward: {std_reward:.2f}")
    trainer.plot_results(max_timesteps=max_timesteps)


@app.command()
def a2c(experiment_name: Annotated[str, typer.Argument(help="The name of the experiment")],
              max_timesteps: Annotated[float, typer.Argument(help="The maximum number of timesteps")] = 30e6,
              save_freq: Annotated[int, typer.Argument(help="The frequency of saving the model (in timesteps)")] = 100000,
              use_expecteds: Annotated[bool, typer.Option(help="Whether to use expecteds")] = False,
              use_probabilities: Annotated[bool, typer.Option(help="Whether to use probabilities")] = True,
              invalid_action_penalty: Annotated[float, typer.Argument(help="Reward penalty applied when an invalid action is substituted")] = -20.0,
              n_steps: Annotated[int, typer.Argument(help="The number of steps")] = 2512,
              gamma: Annotated[float, typer.Argument(help="The discount factor")] = 0.99,
              ent_coef: Annotated[float, typer.Argument(help="The entropy coefficient")] = 0.02,
              vec_normalize: Annotated[bool, typer.Option(help="Whether to use vector normalization")] = True,
              gae_lambda_initial: Annotated[float, typer.Argument(help="The initial GAE lambda")] = 0.3,
              gae_lambda_final: Annotated[float, typer.Argument(help="The GAE lambda")] = 0.95,
              normalize_advantage: Annotated[bool, typer.Option(help="Whether to normalize the advantage")] = False):
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

    Side Effects:
        Writes checkpoints, the final model, VecNormalize stats (if enabled), and
        a learning-curve plot under ``artifacts/<experiment_name>/<date>/``. Prints
        the mean and standard deviation of episode rewards over a 10-episode
        evaluation pass.
    """

    print("Training A2C with the following parameters")
    output_params = {
        "experiment_name": experiment_name,
        "max_timesteps": max_timesteps,
        "save_freq": save_freq,
        "use_expecteds": use_expecteds,
        "use_probabilities": use_probabilities,
        "invalid_action_substitute": True, # Has to be set, otherwise the environment will raise an error
        "invalid_action_penalty": invalid_action_penalty,
        "n_steps": n_steps,
        "gamma": gamma,
        "ent_coef": ent_coef,
        "vec_normalize": vec_normalize,
        "gae_lambda": (gae_lambda_initial, gae_lambda_final),
        "normalize_advantage": normalize_advantage,
    }
    print(output_params)
    env = YahtzeeEnv(
        lambda_upper=0.05, lambda_yahtzee=0.2,
        use_expecteds=use_expecteds, use_probabilities=use_probabilities,
        invalid_action_substitute=True,
        invalid_action_penalty=invalid_action_penalty,
    )
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))

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
    )
    trainer.train(max_timesteps=max_timesteps, save_freq=save_freq)
    mean_reward, std_reward = trainer.evaluate(num_episodes=10)
    print(f"Mean reward: {mean_reward:.2f}, Std reward: {std_reward:.2f}")
    trainer.plot_results(max_timesteps=max_timesteps)


if __name__ == "__main__":
    app()

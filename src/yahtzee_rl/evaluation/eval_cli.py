import typer
from rich import print
from typing import Annotated
from yahtzee_rl.train.train_baselines import TrainerBaselines, ModelType
from yahtzee_rl.envs.yahtzee_env import YahtzeeEnv
from yahtzee_rl.config import Category
from yahtzee_rl.run_config import PPORunConfig, load_run_config
from yahtzee_rl.display.metrics import plot_standard_metrics
from yahtzee_rl.paths import artifact_dir
import numpy as np


app = typer.Typer()


@app.command()
def model(experiment_name: Annotated[str, typer.Argument(help="The name of the experiment")],
             run_date: Annotated[str, typer.Argument(help="The date of the run")],
             num_episodes: Annotated[int, typer.Argument(help="The number of episodes to evaluate")] = 500):
    """
    Evaluate a trained model on the Yahtzee environment.

    Args:
        experiment_name: The name of the experiment.
        run_date: The date of the run.
    """
    full_path = artifact_dir() / experiment_name / run_date
    run_config = load_run_config(full_path)
    env = YahtzeeEnv(
            lambda_upper=run_config.env.lambda_upper,
            lambda_yahtzee=run_config.env.lambda_yahtzee,
            use_expecteds=run_config.env.use_expecteds,
            use_probabilities=run_config.env.use_probabilities,
            invalid_action_substitute=run_config.env.invalid_action_substitute,
            invalid_action_penalty=run_config.env.invalid_action_penalty,
    )
    match run_config.model_type:
        case "MASKABLE_PPO":
            print("Evaluating MASKABLE_PPO model")
            policy_kwargs = dict(net_arch=run_config.policy_net_arch)
            trainer = TrainerBaselines(ModelType.MASKABLE_PPO,
                                       env, experiment_name, batch_size=run_config.batch_size, n_steps=run_config.n_steps,
                                       gamma=run_config.gamma, n_epochs=run_config.n_epochs, policy_kwargs=policy_kwargs, ent_coef=run_config.ent_coef,
                                       vec_normalize=run_config.vec_normalize, 
                                       clip_range=run_config.clip_range, 
                                       gae_lambda=(run_config.gae_lambda_initial, run_config.gae_lambda_final), 
                                       normalize_advantage=run_config.normalize_advantage)
        case "DQN":
            print("Evaluating DQN model")
            policy_kwargs = dict(net_arch=run_config.policy_net_arch)
            trainer = TrainerBaselines(ModelType.DQN,
                                       env, experiment_name, buffer_size=run_config.buffer_size, 
                                       learning_starts=run_config.learning_starts, 
                                       batch_size=run_config.batch_size, 
                                       gamma=run_config.gamma, 
                                       train_freq=run_config.train_freq, 
                                       gradient_steps=run_config.gradient_steps, 
                                       exploration_fraction=run_config.exploration_fraction,
                                        tau=run_config.tau)
        case "A2C":
            print("Evaluating A2C model")
            policy_kwargs = dict(net_arch=run_config.policy_net_arch)
            trainer = TrainerBaselines(ModelType.A2C,
                                       env, experiment_name, n_steps=run_config.n_steps,
                                       gamma=run_config.gamma, policy_kwargs=policy_kwargs, ent_coef=run_config.ent_coef,
                                       vec_normalize=run_config.vec_normalize, gae_lambda=(run_config.gae_lambda_initial, run_config.gae_lambda_final), normalize_advantage=run_config.normalize_advantage)
        case _:
            print(f"Unsupported model type: {run_config.model_type}")
            raise ValueError(f"Unsupported model type: {run_config.model_type}")
    
    if run_config.vec_normalize:
        trainer.load(model_path=full_path / "model.zip", vecnormalize_path=full_path / "vecnormalize.pkl")
    else:
        trainer.load(model_path=full_path / "model.zip")
    
    X = np.arange(0, num_episodes, 1)
    Y = []
    action_counts = np.zeros((len(Category), len(Category)), dtype=int)
    category_labels = [category.value for category in Category]
    category_index = {category.value: idx for idx, category in enumerate(Category)}
    print(f"Evaluating over {num_episodes} episodes...")
    for _ in range(num_episodes):
        obs = trainer.env.reset()
        done = False
        total_reward = 0
        step_count = 0
        turn_index = 0
        while not done:
            if run_config.vec_normalize:
                raw = trainer.env.get_original_obs()          # shape (n_envs, obs_dim)
                parsed = YahtzeeEnv.parse_observation(raw[0], 
                                                    use_expecteds=run_config.env.use_expecteds, 
                                                    use_probabilities=run_config.env.use_probabilities)
            else:
                parsed = YahtzeeEnv.parse_observation(obs[0], use_expecteds=run_config.env.use_expecteds, use_probabilities=run_config.env.use_probabilities)
            action_masks = trainer.env.get_attr('action_masks')[0]()
            if parsed['time_to_score']:
                action, _ = trainer.model.predict(obs, action_masks=action_masks, deterministic=True)
                action_counts[turn_index, action] += 1
                turn_index += 1
                obs, reward, done, info = trainer.env.step(action)
                if done[0]:
                    Y.append(reward[0])
            else:
                action, _ = trainer.model.predict(obs, action_masks=action_masks, deterministic=True)
                obs, reward, done, info = trainer.env.step(action)
            total_reward += reward
            step_count += 1

    plot_standard_metrics(
        X,
        Y,
        num_episodes,
        f"Evaluation of {experiment_name} on {run_date}",
        action_counts=action_counts,
        action_labels=category_labels,
    )
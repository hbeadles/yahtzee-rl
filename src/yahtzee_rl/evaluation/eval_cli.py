import random as _random
import torch
import typer
from rich import print
from typing import Annotated
from yahtzee_rl.train.train_baselines import TrainerBaselines, ModelType
from yahtzee_rl.envs.yahtzee_env import YahtzeeEnv
from yahtzee_rl.config import Category
from yahtzee_rl.run_config import PPORunConfig, load_run_config
from yahtzee_rl.display.metrics import plot_standard_metrics
from yahtzee_rl.display.diagnostics import (
    plot_diagnostic_dashboard,
    plot_score_sankey,
)
from yahtzee_rl.evaluation.diagnostics import (
    EpisodeReport,
    aggregate_reports,
    report_from_scorecard,
    save_reports,
)
from yahtzee_rl.strategies.markov import MarkovStrategy
from yahtzee_rl.paths import artifact_dir
from yahtzee_rl.config import ACTION_TO_CATEGORY, CATEGORY_TO_ACTION
import numpy as np


app = typer.Typer()


def load_agent(experiment_name: str, run_date: str):
    """Load a trained agent + its env from ``artifact_dir/<name>/<date>``.

    Returns ``(trainer, run_config, mask_action)``. Encapsulates the model-type
    dispatch shared by the ``model`` and ``diagnose`` commands.
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
            s_ref=run_config.env.s_ref,
            reward_exponent=run_config.env.reward_exponent,
    )
    mask_action = False
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
                                       normalize_advantage=run_config.normalize_advantage,
                                       target_kl=run_config.target_kl)
            mask_action = True
        case "DQN":
            print("Evaluating DQN model")
            policy_kwargs = dict(net_arch=run_config.policy_net_arch)
            trainer = TrainerBaselines(
                    ModelType.DQN,
                    env, experiment_name,
                    hidden_dim=run_config.hidden_dim,
                    learning_rate=run_config.learning_rate,
                    buffer_size=run_config.buffer_size,
                    batch_size=run_config.batch_size,
                    gamma=run_config.gamma,
                    epsilon_start=run_config.epsilon_start,
                    epsilon_end=run_config.epsilon_end,
                    epsilon_decay=run_config.epsilon_decay,
                    target_update_freq=run_config.target_update_freq,
                    tau=run_config.tau
                )
            mask_action = True
        case "A2C":
            print("Evaluating A2C model")
            policy_kwargs = dict(net_arch=run_config.policy_net_arch)
            trainer = TrainerBaselines(ModelType.A2C,
                                       env, experiment_name, n_steps=run_config.n_steps,
                                       gamma=run_config.gamma, policy_kwargs=policy_kwargs, ent_coef=run_config.ent_coef,
                                       vec_normalize=run_config.vec_normalize, gae_lambda=(run_config.gae_lambda_initial, run_config.gae_lambda_final), normalize_advantage=run_config.normalize_advantage)
        case "BC":
            print("Evaluating BC checkpoint (MaskablePPO policy)")
            policy_kwargs = dict(net_arch=run_config.policy_net_arch)
            trainer = TrainerBaselines(ModelType.MASKABLE_PPO,
                                       env, experiment_name,
                                       policy_kwargs=policy_kwargs,
                                       vec_normalize=run_config.vec_normalize)
            mask_action = True
        case _:
            print(f"Unsupported model type: {run_config.model_type}")
            raise ValueError(f"Unsupported model type: {run_config.model_type}")
    if run_config.model_type == "DQN":
        trainer.load_custom_dqn(model_path=full_path / "best_model.pt")
    elif getattr(run_config, "vec_normalize", False):
        trainer.load(model_path=full_path / "model.zip", vecnormalize_path=full_path / "vecnormalize.pkl")
    else:
        trainer.load(model_path=full_path / "model.zip")

    return trainer, run_config, mask_action


def _model_action(trainer, obs, mask_action):
    """Predict a deterministic action for the (vectorized) trained model env."""
    if mask_action:
        action_masks = trainer.env.get_attr("action_masks")[0]()
        action, _ = trainer.model.predict(obs, action_masks=action_masks, deterministic=True)
    else:
        action, _ = trainer.model.predict(obs, deterministic=True)
    return action


def collect_reports_random(env: YahtzeeEnv, num_episodes: int, seed=None) -> list[EpisodeReport]:
    """Roll out a uniform-random (mask-respecting) agent on a raw env."""
    reports: list[EpisodeReport] = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=None if seed is None else seed + ep)
        done = False
        turn_index = 0
        turn_filled: dict[Category, int] = {}
        while not done:
            mask = env.action_masks()
            valid = [i for i, ok in enumerate(mask) if ok]
            action = _random.choice(valid)
            parsed = YahtzeeEnv.parse_observation(
                obs, use_expecteds=env.use_expecteds, use_probabilities=env.use_probabilities
            )
            if parsed["time_to_score"] and action in range(13):
                turn_index += 1
                turn_filled[ACTION_TO_CATEGORY[action]] = turn_index
            obs, _, done, _, _ = env.step(action)
        reports.append(report_from_scorecard(env.game.scorecard, turn_filled))
    return reports


def _collect_markov_reports(env: YahtzeeEnv, strategy: MarkovStrategy, num_episodes: int) -> list[EpisodeReport]:
    reports: list[EpisodeReport] = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        turn_index = 0
        turn_filled: dict[Category, int] = {}
        while not done:
            parsed = YahtzeeEnv.parse_observation(obs)
            raw = strategy.strategy(obs, env.game.scorecard)
            if parsed["time_to_score"]:
                category = raw if isinstance(raw, Category) else Category(raw)
                turn_index += 1
                turn_filled[category] = turn_index
                obs, _, done, _, _ = env.step(CATEGORY_TO_ACTION[category])
            else:
                obs, _, done, _, _ = env.step(raw)
        reports.append(report_from_scorecard(env.game.scorecard, turn_filled))
    return reports


def _collect_model_reports(trainer, run_config, mask_action, num_episodes: int) -> list[EpisodeReport]:
    reports: list[EpisodeReport] = []
    uses_vecnorm = getattr(run_config, "vec_normalize", False)
    for _ in range(num_episodes):
        obs = trainer.env.reset()
        done = False
        turn_index = 0
        turn_filled: dict[Category, int] = {}
        while not done:
            obs_for_parse = trainer.env.get_original_obs()[0] if uses_vecnorm else obs[0]
            parsed = YahtzeeEnv.parse_observation(
                obs_for_parse,
                use_expecteds=run_config.env.use_expecteds,
                use_probabilities=run_config.env.use_probabilities,
            )
            action = _model_action(trainer, obs, mask_action)
            a = int(np.asarray(action).reshape(-1)[0])
            if parsed["time_to_score"] and a in range(13):
                turn_index += 1
                turn_filled[ACTION_TO_CATEGORY[a]] = turn_index
            obs, _, done, info = trainer.env.step(action)
            done = bool(np.asarray(done).reshape(-1)[0])
        scorecard = info[0]["final_scorecard"]
        reports.append(report_from_scorecard(scorecard, turn_filled))
    return reports


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
    trainer, run_config, mask_action = load_agent(experiment_name, run_date)

    X = np.arange(0, num_episodes, 1)
    Y = []
    action_counts = np.zeros((len(Category), len(Category)), dtype=int)
    category_labels = [category.value for category in Category]
    print(f"Evaluating over {num_episodes} episodes...")
    if run_config.model_type == "DQN":
        for _ in range(num_episodes):
            obs, _ = trainer.env.reset()
            done = False
            total_reward = 0
            step_count = 0
            turn_index = 0
            while not done:
                parsed = YahtzeeEnv.parse_observation(obs, use_expecteds=run_config.env.use_expecteds, use_probabilities=run_config.env.use_probabilities)
                if parsed["time_to_score"]:
                    state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    if mask_action:
                        action_masks = torch.tensor(trainer.env.unwrapped.action_masks(), dtype=torch.bool).unsqueeze(0)
                        action = trainer.model.evaluate(state, action_mask=action_masks)
                    else:
                        action = trainer.model.evaluate(state)
                    bonus_achieved = trainer.model.predict_bonus(state)
                    if action in range(13):
                        action_counts[turn_index, action] += 1
                    turn_index += 1
                    obs, reward, done, truncated, info = trainer.env.step(action)
                    reward = reward + bonus_achieved
                    if done:
                        Y.append(info["total_score"])
                else:
                    state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    if mask_action:
                        action_masks = torch.tensor(trainer.env.unwrapped.action_masks(), dtype=torch.bool).unsqueeze(0)
                        action = trainer.model.evaluate(state, action_mask=action_masks)
                    else:
                        action = trainer.model.evaluate(state)
                    bonus_achieved = trainer.model.predict_bonus(state)
                    obs, reward, done, truncated, info = trainer.env.step(action)
                    reward = reward + bonus_achieved
                
                total_reward += reward
                step_count += 1

    else:
        for _ in range(num_episodes):
            obs = trainer.env.reset()
            done = False
            total_reward = 0
            step_count = 0
            turn_index = 0
            while not done:
                if getattr(run_config, "vec_normalize", False):
                    raw = trainer.env.get_original_obs()  # (n_envs, obs_dim)
                    obs_for_parse = raw[0]
                else:
                    obs_for_parse = obs[0]
                parsed = YahtzeeEnv.parse_observation(obs_for_parse, use_expecteds=run_config.env.use_expecteds, use_probabilities=run_config.env.use_probabilities)
                if parsed['time_to_score']:
                    if mask_action:
                        action_masks = trainer.env.get_attr('action_masks')[0]()
                        action, _ = trainer.model.predict(obs, action_masks=action_masks, deterministic=True)
                    else:
                        action, _ = trainer.model.predict(obs, deterministic=True)
                    if action in range(13):
                        action_counts[turn_index, action] += 1
                    turn_index += 1
                    obs, reward, done, info = trainer.env.step(action)
                    if done[0]:
                        Y.append(info[0]["total_score"])
                else:
                    if mask_action:
                        action_masks = trainer.env.get_attr('action_masks')[0]()
                        action, _ = trainer.model.predict(obs, action_masks=action_masks, deterministic=True)
                    else:
                        action, _ = trainer.model.predict(obs, deterministic=True)
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



@app.command()
def markov(num_episodes: Annotated[int, typer.Argument(help="The number of episodes to evaluate")] = 500):
    """
    Evaluate the Markov Agent on the Yahtzee environment.
    """
    X = np.arange(0, num_episodes, 1)
    Y = []
    action_counts = np.zeros((len(Category), len(Category)), dtype=int)
    category_labels = [category.value for category in Category]
    category_index = {category.value: idx for idx, category in enumerate(Category)}
    env = YahtzeeEnv()
    strategy = MarkovStrategy(env)
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        turn_index = 0
        while not done:
            parsed = YahtzeeEnv.parse_observation(obs)
            if parsed['time_to_score']:
                action = strategy.strategy(obs, env.game.scorecard)
                action_key = action.value if isinstance(action, Category) else action
                if action_key in category_index and turn_index < action_counts.shape[0]:
                    action_counts[turn_index, category_index[action_key]] += 1
                    turn_index += 1
                obs, reward, done, truncated, info = env.step(CATEGORY_TO_ACTION[action])
            else:
                action = strategy.strategy(obs, env.game.scorecard)
                obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
        Y.append(env.game.get_final_score())

    plot_standard_metrics(
        X,
        Y,
        num_episodes,
        f"Evaluation of Markov Agent on Yahtzee Environment",
        action_counts=action_counts,
        action_labels=category_labels,
    )


@app.command()
def diagnose(
    experiment_name: Annotated[str, typer.Argument(help="The experiment name (ignored for --agent markov/random)")],
    run_date: Annotated[str, typer.Argument(help="The run date (ignored for --agent markov/random)")],
    num_episodes: Annotated[int, typer.Argument(help="Number of episodes to evaluate")] = 500,
    save: Annotated[bool, typer.Option(help="Dump per-episode reports to JSON")] = True,
    sankey: Annotated[bool, typer.Option(help="Render the category->outcome Sankey HTML")] = True,
):
    """Score diagnostics: upper-bonus rate, per-category dumps, joker status.

    Renders a matplotlib dashboard and (optionally) a Plotly Sankey, and dumps
    per-episode reports. Works for the trained model, the Markov baseline, or a
    uniform-random baseline so all three are viewable on the same axes.
    """

    trainer, run_config, mask_action = load_agent(experiment_name, run_date)
    reports = _collect_model_reports(trainer, run_config, mask_action, num_episodes)
    out_dir = artifact_dir() / experiment_name / run_date / "diagnostics"

    summary = aggregate_reports(reports)
    print(
        f"episodes={summary.n_episodes} mean={summary.score_mean:.1f} "
        f"median={summary.score_median:.1f} bonus_rate={summary.bonus_rate:.0%} "
        f"joker_unlocked={summary.joker_unlock_rate:.0%}"
    )
    if save:
        path = save_reports(reports, out_dir / "reports.json")
        print(f"Saved reports to {path}")
    if sankey:
        path = plot_score_sankey(reports, out_dir / "sankey.html")
        print(f"Saved Sankey to {path}")
    plot_diagnostic_dashboard(
        summary, f"Diagnostics: {experiment_name} {run_date}"
    )

@app.command()
def diagnose_markov(
    num_episodes: Annotated[int, typer.Argument(help="Number of episodes to evaluate")] = 500,
    save: Annotated[bool, typer.Option(help="Dump per-episode reports to JSON")] = True,
    sankey: Annotated[bool, typer.Option(help="Render the category->outcome Sankey HTML")] = True,
):
    """Score diagnostics for the Markov Agent."""
    env = YahtzeeEnv()
    reports = _collect_markov_reports(env, MarkovStrategy(env), num_episodes)
    out_dir = artifact_dir() / "diagnostics" / "markov"

    summary = aggregate_reports(reports)
    print(
        f"episodes={summary.n_episodes} mean={summary.score_mean:.1f} "
        f"median={summary.score_median:.1f} bonus_rate={summary.bonus_rate:.0%} "
        f"joker_unlocked={summary.joker_unlock_rate:.0%}"
    )
    if save:
        path = save_reports(reports, out_dir / "reports.json")
        print(f"Saved reports to {path}")
    if sankey:
        path = plot_score_sankey(reports, out_dir / "sankey.html")
        print(f"Saved Sankey to {path}")
    plot_diagnostic_dashboard(
        summary, f"Diagnostics: Markov Agent"
    )
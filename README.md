## Yahtzee RL

An end to end environment, training, and interactive suite of tools for training agents against Yahtzee and playing the game, including:

1. A Markov chain agent (using a greedy policy)
2. A MaskedPPO Agent (Yahtzee env)
3. Other RL Agents (A2C, DQN)
4. Game Interface provided by Raylib Python wrapper

These agents are trained using a custom Yahtzee env gymnasium environment, with a $49$ N-length observation state, and a $32$ Discrete choice action space. The action space changes based on the decision in question. For our env, we force Yahtzee to choose to reroll or not, so the first $2$ rounds are spent on the roll decision, the last decision state is to choose the score target the agent will score against. 

In addtion, we also include a fully playable interface via Raylib that allows you to play the game of Yahtzee with Markov Agent assistance. 

## How to Train / Evaluate

You can utilize an existing agent or train your own using the cli tools that come with this library. This requires the `uv` package manager. 

1. Install `uv` if you don't have it already - [UV Install](https://docs.astral.sh/uv/getting-started/installation/)
2. Download this repository using git clone
3. Install dependencies 
    1. `uv sync` to install and setup the env


### Training the agent

There are multiple training agents you can use. Each model has specific default parameters that were chosen based on the specification of the model and early testing; they can be modified via the cli. 

#### MaskedPPO

MaskedPPO is a version of PPO that invalidates "wrong" actions within the model itself. For a deterministic game like Yahtzee, this can occur when chosing a scoring category that has already been chosen, or trying to choose a dice related discrete action that is out of bounds for the scoring action. 

(Rolling a dice is encoded as a number between 0 and 32 (2^5), whereas scoring is number up to 13). `src/yahtzee_rl/evs/yahtzee_env.py` contains more details.

**Command**

```bash
uv run yahtzee-rl train ppo <experiment_name> [max_timesteps] [save_freq] [OPTIONS]
```

Artifacts are written to `artifacts/<experiment_name>/<YYYY-MM-DD>/` (model, checkpoints, `VecNormalize` stats, learning-curve plot, run config).

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `experiment_name` | str | — | Artifact subdirectory name (required). |
| `max_timesteps` | float | `30e6` | Total environment steps to train for. |
| `save_freq` | int | `100000` | Checkpoint frequency, in env steps. |
| `--use-expecteds / --no-use-expecteds` | bool | `True` | Include per-category expected-score features in the observation. |
| `--use-probabilities / --no-use-probabilities` | bool | `True` | Include per-category probability features in the observation. |
| `batch_size` | int | `128` | Minibatch size for each PPO update. |
| `n_steps` | int | `2512` | Env steps collected per rollout before an update. |
| `gamma` | float | `0.99` | Discount factor. |
| `n_epochs` | int | `8` | Optimization epochs per rollout. |
| `ent_coef` | float | `0.02` | Entropy regularization coefficient. |
| `--vec-normalize / --no-vec-normalize` | bool | `True` | Wrap env in `VecNormalize` for observation normalization. |
| `clip_range` | float | `0.1` | PPO surrogate-objective clip range. |
| `gae_lambda_initial` | float | `0.2` | Starting value for linearly-scheduled GAE lambda. |
| `gae_lambda_final` | float | `0.95` | Ending value for linearly-scheduled GAE lambda. |
| `--normalize-advantage / --no-normalize-advantage` | bool | `False` | Normalize advantages within each minibatch. |

**Example**

```bash
uv run yahtzee-rl train ppo ppo_yahtzee_full 30000000 100000
```

#### DQN

DQN is value-based control that learns Q-values over the discrete action space without masking; invalid picks can be substituted with a valid action at the cost of an explicit penalty (`invalid_action_substitute` / `invalid_action_penalty`).

**Command**

```bash
uv run yahtzee-rl train dqn <experiment_name> [max_timesteps] [save_freq] [OPTIONS]
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `experiment_name` | str | — | Artifact subdirectory name (required). |
| `max_timesteps` | float | `30e6` | Total environment steps to train for. |
| `save_freq` | int | `100000` | Checkpoint frequency, in env steps. |
| `--use-expecteds / --no-use-expecteds` | bool | `True` | Include per-category expected-score features in the observation. |
| `--use-probabilities / --no-use-probabilities` | bool | `True` | Include per-category probability features in the observation. |
| `--invalid-action-substitute / --no-invalid-action-substitute` | bool | `True` | Substitute a valid action when the agent picks an invalid one. |
| `invalid_action_penalty` | float | `-20.0` | Reward penalty when an invalid action is substituted. |
| `buffer_size` | int | `1000000` | Replay buffer capacity. |
| `learning_starts` | int | `10000` | Steps collected before the first gradient update. |
| `batch_size` | int | `40` | Minibatch size sampled from the replay buffer. |
| `gamma` | float | `0.99` | Discount factor. |
| `train_freq` | int | `4` | Perform a training update every `train_freq` env steps. |
| `gradient_steps` | int | `1` | Gradient steps per training update. |
| `exploration_fraction` | float | `0.2` | Fraction of training over which epsilon-greed anneals. |
| `tau` | float | `1.0` | Target-network soft-update coefficient (`1.0` = hard update). |

**Example**

```bash
uv run yahtzee-rl train dqn dqn_yahtzee_full 30000000 100000
```

#### A2C

A2C is a synchronous actor–critic policy-gradient method without action masking; invalid actions are always substituted with a valid one with penalty (`invalid_action_substitute` is fixed `True` in the trainer — no CLI flag).

**Command**

```bash
uv run yahtzee-rl train a2c <experiment_name> [max_timesteps] [save_freq] [OPTIONS]
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `experiment_name` | str | — | Artifact subdirectory name (required). |
| `max_timesteps` | float | `30e6` | Total environment steps to train for. |
| `save_freq` | int | `100000` | Checkpoint frequency, in env steps. |
| `--use-expecteds / --no-use-expecteds` | bool | `False` | Include per-category expected-score features in the observation. |
| `--use-probabilities / --no-use-probabilities` | bool | `True` | Include per-category probability features in the observation. |
| `invalid_action_penalty` | float | `-20.0` | Reward penalty when an invalid action is substituted. |
| `n_steps` | int | `2512` | Env steps collected per rollout before an update. |
| `gamma` | float | `0.99` | Discount factor. |
| `ent_coef` | float | `0.02` | Entropy regularization coefficient. |
| `--vec-normalize / --no-vec-normalize` | bool | `True` | Wrap env in `VecNormalize` for observation normalization. |
| `gae_lambda_initial` | float | `0.3` | Starting value for linearly-scheduled GAE lambda. |
| `gae_lambda_final` | float | `0.95` | Ending value for linearly-scheduled GAE lambda. |
| `--normalize-advantage / --no-normalize-advantage` | bool | `False` | Normalize advantages within each minibatch. |

**Example**

```bash
uv run yahtzee-rl train a2c a2c_yahtzee_full 30000000 100000
```


### Evaluation Results

1. **MaskedPPO**

`uv run model ppo_yahtzee_full 2026-04-22`

<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/hbeadles/yahtzee-rl/eb7b9a430763a59a54b790840656d5a4dcc02a76/imgs/yahtzee_metrics_masked_ppo.png" alt="PPO Metrics" width="400"/></td>
    <td><img src="https://raw.githubusercontent.com/hbeadles/yahtzee-rl/eb7b9a430763a59a54b790840656d5a4dcc02a76/imgs/yahtzee_scorecard_map_masked_ppo.png" alt="PPO Scorecard" width="400"/></td>
  </tr>
  <tr>
    <td align="center">Masked PPO Metrics - 500 games</td>
    <td align="center">Masked PPO Scorecard Heatmap</td>
  </tr>
</table>





## How to Play

### Yahtzee Game

*13 Rounds* - Choose a score-target each round. There are two categories of scores, in an *Upper* section and *Lower* section. 

#### Upper

All of these comprise of the sum of dice that contains a particular die value, whether 1s, 2s.. up to 6s. Getting more than $63$ points here yields an upper score bonus of 35 at the end of the game. 

1. 

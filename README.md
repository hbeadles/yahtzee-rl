## Yahtzee RL

An end to end environment, training, and interactive suite of tools for training agents against Yahtzee and playing the game, including:

1. A Markov chain agent (using a greedy policy)
2. A MaskedPPO Agent (Yahtzee env)
3. Other RL Agents (A2C, DQN)
4. Game Interface provided by Raylib Python wrapper

These agents are trained using a custom Yahtzee env gymnasium environment, with a $52$ N-length observation state, and a $32$ Discrete choice action space. The action space changes based on the decision in question. For our env, we force the agent to reroll each time, so the first $2$ rounds are spent on the roll decision, the last decision state is to choose the score target the agent will score against, for a total of 39 timesteps per episode, ((2 rolls + score action) * 13)

In addtion, we also include a fully playable interface via Raylib that allows you to play the game of Yahtzee with Markov Agent assistance. The Markov Agent generally trends on the maximization strategy for Yahtzee, which favors getting the lower-score categories over the higher score categories. 

## Yahtzee RL - Game

You can play a game of Yahtzee with an interactive game I developed here: 

1. [Yahtzee RL](https://hbeadles.github.io/yahtzee-rl)

It uses Raylib and pygbag to deploy to WebGL. Developing the game took some thought on state management and interaction between elements. 

The AI uses the Markov agent, which doesn't require torch as a dependency

## Yahtzee RL - Methodology

Several reference resources were used for this project. I'll reference a few of them with links for attribution. 

1. [Yahtzee: Reinforcement Learning Techniques for Stochastic
Combinatorial Games](https://arxiv.org/pdf/2601.00007): An excellent reference paper I would hold as state of the art for Yahtzee. Contains strong research into neural network setup, environment, and ablation studies of the effect of various parameters. I used this as a reference and implemented the dual-channel approach in DQN. 
2. [Learning to play Yahtzee with Advantage Actor-Critic (A2C)](https://dionhaefner.github.io/2021/04/yahtzotron-learning-to-play-yahtzee-with-advantage-actor-critic/): An excellent reference that trains an A2C model on Yahtzee directly, with final scores around 236+. The observation state was helpful, as well as notes around creating an action mask for the model for incorrect actions. For Yahtzee, this is especially important, since 


## How to Train / Evaluate

You can utilize an existing agent or train your own using the cli tools that come with this library. This requires the `uv` package manager. 

1. Install `uv` if you don't have it already - [UV Install](https://docs.astral.sh/uv/getting-started/installation/)
2. Download this repository using git clone
3. Install dependencies 
    1. `uv sync` to install and setup the env


### Training the agent

There are multiple training agents you can use. Each model has specific default parameters that were chosen based on the specification of the model and early testing; they can be modified via the cli. 

### Markov Agent

We setup a Markov agent using the expected probabilities for dice, which can be formed via a lower upper triangular matrix for different combinations. Answering questions like:

1. If I hold 3 dice, and roll for 2 dice, what is probability of getting 4 of a kind?
2. Probability of rolling a small or large straight from a combination of dice in hand

We can combine those to get a set of probabilities, and then construct a simple scoring mechanism 

#### Transition Matrices

#### Upper Section

Upper section utilizes a Markov transition matrix which calculates the probability of increasing the number of dice with a certain face value (0 - 5 (1-6)). For example, if I have zero dice with the face value I want, what is the probability I say where I am? That is $\frac{5^5}{6^5}$ or $3125/7776$. That is, what is the probability I get zero matches with the face I want? (Around $40%$). 
```python
return np.array([[3125/7776,0,0,0,0,0],
        [3125/7776, 625/1296, 0, 0, 0, 0],
        [ 625/3888, 125/324, 125/216, 0, 0, 0],
        [ 125/3888, 25/216, 25/72, 25/36, 0, 0],
        [  25/7776, 5/324, 5/72, 5/18, 5/6, 0],
        [   1/7776, 1/1296, 1/216, 1/36, 1/6, 1]])

```

Each of the diagonal entries represent the probability of staying where you were. 

To determine the probability of moving up a column, we form a state vector $S_t$ of length 6 that current matches for an upper category. For example, if I have two fours, then $S_2 = 1$, and the vector would be $[0, 0, 1, 0, 0, 0]$. I multiply it by the transition matrix: 

$$ M^{(\text{remaining\_rolls})} @ S_t$$

That gives the distribution over all counts. In our case we're interested in $P(C >= 3)$, which is formed as 

```python
three_sum_p = float(np.sum(dist[3:]))
```
From this distribution, we can give assign a score, which can allow our agent to greedily pick a strategy based on the dice selected

#### Lower Section

Lower section probabilities consist a separate transition matrix for small straights, large straights, and runs. Again, the theory is the same as above. 

For three of a kind, and four of a kind, and full house, we use the following matrix:

```python

return np.array([[120 / 1296, 0, 0, 0, 0],
          [900 / 1296, 120 / 216, 0, 0, 0],
          [250 / 1296, 80 / 216, 25 / 36, 0, 0],
          [25 / 1296, 15 / 216, 10 / 36, 5 / 6, 0],
          [1 / 1296, 1 / 216, 1 / 36, 1 / 6, 1]])

```
Column 0 asumes you keep 1 dice and reroll the other four. The first cell of that column is the probability of staying where I am $\frac{120}{1296}$. The numerator is found by taking the number of ways we can get 4 different faces, so $5 * 4! = 5 * 24 = 120$

As before, we set our state based on the count of dice we have for a particular face, and multiply it. 

$$S_t = [0, 0, 0, 0, 0]$$

Suppose I have two dice with value 2 in my hand. I would set $S_1 = 1$, and then multiply

$$ M^{(\text{remaining\_rolls})} @ S_t$$

#### Lower Section - Large / Small Straight

The small straight / large straight matrices behave similarly as before, but in this case the transitions are based on getting consecutive numbers. 



Usage is very similar to before. For brevity I'll omit getting into more of the details. Please refer to `src/yahtzee_rl/markov/probabilities.py` for reference.

#### MaskedPPO

MaskedPPO is a version of PPO that invalidates "wrong" actions within the model itself. For a deterministic game like Yahtzee, this can occur when chosing a scoring category that has already been chosen, or trying to choose a dice related discrete action that is out of bounds for the scoring action. This is achieved via a boolean mask that we pass into the model. The model can then invalidate, or negate the logits associated with the action

(Rolling a dice is encoded as a number between 0 and 32 (2^5), whereas scoring is number up to 13). `src/yahtzee_rl/evs/yahtzee_env.py` contains more details.

**Command**

```bash
uv run yahtzee-rl train ppo <experiment_name> [max_timesteps] [OPTIONS]
```

Artifacts are written to `artifacts/<experiment_name>/<YYYY-MM-DD>/` (model, checkpoints, `VecNormalize` stats, learning-curve plot, run config).

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `experiment_name` | str (positional) | — | Artifact subdirectory name (required). |
| `max_timesteps` | float (positional) | `30e6` | Total environment steps to train for. |
| `--save-freq` | int | `100000` | Checkpoint frequency, in env steps. |
| `--eval-freq` | int | `save_freq` | Mid-training eval cadence in env steps; `0` disables mid-training eval. Defaults to `--save-freq`. |
| `--n-eval-episodes` | int | `5` | Episodes per mid-training eval pass. |
| `--resume-from` | path | `None` | BC or checkpoint directory containing `model.zip` to resume from. |
| `--use-expecteds / --no-use-expecteds` | bool | `True` | Include per-category expected-score features in the observation. |
| `--use-probabilities / --no-use-probabilities` | bool | `True` | Include per-category probability features in the observation. |
| `--batch-size` | int | `96` | Minibatch size for each PPO update. |
| `--n-steps` | int | `512` | Env steps collected per rollout before an update. |
| `--gamma` | float | `0.99` | Discount factor. |
| `--n-epochs` | int | `5` | Optimization epochs per rollout. |
| `--ent-coef` | float | `0.02` | Entropy regularization coefficient. |
| `--vec-normalize / --no-vec-normalize` | bool | `True` | Wrap env in `VecNormalize` for observation normalization. |
| `--clip-range` | float | `0.1` | PPO surrogate-objective clip range. |
| `--gae-lambda-initial` | float | `0.2` | Starting value for linearly-scheduled GAE lambda. |
| `--gae-lambda-final` | float | `0.95` | Ending value for linearly-scheduled GAE lambda. |
| `--normalize-advantage / --no-normalize-advantage` | bool | `False` | Normalize advantages within each minibatch. |
| `--target-kl` | float | `None` | Early-stop the PPO epoch loop when approx KL exceeds this. Recommended ~0.02 for BC-resume runs; leave unset for from-scratch training. |
| `--s-ref` | float | `150.0` | Reference score used to normalize the reward signal. |
| `--reward-exponent` | float | `6.0` | Exponent applied to the normalized score in the reward shaping function. |

**Example**

```bash
uv run yahtzee-rl train ppo ppo_yahtzee_full 30000000 --save-freq 100000
```

The best-performing MaskedPPO run to date used the configuration checked into
[`artifacts/ppo_yahtzee_full_v2/2026-07-21/config.json`](artifacts/ppo_yahtzee_full_v2/2026-07-21/config.json).
Its CLI-exposed settings can be reproduced with:

```bash
uv run yahtzee-rl train ppo ppo_yahtzee_full_v2 20000000 \
  --save-freq 100000 \
  --eval-freq 100000 \
  --n-eval-episodes 5 \
  --batch-size 78 \
  --n-steps 390 \
  --normalize-advantage \
  --s-ref 200.0 \
  --reward-exponent 3.0
```

#### DQN

DQN is value-based control that learns Q-values over the discrete action space without masking; invalid picks can be substituted with a valid action at the cost of an explicit penalty (`invalid_action_substitute` / `invalid_action_penalty`).

**Command**

```bash
uv run yahtzee-rl train dqn <experiment_name> [max_timesteps] [OPTIONS]
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `experiment_name` | str (positional) | — | Artifact subdirectory name (required). |
| `max_timesteps` | float (positional) | `30e6` | Total environment steps to train for. |
| `--save-freq` | int | `100000` | Checkpoint frequency, in env steps. |
| `--eval-freq` | int | `save_freq` | Mid-training eval cadence in env steps; `0` disables mid-training eval. Defaults to `--save-freq`. |
| `--n-eval-episodes` | int | `30` | Episodes per mid-training eval pass. |
| `--use-expecteds / --no-use-expecteds` | bool | `True` | Include per-category expected-score features in the observation. |
| `--use-probabilities / --no-use-probabilities` | bool | `True` | Include per-category probability features in the observation. |
| `--invalid-action-substitute / --no-invalid-action-substitute` | bool | `True` | Substitute a valid action when the agent picks an invalid one (DQN has no action masking). |
| `--invalid-action-penalty` | float | `-20.0` | Reward penalty when an invalid action is substituted. |
| `--hidden-dim` | int | `128` | Hidden layer width for the Q-network. |
| `--learning-rate` | float | `1e-3` | Adam learning rate. |
| `--buffer-size` | int | `500000` | Replay buffer capacity. |
| `--batch-size` | int | `64` | Minibatch size sampled from the replay buffer. |
| `--gamma` | float | `0.99` | Discount factor. |
| `--epsilon-start` | float | `1.0` | Starting epsilon for epsilon-greedy exploration. |
| `--epsilon-end` | float | `0.01` | Final epsilon for epsilon-greedy exploration. |
| `--exploration-fraction` | float | `0.2` | Fraction of training over which epsilon anneals (used to compute `epsilon_decay`). |
| `--target-update-freq` | int | `100` | Steps between target-network hard updates. |
| `--update-timestep` | int | `8` | Perform a gradient update every N environment steps. |
| `--aux-lambda` | float | `0.2` | Auxiliary loss weight for the value head. |
| `--tau` | float | `1.0` | Target-network soft-update coefficient (`1.0` = hard update). |
| `--s-ref` | float | `150.0` | Reference score used to normalize the reward signal. |
| `--reward-exponent` | float | `6.0` | Exponent applied to the normalized score in the reward shaping function. |

**Example**

```bash
uv run yahtzee-rl train dqn dqn_yahtzee_full 30000000 --save-freq 100000
```

The best-performing DQN run to date used the configuration checked into
[`artifacts/dqn_yahtzee_full/2026-07-22/config.json`](artifacts/dqn_yahtzee_full/2026-07-22/config.json).
It can be reproduced with:

```bash
uv run yahtzee-rl train dqn dqn_yahtzee_full 20000000 \
  --save-freq 50000 \
  --eval-freq 50000 \
  --n-eval-episodes 5 \
  --hidden-dim 256 \
  --batch-size 390 \
  --target-update-freq 780 \
  --tau 0.05 \
  --exploration-fraction 0.08
```

(`--exploration-fraction 0.08` reproduces the saved `epsilon_decay` of `1,600,000`, since
`epsilon_decay = max_timesteps * exploration_fraction`; all other flags match the CLI defaults
listed above.)

#### A2C

A2C is a synchronous actor–critic policy-gradient method without action masking; invalid actions are always substituted with a valid one with penalty (`invalid_action_substitute` is fixed `True` in the trainer — no CLI flag).

**Command**

```bash
uv run yahtzee-rl train a2c <experiment_name> [max_timesteps] [OPTIONS]
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `experiment_name` | str (positional) | — | Artifact subdirectory name (required). |
| `max_timesteps` | float (positional) | `30e6` | Total environment steps to train for. |
| `--save-freq` | int | `100000` | Checkpoint frequency, in env steps. |
| `--use-expecteds / --no-use-expecteds` | bool | `False` | Include per-category expected-score features in the observation. |
| `--use-probabilities / --no-use-probabilities` | bool | `True` | Include per-category probability features in the observation. |
| `--invalid-action-penalty` | float | `-20.0` | Reward penalty when an invalid action is substituted. |
| `--n-steps` | int | `2512` | Env steps collected per rollout before an update. |
| `--gamma` | float | `0.99` | Discount factor. |
| `--ent-coef` | float | `0.02` | Entropy regularization coefficient. |
| `--vec-normalize / --no-vec-normalize` | bool | `True` | Wrap env in `VecNormalize` for observation normalization. |
| `--gae-lambda-initial` | float | `0.3` | Starting value for linearly-scheduled GAE lambda. |
| `--gae-lambda-final` | float | `0.95` | Ending value for linearly-scheduled GAE lambda. |
| `--normalize-advantage / --no-normalize-advantage` | bool | `False` | Normalize advantages within each minibatch. |
| `--s-ref` | float | `150.0` | Reference score used to normalize the reward signal. |
| `--reward-exponent` | float | `6.0` | Exponent applied to the normalized score in the reward shaping function. |

**Example**

```bash
uv run yahtzee-rl train a2c a2c_yahtzee_full 30000000 --save-freq 100000
```


### Evaluation Results

1. **MaskedPPO**

`uv run model ppo_yahtzee_full 2026-04-22`

<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/hbeadles/yahtzee-rl/main/imgs/yahtzee_metrics_masked_ppo.png" alt="PPO Metrics" width="400"/></td>
    <td><img src="https://raw.githubusercontent.com/hbeadles/yahtzee-rl/main/imgs/yahtzee_masked_ppo_map.png" alt="PPO Scorecard" width="400"/></td>
  </tr>
  <tr>
    <td align="center">Masked PPO Metrics - 500 games</td>
    <td align="center">Masked PPO Scorecard Heatmap</td>
  </tr>
</table>

2. **A2C**

`uv run model a2c_yahtzee_v3_full 2026-04-21`

<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/hbeadles/yahtzee-rl/main/imgs/yahtzee_metrics_a2c.png" alt="A2C Metrics" width="400"/></td>
    <td><img src="https://raw.githubusercontent.com/hbeadles/yahtzee-rl/main/imgs/yahtzee_scorecard_map_a2c.png" alt="A2C Scorecard" width="400"/></td>
  </tr>
  <tr>
    <td align="center">A2C Metrics - 500 games</td>
    <td align="center">A2C Scorecard Heatmap</td>
  </tr>
</table>

3. **DQN**

`uv run model dqn_yahtzee_v3_full 2026-04-22`

<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/hbeadles/yahtzee-rl/main/imgs/yahtzee_metrics_dqn.png" alt="DQN Metrics" width="400"/></td>
    <td><img src="https://raw.githubusercontent.com/hbeadles/yahtzee-rl/main/imgs/yahtzee_scorecard_map_dqn.png" alt="DQN Scorecard" width="400"/></td>
  </tr>
  <tr>
    <td align="center">DQN Metrics - 500 games</td>
    <td align="center">DQN Scorecard Heatmap</td>
  </tr>
</table>



## How to Play

Rules can be found below:

1. [Yahtzee](https://pi.math.cornell.edu/~mec/2006-2007/Probability/Yahtzee.htm)

## Acknowledgements

1. [Yahtzotron - Design, DQN, Observation Space]()
2. [Yahtzee: Reinforcement Learning Techniques for Stochastic
Combinatorial Games - Nicholas Pape](https://arxiv.org/pdf/2601.00007)
3. [Snoblin Dice - Dice Texture](https://snoblin.itch.io/pixel-art-dice)
3. [Dice Roll Animation (Kicked in Teeth)](https://kicked-in-teeth.itch.io/dice-roll)
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Conventions (STRICT)

- **Do not stage files.** Never run `git add` (or any equivalent). Leave every change in the working tree unstaged.
- **Do not create commits.** Never run `git commit`, `git commit --amend`, or any other command that produces a commit. This applies even when a plan, prompt, or subtask says "commit this." Ignore those instructions.
- **Do not push or modify remotes.** Never run `git push`, `git fetch --prune`, or anything that mutates `origin`.
- **The human owns the commit boundary.** The user stages and commits all work themselves so they can review the full diff first. Your job ends at "files edited on disk, working tree dirty."
- If a workflow would normally end with a commit step, stop one step earlier and tell the user what you would have committed, including the suggested commit message.

## Reasoning Conventions

- **Don't reinforce biases.** Treat any claim — from the user or from yourself — as a hypothesis, not a conclusion. Look for evidence both supporting and contradicting it before acting, proportional to the stakes. State plainly when the evidence disagrees with the claim, and cite what you checked.
- **Don't let perfect be the enemy of good.** Decisions still have to get made. When evidence is incomplete, time-boxed, or ambiguous, pick the best-supported option available, state the assumptions and remaining uncertainty, and move on — don't stall waiting for ideal certainty.

## Project Overview

Yahtzee RL: a Python project exploring multiple AI agents that play Yahtzee — Markov chain (greedy), Masked PPO (reinforcement learning), partial random, and fully random strategies. Uses `uv` as the package manager.

## Common Commands

```bash
# Install dependencies
uv sync

# Run all tests (always scope to tests/ — see note below)
uv run pytest tests/

# Run a single test file
uv run pytest tests/test_probabilities.py

# Run a single test by name
uv run pytest tests/test_probabilities.py -k "test_name"

# Run the package entry point
uv run yahtzee-rl
```

**Note:** Always run pytest scoped to the `tests/` folder (e.g. `uv run pytest tests/`). Do not run a bare `uv run pytest`, which can pick up unintended files outside the test suite.

## Architecture

### Core Types (`src/yahtzee_rl/__init__.py`)
The root `__init__.py` is load-bearing — it defines `Category` (StrEnum of all 13 scoring categories), `CATEGORIES` (upper/lower enum), `SCORE_TYPES`, and the `ACTION_TO_CATEGORY` / `CATEGORY_TO_ACTION` mappings used everywhere. Most modules import directly from `yahtzee_rl`.

### Scoring (`src/yahtzee_rl/scoring/`)
- `ops.py` — pure functions: individual scoring functions (aces, twos, ..., chance), `combo_satisfied()` for checking if dice meet a category, `dice_count()` returning a `Counter`.
- `scorecard.py` — `Scorecard` class tracking marked categories, scores, and yahtzee bonus count. Computes upper/lower/final scores with bonus logic (35-pt upper bonus at 63+, 100-pt per extra yahtzee).

### Markov Probability Engine (`src/yahtzee_rl/markov/`)
- `probabilities.py` — Markov chain transition matrices for computing probabilities of reaching scoring combos given current dice and remaining rolls. Uses `lru_cache` on matrix powers. Provides both raw probability vectors and expected score vectors (normalized against remaining-score denominator with tunable lambda weights).
- `lookaheads.py` — `determine_keep_positions(dice, move)` returns a 5-bit reroll mask (int 0-31) deciding which dice to keep for a target category. Bit semantics: bit i = 1 means reroll die i, 0 means keep.

### Gymnasium Environment (`src/yahtzee_rl/envs/`)
- `yahtzee_game.py` — `YahtzeeGame`: game logic wrapping Scorecard + dice state. 13 rounds, 3 rolls per round.
- `yahtzee_env.py` — `YahtzeeEnv(gym.Env)`: Gymnasium wrapper. Observation is a flat float32 vector (up to 51 dims) with configurable probability/expected-score features (`use_probabilities`, `use_expecteds`). Always-present fields include `bonus_progress` (clamped upper-section progress `min(raw_sum/63, 1)`) and `joker_active` (Hasbro Joker indicator). Action space is `Discrete(32)` — 2^5 reroll masks when rolling, first 13 actions map to scoring categories when `rolls_remaining == 0`. Has `action_masks()` for masked RL (implements Hasbro Joker priority routing). `parse_observation()` class method converts flat obs back to labeled dict.

### Strategies (`src/yahtzee_rl/strategies/`)
- `base.py` — abstract `Strategy` class.
- `markov.py` — `MarkovStrategy`: picks the category with highest expected score from the Markov engine, then uses `determine_keep_positions` to choose which dice to keep. Falls back to random category selection if no combo is satisfied.

### Training (`src/yahtzee_rl/train/`)
- `train_baselines.py` — `TrainerBaselines`: wraps Stable-Baselines3 (PPO, MaskablePPO, DQN, A2C, SAC) with experiment saving, checkpointing, GAE lambda scheduling, and evaluation. Models save to `experiments/<name>/<date>/`.

## Key Design Patterns

- **Dual-phase action space**: the same `Discrete(32)` action space handles both reroll masks (bits 0-4) and category selection (actions 0-12), switched by `rolls_remaining`. The `action_masks()` method handles validity for both phases.
- **Observation toggles**: `YahtzeeEnv` optionally includes Markov probability and expected-score vectors in observations, controlled by `use_probabilities` and `use_expecteds` constructor flags. This changes the obs dimension.
- **Transition matrices**: probability computations use hardcoded rational transition matrices (runs, straights, upper counts) raised to the power of remaining rolls via `np.linalg.matrix_power`, cached with `lru_cache`.
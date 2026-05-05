"""Typed run-config contract emitted alongside every training artifact.

Each training command writes a ``config.json`` next to the saved model so that
downstream tools (evaluation, plotting, resume) can reconstruct the exact env
and algorithm settings used for that run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter

SCHEMA_VERSION = 1
CONFIG_FILENAME = "config.json"


class EnvConfig(BaseModel):
    """Environment-side knobs for ``YahtzeeEnv``."""

    lambda_upper: float = 0.05
    lambda_yahtzee: float = 0.2
    use_expecteds: bool = True
    use_probabilities: bool = True
    invalid_action_substitute: bool = False
    invalid_action_penalty: float = -20.0


class _BaseRunConfig(BaseModel):
    """Fields shared by every training-run config."""

    schema_version: int = SCHEMA_VERSION
    experiment_name: str
    created_at: Optional[str] = None
    max_timesteps: float
    save_freq: int
    policy_net_arch: dict
    env: EnvConfig


class PPORunConfig(_BaseRunConfig):
    model_type: Literal["MASKABLE_PPO"] = "MASKABLE_PPO"
    batch_size: int
    n_steps: int
    gamma: float
    n_epochs: int
    ent_coef: float
    vec_normalize: bool
    clip_range: float
    gae_lambda_initial: float
    gae_lambda_final: float
    normalize_advantage: bool


class DQNRunConfig(_BaseRunConfig):
    model_type: Literal["DQN"] = "DQN"
    buffer_size: int
    learning_starts: int
    batch_size: int
    gamma: float
    train_freq: int
    gradient_steps: int
    exploration_fraction: float
    tau: float


class A2CRunConfig(_BaseRunConfig):
    model_type: Literal["A2C"] = "A2C"
    n_steps: int
    gamma: float
    ent_coef: float
    vec_normalize: bool
    gae_lambda_initial: float
    gae_lambda_final: float
    normalize_advantage: bool


RunConfig = Annotated[
    Union[PPORunConfig, DQNRunConfig, A2CRunConfig],
    Field(discriminator="model_type"),
]

_RUN_CONFIG_ADAPTER: TypeAdapter = TypeAdapter(RunConfig)


def save_run_config(config: _BaseRunConfig, save_dir: Union[str, Path]) -> Path:
    """Write ``config.json`` inside ``save_dir``. Returns the written file path."""
    path = Path(save_dir) / CONFIG_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.model_dump_json(indent=2))
    return path


def load_run_config(
    save_dir: Union[str, Path],
) -> Union[PPORunConfig, DQNRunConfig, A2CRunConfig]:
    """Load ``config.json`` from ``save_dir`` and return the correct subclass.

    Dispatch is driven by the ``model_type`` discriminator; the returned object
    is one of ``PPORunConfig``, ``DQNRunConfig``, or ``A2CRunConfig``.
    """
    path = Path(save_dir) / CONFIG_FILENAME
    return _RUN_CONFIG_ADAPTER.validate_json(path.read_text())

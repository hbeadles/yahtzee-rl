"""Typed run-config contract emitted alongside every training artifact.

Each training command writes a ``config.json`` next to the saved model so that
downstream tools (evaluation, plotting, resume) can reconstruct the exact env
and algorithm settings used for that run.
"""
from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_serializer, field_validator
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
SCHEMA_VERSION = 1
CONFIG_FILENAME = "config.json"


def _import_features_extractor(dotted_path: str) -> type[BaseFeaturesExtractor]:
    """Resolve a ``module.sub.Class`` dotted path to a ``BaseFeaturesExtractor`` subclass."""
    module_path, _, class_name = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(
            f"features_extractor_class must be a dotted path 'pkg.module.Class', got {dotted_path!r}"
        )
    cls = getattr(import_module(module_path), class_name)
    if not (isinstance(cls, type) and issubclass(cls, BaseFeaturesExtractor)):
        raise TypeError(f"{dotted_path!r} is not a BaseFeaturesExtractor subclass")
    return cls


class EnvConfig(BaseModel):
    """Environment-side knobs for ``YahtzeeEnv``."""

    lambda_upper: float = 0.05
    lambda_yahtzee: float = 0.2
    use_expecteds: bool = True
    use_probabilities: bool = True
    invalid_action_substitute: bool = False
    invalid_action_penalty: float = -20.0
    s_ref: float = 200.0
    reward_exponent: float = 3.0


class _BaseRunConfig(BaseModel):
    """Fields shared by every training-run config."""

    schema_version: int = SCHEMA_VERSION
    experiment_name: str
    created_at: Optional[str] = None
    max_timesteps: float
    save_freq: int
    eval_freq: Optional[int] = None
    n_eval_episodes: int = 5
    policy_net_arch: dict
    env: EnvConfig


class PPORunConfig(_BaseRunConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    features_extractor_class: Optional[type[BaseFeaturesExtractor]] = BaseFeaturesExtractor
    features_extractor_kwargs: Optional[dict] = None
    model_type: Literal["MASKABLE_PPO"] = "MASKABLE_PPO"

    @field_validator("features_extractor_class", mode="before")
    @classmethod
    def _resolve_features_extractor_class(cls, value: Any) -> Optional[type[BaseFeaturesExtractor]]:
        if value is None:
            return None
        if isinstance(value, str):
            return _import_features_extractor(value)
        if isinstance(value, type) and issubclass(value, BaseFeaturesExtractor):
            return value
        raise TypeError(
            f"features_extractor_class must be None, a dotted path, or a BaseFeaturesExtractor subclass; got {value!r}"
        )

    @field_serializer("features_extractor_class")
    def _serialize_features_extractor_class(
        self, value: Optional[type[BaseFeaturesExtractor]]
    ) -> Optional[str]:
        if value is None:
            return None
        return f"{value.__module__}.{value.__qualname__}"
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
    target_kl: Optional[float] = None


class DQNRunConfig(_BaseRunConfig):
    model_type: Literal["DQN"] = "DQN"
    hidden_dim: int = 128
    learning_rate: float = 1e-3
    buffer_size: int
    batch_size: int
    gamma: float
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int
    target_update_freq: int = 100
    tau: float = 1.0
    update_timestep: int = 4000
    aux_lambda: float = 0.5


class A2CRunConfig(_BaseRunConfig):
    model_type: Literal["A2C"] = "A2C"
    n_steps: int
    gamma: float
    ent_coef: float
    vec_normalize: bool
    gae_lambda_initial: float
    gae_lambda_final: float
    normalize_advantage: bool


class CollectMarkovRunConfig(BaseModel):
    """Config written alongside Markov demonstration collection."""

    schema_version: int = SCHEMA_VERSION
    model_type: Literal["COLLECT_MARKOV"] = "COLLECT_MARKOV"
    experiment_name: str
    created_at: Optional[str] = None
    num_episodes: int
    output_path: str
    seed: Optional[int] = None
    env: EnvConfig


class BCRunConfig(BaseModel):
    """Config written alongside behavioral-cloning pretraining."""

    schema_version: int = SCHEMA_VERSION
    model_type: Literal["BC"] = "BC"
    experiment_name: str
    created_at: Optional[str] = None
    demos_path: str
    n_epochs: int
    batch_size: int
    learning_rate: float
    vec_normalize: bool
    policy_net_arch: dict
    env: EnvConfig
    gamma: float = 0.99
    value_epochs: int = 10
    value_learning_rate: float = 1e-3


RunConfig = Annotated[
    Union[PPORunConfig, DQNRunConfig, A2CRunConfig, CollectMarkovRunConfig, BCRunConfig],
    Field(discriminator="model_type"),
]

_RUN_CONFIG_ADAPTER: TypeAdapter = TypeAdapter(RunConfig)


def save_run_config(config: BaseModel, save_dir: Union[str, Path]) -> Path:
    """Write ``config.json`` inside ``save_dir``. Returns the written file path."""
    path = Path(save_dir) / CONFIG_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.model_dump_json(indent=2))
    return path


def load_run_config(
    save_dir: Union[str, Path],
) -> Union[PPORunConfig, DQNRunConfig, A2CRunConfig, CollectMarkovRunConfig, BCRunConfig]:
    """Load ``config.json`` from ``save_dir`` and return the correct subclass.

    Dispatch is driven by the ``model_type`` discriminator; the returned object
    is one of ``PPORunConfig``, ``DQNRunConfig``, or ``A2CRunConfig``.
    """
    path = Path(save_dir) / CONFIG_FILENAME
    return _RUN_CONFIG_ADAPTER.validate_json(path.read_text())

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from platformdirs import user_data_dir


_MARKERS = ("pyproject.toml", ".git", ".project-root")


class ProjectRootNotFound(RuntimeError):
    """Raised when the project root directory cannot be located."""

    pass


@lru_cache(maxsize=1)
def get_project_root(depth: int = 5) -> Path:
    """Walk up from this file, bounded by ``depth``, looking for a marker."""
    current_dir = Path(__file__).resolve().parent
    for _ in range(depth):
        if any((current_dir / m).exists() for m in _MARKERS):
            return current_dir
        current_dir = current_dir.parent
    raise ProjectRootNotFound(
        f"No {_MARKERS} found walking up {depth} levels from {Path(__file__).resolve()}"
    )


def project_root_exists() -> bool:
    """Check whether a project root can be located without raising."""
    try:
        get_project_root()
        return True
    except ProjectRootNotFound:
        return False


@lru_cache(maxsize=1)
def artifact_dir() -> Path:
    """Return the directory used for training/evaluation artifacts.

    Falls back to a platform-appropriate user data directory when the project
    root cannot be located (e.g. the package has been installed into a venv
    outside its source checkout).
    """
    if project_root_exists():
        d = get_project_root() / "artifacts"
    else:
        d = Path(user_data_dir("yahtzee-rl")) / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_under_root(p: str | Path) -> Path:
    """Resolve ``p`` under the project root unless it is already absolute."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else (get_project_root() / p).resolve()

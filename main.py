# /// script
# dependencies = [
#     "cffi",
#     "raylib",
#     "numpy",
#     "matplotlib", 
#     "gymnasium",
#     "cloudpickle",
#     "typing-extensions",
#     "farama-notifications",
#     "typer"
# ]
# ///
import os
import sys

# The `yahtzee_rl` package lives under src/ (src-layout). On desktop it is
# importable via the editable install in .venv, but the pygbag/WASM bundle has
# no install step - it just ships the raw files with cwd at the bundle root -
# so we put src/ on sys.path ourselves.
_src = os.path.join(os.getcwd(), "src")
if os.path.isdir(_src) and _src not in sys.path:
    sys.path.insert(0, _src)


def _shim_missing_multiprocessing() -> None:
    """Let `import gymnasium` succeed under pygbag's WASM CPython.

    `import gymnasium` eagerly imports its vector subsystem, which does
    top-level `from multiprocessing[.sharedctypes|.connection] import ...`.
    The browser's single-threaded CPython omits those pieces. The game never
    uses vector envs, and gymnasium annotates them lazily (PEP 563), so we only
    need the *names* to exist at import time. No-op on desktop where the real
    modules are present.
    """
    import types
    import multiprocessing as mp

    def ensure_submodule(name: str, attrs: list[str]) -> None:
        full = f"multiprocessing.{name}"
        try:
            __import__(full)
            mod = sys.modules[full]
        except Exception:
            mod = types.ModuleType(full)
            sys.modules[full] = mod
            setattr(mp, name, mod)
        for attr in attrs:
            if not hasattr(mod, attr):
                setattr(mod, attr, type(attr, (), {}))

    ensure_submodule("sharedctypes", ["SynchronizedArray"])
    ensure_submodule("connection", ["Connection"])
    if not hasattr(mp, "Queue"):
        mp.Queue = type("Queue", (), {})  # type: ignore[assignment]


try:
    _shim_missing_multiprocessing()
except Exception as _shim_err:  # pragma: no cover - best-effort on desktop
    print(f"multiprocessing shim skipped: {_shim_err}")

import asyncio
from pathlib import Path
from game.base import Game

async def main() -> None:
    game = Game(
        base_shader_path=Path("game/shaders/background_es.frag"),
        title=b"Yahtzee RL Simulation",
        target_fps=60
    )
    game.initialize()
    
    # 3. Setup states (registers states with the StateManager)
    game.setup_states()

    try:
        await game.run_loop_async()
    finally:
        game.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
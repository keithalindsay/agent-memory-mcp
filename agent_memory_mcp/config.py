"""Configuration resolution (DESIGN.md §7.3).

Precedence: CLI flags (handled in cli.py) > env vars > agent-memory-mcp.toml
in the cwd > built-in defaults. The Anthropic model string is defined ONLY
here (never hardcoded elsewhere).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:  # Python 3.11+
    import tomllib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - 3.10 fallback
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover
        tomllib = None  # type: ignore

# The ONLY place the default model id is written.
DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_DB = "~/.agent-memory-mcp/memory.sqlite"
DEFAULT_EXTRACTOR = "rules"
DEFAULT_EMBEDDER = "hash"
DEFAULT_DIM = 256
DEFAULT_K = 4
MODEL_ENV = "AGENT_MEMORY_MODEL"
DB_ENV = "AGENT_MEMORY_DB"


@dataclass
class Config:
    db: str = DEFAULT_DB
    extractor: str = DEFAULT_EXTRACTOR
    embedder: str = DEFAULT_EMBEDDER
    dim: int = DEFAULT_DIM
    k: int = DEFAULT_K
    model: str = DEFAULT_MODEL

    def resolved_db(self) -> str:
        if self.db == ":memory:":
            return self.db
        return str(Path(self.db).expanduser())


def _load_toml(path: Path) -> dict:
    if tomllib is None or not path.exists():
        return {}
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except Exception:  # pragma: no cover - malformed toml is non-fatal
        return {}


def load_config(toml_path: Optional[str | Path] = None) -> Config:
    """Resolve config from defaults, optional TOML file, and env vars."""
    cfg = Config()

    path = Path(toml_path) if toml_path else Path.cwd() / "agent-memory-mcp.toml"
    data = _load_toml(path)
    defaults = data.get("defaults", {})
    if "db" in defaults:
        cfg.db = str(defaults["db"])
    if "extractor" in defaults:
        cfg.extractor = str(defaults["extractor"])
    if "embedder" in defaults:
        cfg.embedder = str(defaults["embedder"])
    if "dim" in defaults:
        cfg.dim = int(defaults["dim"])
    if "k" in defaults:
        cfg.k = int(defaults["k"])

    # LLM model: env var named by [llm].model_env (default AGENT_MEMORY_MODEL).
    model_env = data.get("llm", {}).get("model_env", MODEL_ENV)
    cfg.model = os.environ.get(model_env) or os.environ.get(MODEL_ENV) or DEFAULT_MODEL

    # Direct env overrides.
    if os.environ.get(DB_ENV):
        cfg.db = os.environ[DB_ENV]

    return cfg

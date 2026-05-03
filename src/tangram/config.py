from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, PositiveInt


DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_THINKING_BUDGET = 2000


class ModelConfig(BaseModel):
    model: str = Field(default_factory=lambda: os.getenv("TANGRAM_MODEL", DEFAULT_MODEL))
    max_tokens: PositiveInt = DEFAULT_MAX_TOKENS
    thinking_budget_tokens: int = Field(default=DEFAULT_THINKING_BUDGET, ge=0)

    @property
    def thinking(self) -> dict[str, Any] | None:
        if self.thinking_budget_tokens <= 0:
            return None
        return {"type": "enabled", "budget_tokens": self.thinking_budget_tokens}


class ExperimentConfig(BaseModel):
    pairs: PositiveInt = 8
    trials: PositiveInt = 6
    figures: PositiveInt = 12
    max_turns_per_trial: PositiveInt = 200
    max_continue_chain: PositiveInt = 5
    concurrency: PositiveInt = 4
    seed: int | None = None
    run_id: str | None = None
    model: ModelConfig = Field(default_factory=ModelConfig)
    prompt_variant: str = "default"
    use_fake_client: bool = False

    def resolved_run_id(self) -> str:
        if self.run_id:
            return self.run_id
        stamp = datetime.now(UTC).strftime("%Y-%m-%d-T%H-%M-%S")
        return f"{stamp}-tangram"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_stimuli_dir() -> Path:
    return project_root() / "stimuli" / "tangrams"


def default_results_dir() -> Path:
    return project_root() / "results"


def load_dotenv(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE lines without adding a dotenv dependency."""

    env_path = Path(path)
    if not env_path.is_absolute():
        env_path = project_root() / env_path
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


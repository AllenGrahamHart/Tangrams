from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, PositiveInt, model_validator


ModelProvider = Literal["anthropic", "openai"]
ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh"]

DEFAULT_PROVIDER: ModelProvider = "anthropic"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5"
DEFAULT_OPENAI_MODEL = "gpt-5.2"
DEFAULT_MAX_TOKENS = 4096


def default_provider() -> ModelProvider:
    value = os.getenv("TANGRAM_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    if value not in ("anthropic", "openai"):
        raise ValueError("TANGRAM_PROVIDER must be 'anthropic' or 'openai'.")
    return value


def default_model_for_provider(provider: ModelProvider) -> str:
    if provider == "openai":
        return DEFAULT_OPENAI_MODEL
    return DEFAULT_ANTHROPIC_MODEL


class ModelConfig(BaseModel):
    provider: ModelProvider = Field(default_factory=default_provider)
    model: str | None = Field(default_factory=lambda: os.getenv("TANGRAM_MODEL") or None)
    max_tokens: PositiveInt | None = None
    thinking_budget_tokens: int | None = Field(default=None, ge=0)
    reasoning_effort: ReasoningEffort | None = None

    @model_validator(mode="after")
    def fill_default_model(self) -> "ModelConfig":
        if self.model is None:
            self.model = default_model_for_provider(self.provider)
        return self

    @property
    def thinking(self) -> dict[str, Any] | None:
        if not self.thinking_budget_tokens:
            return None
        return {"type": "enabled", "budget_tokens": self.thinking_budget_tokens}

    @property
    def reasoning(self) -> dict[str, Any] | None:
        if self.reasoning_effort is None:
            return None
        return {"effort": self.reasoning_effort}


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

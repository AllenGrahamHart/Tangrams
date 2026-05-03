from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from tangram.protocol import Handoff, PlacementAction


class TokenUsage(BaseModel):
    input: int = 0
    output: int = 0
    thinking: int = 0

    def add(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input=self.input + other.input,
            output=self.output + other.output,
            thinking=self.thinking + other.thinking,
        )


class TurnLog(BaseModel):
    turn_index: int
    speaker: Literal["director", "matcher"]
    position: int | None = None
    text: str
    raw_text: str | None = None
    thinking: str | None = None
    actions: list[PlacementAction] = Field(default_factory=list)
    handoff: Handoff
    parse_errors: list[str] = Field(default_factory=list)
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    wall_time_ms: int = 0


class TrialLog(BaseModel):
    run_id: str
    pair_id: int
    trial: int
    timestamp_start: str
    timestamp_end: str | None = None
    model: str
    prompt_version: str
    director_target: list[str]
    matcher_initial: list[str]
    matcher_image_mapping: dict[str, str]
    director_image_mapping: dict[str, str]
    turns: list[TurnLog] = Field(default_factory=list)
    final_placements: list[str] = Field(default_factory=list)
    accuracy_per_position: list[bool] = Field(default_factory=list)
    accuracy_overall: float = 0.0
    termination: Literal["done", "turn_cap", "error"] = "turn_cap"
    total_tokens: TokenUsage = Field(default_factory=TokenUsage)
    estimated_cost_usd: float = 0.0


class Manifest(BaseModel):
    run_id: str
    timestamp_start: str
    timestamp_end: str | None = None
    config: dict[str, Any]
    pair_ids: list[int]
    trial_files: list[str] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def write_json(path: Path, model: BaseModel | dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, BaseModel):
        data = model.model_dump(mode="json")
    else:
        data = model
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def trial_path(results_dir: Path, run_id: str, pair_id: int, trial: int) -> Path:
    return results_dir / run_id / f"pair_{pair_id}" / f"trial_{trial}.json"


def write_trial_log(results_dir: Path, trial_log: TrialLog) -> Path:
    path = trial_path(results_dir, trial_log.run_id, trial_log.pair_id, trial_log.trial)
    write_json(path, trial_log)
    return path


def write_manifest(results_dir: Path, manifest: Manifest) -> Path:
    path = results_dir / manifest.run_id / "manifest.json"
    write_json(path, manifest)
    return path


def load_trial_logs(run_dir: Path) -> list[TrialLog]:
    logs: list[TrialLog] = []
    for path in sorted(run_dir.glob("pair_*/trial_*.json")):
        logs.append(TrialLog.model_validate(read_json(path)))
    return logs


from __future__ import annotations

import random
import time
from collections.abc import Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from tangram.config import ModelConfig
from tangram.logging import TokenUsage


Speaker = Literal["director", "matcher"]


class LLMResponse(BaseModel):
    text: str
    raw_content: list[dict[str, Any]] = Field(default_factory=list)
    thinking: str | None = None
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    wall_time_ms: int = 0


class TurnClient(Protocol):
    def create_turn(
        self,
        *,
        speaker: Speaker,
        system: str,
        messages: list[dict[str, Any]],
        config: ModelConfig,
        trial: int,
        position: int,
    ) -> LLMResponse:
        ...


def _dump_content_block(block: Any) -> dict[str, Any]:
    if hasattr(block, "model_dump"):
        return block.model_dump(mode="json", exclude_none=True)
    if isinstance(block, dict):
        return block
    data: dict[str, Any] = {"type": getattr(block, "type", "text")}
    for attr in ("text", "thinking", "signature"):
        if hasattr(block, attr):
            value = getattr(block, attr)
            if value is not None:
                data[attr] = value
    return data


class AnthropicTurnClient:
    def __init__(self, *, max_retries: int = 5, initial_backoff: float = 1.0):
        from anthropic import Anthropic

        self.client = Anthropic()
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff

    def create_turn(
        self,
        *,
        speaker: Speaker,
        system: str,
        messages: list[dict[str, Any]],
        config: ModelConfig,
        trial: int,
        position: int,
    ) -> LLMResponse:
        del speaker, trial, position
        kwargs: dict[str, Any] = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "system": system,
            "messages": messages,
        }
        if config.thinking:
            kwargs["thinking"] = config.thinking

        start = time.monotonic()
        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(**kwargs)
                return self._to_llm_response(response, start)
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                retryable = status_code == 429 or (isinstance(status_code, int) and status_code >= 500)
                if not retryable or attempt == self.max_retries - 1:
                    raise
                sleep_for = self.initial_backoff * (2**attempt) + random.random()
                time.sleep(sleep_for)
        raise RuntimeError("Unreachable Anthropic retry state.")

    def _to_llm_response(self, response: Any, start: float) -> LLMResponse:
        raw_content = [_dump_content_block(block) for block in getattr(response, "content", [])]
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        for block in raw_content:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "thinking":
                thinking_parts.append(block.get("thinking", ""))

        usage = getattr(response, "usage", None)
        tokens = TokenUsage(
            input=int(getattr(usage, "input_tokens", 0) or 0),
            output=int(getattr(usage, "output_tokens", 0) or 0),
            thinking=int(getattr(usage, "thinking_tokens", 0) or 0),
        )
        return LLMResponse(
            text="\n".join(part for part in text_parts if part).strip(),
            raw_content=raw_content,
            thinking="\n".join(part for part in thinking_parts if part).strip() or None,
            tokens=tokens,
            wall_time_ms=int((time.monotonic() - start) * 1000),
        )


class FakeTangramClient:
    """Deterministic local participant pair for tests and dry-run transcripts."""

    DESCRIPTIONS: dict[str, tuple[str, str, str]] = {
        "A": (
            "the tall person leaning forward with a blocky head and one bent leg, almost like someone stepping up",
            "the stepping person",
            "stepping person",
        ),
        "B": (
            "the squat little figure with a wide skirt base and a tilted square head, kind of like a hunched witch",
            "the hunched witch",
            "witch",
        ),
        "C": (
            "the flying bird shape stretched out sideways, with a little square on top and a long tail down to the right",
            "the flying bird",
            "bird",
        ),
        "D": (
            "the upright person with a diamond head and a flat arm sticking straight out to the left",
            "the diamond-head standing one",
            "diamond head",
        ),
        "E": (
            "the tall shape with two pointy pieces like ears or wings at the top and a long base sweeping right",
            "the pointy-eared one",
            "pointy ears",
        ),
        "F": (
            "the low crawling animal shape, with a square head up on the left and a long body going right",
            "the crawling animal",
            "crawler",
        ),
        "G": (
            "the dancer with a big triangular dress and two little arms or flags near the top",
            "the dress dancer",
            "dancer",
        ),
        "H": (
            "the straight upright bottle shape with a diamond top and a slanted foot at the bottom",
            "the bottle person",
            "bottle",
        ),
        "I": (
            "the figure sitting at a little table or bench, with a diamond head and one long leg down",
            "the table sitter",
            "sitter",
        ),
        "J": (
            "the blocky pedestal shape with a diamond floating on top and a small point on the lower right",
            "the pedestal with a diamond",
            "pedestal",
        ),
        "K": (
            "the kneeling person with a diamond head, one sharp knee forward, and a jagged body underneath",
            "the kneeling person",
            "kneeler",
        ),
        "L": (
            "the zigzag standing person with a diamond head and a long triangular foot down to the left",
            "the zigzag person",
            "zigzag",
        ),
    }

    def __init__(self) -> None:
        self.target_order: list[str] = []
        self.matcher_image_mapping: dict[int, str] = {}
        self.next_position = 1

    def set_trial_context(
        self,
        *,
        target_order: Sequence[str],
        matcher_image_mapping: dict[int, str],
        trial: int,
    ) -> None:
        del trial
        self.target_order = list(target_order)
        self.matcher_image_mapping = dict(matcher_image_mapping)
        self.next_position = 1

    def create_turn(
        self,
        *,
        speaker: Speaker,
        system: str,
        messages: list[dict[str, Any]],
        config: ModelConfig,
        trial: int,
        position: int,
    ) -> LLMResponse:
        del system, messages, config
        if speaker == "director":
            text = self._director_text(trial, self.next_position)
        else:
            text = self._matcher_text(self.next_position)
            self.next_position += 1
        return LLMResponse(
            text=text,
            raw_content=[{"type": "text", "text": text}],
            tokens=TokenUsage(input=100, output=len(text.split()), thinking=0),
            wall_time_ms=1,
        )

    def _director_text(self, trial: int, position: int) -> str:
        if position > len(self.target_order):
            return "Okay, that is all twelve. We are done. <done/>"
        figure_id = self.target_order[position - 1]
        variants = self.DESCRIPTIONS[figure_id]
        if trial <= 1:
            desc = variants[0]
        elif trial <= 3:
            desc = variants[1]
        else:
            desc = variants[2]
        return f"Position {position} is {desc}. <yield/>"

    def _matcher_text(self, position: int) -> str:
        if position > len(self.target_order):
            return "Okay. <yield/>"
        figure_id = self.target_order[position - 1]
        inverse = {figure: image_number for image_number, figure in self.matcher_image_mapping.items()}
        image_number = inverse[figure_id]
        return (
            f"Okay, I know the one you mean and I am putting it in position {position}. "
            f"<place figure=\"{image_number}\" position=\"{position}\"/><yield/>"
        )

from __future__ import annotations

import random
import time
from collections.abc import Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from tangram.config import DEFAULT_MAX_TOKENS, ModelConfig
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


def make_turn_client(config: ModelConfig) -> TurnClient:
    if config.provider == "openai":
        return OpenAITurnClient()
    return AnthropicTurnClient()


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


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_int_value(obj: Any, key: str) -> int:
    return int(_get_value(obj, key, 0) or 0)


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
        if config.model is None:
            raise ValueError("Anthropic model must be configured.")
        kwargs: dict[str, Any] = {
            "model": config.model,
            "max_tokens": config.max_tokens or DEFAULT_MAX_TOKENS,
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
            input=_get_int_value(usage, "input_tokens"),
            output=_get_int_value(usage, "output_tokens"),
            thinking=_get_int_value(usage, "thinking_tokens"),
        )
        return LLMResponse(
            text="\n".join(part for part in text_parts if part).strip(),
            raw_content=raw_content,
            thinking="\n".join(part for part in thinking_parts if part).strip() or None,
            tokens=tokens,
            wall_time_ms=int((time.monotonic() - start) * 1000),
        )


class OpenAITurnClient:
    def __init__(self, *, max_retries: int = 5, initial_backoff: float = 1.0):
        from openai import OpenAI

        self.client = OpenAI()
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
        if config.model is None:
            raise ValueError("OpenAI model must be configured.")
        kwargs: dict[str, Any] = {
            "model": config.model,
            "instructions": system,
            "input": [_to_openai_message(message) for message in messages],
        }
        if config.max_tokens is not None:
            kwargs["max_output_tokens"] = config.max_tokens
        if config.reasoning:
            kwargs["reasoning"] = config.reasoning

        start = time.monotonic()
        for attempt in range(self.max_retries):
            try:
                response = self.client.responses.create(**kwargs)
                return self._to_llm_response(response, start)
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                retryable = status_code == 429 or (isinstance(status_code, int) and status_code >= 500)
                if not retryable or attempt == self.max_retries - 1:
                    raise
                sleep_for = self.initial_backoff * (2**attempt) + random.random()
                time.sleep(sleep_for)
        raise RuntimeError("Unreachable OpenAI retry state.")

    def _to_llm_response(self, response: Any, start: float) -> LLMResponse:
        text = str(getattr(response, "output_text", "") or "").strip()
        raw_content = [{"type": "text", "text": text}] if text else []

        usage = getattr(response, "usage", None)
        output_details = _get_value(usage, "output_tokens_details")
        tokens = TokenUsage(
            input=_get_int_value(usage, "input_tokens"),
            output=_get_int_value(usage, "output_tokens"),
            thinking=_get_int_value(output_details, "reasoning_tokens"),
        )
        return LLMResponse(
            text=text,
            raw_content=raw_content,
            tokens=tokens,
            wall_time_ms=int((time.monotonic() - start) * 1000),
        )


def _to_openai_message(message: dict[str, Any]) -> dict[str, Any]:
    role = message.get("role", "user")
    content = message.get("content", "")
    if isinstance(content, str):
        return {"role": role, "content": content}
    if role == "assistant":
        return {"role": role, "content": _text_from_content_blocks(content)}
    return {"role": role, "content": _to_openai_content(content)}


def _to_openai_content(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return [{"type": "input_text", "text": str(content)}]

    converted: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            converted.append({"type": "input_text", "text": str(block)})
            continue
        if block.get("type") == "text":
            converted.append({"type": "input_text", "text": str(block.get("text", ""))})
        elif block.get("type") == "image":
            source = block.get("source", {})
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/png")
                data = source.get("data", "")
                converted.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:{media_type};base64,{data}",
                        "detail": "auto",
                    }
                )
    return converted


def _text_from_content_blocks(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(part for part in parts if part).strip()


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

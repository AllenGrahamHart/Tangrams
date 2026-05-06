from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import BaseModel

from tangram.client import LLMResponse, Speaker, TurnClient
from tangram.config import ModelConfig
from tangram.logging import TurnLog
from tangram.prompts import (
    DIRECTOR_SYSTEM,
    MATCHER_SYSTEM,
    between_trials_text,
    director_trial_text,
    matcher_trial_text,
)
from tangram.stimuli import Stimulus, image_mapping_content


class TrialContext(BaseModel):
    trial: int
    target_order: list[str]
    matcher_initial: list[str]
    director_image_mapping: dict[int, str]
    matcher_image_mapping: dict[int, str]


class Participant(Protocol):
    role: Speaker

    def observe_trial_context(self, context: TrialContext, stimuli: dict[str, Stimulus]) -> None:
        ...

    def add_orchestrator_message(self, text: str) -> None:
        ...

    def create_turn(self, *, config: ModelConfig, trial: int, position: int) -> LLMResponse:
        ...

    def record_own_turn(self, *, response: LLMResponse, turn: TurnLog) -> None:
        ...

    def record_partner_turn(self, *, turn: TurnLog) -> None:
        ...


class ClientParticipant:
    needs_position_hint = False

    def __init__(self, *, role: Speaker, client: TurnClient):
        self.role = role
        self.client = client
        self.history: list[dict[str, Any]] = []

    @property
    def system_prompt(self) -> str:
        return DIRECTOR_SYSTEM if self.role == "director" else MATCHER_SYSTEM

    def observe_trial_context(self, context: TrialContext, stimuli: dict[str, Stimulus]) -> None:
        if hasattr(self.client, "set_trial_context"):
            self.client.set_trial_context(
                target_order=context.target_order,
                matcher_image_mapping=context.matcher_image_mapping,
                trial=context.trial,
            )

        if context.trial > 1:
            self.history.append(
                {"role": "user", "content": between_trials_text(context.trial - 1, context.trial)}
            )

        if self.role == "director":
            content = image_mapping_content(
                stimuli,
                context.director_image_mapping,
                heading="Your private numbered images for this trial are below.",
            )
            content.append(
                {
                    "type": "text",
                    "text": director_trial_text(
                        context.trial,
                        context.target_order,
                        context.director_image_mapping,
                    ),
                }
            )
        else:
            content = image_mapping_content(
                stimuli,
                context.matcher_image_mapping,
                heading="Your private numbered images for this trial are below.",
            )
            content.append(
                {
                    "type": "text",
                    "text": matcher_trial_text(
                        context.trial,
                        context.matcher_initial,
                        context.matcher_image_mapping,
                    ),
                }
            )
        self.history.append({"role": "user", "content": content})

    def add_orchestrator_message(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})

    def create_turn(self, *, config: ModelConfig, trial: int, position: int) -> LLMResponse:
        return self.client.create_turn(
            speaker=self.role,
            system=self.system_prompt,
            messages=self.history,
            config=config,
            trial=trial,
            position=position,
        )

    def record_own_turn(self, *, response: LLMResponse, turn: TurnLog) -> None:
        assistant_content = response.raw_content or [{"type": "text", "text": response.text}]
        self.history.append({"role": "assistant", "content": assistant_content})

    def record_partner_turn(self, *, turn: TurnLog) -> None:
        self.history.append({"role": "user", "content": turn.partner_visible_text or turn.text})


def make_client_participants(client: TurnClient) -> dict[Speaker, Participant]:
    return {
        "director": ClientParticipant(role="director", client=client),
        "matcher": ClientParticipant(role="matcher", client=client),
    }

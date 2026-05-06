from __future__ import annotations

import queue
import threading
import time
from typing import Literal

from pydantic import BaseModel, Field

from tangram.client import LLMResponse, Speaker
from tangram.config import ModelConfig
from tangram.logging import TurnLog
from tangram.participants import Participant, TrialContext
from tangram.protocol import PlacementAction, swap_place
from tangram.stimuli import Stimulus, invert_image_mapping


class HumanTurnSubmission(BaseModel):
    text: str = ""
    handoff: Literal["yield", "continue", "done"] = "yield"
    figure_image_n: int | None = Field(default=None, ge=1, le=12)
    position: int | None = Field(default=None, ge=1, le=12)


class HumanSessionState:
    def __init__(self, *, role: Speaker):
        self.role = role
        self._lock = threading.RLock()
        self._turn_queue: queue.Queue[HumanTurnSubmission] = queue.Queue()
        self.events: list[dict] = []
        self.waiting_for_turn = False
        self.current_position: int | None = None
        self.trial: int | None = None
        self.images: list[dict] = []
        self.target_slots: list[dict] = []
        self.arrangement_slots: list[dict] = []
        self._stimuli: dict[str, Stimulus] = {}
        self._image_mapping: dict[int, str] = {}
        self._inverse_mapping: dict[str, int] = {}
        self._matcher_arrangement: list[str] = []

    def set_trial_context(self, context: TrialContext, stimuli: dict[str, Stimulus]) -> None:
        with self._lock:
            self._stimuli = stimuli
            self.trial = context.trial
            if self.role == "director":
                self._image_mapping = dict(context.director_image_mapping)
                self._inverse_mapping = invert_image_mapping(context.director_image_mapping)
                self.target_slots = [
                    self._slot(position, figure_id)
                    for position, figure_id in enumerate(context.target_order, start=1)
                ]
                self.arrangement_slots = []
            else:
                self._image_mapping = dict(context.matcher_image_mapping)
                self._inverse_mapping = invert_image_mapping(context.matcher_image_mapping)
                self._matcher_arrangement = list(context.matcher_initial)
                self.arrangement_slots = [
                    self._slot(position, figure_id)
                    for position, figure_id in enumerate(self._matcher_arrangement, start=1)
                ]
                self.target_slots = []

            self.images = [
                self._image_payload(image_number, figure_id)
                for image_number, figure_id in sorted(self._image_mapping.items())
            ]
            self.waiting_for_turn = False
            self.current_position = None
            self.events.append(
                {
                    "kind": "system",
                    "speaker": "system",
                    "text": f"Beginning trial {context.trial}.",
                    "time": time.time(),
                }
            )

    def request_turn(self, *, trial: int, position: int) -> HumanTurnSubmission:
        with self._lock:
            self.trial = trial
            self.current_position = position
            self.waiting_for_turn = True
            self.events.append(
                {
                    "kind": "system",
                    "speaker": "system",
                    "text": "Your turn.",
                    "time": time.time(),
                }
            )
        submission = self._turn_queue.get()
        with self._lock:
            self.waiting_for_turn = False
        return submission

    def submit_turn(self, submission: HumanTurnSubmission) -> None:
        with self._lock:
            if not self.waiting_for_turn:
                raise ValueError("It is not currently this participant's turn.")
            if self.role == "director" and submission.figure_image_n is not None:
                raise ValueError("Director turns cannot include placement actions.")
            if self.role == "matcher" and (submission.figure_image_n is None) != (submission.position is None):
                raise ValueError("Matcher placement requires both figure_image_n and position.")
            if self.role == "matcher" and submission.handoff == "done":
                raise ValueError("Matcher cannot use done handoff.")
        self._turn_queue.put(submission)

    def add_orchestrator_message(self, text: str) -> None:
        with self._lock:
            self.events.append(
                {"kind": "system", "speaker": "system", "text": text, "time": time.time()}
            )

    def record_own_turn(self, turn: TurnLog) -> None:
        with self._lock:
            if self.role == "matcher":
                self._apply_actions(turn.actions)
            self.events.append(
                {
                    "kind": "message",
                    "speaker": self.role,
                    "text": turn.text,
                    "handoff": turn.handoff,
                    "position": turn.position,
                    "actions": [action.model_dump(mode="json") for action in turn.actions],
                    "time": time.time(),
                }
            )

    def record_partner_turn(self, turn: TurnLog) -> None:
        with self._lock:
            visible_text = turn.partner_visible_text or turn.text
            for prefix in ("Director: ", "Matcher: "):
                if visible_text.startswith(prefix):
                    visible_text = visible_text[len(prefix) :]
                    break
            self.events.append(
                {
                    "kind": "message",
                    "speaker": turn.speaker,
                    "text": visible_text,
                    "handoff": turn.handoff,
                    "position": turn.position,
                    "actions": [],
                    "time": time.time(),
                }
            )

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "role": self.role,
                "trial": self.trial,
                "waiting_for_turn": self.waiting_for_turn,
                "current_position": self.current_position,
                "images": list(self.images),
                "target_slots": list(self.target_slots),
                "arrangement_slots": list(self.arrangement_slots),
                "events": list(self.events),
            }

    def _apply_actions(self, actions: list[PlacementAction]) -> None:
        for action in actions:
            if action.resolved_id:
                swap_place(self._matcher_arrangement, action.resolved_id, action.position)
        self.arrangement_slots = [
            self._slot(position, figure_id)
            for position, figure_id in enumerate(self._matcher_arrangement, start=1)
        ]

    def _slot(self, position: int, figure_id: str) -> dict:
        image_number = self._inverse_mapping[figure_id]
        return {
            "position": position,
            "image_number": image_number,
            "data_url": self._data_url(figure_id),
        }

    def _image_payload(self, image_number: int, figure_id: str) -> dict:
        return {
            "image_number": image_number,
            "data_url": self._data_url(figure_id),
        }

    def _data_url(self, figure_id: str) -> str:
        stimulus = self._stimuli[figure_id]
        return f"data:{stimulus.media_type};base64,{stimulus.data_base64}"


class HumanParticipant:
    needs_position_hint = True

    def __init__(self, *, role: Speaker, session: HumanSessionState):
        self.role = role
        self.session = session

    def observe_trial_context(self, context: TrialContext, stimuli: dict[str, Stimulus]) -> None:
        self.session.set_trial_context(context, stimuli)

    def add_orchestrator_message(self, text: str) -> None:
        self.session.add_orchestrator_message(text)

    def create_turn(self, *, config: ModelConfig, trial: int, position: int) -> LLMResponse:
        del config
        submission = self.session.request_turn(trial=trial, position=position)
        text = submission.text.strip()
        if self.role == "matcher" and submission.figure_image_n is not None and submission.position is not None:
            text = (
                f"{text}\n"
                f"<place figure=\"{submission.figure_image_n}\" position=\"{submission.position}\"/>"
            ).strip()
        text = f"{text}\n<{submission.handoff}/>".strip()
        return LLMResponse(text=text)

    def record_own_turn(self, *, response: LLMResponse, turn: TurnLog) -> None:
        del response
        self.session.record_own_turn(turn)

    def record_partner_turn(self, *, turn: TurnLog) -> None:
        self.session.record_partner_turn(turn)


class HumanSessionManager:
    def __init__(self, *, run_id: str):
        self.run_id = run_id
        self.sessions: dict[Speaker, HumanSessionState] = {}
        self.status = "starting"
        self.error: str | None = None
        self.result_summary: dict | None = None
        self._lock = threading.RLock()

    def create_session(self, role: Speaker) -> HumanSessionState:
        session = HumanSessionState(role=role)
        self.sessions[role] = session
        return session

    def get_session(self, role: Speaker) -> HumanSessionState:
        if role not in self.sessions:
            raise KeyError(f"No human session for role {role}.")
        return self.sessions[role]

    def set_status(self, status: str, *, error: str | None = None, summary: dict | None = None) -> None:
        with self._lock:
            self.status = status
            self.error = error
            self.result_summary = summary

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "run_id": self.run_id,
                "status": self.status,
                "error": self.error,
                "summary": self.result_summary,
                "roles": sorted(self.sessions),
            }

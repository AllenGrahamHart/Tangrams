from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field


Handoff = Literal["yield", "continue", "done"]

HANDOFF_RE = re.compile(r"<(yield|continue|done)\s*/>\s*$", re.IGNORECASE)
ANY_HANDOFF_RE = re.compile(r"<(yield|continue|done)\s*/>", re.IGNORECASE)
PLACE_RE = re.compile(
    r"<place\s+figure=[\"'](?P<figure>\d{1,2})[\"']\s+position=[\"'](?P<position>\d{1,2})[\"']\s*/>",
    re.IGNORECASE,
)
POSITION_RE = re.compile(
    r"\b(?:position|pos\.?|slot|place)\s*(?:number|#|no\.)?\s*(?P<position>1[0-2]|[1-9])\b",
    re.IGNORECASE,
)
PRIVATE_IMAGE_REF_RE = re.compile(
    r"\b(?:(?:my|your|private)\s+)?image\s*(?:number|#|no\.)?\s*(?:1[0-2]|[1-9])\b",
    re.IGNORECASE,
)


class PlacementAction(BaseModel):
    type: Literal["place"] = "place"
    figure_image_n: int = Field(ge=1, le=12)
    position: int = Field(ge=1, le=12)
    resolved_id: str | None = None


class ParsedTurn(BaseModel):
    raw_text: str
    text: str
    actions: list[PlacementAction] = Field(default_factory=list)
    handoff: Handoff = "yield"
    parse_errors: list[str] = Field(default_factory=list)


def parse_model_response(raw_text: str, speaker: Literal["director", "matcher"]) -> ParsedTurn:
    parse_errors: list[str] = []
    raw_text = raw_text or ""

    handoff_matches = list(ANY_HANDOFF_RE.finditer(raw_text))
    final_match = HANDOFF_RE.search(raw_text)
    if final_match:
        handoff = final_match.group(1).lower()
        if len(handoff_matches) != 1:
            parse_errors.append("Expected exactly one handoff tag.")
    else:
        handoff = "yield"
        parse_errors.append("Missing final handoff tag; defaulted to <yield/>.")

    if handoff == "done" and speaker != "director":
        parse_errors.append("Matcher emitted <done/>; converted to <yield/>.")
        handoff = "yield"

    actions: list[PlacementAction] = []
    for match in PLACE_RE.finditer(raw_text):
        figure = int(match.group("figure"))
        position = int(match.group("position"))
        if speaker != "matcher":
            parse_errors.append("Director emitted a placement action; ignored.")
            continue
        try:
            actions.append(PlacementAction(figure_image_n=figure, position=position))
        except ValueError:
            parse_errors.append(f"Ignored invalid placement action figure={figure} position={position}.")

    if speaker == "matcher" and len(actions) > 1:
        parse_errors.append("Matcher emitted more than one placement action in one turn.")

    text = PLACE_RE.sub("", raw_text)
    text = ANY_HANDOFF_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return ParsedTurn(raw_text=raw_text, text=text, actions=actions, handoff=handoff, parse_errors=parse_errors)


def count_words(text: str) -> int:
    return len([part for part in text.split() if part.strip()])


def visible_partner_message(speaker: Literal["director", "matcher"], text: str) -> str:
    spoken = sanitize_visible_text(text.strip()) if text.strip() else "[no spoken text]"
    label = "Director" if speaker == "director" else "Matcher"
    return f"{label}: {spoken}"


def sanitize_visible_text(text: str) -> str:
    """Remove private image-number references before broadcasting to the partner."""

    return PRIVATE_IMAGE_REF_RE.sub("[private image number omitted]", text)


def infer_position_from_text(text: str) -> int | None:
    matches = [int(match.group("position")) for match in POSITION_RE.finditer(text)]
    return matches[-1] if matches else None


def swap_place(ordering: list[str], figure_id: str, position: int) -> None:
    target_index = position - 1
    if target_index < 0 or target_index >= len(ordering):
        raise ValueError(f"Position out of range: {position}")
    try:
        old_index = ordering.index(figure_id)
    except ValueError:
        ordering[target_index] = figure_id
        return
    ordering[old_index], ordering[target_index] = ordering[target_index], ordering[old_index]

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pandas as pd

from tangram.config import ModelConfig
from tangram.logging import TrialLog


NP_TYPES = [
    "elementary",
    "episodic",
    "installment",
    "provisional",
    "dummy",
    "proxy",
    "description",
    "unclassified",
]


CLASSIFICATION_SYSTEM = """Classify the initial noun phrase strategy in a director's Tangram reference.

Use exactly one label:
elementary: one direct referring expression offered as adequate.
episodic: a single reference delivered in multiple sentence-like chunks.
installment: a reference offered piece by piece with explicit acceptance expected between pieces.
provisional: a tentative or incomplete expression that projects self-expansion.
dummy: placeholder such as whatchamacallit or thingy while preparing a real description.
proxy: the matcher supplies the expression the director was reaching for.
description: first-trial style description/categorization rather than a settled definite reference.
unclassified: cannot tell.

Return only the label."""


def utterance_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def heuristic_np_type(text: str, trial: int) -> str:
    lowered = text.lower()
    if re.search(r"\b(thingy|whatchamacallit|what's it called|whatever)\b", lowered):
        return "dummy"
    if "..." in lowered or lowered.count(",") >= 3:
        return "episodic"
    if re.search(r"\b(kind of|sort of|maybe|i guess|looks like)\b", lowered):
        return "provisional" if trial > 1 else "description"
    if trial == 1:
        return "description"
    if re.search(r"\b(the|that)\b", lowered):
        return "elementary"
    return "unclassified"


def classify_logs(
    logs: list[TrialLog],
    *,
    cache_path: Path,
    llm: bool = False,
    model_config: ModelConfig | None = None,
) -> pd.DataFrame:
    cache = _read_cache(cache_path)
    rows = []
    for log in logs:
        for position in range(1, len(log.director_target) + 1):
            first_director = next(
                (
                    turn
                    for turn in log.turns
                    if turn.speaker == "director" and turn.position == position and turn.text.strip()
                ),
                None,
            )
            if first_director is None:
                label = "unclassified"
                text = ""
            else:
                text = first_director.text
                key = utterance_hash(text)
                if key in cache:
                    label = cache[key]
                elif llm:
                    label = classify_utterance_llm(text, model_config or ModelConfig())
                    cache[key] = label
                else:
                    label = heuristic_np_type(text, log.trial)
                    cache[key] = label
            rows.append(
                {
                    "pair_id": log.pair_id,
                    "trial": log.trial,
                    "position": position,
                    "utterance": text,
                    "np_type": label if label in NP_TYPES else "unclassified",
                }
            )
    _write_cache(cache_path, cache)
    return pd.DataFrame(rows)


def np_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["trial", "np_type", "percentage"])
    counts = df.groupby(["trial", "np_type"]).size().reset_index(name="count")
    totals = counts.groupby("trial")["count"].transform("sum")
    counts["percentage"] = counts["count"] / totals * 100
    return counts


def classify_utterance_llm(text: str, model_config: ModelConfig) -> str:
    from anthropic import Anthropic

    if not os.getenv("ANTHROPIC_API_KEY"):
        return "unclassified"
    client = Anthropic()
    response = client.messages.create(
        model=model_config.model,
        max_tokens=20,
        system=CLASSIFICATION_SYSTEM,
        messages=[{"role": "user", "content": text}],
    )
    parts = [getattr(block, "text", "") for block in response.content if getattr(block, "type", "") == "text"]
    label = " ".join(parts).strip().lower()
    return label if label in NP_TYPES else "unclassified"


def _read_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


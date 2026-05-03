from __future__ import annotations

from pathlib import Path

import pandas as pd

from tangram.logging import TrialLog, load_trial_logs
from tangram.protocol import count_words


HUMAN_WORDS_PER_FIGURE = {1: 41, 2: 19, 3: 13, 4: 11, 5: 9, 6: 8}
HUMAN_TURNS_PER_FIGURE = {1: 3.7, 2: 1.9, 3: 1.4, 4: 1.3, 5: 1.1, 6: 1.2}
HUMAN_BASIC_EXCHANGE_RATE = {1: 0.18, 2: 0.55, 3: 0.75, 4: 0.80, 5: 0.88, 6: 0.84}


def logs_from_run(run_dir: str | Path) -> list[TrialLog]:
    return load_trial_logs(Path(run_dir))


def words_per_figure(logs: list[TrialLog]) -> pd.DataFrame:
    rows = []
    for log in logs:
        for position in range(1, len(log.director_target) + 1):
            words = sum(
                count_words(turn.text)
                for turn in log.turns
                if turn.speaker == "director" and turn.position == position
            )
            rows.append(
                {
                    "pair_id": log.pair_id,
                    "trial": log.trial,
                    "position": position,
                    "figure_id": log.director_target[position - 1],
                    "words": words,
                }
            )
    return pd.DataFrame(rows)


def turns_per_figure(logs: list[TrialLog]) -> pd.DataFrame:
    rows = []
    for log in logs:
        for position in range(1, len(log.director_target) + 1):
            turns = sum(
                1 for turn in log.turns if turn.speaker == "director" and turn.position == position
            )
            rows.append(
                {
                    "pair_id": log.pair_id,
                    "trial": log.trial,
                    "position": position,
                    "figure_id": log.director_target[position - 1],
                    "turns": turns,
                }
            )
    return pd.DataFrame(rows)


def words_by_trial(logs: list[TrialLog]) -> pd.DataFrame:
    df = words_per_figure(logs)
    if df.empty:
        return pd.DataFrame(columns=["trial", "mean_words_per_figure"])
    return (
        df.groupby("trial", as_index=False)["words"]
        .mean()
        .rename(columns={"words": "mean_words_per_figure"})
    )


def turns_by_trial(logs: list[TrialLog]) -> pd.DataFrame:
    df = turns_per_figure(logs)
    if df.empty:
        return pd.DataFrame(columns=["trial", "mean_turns_per_figure"])
    return (
        df.groupby("trial", as_index=False)["turns"]
        .mean()
        .rename(columns={"turns": "mean_turns_per_figure"})
    )


def words_by_position(logs: list[TrialLog], trials: tuple[int, ...] = (1, 2, 6)) -> pd.DataFrame:
    df = words_per_figure(logs)
    if df.empty:
        return pd.DataFrame(columns=["trial", "position", "mean_words_per_figure"])
    df = df[df["trial"].isin(trials)]
    return (
        df.groupby(["trial", "position"], as_index=False)["words"]
        .mean()
        .rename(columns={"words": "mean_words_per_figure"})
    )


def basic_exchange_by_trial(logs: list[TrialLog]) -> pd.DataFrame:
    rows = []
    for log in logs:
        for position in range(1, len(log.director_target) + 1):
            position_turns = [turn for turn in log.turns if turn.position == position]
            director_before_first_matcher = 0
            first_matcher_has_placement = False
            seen_matcher = False
            for turn in position_turns:
                if turn.speaker == "director" and not seen_matcher:
                    director_before_first_matcher += 1
                if turn.speaker == "matcher":
                    seen_matcher = True
                    first_matcher_has_placement = any(
                        action.position == position for action in turn.actions
                    )
                    break
            rows.append(
                {
                    "pair_id": log.pair_id,
                    "trial": log.trial,
                    "position": position,
                    "basic_exchange": bool(
                        director_before_first_matcher == 1 and first_matcher_has_placement
                    ),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["trial", "basic_exchange_rate"])
    return (
        df.groupby("trial", as_index=False)["basic_exchange"]
        .mean()
        .rename(columns={"basic_exchange": "basic_exchange_rate"})
    )


def accuracy_by_trial(logs: list[TrialLog]) -> pd.DataFrame:
    rows = [
        {"pair_id": log.pair_id, "trial": log.trial, "accuracy": log.accuracy_overall}
        for log in logs
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["trial", "accuracy"])
    return df.groupby("trial", as_index=False)["accuracy"].mean()


def pair_accuracy(logs: list[TrialLog]) -> pd.DataFrame:
    rows = [
        {"pair_id": log.pair_id, "trial": log.trial, "accuracy": log.accuracy_overall}
        for log in logs
    ]
    return pd.DataFrame(rows)


def comparison_table(logs: list[TrialLog]) -> pd.DataFrame:
    words = words_by_trial(logs).set_index("trial")
    turns = turns_by_trial(logs).set_index("trial")
    accuracy = accuracy_by_trial(logs).set_index("trial")
    rows = []
    for trial in sorted(set(words.index) | set(turns.index) | set(accuracy.index)):
        rows.append(
            {
                "trial": trial,
                "llm_words_per_figure": round(float(words.loc[trial, "mean_words_per_figure"]), 2)
                if trial in words.index
                else None,
                "human_words_per_figure": HUMAN_WORDS_PER_FIGURE.get(trial),
                "llm_turns_per_figure": round(float(turns.loc[trial, "mean_turns_per_figure"]), 2)
                if trial in turns.index
                else None,
                "human_turns_per_figure": HUMAN_TURNS_PER_FIGURE.get(trial),
                "llm_accuracy": round(float(accuracy.loc[trial, "accuracy"]), 3)
                if trial in accuracy.index
                else None,
            }
        )
    return pd.DataFrame(rows)


from __future__ import annotations

from tangram.analysis import metrics
from tangram.logging import TokenUsage, TrialLog, TurnLog
from tangram.protocol import PlacementAction


def make_log() -> TrialLog:
    return TrialLog(
        run_id="r",
        pair_id=0,
        trial=1,
        timestamp_start="2026-05-03T00:00:00Z",
        timestamp_end="2026-05-03T00:00:01Z",
        model="fake",
        prompt_version="v1",
        director_target=["A", "B"],
        matcher_initial=["B", "A"],
        matcher_image_mapping={"1": "A", "2": "B"},
        director_image_mapping={"1": "A", "2": "B"},
        turns=[
            TurnLog(turn_index=0, speaker="director", position=1, text="position one angel", handoff="yield"),
            TurnLog(
                turn_index=1,
                speaker="matcher",
                position=1,
                text="ok",
                actions=[PlacementAction(figure_image_n=1, position=1, resolved_id="A")],
                handoff="yield",
            ),
            TurnLog(turn_index=2, speaker="director", position=2, text="position two dancer", handoff="yield"),
            TurnLog(
                turn_index=3,
                speaker="matcher",
                position=2,
                text="ok",
                actions=[PlacementAction(figure_image_n=2, position=2, resolved_id="B")],
                handoff="yield",
            ),
        ],
        final_placements=["A", "B"],
        accuracy_per_position=[True, True],
        accuracy_overall=1.0,
        termination="done",
        total_tokens=TokenUsage(input=1, output=1),
    )


def test_words_and_turns_by_trial():
    log = make_log()
    words = metrics.words_by_trial([log])
    turns = metrics.turns_by_trial([log])
    assert words.loc[0, "mean_words_per_figure"] == 3
    assert turns.loc[0, "mean_turns_per_figure"] == 1


def test_basic_exchange_and_accuracy():
    log = make_log()
    basic = metrics.basic_exchange_by_trial([log])
    accuracy = metrics.accuracy_by_trial([log])
    assert basic.loc[0, "basic_exchange_rate"] == 1.0
    assert accuracy.loc[0, "accuracy"] == 1.0


from __future__ import annotations

import threading
import time

from tangram.config import ExperimentConfig, ModelConfig
from tangram.human import HumanParticipant, HumanSessionManager, HumanSessionState, HumanTurnSubmission
from tangram.logging import TurnLog
from tangram.participants import TrialContext
from tangram.protocol import PlacementAction
from tangram.runner import PairRunner
from tangram.stimuli import load_tangrams

from tests.test_stimuli import PNG_1X1


def write_test_stimuli(path) -> None:
    path.mkdir()
    for figure_id in "ABCDEFGHIJKL":
        (path / f"{figure_id}.png").write_bytes(PNG_1X1)


def wait_until_waiting(session: HumanSessionState) -> None:
    for _ in range(100):
        if session.snapshot()["waiting_for_turn"]:
            return
        time.sleep(0.01)
    raise AssertionError("Session did not request a turn.")


def test_human_participant_waits_for_submitted_turn(tmp_path):
    stimuli_dir = tmp_path / "stimuli"
    write_test_stimuli(stimuli_dir)
    stimuli = load_tangrams(stimuli_dir)
    session = HumanSessionState(role="matcher")
    participant = HumanParticipant(role="matcher", session=session)
    participant.observe_trial_context(
        TrialContext(
            trial=1,
            target_order=list("ABCDEFGHIJKL"),
            matcher_initial=list("ABCDEFGHIJKL"),
            director_image_mapping={i + 1: figure for i, figure in enumerate("ABCDEFGHIJKL")},
            matcher_image_mapping={i + 1: figure for i, figure in enumerate("ABCDEFGHIJKL")},
        ),
        stimuli,
    )

    result = {}

    def run_turn() -> None:
        result["response"] = participant.create_turn(
            config=ModelConfig(thinking_budget_tokens=0),
            trial=1,
            position=2,
        )

    thread = threading.Thread(target=run_turn)
    thread.start()
    wait_until_waiting(session)

    session.submit_turn(
        HumanTurnSubmission(
            text="I will place that one.",
            handoff="yield",
            figure_image_n=3,
            position=2,
        )
    )
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert '<place figure="3" position="2"/>' in result["response"].text
    assert result["response"].text.endswith("<yield/>")


def test_runner_accepts_human_participant_turns(tmp_path):
    stimuli_dir = tmp_path / "stimuli"
    results_dir = tmp_path / "results"
    write_test_stimuli(stimuli_dir)
    manager = HumanSessionManager(run_id="human_runner_test")
    director_session = manager.create_session("director")
    matcher_session = manager.create_session("matcher")
    participants = {
        "director": HumanParticipant(role="director", session=director_session),
        "matcher": HumanParticipant(role="matcher", session=matcher_session),
    }
    runner = PairRunner(
        run_id="human_runner_test",
        pair_id=0,
        config=ExperimentConfig(
            pairs=1,
            trials=1,
            figures=2,
            max_turns_per_trial=4,
            run_id="human_runner_test",
            model=ModelConfig(thinking_budget_tokens=0),
        ),
        participants=participants,
        stimuli_dir=stimuli_dir,
        results_dir=results_dir,
    )
    result = {}

    thread = threading.Thread(target=lambda: result.setdefault("log", runner.run_trial(1)))
    thread.start()

    wait_until_waiting(director_session)
    director_session.submit_turn(
        HumanTurnSubmission(text="Put this one in position 1.", handoff="yield")
    )

    wait_until_waiting(matcher_session)
    matcher_state = matcher_session.snapshot()
    matcher_session.submit_turn(
        HumanTurnSubmission(
            text="Placed.",
            handoff="yield",
            figure_image_n=matcher_state["images"][0]["image_number"],
            position=matcher_state["current_position"],
        )
    )

    wait_until_waiting(director_session)
    director_session.submit_turn(HumanTurnSubmission(text="Done.", handoff="done"))

    thread.join(timeout=2)
    assert not thread.is_alive()
    assert result["log"].termination == "done"
    assert len(result["log"].turns) == 3


def test_human_matcher_arrangement_updates_after_own_placement(tmp_path):
    stimuli_dir = tmp_path / "stimuli"
    write_test_stimuli(stimuli_dir)
    stimuli = load_tangrams(stimuli_dir)
    session = HumanSessionState(role="matcher")
    participant = HumanParticipant(role="matcher", session=session)
    participant.observe_trial_context(
        TrialContext(
            trial=1,
            target_order=list("ABCDEFGHIJKL"),
            matcher_initial=list("ABCDEFGHIJKL"),
            director_image_mapping={i + 1: figure for i, figure in enumerate("ABCDEFGHIJKL")},
            matcher_image_mapping={i + 1: figure for i, figure in enumerate("ABCDEFGHIJKL")},
        ),
        stimuli,
    )
    action = PlacementAction(figure_image_n=3, position=1, resolved_id="C")
    participant.record_own_turn(
        response=None,
        turn=TurnLog(
            turn_index=0,
            speaker="matcher",
            position=1,
            text="Done.",
            actions=[action],
            handoff="yield",
        ),
    )
    slots = session.snapshot()["arrangement_slots"]
    assert slots[0]["image_number"] == 3
    assert slots[2]["image_number"] == 1


def test_human_partner_message_uses_visible_text():
    session = HumanSessionState(role="matcher")
    participant = HumanParticipant(role="matcher", session=session)
    participant.record_partner_turn(
        turn=TurnLog(
            turn_index=0,
            speaker="director",
            position=1,
            text="Position 1 is my image 4.",
            partner_visible_text="Director: Position 1 is [private image number omitted].",
            handoff="yield",
        )
    )
    event = session.snapshot()["events"][0]
    assert "my image 4" not in event["text"]
    assert event["text"] == "Position 1 is [private image number omitted]."

from __future__ import annotations

from typing import Any

from tangram.client import FakeTangramClient, LLMResponse
from tangram.config import ExperimentConfig, ModelConfig
from tangram.runner import PairRunner

from tests.test_stimuli import PNG_1X1


def write_test_stimuli(path) -> None:
    path.mkdir()
    for figure_id in "ABCDEFGHIJKL":
        (path / f"{figure_id}.png").write_bytes(PNG_1X1)


def test_single_trial_fake_client_runs_end_to_end(tmp_path):
    stimuli_dir = tmp_path / "stimuli"
    results_dir = tmp_path / "results"
    write_test_stimuli(stimuli_dir)
    config = ExperimentConfig(
        pairs=1,
        trials=1,
        max_turns_per_trial=40,
        seed=7,
        run_id="test",
        model=ModelConfig(thinking_budget_tokens=0),
        use_fake_client=True,
    )
    runner = PairRunner(
        run_id="test",
        pair_id=0,
        config=config,
        client=FakeTangramClient(),
        stimuli_dir=stimuli_dir,
        results_dir=results_dir,
    )
    log = runner.run_trial(1)
    assert log.termination == "done"
    assert log.accuracy_overall == 1.0
    assert len(log.turns) == 25
    director_partner_messages = [
        message["content"]
        for message in runner.histories["director"]
        if message["role"] == "user" and isinstance(message["content"], str)
    ]
    assert not any("<place" in message for message in director_partner_messages)


class AlwaysContinueClient:
    def create_turn(
        self,
        *,
        speaker: str,
        system: str,
        messages: list[dict[str, Any]],
        config: ModelConfig,
        trial: int,
        position: int,
    ) -> LLMResponse:
        return LLMResponse(text="Still talking. <continue/>")


def test_continue_chain_cap_forces_yield(tmp_path):
    stimuli_dir = tmp_path / "stimuli"
    results_dir = tmp_path / "results"
    write_test_stimuli(stimuli_dir)
    config = ExperimentConfig(
        pairs=1,
        trials=1,
        max_turns_per_trial=4,
        max_continue_chain=2,
        run_id="test",
        model=ModelConfig(thinking_budget_tokens=0),
    )
    runner = PairRunner(
        run_id="test",
        pair_id=0,
        config=config,
        client=AlwaysContinueClient(),
        stimuli_dir=stimuli_dir,
        results_dir=results_dir,
    )
    log = runner.run_trial(1)
    assert log.turns[1].handoff == "yield"
    assert any("forced yield" in error for error in log.turns[1].parse_errors)


class PrivateImageMentionClient:
    def create_turn(
        self,
        *,
        speaker: str,
        system: str,
        messages: list[dict[str, Any]],
        config: ModelConfig,
        trial: int,
        position: int,
    ) -> LLMResponse:
        if speaker == "director":
            return LLMResponse(text="Position 1 is my image 4. <yield/>")
        return LLMResponse(text='Okay. <place figure="1" position="1"/><yield/>')


def test_partner_visible_text_is_logged_after_scrubbing(tmp_path):
    stimuli_dir = tmp_path / "stimuli"
    results_dir = tmp_path / "results"
    write_test_stimuli(stimuli_dir)
    config = ExperimentConfig(
        pairs=1,
        trials=1,
        max_turns_per_trial=2,
        run_id="test",
        model=ModelConfig(thinking_budget_tokens=0),
    )
    runner = PairRunner(
        run_id="test",
        pair_id=0,
        config=config,
        client=PrivateImageMentionClient(),
        stimuli_dir=stimuli_dir,
        results_dir=results_dir,
    )
    log = runner.run_trial(1)
    assert "my image 4" in log.turns[0].text
    assert "my image 4" not in log.turns[0].partner_visible_text
    assert "[private image number omitted]" in log.turns[0].partner_visible_text

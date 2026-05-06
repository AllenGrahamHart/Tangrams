from __future__ import annotations

import os
import random
from pathlib import Path

from tangram.client import Speaker, TurnClient
from tangram.config import ExperimentConfig
from tangram.logging import TokenUsage, TrialLog, TurnLog, utc_now_iso, write_trial_log
from tangram.participants import Participant, TrialContext, make_client_participants
from tangram.prompts import PROMPT_VERSION
from tangram.protocol import infer_position_from_text, parse_model_response, swap_place, visible_partner_message
from tangram.stimuli import FIGURE_IDS, load_tangrams


class PairRunner:
    def __init__(
        self,
        *,
        run_id: str,
        pair_id: int,
        config: ExperimentConfig,
        client: TurnClient | None = None,
        participants: dict[Speaker, Participant] | None = None,
        stimuli_dir: Path,
        results_dir: Path,
        rng: random.Random | None = None,
    ) -> None:
        self.run_id = run_id
        self.pair_id = pair_id
        self.config = config
        if participants is None:
            if client is None:
                raise ValueError("Either client or participants must be provided.")
            participants = make_client_participants(client)
        self.participants = participants
        self.stimuli_dir = stimuli_dir
        self.results_dir = results_dir
        self.rng = rng or random.Random(config.seed)
        self.figure_ids = tuple(FIGURE_IDS[: config.figures])
        if len(self.figure_ids) != config.figures:
            raise ValueError(f"Only {len(FIGURE_IDS)} figures are available; requested {config.figures}.")
        self.stimuli = load_tangrams(stimuli_dir, self.figure_ids)
        self.histories = {
            role: getattr(participant, "history", [])
            for role, participant in self.participants.items()
        }

    def run_pair(self) -> list[TrialLog]:
        logs: list[TrialLog] = []
        for trial in range(1, self.config.trials + 1):
            log = self.run_trial(trial)
            write_trial_log(self.results_dir, log)
            logs.append(log)
        return logs

    def run_trial(self, trial: int) -> TrialLog:
        target_order = self._permutation()
        matcher_initial = self._permutation()
        while matcher_initial == target_order:
            matcher_initial = self._permutation()
        director_image_mapping = self._image_mapping()
        matcher_image_mapping = self._image_mapping()

        context = TrialContext(
            trial=trial,
            target_order=list(target_order),
            matcher_initial=list(matcher_initial),
            director_image_mapping=director_image_mapping,
            matcher_image_mapping=matcher_image_mapping,
        )
        for participant in self.participants.values():
            participant.observe_trial_context(context, self.stimuli)

        log = TrialLog(
            run_id=self.run_id,
            pair_id=self.pair_id,
            trial=trial,
            timestamp_start=utc_now_iso(),
            model=self.config.model.model,
            prompt_version=PROMPT_VERSION,
            director_target=list(target_order),
            matcher_initial=list(matcher_initial),
            matcher_image_mapping={str(k): v for k, v in matcher_image_mapping.items()},
            director_image_mapping={str(k): v for k, v in director_image_mapping.items()},
        )

        placements = list(matcher_initial)
        current_position = 1
        speaker: Speaker = "director"
        continue_speaker: Speaker | None = None
        continue_chain = 0
        done_cue_sent = False
        placed_positions: set[int] = set()

        try:
            while len(log.turns) < self.config.max_turns_per_trial:
                if (
                    len(placed_positions) == self.config.figures
                    and speaker == "director"
                    and not done_cue_sent
                ):
                    self.participants["director"].add_orchestrator_message(
                        "All 12 positions have now received matcher placement acknowledgments. If you are satisfied, end the trial with <done/>."
                    )
                    done_cue_sent = True

                response = self.participants[speaker].create_turn(
                    config=self.config.model,
                    trial=trial,
                    position=current_position,
                )
                parsed = parse_model_response(response.text, speaker)

                inferred_position = infer_position_from_text(parsed.text) if speaker == "director" else None
                if inferred_position is not None:
                    current_position = inferred_position

                turn_position = (
                    inferred_position
                    if speaker == "director"
                    else parsed.actions[0].position
                    if parsed.actions
                    else current_position if current_position <= self.config.figures else None
                )
                if speaker == "matcher":
                    self._resolve_actions(parsed.actions, matcher_image_mapping, placements)
                    placed_positions.update(action.position for action in parsed.actions)
                    if getattr(self.participants["director"], "needs_position_hint", False):
                        current_position = self._next_unplaced_position(placed_positions)

                effective_handoff = parsed.handoff
                if effective_handoff == "continue":
                    if continue_speaker == speaker:
                        continue_chain += 1
                    else:
                        continue_speaker = speaker
                        continue_chain = 1
                    if continue_chain >= self.config.max_continue_chain:
                        parsed.parse_errors.append("Continue-chain cap reached; forced yield.")
                        effective_handoff = "yield"
                else:
                    continue_speaker = None
                    continue_chain = 0

                partner_visible_text = visible_partner_message(speaker, parsed.text)
                turn = TurnLog(
                    turn_index=len(log.turns),
                    speaker=speaker,
                    position=turn_position,
                    text=parsed.text,
                    raw_text=parsed.raw_text,
                    partner_visible_text=partner_visible_text,
                    thinking=response.thinking,
                    actions=parsed.actions,
                    handoff=effective_handoff,
                    parse_errors=parsed.parse_errors,
                    tokens=response.tokens,
                    wall_time_ms=response.wall_time_ms,
                )
                log.turns.append(turn)
                log.total_tokens = log.total_tokens.add(response.tokens)

                self.participants[speaker].record_own_turn(response=response, turn=turn)
                partner: Speaker = "matcher" if speaker == "director" else "director"
                self.participants[partner].record_partner_turn(turn=turn)

                if speaker == "director" and parsed.handoff == "done":
                    log.termination = "done"
                    break

                if effective_handoff == "continue":
                    self.participants[speaker].add_orchestrator_message("Continue your turn.")
                else:
                    speaker = "matcher" if speaker == "director" else "director"
            else:
                log.termination = "turn_cap"
        except Exception as exc:
            log.termination = "error"
            log.turns.append(
                TurnLog(
                    turn_index=len(log.turns),
                    speaker=speaker,
                    position=current_position if current_position <= self.config.figures else None,
                    text=f"ERROR: {exc}",
                    handoff="yield",
                    parse_errors=[repr(exc)],
                )
            )

        log.timestamp_end = utc_now_iso()
        log.final_placements = placements
        log.accuracy_per_position = [
            placed == target for placed, target in zip(placements, target_order, strict=True)
        ]
        log.accuracy_overall = sum(log.accuracy_per_position) / len(log.accuracy_per_position)
        log.estimated_cost_usd = estimate_cost_usd(log.total_tokens)
        return log

    def _resolve_actions(
        self,
        actions: list,
        matcher_image_mapping: dict[int, str],
        placements: list[str],
    ) -> None:
        for action in actions:
            resolved = matcher_image_mapping.get(action.figure_image_n)
            action.resolved_id = resolved
            if resolved:
                swap_place(placements, resolved, action.position)

    def _permutation(self) -> list[str]:
        values = list(self.figure_ids)
        self.rng.shuffle(values)
        return values

    def _image_mapping(self) -> dict[int, str]:
        values = self._permutation()
        return {index: figure_id for index, figure_id in enumerate(values, start=1)}

    def _next_unplaced_position(self, placed_positions: set[int]) -> int:
        for position in range(1, self.config.figures + 1):
            if position not in placed_positions:
                return position
        return self.config.figures + 1


def estimate_cost_usd(tokens: TokenUsage) -> float:
    """Rough configurable estimate; Anthropic invoices are the source of truth."""

    input_rate = float(os.getenv("TANGRAM_INPUT_USD_PER_MTOKEN", "3.0"))
    output_rate = float(os.getenv("TANGRAM_OUTPUT_USD_PER_MTOKEN", "15.0"))
    thinking_rate = float(os.getenv("TANGRAM_THINKING_USD_PER_MTOKEN", str(output_rate)))
    return round(
        (tokens.input / 1_000_000 * input_rate)
        + (tokens.output / 1_000_000 * output_rate)
        + (tokens.thinking / 1_000_000 * thinking_rate),
        6,
    )

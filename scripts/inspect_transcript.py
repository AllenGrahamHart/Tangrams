from __future__ import annotations

import argparse
from pathlib import Path

from tangram.config import default_results_dir
from tangram.logging import TrialLog, read_json, trial_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretty-print a Tangram trial transcript.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pair", type=int, required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--results-dir", type=Path, default=default_results_dir())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = trial_path(args.results_dir, args.run_id, args.pair, args.trial)
    log = TrialLog.model_validate(read_json(path))
    print(f"Run {log.run_id} / pair {log.pair_id} / trial {log.trial}")
    print(f"Target: {' '.join(log.director_target)}")
    print(f"Initial: {' '.join(log.matcher_initial)}")
    print(f"Final: {' '.join(log.final_placements)}")
    print(f"Accuracy: {log.accuracy_overall:.3f} ({log.termination})")
    print()
    for turn in log.turns:
        label = "D" if turn.speaker == "director" else "M"
        position = f"p{turn.position}" if turn.position else "--"
        print(f"{turn.turn_index:03d} {label} {position} [{turn.handoff}] {turn.text}")
        for action in turn.actions:
            print(
                f"    action: place image {action.figure_image_n} "
                f"({action.resolved_id}) at position {action.position}"
            )
        for error in turn.parse_errors:
            print(f"    parse: {error}")


if __name__ == "__main__":
    main()


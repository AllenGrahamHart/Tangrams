from __future__ import annotations

import hashlib
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from tangram.client import AnthropicTurnClient, TurnClient
from tangram.config import ExperimentConfig, default_results_dir, default_stimuli_dir
from tangram.logging import Manifest, TrialLog, utc_now_iso, write_manifest
from tangram.prompts import DIRECTOR_SYSTEM, MATCHER_SYSTEM, PROMPT_VERSION
from tangram.runner import PairRunner
from tangram.stimuli import FIGURE_IDS, load_tangrams


ClientFactory = Callable[[], TurnClient]


def run_experiment(
    config: ExperimentConfig,
    *,
    client_factory: ClientFactory | None = None,
    stimuli_dir: Path | None = None,
    results_dir: Path | None = None,
) -> Manifest:
    run_id = config.resolved_run_id()
    results_root = results_dir or default_results_dir()
    stimuli_root = stimuli_dir or default_stimuli_dir()
    factory = client_factory or (lambda: AnthropicTurnClient())

    manifest = Manifest(
        run_id=run_id,
        timestamp_start=utc_now_iso(),
        config=config.model_dump(mode="json"),
        pair_ids=list(range(config.pairs)),
        git_commit=current_git_commit(),
        git_dirty=current_git_dirty(),
        prompt_version=PROMPT_VERSION,
        prompt_sha256=prompt_sha256(),
        stimuli_sha256=stimuli_sha256(stimuli_root, config.figures),
    )
    write_manifest(results_root, manifest)

    all_logs: list[TrialLog] = []
    with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
        futures = []
        for pair_id in range(config.pairs):
            pair_seed = None if config.seed is None else config.seed + pair_id
            runner = PairRunner(
                run_id=run_id,
                pair_id=pair_id,
                config=config,
                client=factory(),
                stimuli_dir=stimuli_root,
                results_dir=results_root,
                rng=random.Random(pair_seed),
            )
            futures.append(executor.submit(runner.run_pair))

        for future in as_completed(futures):
            all_logs.extend(future.result())

    all_logs.sort(key=lambda item: (item.pair_id, item.trial))
    manifest.trial_files = [
        str(Path(f"pair_{log.pair_id}") / f"trial_{log.trial}.json") for log in all_logs
    ]
    manifest.timestamp_end = utc_now_iso()
    manifest.summary = summarize_logs(all_logs)
    write_manifest(results_root, manifest)
    return manifest


def prompt_sha256() -> str:
    prompt_blob = "\n\n".join([PROMPT_VERSION, DIRECTOR_SYSTEM, MATCHER_SYSTEM])
    return hashlib.sha256(prompt_blob.encode("utf-8")).hexdigest()


def stimuli_sha256(stimuli_dir: Path, figures: int) -> dict[str, str]:
    figure_ids = tuple(FIGURE_IDS[:figures])
    return {
        figure_id: stimulus.sha256
        for figure_id, stimulus in load_tangrams(stimuli_dir, figure_ids).items()
    }


def current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def current_git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def summarize_logs(logs: list[TrialLog]) -> dict:
    if not logs:
        return {}
    total_cost = sum(log.estimated_cost_usd for log in logs)
    total_input = sum(log.total_tokens.input for log in logs)
    total_output = sum(log.total_tokens.output for log in logs)
    total_thinking = sum(log.total_tokens.thinking for log in logs)
    mean_accuracy = sum(log.accuracy_overall for log in logs) / len(logs)
    by_trial: dict[int, list[float]] = {}
    for log in logs:
        by_trial.setdefault(log.trial, []).append(log.accuracy_overall)
    return {
        "trials": len(logs),
        "mean_accuracy": round(mean_accuracy, 4),
        "accuracy_by_trial": {
            str(trial): round(sum(values) / len(values), 4) for trial, values in sorted(by_trial.items())
        },
        "total_tokens": {"input": total_input, "output": total_output, "thinking": total_thinking},
        "estimated_cost_usd": round(total_cost, 6),
    }

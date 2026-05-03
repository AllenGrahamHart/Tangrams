from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from tangram.analysis import metrics, plots
from tangram.config import ModelConfig, default_results_dir, load_dotenv
from tangram.logging import load_trial_logs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Tangram experiment results.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-dir", type=Path, default=default_results_dir())
    parser.add_argument("--llm-coding", action="store_true", help="Use Anthropic for NP-type coding.")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    run_dir = args.results_dir / args.run_id
    outputs = plots.generate_all(run_dir, llm_coding=args.llm_coding, model_config=ModelConfig())
    summary_path = write_summary(run_dir, llm_coding=args.llm_coding)
    print(f"Wrote summary: {summary_path}")
    for name, path in outputs.items():
        print(f"{name}: {path}")


def write_summary(run_dir: Path, *, llm_coding: bool) -> Path:
    logs = load_trial_logs(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    comparison = metrics.comparison_table(logs)
    basic = metrics.basic_exchange_by_trial(logs)
    accuracy = metrics.accuracy_by_trial(logs)
    pair_accuracy = metrics.pair_accuracy(logs)

    lines = [
        f"# Results Summary: {run_dir.name}",
        "",
        "## Configuration",
        "",
        f"- Model: {manifest.get('config', {}).get('model', {}).get('model', 'unknown')}",
        f"- Pairs: {manifest.get('config', {}).get('pairs', 'unknown')}",
        f"- Trials: {manifest.get('config', {}).get('trials', 'unknown')}",
        f"- NP coding: {'Anthropic classifier' if llm_coding else 'deterministic heuristic'}",
        f"- Estimated cost USD: {manifest.get('summary', {}).get('estimated_cost_usd', 'unknown')}",
        "",
        "## LLM vs Human Headline Metrics",
        "",
        markdown_table(comparison),
        "",
        "## Basic Exchange Rate",
        "",
        markdown_table(basic),
        "",
        "## Accuracy By Trial",
        "",
        markdown_table(accuracy),
        "",
        "## Per-Pair Accuracy",
        "",
        markdown_table(pair_accuracy),
        "",
        "## Plots",
        "",
        "![Words per trial](plots/figure_2_words_per_trial.png)",
        "",
        "![Turns per trial](plots/figure_3_turns_per_trial.png)",
        "",
        "![Words per position](plots/figure_4_words_per_position.png)",
        "",
        "![Accuracy by trial](plots/accuracy_by_trial.png)",
        "",
        "![NP type distribution](plots/np_type_distribution.png)",
        "",
    ]
    path = run_dir / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data._"
    columns = list(df.columns)
    rows: list[list[str]] = []
    for _, row in df.iterrows():
        rows.append([format_value(row[column]) for column in columns])
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
    )


def format_value(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


if __name__ == "__main__":
    main()


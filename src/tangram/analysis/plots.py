from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from tangram.analysis import coding, metrics
from tangram.config import ModelConfig
from tangram.logging import TrialLog, load_trial_logs


def generate_all(
    run_dir: str | Path,
    *,
    llm_coding: bool = False,
    model_config: ModelConfig | None = None,
) -> dict[str, Path]:
    run_path = Path(run_dir)
    logs = load_trial_logs(run_path)
    plot_dir = run_path / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}
    outputs["figure_2"] = _plot_words_per_trial(logs, plot_dir)
    outputs["figure_3"] = _plot_turns_per_trial(logs, plot_dir)
    outputs["figure_4"] = _plot_words_per_position(logs, plot_dir)
    outputs["accuracy"] = _plot_accuracy(logs, plot_dir)

    coded = coding.classify_logs(
        logs,
        cache_path=run_path / "np_type_cache.json",
        llm=llm_coding,
        model_config=model_config,
    )
    coded.to_csv(plot_dir / "np_type_classifications.csv", index=False)
    distribution = coding.np_distribution(coded)
    distribution.to_csv(plot_dir / "np_type_distribution.csv", index=False)
    outputs["np_type_distribution"] = _plot_np_distribution(distribution, plot_dir)
    return outputs


def _plot_words_per_trial(logs: list[TrialLog], plot_dir: Path) -> Path:
    df = metrics.words_by_trial(logs)
    df.to_csv(plot_dir / "figure_2_words_per_trial.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    if not df.empty:
        ax.plot(df["trial"], df["mean_words_per_figure"], marker="o", label="LLM")
        human = pd.DataFrame(
            {"trial": list(metrics.HUMAN_WORDS_PER_FIGURE), "human": list(metrics.HUMAN_WORDS_PER_FIGURE.values())}
        )
        ax.plot(human["trial"], human["human"], marker="s", linestyle="--", label="Human paper")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Mean director words per figure")
    ax.set_title("Words Per Figure")
    ax.legend()
    ax.grid(True, alpha=0.25)
    path = plot_dir / "figure_2_words_per_trial.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_turns_per_trial(logs: list[TrialLog], plot_dir: Path) -> Path:
    df = metrics.turns_by_trial(logs)
    df.to_csv(plot_dir / "figure_3_turns_per_trial.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    if not df.empty:
        ax.plot(df["trial"], df["mean_turns_per_figure"], marker="o", label="LLM")
        human = pd.DataFrame(
            {"trial": list(metrics.HUMAN_TURNS_PER_FIGURE), "human": list(metrics.HUMAN_TURNS_PER_FIGURE.values())}
        )
        ax.plot(human["trial"], human["human"], marker="s", linestyle="--", label="Human paper")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Mean director turns per figure")
    ax.set_title("Turns Per Figure")
    ax.legend()
    ax.grid(True, alpha=0.25)
    path = plot_dir / "figure_3_turns_per_trial.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_words_per_position(logs: list[TrialLog], plot_dir: Path) -> Path:
    df = metrics.words_by_position(logs)
    df.to_csv(plot_dir / "figure_4_words_per_position.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    if not df.empty:
        for trial, group in df.groupby("trial"):
            ax.plot(group["position"], group["mean_words_per_figure"], marker="o", label=f"Trial {trial}")
    ax.set_xlabel("Position")
    ax.set_ylabel("Mean director words")
    ax.set_title("Words Per Position")
    ax.legend()
    ax.grid(True, alpha=0.25)
    path = plot_dir / "figure_4_words_per_position.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_accuracy(logs: list[TrialLog], plot_dir: Path) -> Path:
    df = metrics.accuracy_by_trial(logs)
    df.to_csv(plot_dir / "accuracy_by_trial.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    if not df.empty:
        ax.plot(df["trial"], df["accuracy"], marker="o")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy By Trial")
    ax.grid(True, alpha=0.25)
    path = plot_dir / "accuracy_by_trial.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_np_distribution(df: pd.DataFrame, plot_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not df.empty:
        pivot = df.pivot(index="trial", columns="np_type", values="percentage").fillna(0)
        pivot.plot(kind="bar", stacked=True, ax=ax, width=0.8)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Percent")
    ax.set_title("Initial NP Type Distribution")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")
    path = plot_dir / "np_type_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


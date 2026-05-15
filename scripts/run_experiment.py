from __future__ import annotations

import argparse
from pathlib import Path

from tangram.client import FakeTangramClient
from tangram.config import ExperimentConfig, ModelConfig, default_results_dir, load_dotenv
from tangram.experiment import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM Tangram replication experiments.")
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--provider", choices=["anthropic", "openai"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help="Optional Anthropic extended-thinking budget. Omitted by default.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh"],
        default=None,
        help="Optional OpenAI reasoning effort. Omitted by default.",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--results-dir", type=Path, default=default_results_dir())
    parser.add_argument("--fake", action="store_true", help="Use deterministic fake participants.")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    default_model = ModelConfig()
    provider = args.provider or default_model.provider
    model_name = args.model or (default_model.model if args.provider is None else None)
    model = ModelConfig(
        provider=provider,
        model=model_name,
        max_tokens=args.max_tokens,
        thinking_budget_tokens=args.thinking_budget,
        reasoning_effort=args.reasoning_effort,
    )
    config = ExperimentConfig(
        pairs=args.pairs,
        trials=args.trials,
        max_turns_per_trial=args.max_turns,
        concurrency=args.concurrency,
        seed=args.seed,
        run_id=args.run_id,
        model=model,
        use_fake_client=args.fake,
    )
    client_factory = (lambda: FakeTangramClient()) if args.fake else None
    manifest = run_experiment(config, client_factory=client_factory, results_dir=args.results_dir)
    print(f"Wrote run {manifest.run_id} to {args.results_dir / manifest.run_id}")
    print(f"Trials: {manifest.summary.get('trials', 0)}")
    print(f"Mean accuracy: {manifest.summary.get('mean_accuracy')}")
    print(f"Estimated cost USD: {manifest.summary.get('estimated_cost_usd')}")


if __name__ == "__main__":
    main()

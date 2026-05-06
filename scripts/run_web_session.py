from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from tangram.config import ExperimentConfig, ModelConfig, default_results_dir, load_dotenv
from tangram.human import HumanSessionManager
from tangram.webapp import build_participants, create_human_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local human/LLM Tangram web session.")
    parser.add_argument("--director", choices=["human", "llm"], default="human")
    parser.add_argument("--matcher", choices=["human", "llm"], default="human")
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--thinking-budget", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--results-dir", type=Path, default=default_results_dir())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    model = ModelConfig(
        model=args.model or ModelConfig().model,
        max_tokens=args.max_tokens,
        thinking_budget_tokens=args.thinking_budget,
    )
    config = ExperimentConfig(
        pairs=1,
        trials=args.trials,
        max_turns_per_trial=args.max_turns,
        concurrency=1,
        seed=args.seed,
        run_id=args.run_id,
        model=model,
    )
    run_id = config.resolved_run_id()
    manager = HumanSessionManager(run_id=run_id)
    participants = build_participants(
        manager=manager,
        director=args.director,
        matcher=args.matcher,
    )
    app = create_human_app(
        manager=manager,
        config=config,
        participants=participants,
        results_dir=args.results_dir,
    )
    print(f"Run ID: {run_id}")
    print(f"Open: http://{args.host}:{args.port}/")
    if args.director == "human":
        print(f"Director: http://{args.host}:{args.port}/session/director")
    if args.matcher == "human":
        print(f"Matcher: http://{args.host}:{args.port}/session/matcher")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

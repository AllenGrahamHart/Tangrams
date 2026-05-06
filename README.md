# Tangram LLM Replication

This repository runs a Tangram communication-task replication inspired by Clark & Wilkes-Gibbs (1986), "Referring as a Collaborative Process." It supports LLM and/or human participants in the director and matcher roles, using the original 12 tangram figures across repeated trials, logging each dialogue, and computing the headline measures from the paper.

## Setup

```bash
uv sync --extra dev
cp .env.example .env
```

Fill in `ANTHROPIC_API_KEY` in `.env` before any run involving an LLM participant. A fake deterministic mode is available for tests and dry runs.

The stimuli live in `stimuli/tangrams/A.png` through `stimuli/tangrams/L.png`. They were split from the included paper screenshot so each API image block contains one tangram figure.

## Run

Fake development run:

```bash
uv run python -m scripts.run_experiment --pairs 1 --trials 6 --run-id sample_run --fake
uv run python -m scripts.analyze_results --run-id sample_run
uv run python -m scripts.inspect_transcript --run-id sample_run --pair 0 --trial 1
```

Real Anthropic run:

```bash
uv run python -m scripts.run_experiment \
  --pairs 8 \
  --trials 6 \
  --model claude-sonnet-4-5 \
  --thinking-budget 2000 \
  --concurrency 4 \
  --run-id my_first_run
```

Then analyze:

```bash
uv run python -m scripts.analyze_results --run-id my_first_run
```

## Local Human Web Sessions

The same runner can also be used with human participants through a local browser UI. Start a local session:

```bash
uv run python -m scripts.run_web_session \
  --director human \
  --matcher human \
  --trials 1 \
  --run-id local_human_test
```

Open the printed links in separate browser windows:

```text
http://127.0.0.1:8765/session/director
http://127.0.0.1:8765/session/matcher
```

Supported pairings:

```bash
--director human --matcher human
--director human --matcher llm
--director llm --matcher human
--director llm --matcher llm
```

For mixed human/LLM sessions, fill in `ANTHROPIC_API_KEY` in `.env` first.

## Outputs

Each trial is written to:

```text
results/{run_id}/pair_{pair_id}/trial_{trial}.json
```

The run manifest is:

```text
results/{run_id}/manifest.json
```

Analysis writes plots, CSVs, and a summary report under `results/{run_id}/`.

## Notes

Word counts use whitespace tokenization over director dialogue, not API token counts. Image letters `A` through `L` are only internal IDs; prompts show each participant private numbered images with independently randomized image orders.

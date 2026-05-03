# Sample Run Notes

Run: `results/sample_run/`

This sample was generated with `--fake`, so it is a deterministic smoke test of the orchestration, logging, and analysis pipeline. It is not evidence about real LLM behavior.

## Observations

- The director and matcher completed 1 pair x 6 trials with `done` termination on every trial.
- Accuracy was 100% on every trial because the fake matcher deterministically resolves the correct internal figure ID.
- Director descriptions shortened across trials by design:
  - Trial 1: long holistic descriptions.
  - Trials 2-3: shorter conventionalized names.
  - Trials 4-6: compact labels.
- The generated plots and CSVs were written under `results/sample_run/plots/`.

## Failure Modes This Sample Cannot Test

- Whether real models use private image numbers as dialogue shortcuts.
- Whether real matchers ask useful clarification questions instead of rubber-stamping.
- Whether real directors reliably emit `<done/>`.
- Whether descriptions shorten naturally rather than by scripted fake-client behavior.
- Whether NP-type coding is reliable enough for real transcripts.

The next useful check is a real 1 pair x 6 trial run with Anthropic, followed by manual transcript review before scaling to 8 pairs.


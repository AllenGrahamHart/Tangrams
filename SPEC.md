# LLM Replication of Clark & Wilkes-Gibbs (1986): Referring as a Collaborative Process

## Project overview

We are replicating the experiment from Clark & Wilkes-Gibbs (1986), "Referring as a collaborative process," *Cognition* 22, 1–39 — but with LLMs as participants instead of humans. The original paper is summarised in `clark_wilkes_gibbs_1986.md` (in this repo); read it before starting.

The original study had pairs of humans collaborate to arrange abstract Tangram figures via voice-only dialogue, across 6 trials. Key findings: words-per-figure dropped from ~41 (trial 1) to ~8 (trial 6), turns dropped from ~3.7 to ~1.2, and the partners converged on shorter, more holistic descriptions through iterative mutual acceptance.

Our goal is to test whether LLM pairs reproduce these effects.

---

## Scope

Build a self-contained Python repository that:

1. Runs dyadic LLM-vs-LLM Tangram dialogues via the Anthropic API.
2. Logs everything needed to replicate the paper's analyses.
3. Computes the headline metrics (words/figure, turns/figure, accuracy, NP type distribution) and produces plots analogous to the paper's Figures 2–4.
4. Is parameterizable: number of pairs, number of trials, model choice, thinking budget, system prompt variants.

Do **not** build:
- A multi-agent orchestration framework (no LangGraph, no AutoGen, no agent runtimes — just direct API calls).
- A web UI.
- Sandboxed execution environments — the LLMs only emit text + structured actions.

---

## Tech stack

- **Python 3.11+**
- **Anthropic SDK** (`anthropic`) — use the official Python client.
- **Pydantic** for data models / structured outputs.
- **uv** for dependency management (preferred over pip/poetry).
- **pytest** for tests.
- **matplotlib** for plots.
- **pandas** for analysis.

Keep dependencies minimal. No frameworks beyond these.

Default model: `claude-sonnet-4-5` with extended thinking enabled (medium budget). Make this configurable.

---

## Repository structure

```
tangram-replication/
├── README.md
├── pyproject.toml
├── .env.example
├── clark_wilkes_gibbs_1986.md         # the paper summary, already exists
├── stimuli/
│   └── tangrams/                       # 12 Tangram images, A.png ... L.png
├── src/
│   └── tangram/
│       ├── __init__.py
│       ├── config.py                   # experiment config, model settings
│       ├── stimuli.py                  # loads + encodes the 12 figures
│       ├── prompts.py                  # system prompts for director / matcher
│       ├── protocol.py                 # turn-taking, message structure
│       ├── runner.py                   # runs a single pair through N trials
│       ├── experiment.py               # runs N pairs (the full study)
│       ├── logging.py                  # transcript schema + persistence
│       └── analysis/
│           ├── __init__.py
│           ├── metrics.py              # words/turns/accuracy
│           ├── coding.py               # NP type classifier (LLM-based)
│           └── plots.py                # Figures 2–4 analogues
├── tests/
│   ├── test_stimuli.py
│   ├── test_protocol.py
│   ├── test_runner.py                  # uses mocked API
│   └── test_metrics.py
├── scripts/
│   ├── run_experiment.py               # CLI entry point
│   ├── analyze_results.py              # generates plots + summary stats
│   └── inspect_transcript.py           # pretty-prints a single trial
└── results/
    └── .gitkeep                        # outputs go here, gitignored except .gitkeep
```

---

## Stimuli

12 Tangram images will be placed in `stimuli/tangrams/` as `A.png` through `L.png` before you run anything. Don't try to fetch them. Verify their presence on startup; fail loudly if missing.

Each figure should be loaded as base64-encoded PNG and included in both the director's and matcher's system prompts as image content blocks. Both participants see the same 12 images with the same A–L labels — but **the labels are for our internal tracking only**. Critical: do **not** reveal these labels to the participants. The whole point of the experiment is that they have to invent their own descriptions. Internally we use A–L to identify figures; in the prompts shown to the models, figures are presented as anonymous images.

Each participant's prompt also contains a separate **position grid**, which is a 1–12 ordering. For the director this is the *target* ordering. For the matcher this is their *current* ordering, which starts random.

---

## The experimental protocol

### One pair, one trial

A single trial proceeds like this:

1. The orchestrator generates two random permutations of A–L: one is the director's target, one is the matcher's starting arrangement. (Different permutations — they should not coincidentally match.)
2. The director is prompted: "Trial X. Your target ordering is [position 1 = figure shown in image_1, position 2 = ..., ...]. Begin with position 1."
3. The matcher is prompted with their starting ordering.
4. The director takes a turn (text only). They reference position 1 and describe a figure.
5. The matcher takes a turn. They may ask clarifying questions, propose alternative descriptions, or place a figure. Placement is a structured action (see below).
6. Turns alternate, with speaker-controlled handoff (see below), until the director moves to position 2.
7. Repeat through position 12.
8. Trial ends when the director declares completion (e.g., emits a `done()` action) or after a hard turn cap (say 200 turns total).
9. Compute placements vs. target → record per-figure and per-trial accuracy.

### Cross-trial structure

After trial 1:
- Both grids are re-randomized (new permutations for both director's target and matcher's starting state).
- The director's new target becomes the new ground truth for trial 2.
- **Both participants retain the full conversation history from trial 1.** This is non-negotiable — the shortening effect depends on it.
- The orchestrator inserts a separator turn: "End of trial X. Beginning trial X+1. Your new arrangement: [...]."

Run 6 trials per pair.

### Turn-taking: speaker-controlled handoff

We can't replicate human overlap, but we can approximate installment NPs and continuers. Mechanism:

Each model response ends with one of three structured signals:
- `<yield/>` — passes the floor to the partner. Default for normal turns.
- `<continue/>` — model wants to extend its own turn (e.g., installment NPs); orchestrator immediately calls the same model again with the appended response in history.
- `<done/>` — director only; signals end of trial.

The matcher additionally has access to a `<place figure="..." position="..."/>` action, which can be emitted alongside (before) the yield/continue signal. Multiple `<place/>` calls in a single turn are allowed but discouraged in the prompt.

Encode these as a structured tool/output the model emits at the end of each text response. Implementation choice: use simple XML-style tags parsed from the response text (cheaper than tool-use API). Validate strictly.

Hard cap: a single uninterrupted speaker chain (multiple consecutive `<continue/>`s without a partner turn) is capped at 5. After that, force a yield.

### Trial termination

The director should be instructed to emit `<done/>` once they've walked through position 12 and the matcher has acknowledged. If they never emit `<done/>`, the orchestrator forces termination at the hard turn cap (200 turns). Log the termination reason.

---

## Prompts

### Director system prompt (template)

```
You are participating in a communication experiment. You and a partner have the same 12 abstract figures in front of you, but in different orderings. Your job is to direct your partner to rearrange their figures so the order matches yours.

You will go through positions 1 to 12 in order. For each position, describe the figure that goes there, well enough that your partner can identify it from their set and place it correctly.

You cannot see your partner's arrangement. They cannot see yours. You can only communicate by talking back and forth.

After your partner indicates they have placed the figure correctly, move on to the next position.

You are speaking, not writing. Use natural conversational language. It is fine to be tentative, to use phrases like "kind of like..." or "looks like...", to pause mid-sentence, to ask "you know the one I mean?", or to revise yourself. Your partner can interrupt you, ask for clarification, or propose their own description.

When your partner makes a placement, evaluate whether you believe it's correct based on what they say. If you're confident, move on. If not, address the disagreement.

Your turn ends with one of:
- <yield/>  — pass the floor to your partner
- <continue/>  — you want to keep talking (e.g. you paused mid-thought)
- <done/>  — you've completed all 12 positions and your partner has acknowledged the last one

Emit exactly one of these tags at the very end of every response.

[12 IMAGES INSERTED HERE — figure 1, figure 2, ..., figure 12 in random order, no labels visible to you]

Your target ordering for trial 1 is:
  Position 1: [figure shown in image #N1]
  Position 2: [figure shown in image #N2]
  ...
  Position 12: [figure shown in image #N12]

Begin with position 1.
```

Important: the image ordering shown to the director is *also* randomized so the model cannot use image position as a reference shortcut. Image ordering should be regenerated for each trial.

### Matcher system prompt (template)

```
You are participating in a communication experiment. You and a partner have the same 12 abstract figures in front of you, but in different orderings. Your partner's job is to direct you to rearrange your figures so the order matches theirs.

You cannot see your partner's arrangement. They cannot see yours. You can only communicate by talking back and forth.

For each position your partner describes, identify which figure they mean and place it. You can:
- Ask clarifying questions
- Propose your own description ("oh, the one that looks like X?")
- Acknowledge and move on
- Push back if you think there's a mistake

When you decide to place a figure, emit a placement action:
  <place figure="N" position="P"/>
where N is the image number you're placing (1–12) and P is the position your partner asked about (1–12). You can place at most one figure per turn.

You are speaking, not writing. Use natural conversational language.

Your turn ends with one of:
- <yield/>  — pass the floor to your partner
- <continue/>  — you want to keep talking

Emit exactly one of these tags at the very end of every response.

[12 IMAGES INSERTED HERE — figure 1, figure 2, ..., figure 12 in random order]

Your starting arrangement for trial 1 is:
  Position 1: [figure shown in image #M1]
  Position 2: [figure shown in image #M2]
  ...
  Position 12: [figure shown in image #M12]

Wait for your partner to begin.
```

### Critical: image labels and position references

The director's image positions and the matcher's image positions are independently randomized. Each side internally references "image #5" but these refer to *different actual figures*. Internal A–L IDs are kept by the orchestrator and never appear in either prompt.

When a matcher emits `<place figure="3" position="7"/>`, the orchestrator translates: matcher's image #3 → A–L ID → check if it matches the director's target for position 7.

### Prompt versioning

Store the system prompts in `prompts.py` as constants with a version number. Log the version into every transcript. We may iterate on prompts and need to know which version produced which results.

---

## API integration

Use the Anthropic Python SDK. Each turn is a single `messages.create` call. Maintain two parallel message histories (one per participant). The orchestrator decides whose turn it is and calls the appropriate API.

### Model & thinking config

```python
DEFAULT_CONFIG = {
    "model": "claude-sonnet-4-5",
    "max_tokens": 4096,
    "thinking": {"type": "enabled", "budget_tokens": 2000},
}
```

Make this overridable per-experiment via `config.py`.

### Thinking traces

**Critical**: extended thinking output must NOT be included in the dialogue history visible to the partner. Each model sees only their own thinking + the partner's final text responses. Implementation: when adding a turn to the partner's message history, strip `thinking` blocks; when adding to the speaker's own history (for their next turn), preserve them per Anthropic's API requirements.

### Rate limiting & retries

Use exponential backoff on 429s and 5xxs. Allow concurrent pair runs (a single trial is sequential by nature, but different pairs can run in parallel). Cap concurrency at 4 by default; configurable.

### Cost tracking

Log token usage (input + output + thinking) per turn. Include a final cost estimate per pair and per experiment in the run summary.

---

## Logging schema

Every trial produces a JSON file at `results/{run_id}/pair_{i}/trial_{j}.json`:

```json
{
  "run_id": "2026-05-03-T14-30-experiment-name",
  "pair_id": 0,
  "trial": 1,
  "timestamp_start": "2026-05-03T14:30:12Z",
  "timestamp_end": "2026-05-03T14:32:45Z",
  "model": "claude-sonnet-4-5",
  "prompt_version": "v1",
  "director_target": ["A", "F", "C", ...],         // length 12, A–L IDs
  "matcher_initial": ["G", "B", "L", ...],
  "matcher_image_mapping": {"1": "G", "2": "B", ...},   // matcher's image-N → A–L
  "director_image_mapping": {"1": "A", "2": "F", ...},  // director's image-N → A–L
  "turns": [
    {
      "turn_index": 0,
      "speaker": "director",
      "text": "Okay, position one looks like a person ice skating, with two arms out front.",
      "thinking": "...",                  // present only for own thread, never broadcast
      "actions": [],
      "handoff": "yield",
      "tokens": {"input": 1234, "output": 89, "thinking": 412},
      "wall_time_ms": 3214
    },
    {
      "turn_index": 1,
      "speaker": "matcher",
      "text": "Got it — placing.",
      "actions": [{"type": "place", "figure_image_n": 7, "position": 1, "resolved_id": "A"}],
      "handoff": "yield",
      "tokens": {...},
      "wall_time_ms": 2103
    }
  ],
  "final_placements": ["A", "F", "X", ...],  // matcher's final ordering, A–L
  "accuracy_per_position": [true, true, false, ...],
  "accuracy_overall": 0.917,
  "termination": "done",                     // or "turn_cap" or "error"
  "total_tokens": {"input": ..., "output": ..., "thinking": ...},
  "estimated_cost_usd": 0.42
}
```

Also emit a top-level `results/{run_id}/manifest.json` with the experiment config, all pair IDs, summary stats, and references to the per-trial files.

---

## Analysis module

The analysis module reads the JSON logs and computes:

### Metrics (mirror the paper's Figures 2–4 and Tables 1–2)

In `analysis/metrics.py`:

1. **Words per figure × trial**: tokenize director utterances per figure (use whitespace tokenization to mirror the paper's word counts; note in docs that this differs from API tokens). Plot mean across pairs by trial. Target: replicate Figure 2.

2. **Turns per figure × trial**: count director turns per figure-placement. Plot mean by trial. Target: Figure 3.

3. **Words per figure × position × trial**: same as (1) but stratified by position (1–12). Plot for trials 1, 2, 6. Target: Figure 4.

4. **Basic exchange rate by trial**: a "basic exchange" = director presents NP → matcher places + acknowledges in one turn → director moves on. Detect heuristically: a placement happens within the matcher's first response after a director-NP, with no intervening clarification turns.

5. **Accuracy by trial**: mean across pairs.

### NP-type coding

In `analysis/coding.py`: for each director utterance that introduces a figure, label the initial NP type (elementary / episodic / installment / provisional / dummy / proxy / description / unclassified) using a separate Claude call as a classifier. Define a clear classification prompt with the definitions from the paper. Cache classifications by utterance hash so we don't re-classify on re-runs.

Output Table 2 analogue: percentages of each NP type by trial.

### Plots

In `analysis/plots.py`: matplotlib plots with simple, clean styling. Save as PNG to `results/{run_id}/plots/`.

- `figure_2_words_per_trial.png`
- `figure_3_turns_per_trial.png`
- `figure_4_words_per_position.png`
- `accuracy_by_trial.png`
- `np_type_distribution.png`

Each plot should also save its underlying data as CSV alongside.

### Summary report

`scripts/analyze_results.py` produces `results/{run_id}/summary.md` with:
- Experiment config recap.
- Headline numbers (mean words/turns by trial, accuracy by trial).
- Comparison table: LLM results vs. the paper's human numbers (which are documented in `clark_wilkes_gibbs_1986.md` §4).
- Inline plot embeds.
- Per-pair accuracy breakdown.
- Cost summary.

---

## CLI

`scripts/run_experiment.py`:

```bash
python -m scripts.run_experiment \
    --pairs 8 \
    --trials 6 \
    --model claude-sonnet-4-5 \
    --thinking-budget 2000 \
    --concurrency 4 \
    --run-id my_first_run
```

`scripts/analyze_results.py`:

```bash
python -m scripts.analyze_results --run-id my_first_run
```

`scripts/inspect_transcript.py`:

```bash
python -m scripts.inspect_transcript --run-id my_first_run --pair 0 --trial 1
```

The inspect script should pretty-print a transcript with speaker labels, timestamps, placement events, and handoff signals — useful for debugging and for the qualitative read-through of trial 1 transcripts.

---

## Tests

Mock the Anthropic API in tests using a fake client that returns canned responses. Test:

- `test_stimuli.py`: 12 figures load correctly, base64 encoding works, missing files raise.
- `test_protocol.py`: handoff parsing, structured action parsing, turn-cap enforcement, malformed responses are handled gracefully.
- `test_runner.py`: a single trial runs end-to-end with mocked responses, transcript schema is valid, accuracy computation is correct.
- `test_metrics.py`: metric functions handle edge cases (zero turns, perfect accuracy, all basic exchanges).

Don't test the actual model behavior — that's what running the experiment is for.

---

## Implementation order

Suggested sequence:

1. Stimuli loading + image encoding. Get the 12 figures into base64 reliably.
2. Prompt templates with image insertion + per-side ordering randomization.
3. Single API call wrapper that handles thinking, retries, token tracking.
4. Turn parser (handoff signals + place actions).
5. Trial runner: orchestrate two threads through one trial.
6. Multi-trial runner: preserve history across trials, re-randomize.
7. Multi-pair runner with concurrency.
8. Logging + transcript persistence.
9. Metrics module (words, turns, accuracy).
10. Plot module.
11. NP-type classifier module (last — needs real transcripts to validate against).
12. CLI scripts.
13. Tests throughout, not at the end.

After step 5, manually run a single pair through 6 trials and read the transcripts. If the dialogue looks plausibly Tangram-like — directors describing figures, matchers placing, descriptions shortening over trials — proceed. If it doesn't, fix the prompts before scaling.

---

## Things to watch for

- **Thinking traces leaking into partner context** — easy to introduce, hard to detect. Add an assertion in the runner that strips thinking before broadcasting.
- **Image ordering as a referent shortcut** — if the director says "the third image" and the matcher's third image is the same figure, they're cheating the experiment. Re-randomize image ordering per side per trial.
- **Matcher rubber-stamping** — if the matcher places without engaging, accuracy will be near-chance. Watch for this in the first trial run; consider a system-prompt nudge.
- **Director never emitting `<done/>`** — set the turn cap and log termination reasons. If most trials hit the cap, the protocol prompt needs work.
- **Token counts as a proxy for word count** — the paper measures whitespace-split words. Use the same in metrics, not API tokens. Document this clearly.
- **Cost** — at thinking-budget 2000 with Sonnet, a single trial of ~50 turns is roughly $0.50–1.00. 8 pairs × 6 trials ≈ $25–50 per run. Log it; don't be surprised.

---

## Out of scope (explicitly)

- Multi-party (3+ participants) — keep it strictly dyadic.
- Cross-model pairs (e.g. Sonnet director + Opus matcher) — interesting but a follow-up.
- Voice / TTS — text only.
- Live observation / streaming — batch only.
- Comparison to human baseline data beyond what the paper publishes.

These are good follow-up studies once the dyadic baseline is solid.

---

## Deliverables

When you finish:

1. The repo as specified.
2. A successful end-to-end run of 1 pair × 6 trials with transcripts and plots, committed to `results/sample_run/`.
3. A `RESULTS_SAMPLE.md` at the repo root summarising what you observed in the sample run — was the dialogue plausible? Did references shorten? Any obvious failure modes? This is qualitative.
4. Confirmation that all tests pass.

After delivery, we'll do a manual review of the sample run and iterate on prompts before scaling up to a full 8-pair experiment.

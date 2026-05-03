# Analysis Scope

This project aims to replicate Clark and Wilkes-Gibbs (1986) as closely as is practical with text-only LLM participants. The original study analyzed tape-recorded human dialogues and checked transcripts for speaker changes, back-channels, interruptions, hesitations, false starts, parentheticals, and basic intonational features. Our LLM runs produce complete text transcripts, but not audio, timing of overlap, gesture, gaze, or intonation.

## Primary Quantitative Comparisons

These can be computed directly from trial logs and should be reported for every model/run.

| Analysis | Paper analogue | Our measure |
|---|---|---|
| Words per figure by trial | Figure 2 | Whitespace word count over director turns assigned to each figure position |
| Director turns per figure by trial | Figure 3 | Count of director turns assigned to each figure position |
| Words per figure by position within trial | Figure 4 | Director word count by position 1-12, especially trials 1, 2, and 6 |
| Accuracy / error rate | Reported 2% human error rate | Final matcher ordering vs. director target by position |
| Figure-level difficulty | Figure B hardest, C easiest in paper | Mean words, turns, and error rate by internal figure ID |
| Basic exchange rate | Paper's basic exchange percentages | Heuristic: first matcher response after director presentation includes placement with no intervening clarification turns |
| Cost and token use | No human analogue | API usage and estimated cost by turn, trial, pair, and run |

## Manual Qualitative Coding

These concepts are central to the paper, but they require linguistic judgment. We should code them manually from transcripts rather than treating simple heuristics as ground truth. LLM-assisted coding can be used for triage, but any reported result should identify the coding procedure.

| Coding target | Notes |
|---|---|
| Initial NP type | Elementary, episodic, installment, provisional, dummy, proxy, description, unclassified |
| Repair / refashioning | Self-repair, repeated words/covert repair, self-expansion, matcher-prompted expansion, matcher-supplied expansion, replacement |
| Acceptance process | Presupposed acceptance, asserted acceptance, rejection, postponement |
| Follow-ups | Director checks or additions after apparent matcher acceptance |
| Perspective strategy | Resemblance, categorization, attribution, action |
| Perspective basis | Permanent/enduring properties vs. temporary/procedural properties |
| Analogical vs. literal | Whether the reference treats the figure as resembling an object/action/person or describes geometric parts literally |
| Holistic vs. segmental | Whether the figure is characterized as a whole or as a set of parts |
| Simplification and narrowing | Cross-trial changes in a pair's accepted descriptions for the same figure |
| Interface violations | Cases where a participant tries to use private image numbers or otherwise relies on the software interface rather than task dialogue |

## Explicitly Out of Scope

These analyses depend on data that text-only LLM runs do not provide.

- True intonation, including rising intonation / try markers as acoustic events.
- Overlap, simultaneous speech, and interruption timing.
- Non-verbal behavior such as head nods, gaze, gesture, or manipulation timing.
- Tape-transcription checks that require audio replay.

Textual substitutes can still be noted qualitatively, for example question marks, written hesitations, or explicit continuers, but they should not be presented as direct equivalents to the paper's audio-based evidence.

## Recommended Reporting Structure

1. Report the primary quantitative comparisons for each model/run.
2. Compare those values to the published human numbers where available.
3. Manually inspect a stratified set of transcripts: early vs. late trials, correct vs. incorrect placements, easy vs. difficult figures.
4. Qualitatively discuss whether LLM pairs appear to form shared referring expressions across trials.
5. Only then discuss model-human differences, keeping prompt version, model ID, and accuracy constraints explicit.


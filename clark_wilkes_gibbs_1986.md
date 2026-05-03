# Referring as a Collaborative Process

**Authors:** Herbert H. Clark & Deanna Wilkes-Gibbs (Stanford University)
**Citation:** *Cognition*, 22 (1986), 1–39

---

## Abstract (paraphrased)

In conversation, speakers and addressees jointly construct definite references. The authors propose that a speaker initiates the process by *presenting* (or inviting) a noun phrase. Before moving to the next contribution, both participants — if necessary — repair, expand, or replace the noun phrase iteratively until they reach a version they *mutually accept*. They try to **minimize joint effort**. The preferred procedure: the speaker presents a simple noun phrase and the addressee accepts it by allowing the next contribution to begin. The paper describes a communication task in which pairs of people arranged complex (Tangram) figures and shows how the proposed model accounts for the references they produced. The model derives from the **mutual responsibility** participants bear for understanding each utterance.

---

## 1. Theoretical Background

### 1.1 The "literary model" of reference (the strawman)

Traditional accounts assume:
1. Reference is expressed via one of three standard noun phrase types: proper noun, definite description, or pronoun.
2. The speaker uses the noun phrase intending the addressee to identify the referent uniquely from common ground.
3. The speaker's intention is satisfied simply by issuing the noun phrase.
4. The course of the process is controlled by the speaker alone.

This view fails for conversation because (a) speakers have limited time for planning and revision, (b) speech is evanescent (must be processed in real time), and (c) listeners are not mute or invisible — they react, and speakers can adapt mid-utterance.

### 1.2 Eight types of "non-literary" noun phrases

The authors enumerate eight phenomena that defeat the literary model. These categories are essential for coding LLM dialogues:

| # | Type | Description | Example |
|---|------|-------------|---------|
| 1 | **Self-corrected NP** | Speaker corrects own NP mid-utterance | "all the people that were gone this year I mean this quarter" |
| 2 | **Expanded NP** | Speaker adds a parenthetical when initial NP feels insufficient | "the spout — the little one that looks like the end of an oil can —" |
| 3 | **Episodic NP** | Single NP delivered in two intonation contours / tone groups | "the other large tube. With the round top." |
| 4 | **Other-corrected NP** | Addressee repairs the speaker's NP across turns | A: "Monday." → B: "oh yih mean like a week f'm tomorrow." → A: "Yah." |
| 5 | **Trial NP** | NP delivered with rising/try-marker intonation, soliciting confirmation | "the small blue cap we talked about before?" |
| 6 | **Installment NP** | NP delivered piece by piece, with explicit confirmation between installments | A: "the hole on the side of that tube" → B: "Yeah" → A: "that is nearest to the top…" |
| 7 | **Dummy NP** | Placeholder term used while the real NP is being prepared | "what's-his-name", "whatchamacallit" |
| 8 | **Proxy NP** | Addressee supplies the NP that the speaker was reaching for | A: "That tree has, uh, uh…" → B: "tentworms." → A: "Yeah." |

### 1.3 Mechanisms for establishing understanding

Listeners signal comprehension through:
- **Continuers** (e.g., "mh hm", "uh huh") inserted while the speaker is still talking.
- **Allowing the next contribution** — by simply continuing the conversation, the listener implicitly accepts the prior reference.
- **Asserted acceptance** — explicit "yeah", "right", "okay", "I see", head nods.
- **Side sequences** — interruptions for repair/clarification before the conversation proceeds.
- **Anticipatory interruption** — listener interrupts as soon as they identify the referent (e.g., "Oh Gina!").

---

## 2. The Collaborative Model

### 2.1 Core idea: mutual acceptance

A and B must *mutually accept* that B has understood A's reference before they let the conversation move on. Common ground accumulates only if each contribution is accepted before the next begins. The two basic elements are:

1. **Presentation** — the speaker offers a noun phrase, presupposing that:
   - the addressee is paying attention and understands the language,
   - the addressee can view referent *r* under description *d*,
   - description *d* plus common ground uniquely picks out *r*.
2. **Acceptance** — the addressee either *presupposes* acceptance (by continuing) or *asserts* it (with continuers/affirmatives).

If acceptance fails, the NP must be **refashioned**.

### 2.2 Acceptance is recursive

Mutual acceptance can be modeled as a recursive process (Table 3 of the paper):

```
Initiating a reference:
    To initiate a reference         → present x₁  OR  invite x₁
    If x₁ is invited                → present x₁

Refashioning a noun phrase:
    If xᵢ is inadequate             → present revision x'ᵢ
                                      OR expansion yᵢ
                                      OR replacement zᵢ
                                      OR request x'ᵢ, yᵢ, or zᵢ
    If x'ᵢ, yᵢ, or zᵢ is requested  → present x'ᵢ, yᵢ, or zᵢ
    If x'ᵢ, yᵢ, or zᵢ is presented  → let x_{i+1} = x'ᵢ, xᵢ + yᵢ, or zᵢ

Concluding a reference:
    If xᵢ is adequate               → accept xᵢ
    If xᵢ is adequate and accepted  → conclude mutual acceptance
```

### 2.3 Projection rules — what each NP type "expects" next

Each NP type projects a preferred next move (Table 1 of the paper):

| Type of NP | Projected next (unmarked) | Projected next (with try marker) |
|---|---|---|
| Elementary | Implicated acceptance | Explicit verdict |
| Episodic | Implicated acceptance | Explicit verdict |
| Installment | Explicit acceptance | Explicit verdict |
| Provisional | Self-expansion | Self-expansion |
| Dummy | Self-expansion | Proxy |
| Proxy | Explicit acceptance | Explicit verdict |

---

## 3. The Experiment

This is the core of what we want to replicate with LLMs.

### 3.1 Materials

- **12 Tangram figures**, each made from 7 elementary geometric shapes, drawn from Elffers (1976)'s collection of 4000 such figures from the ancient Chinese game.
- The 12 figures were selected for varying abstraction and similarity → range of difficulty.
- Two physical copies of each figure, cut from black construction paper, mounted on white cards (15 cm × 20 cm).
- Figures are abstract enough that no canonical name exists — participants must invent descriptions.

### 3.2 Participants

- **8 pairs** of Stanford undergraduates (16 students total: 7 men, 9 women).
- Fulfilling a course requirement.

### 3.3 Procedure

- Two students sat at tables **separated by an opaque screen** (no visual access to each other or to the partner's cards).
- Each had the same 12 figures laid out in a 2 × 6 grid (positions numbered 1–6 on top, 7–12 on bottom).
- Roles drawn by lot:
  - **Director** — has the figures arranged in a *target sequence* and must get the matcher's grid to match.
  - **Matcher** — starts with a *random* arrangement and must rearrange to match the target.
- The director was instructed to go through positions sequentially (1 → 12).
- They could talk back and forth as much as needed.
- After completion, errors were checked, both grids were re-randomized, the director's new arrangement became the new target, and the procedure repeated.
- **6 trials per pair.**
- Each session took ~25 minutes.
- Sessions were timed and tape-recorded.
- Error rate across the experiment: ~2%.

### 3.4 Transcription

- Conversations transcribed including: speaker changes, back-channel responses, parentheticals, interruptions, hesitations, false starts, and basic intonational features.
- Two-pass: one author transcribed; the other checked (especially for intonation).
- Total corpus: **9,792 words**, covering **576 figure placements** (12 figures × 6 trials × 8 pairs).

### 3.5 Roles & assumed pronouns

The paper writes about the director with male pronouns ("he") and the matcher with female ("she") for ease of exposition only — both sexes filled both roles.

---

## 4. Key Quantitative Results

### 4.1 Words per figure (by trial)

The director's average word count per figure dropped sharply across trials (Figure 2):

| Trial | Mean words per figure |
|---|---|
| 1 | ~41 |
| 2 | ~19 |
| 3 | ~13 |
| 4 | ~11 |
| 5 | ~9 |
| 6 | ~8 |

Statistic: F(1,35) = 44.31, p < .001. Steepest drop is from trial 1 → 2; nearly flat by trial 6.

### 4.2 Turns per figure (by trial)

Director's mean turns per figure (Figure 3):

| Trial | Mean turns per figure |
|---|---|
| 1 | ~3.7 |
| 2 | ~1.9 |
| 3 | ~1.4 |
| 4 | ~1.3 |
| 5 | ~1.1 |
| 6 | ~1.2 |

Statistic: F(1,35) = 79.59, p < .001.

### 4.3 Words per figure across positions within a trial

Within a trial, fewer words are needed for figures placed later (because fewer remaining candidates). Slope decreases across trials:

- Trial 1: −4.6 words per position (steep), F(1,77) = 40.01, p < .001
- Trial 2: −1.0 words per position, F(1,77) = 5.83, p < .05
- Trial 6: −0.4 words per position, F(1,77) = 7.16, p < .05

Difference in slopes across trials: F(2,284) = 15.49, p < .001. **This is a key prediction unique to the collaborative view** — pure information-theoretic accounts predict slopes should remain constant across trials.

### 4.4 Figure-level difficulty

- Significant variation across the 12 figures: F(11,77) = 5.94, p < .001.
- Hardest: Figure B (26.5 words/trial average; 39.6 words on trial 1).
- Easiest: Figure C (9.7 words/trial average; 24 words on trial 1).

### 4.5 "Basic exchange" rate by trial

A *basic exchange* = director presents NP → matcher gives one-token acceptance ("okay"). Percentages of placements that were basic exchanges:

| Trial | % basic exchanges |
|---|---|
| 1 | 18% |
| 2 | 55% |
| 3 | 75% |
| 4 | 80% |
| 5 | 88% |
| 6 | 84% |

Statistic: F(1,55) = 84.19, p < .001.

### 4.6 Distribution of initial-NP types (trials 2–6)

Percentages of initial NPs by type across trials 2–6 (N = 96 per column; Table 2):

| Type | T2 | T3 | T4 | T5 | T6 |
|---|---|---|---|---|---|
| Elementary | 52 | 68 | 69 | 80 | 72 |
| Episodic | 11 | 10 | 8 | 6 | 5 |
| Installment | 0 | 0 | 0 | 0 | 1 |
| Provisional | 17 | 14 | 8 | 2 | 6 |
| Dummy | 0 | 0 | 0 | 0 | 0 |
| Proxy | 0 | 1 | 2 | 1 | 1 |
| Description (categorize, not refer) | 17 | 7 | 12 | 9 | 14 |
| Unclassified | 3 | 0 | 1 | 2 | 1 |

- Elementary NPs increase across trials: F(1,28) = 17.02, p < .01.
- Episodic + provisional NPs decrease: F(1,28) = 9.02, p < .01.
- Installment, dummy, and proxy NPs are too rare to test.

### 4.7 Other useful counts

- **Self-repairs** by trial (1→6): 85, 30, 20, 8, 7, 6.
- **Repeated words** (a covert-repair proxy): 47, 14, 10, 4, 7, 1.
- **Self-expansions**: trial 2→6: 25%, 17%, 11%, 6%, 10% of figure placements.
- **Requests for expansion** (matcher prompting director): trial 1→6: 36%, 12%, 8%, 3%, 1%, 3%.
- **Matcher prompts via repetition with rising intonation**: 15%, 3%, 3%, 2%, 1%, 1%.
- **Matcher requests for confirmation** (e.g., "kind of standing up?"): 37%, 12%, 8%, 6%, 1%, 2%.
- **Matcher replacements** (rejecting the director's NP and offering an alternative): 10%, 5%, 0%, 2%, 2%, 0%.
- **Follow-ups** by director after acceptance: 35%, 12%, 6%, 6%, 1%, 5%.

---

## 5. The Acceptance Process — Categories for Coding

A complete coding scheme for analyzing dialogue would track three sub-processes within each reference:

### 5.1 Initiating a reference (six initial-NP types)

(see §1.2 above) — elementary, episodic, installment, provisional, dummy, proxy. Any of these may carry a **try marker** (rising intonation soliciting verdict).

### 5.2 Refashioning a noun phrase

| Mechanism | Description |
|---|---|
| **Repair** (self) | Speaker corrects own NP mid-stream. |
| **Repair** (covert) | Speaker stutters/restarts, possibly correcting silently. |
| **Expansion** (self) | Speaker, judging own NP inadequate, adds a clause (compound *x + y*). |
| **Expansion** (other-prompted) | Matcher signals uncertainty (e.g., "uh—", silence, repetition with rising intonation), speaker complies. |
| **Expansion** (matcher-supplied) | Matcher proposes a clarifying clause as a request for confirmation; speaker accepts. |
| **Replacement** | Matcher rejects director's *x* and offers an alternative *z* under a new perspective. Final accepted form is *z*, not *x + z*. |

### 5.3 Passing judgment on a presentation

| Verdict | Mechanism |
|---|---|
| **Acceptance** | Presupposed (continue) or asserted ("yeah", "okay"). |
| **Rejection** | Asserted ("no, that's not it") or implicated (offering a replacement). |
| **Postponement** | Tentative "okay" — accepts so far, defers final judgment until expansion. |
| **Interruption** | Cut off speaker once enough has been understood. |

### 5.4 Follow-ups

After acceptance, the director may add a check ("you've got that one, right?") if they aren't confident the matcher really has the figure. Frequency drops sharply across trials.

### 5.5 Reconsideration

A previously mutually-accepted reference can later be revoked if either party finds reason to doubt it.

---

## 6. The Principle of Least Collaborative Effort

Classical least-effort theories (Brown, Lenneberg, Krauss & Glucksberg, Olson, Zipf) assume the speaker works alone and minimizes their own NP length. The authors instead propose:

> **Principle of Least Collaborative Effort** — Speakers and addressees try to minimize the *combined* work both parties put in, from initiation of the referential process to its completion.

There is a **trade-off** between effort spent on the initial NP and effort spent refashioning. Why don't speakers always pre-design a perfect NP?

1. **Time pressure** — they cannot design ideal NPs in real time.
2. **Complexity** — the NP is too long/complex for the listener to absorb in one shot.
3. **Ignorance** — they don't know what description the addressee will accept; trial-and-error is forced.

The six initial-NP types and various refashioning devices are all responses to these constraints.

**Predictions of LCE that distinguish it from classical least-effort:**

- References can use *non-standard* / *non-literary* NPs.
- References can use NPs the speaker knows are *inadequate* in context.
- References can use devices that *draw the addressee in*.
- There are trade-offs between initial-NP effort and refashioning effort.
- Self-repair and self-initiated repair are preferred.
- Expansions, replacements, and informative requests for expansion occur.
- The *canonical reference* (elementary NP + presupposed acceptance) is preferred.

---

## 7. Perspective and Change in Perspective

### 7.1 Establishing a common perspective

On **trial 1**, directors did *not* use definite references for the Tangram figures (e.g., they did not say "the ice skater"). Instead, they *described* the figures using one of four strategies:

1. **Resemblance** — "looks like a fat person sitting down…"
2. **Categorization** — "is a diamond on top with a thing that looks like a ripped-up square"
3. **Attribution** — "has a triangle pointing to the left in the bottom left-hand corner"
4. **Action** — "is pointing right"

After trial 1, **89% of initial utterances used identificational definite references** (e.g., "the person ice skating that has two arms?").

Why no definite reference on trial 1? Because no shared perspective on these abstract figures yet exists; trying to presuppose one would cost too much collaborative effort.

### 7.2 Bases for definite reference

Two main bases observed:

- **Permanent / enduring properties** — shape, appearance, identity. Used in **90%** of references.
- **Temporary / procedural properties** — list position, prior mistakes, prior failures. Used in **2%** of references; **7%** combined both.

Permanent properties dominate because they are salient, distinctive, and stable — easier to mutually accept.

### 7.3 Holistic vs. segmental perspectives

- **Analogical** perspectives (the figure as a whole resembles X): 84% introduced with "looks like" or "resembles"; 42% hedged with "sort of", "kind of", "something like".
- **Literal** perspectives (geometric segments): 89% introduced with "is" or "has"; only 4% hedged.

Analogical perspectives are **holistic**; literal perspectives are **segmental**. LCE predicts a preference for holistic. Observed:

- Of 80 trial-1 references introducing an analogical perspective, 93% had the analogical first or alone.
- When parts were also mentioned, parts came after the holistic concept 89% of the time.
- 42% of trial-1 figures got both perspectives; by trial 6, 77% of references used analogical alone, only 19% included literal elements.
- **559 of 576 figure placements (97%)** included some form of analogical depiction.

### 7.4 Refining perspectives over trials

Two main types of refinement:

**Simplification** — details are dropped. Example progression for figure C:
1. "looks like, sort of like an angel flying away or something. It's got two arms"
2. "looks like the angel flying away, or that's what I said last time"
3. "the flying one"
4. "the one that looks like an angel"
5. "the angel one"
6. "the angel"

**Narrowing** — focus moves to one part:
- Central part: "the graduate at the podium" → "the graduate"
- Peripheral but distinctive part: "the guy in a sleeping bag" → "Sleeping bag"

### 7.5 Multinary vs. unitary categories

- **Multinary** category: described via multiple concepts ("person with a leg sticking out back").
- **Unitary** category: a single encompassing concept ("ballerina").

LCE predicts preference for unitary; observed: simplification and narrowing both move from multinary toward unitary.

---

## 8. Generalization: The Principle of Mutual Responsibility

> **Principle of Mutual Responsibility** — The participants in a conversation try to establish, roughly by the initiation of each new contribution, the mutual belief that the listeners have understood what the speaker meant in the last utterance to a criterion sufficient for current purposes.

Two important caveats:

1. **"Roughly by the initiation of each new contribution"** — acceptance may not happen word-by-word; the natural granularity is the *contribution* (~one sentence or quasi-sentence on the topic). Repairs/expansions/replacements typically happen immediately after a presentation and before the next contribution begins.

2. **"To a criterion sufficient for current purposes"** — the criterion varies by context. A high criterion (the Tangram task) demands explicit acceptance; a low criterion (small talk) tolerates vague understanding. Listeners face pressure to *feign* understanding to avoid offending the speaker, revealing incompetence, or adding extra effort.

### 8.1 Modes of language use

The collaborative mode is one of several. Variants include:

- **Distant responsibility** (literary, broadcast, lecture): speaker tries to ensure the addressee *should have been able* to understand, with no real-time feedback. Subtypes:
  - Spontaneous monologue (e.g., recording into a tape recorder) — still produces repairs/expansions/replacements, but no conversational shortening (Krauss & Weinheimer, 1966).
  - Planned writing — typically eliminates everything but elementary NPs.
- **Telephone vs. typed dialogue** (Cohen, 1985) — both are collaborative, but typed produces coarser-grained collaboration: fewer fine-grained interruptions, less use of intonation, slower pacing.
- **Asymmetric / hierarchical** (officer/private, interviewer/applicant, parent/child) — the higher-status party initiates side sequences, requests confirmations, offers replacements; the lower-status party qualifies, hedges, seeks acceptance with "you know" (Ragan, 1983).

---

## 9. Replication Notes for an LLM Version

What we'd need to build:

### 9.1 Stimuli
- Either reuse the original 12 Tangram figures (if obtainable as images), or generate analogous abstract shapes.
- Each figure must be (a) recognizable enough that an analogical description is possible, (b) abstract enough that no canonical name exists, (c) distinct from the others.

### 9.2 Roles
- **Director LLM**: receives a target ordering of 12 figures; must communicate it.
- **Matcher LLM**: starts with a random arrangement; must produce the target arrangement using only the dialogue.
- A "screen" between them = they cannot see each other's grid state, only exchange messages.

### 9.3 Trials
- 6 trials per pair, with re-randomization between trials.
- Director's new arrangement becomes the new target each round.

### 9.4 Measurements
- Words per figure × trial (replicate Figure 2 trend).
- Turns per figure × trial (replicate Figure 3 trend).
- Words per figure × position within trial (replicate Figure 4: slope flattens across trials).
- Basic-exchange rate by trial.
- Distribution of initial-NP types (Table 2).
- Frequency of repairs, expansions, replacements, follow-ups.
- Accuracy (compare to ~2% human error rate).

### 9.5 Coding scheme
For each reference, label:
- Initial NP type (elementary / episodic / installment / provisional / dummy / proxy / description).
- Presence of try marker.
- Refashioning events (self-repair / self-expansion / matcher-prompted expansion / matcher-supplied expansion / replacement).
- Verdict (presupposed accept / asserted accept / reject / postpone / interrupt).
- Follow-up events.
- Perspective type (analogical / literal / both).
- Holistic vs. segmental.
- Multinary vs. unitary category.
- Basis (permanent vs. temporary properties).

### 9.6 Statistical comparisons
- Cross-trial trends (linear F-tests as in original).
- Slope differences across trials (Trial × Position interaction).
- Figure-level difficulty variance.

### 9.7 Open questions for the LLM version
- Do LLMs exhibit the same shortening curve? Or do they jump straight to optimal references (since they have no real-time pressure)?
- Do LLMs use any of the non-elementary NP types (try markers, installments, proxies) without prompting?
- Does the slope-flattening effect (the LCE-unique prediction) replicate, or do LLMs behave more like classical least-effort agents?
- Do LLMs prefer holistic / analogical / unitary perspectives, or do they default to literal / segmental ones?
- Does varying the model on each side, or giving them different "personas" or "roles", affect the collaborative dynamics?

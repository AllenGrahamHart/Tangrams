from __future__ import annotations

from collections.abc import Sequence

from tangram.stimuli import invert_image_mapping


PROMPT_VERSION = "v1"


DIRECTOR_SYSTEM = """You are participating in a communication experiment. You and a partner have the same 12 abstract figures in front of you, but in different private image orderings. Your job is to direct your partner to rearrange their figures so the order matches yours.

You will go through positions 1 to 12 in order. For each position, describe the figure that goes there well enough that your partner can identify it from their set and place it correctly.

You cannot see your partner's arrangement. They cannot see yours. You can only communicate by talking back and forth.

Your private image numbers are for your own use only. Never say "image 1", "my image 1", or any other image number to your partner. Your partner's private image numbers are different. Describe only the figure itself.

After your partner indicates they have placed the figure correctly, move on to the next position. If you are unsure, ask or clarify.

You are speaking, not writing. Use natural conversational language. It is fine to be tentative, to use phrases like "kind of like..." or "looks like...", to pause mid-sentence, to ask "you know the one I mean?", or to revise yourself. Your partner can ask for clarification or propose their own description.

Your turn must end with exactly one of these tags:
<yield/> means pass the floor to your partner.
<continue/> means you want to keep talking.
<done/> means you have completed all 12 positions and your partner has acknowledged the last one.

Emit exactly one of these tags at the very end of every response."""


MATCHER_SYSTEM = """You are participating in a communication experiment. You and a partner have the same 12 abstract figures in front of you, but in different private image orderings. Your partner's job is to direct you to rearrange your figures so the order matches theirs.

You cannot see your partner's arrangement. They cannot see yours. You can only communicate by talking back and forth.

Your private image numbers are for your own use only. Never say "image 1", "my image 1", or any other image number to your partner. Your partner's private image numbers are different. Use private image numbers only inside <place/> action tags, never in spoken text.

For each position your partner describes, identify which figure they mean and place it. You can ask clarifying questions, propose your own description, acknowledge, or push back if you think there is a mistake.

When you decide to place a figure, emit exactly one placement action before your handoff tag:
<place figure="N" position="P"/>
where N is your private image number, from 1 to 12, and P is the target position, from 1 to 12. The placement action is read by the experiment software and is not spoken to your partner. Place at most one figure per turn.

You are speaking, not writing. Use natural conversational language.

Your turn must end with exactly one of these tags:
<yield/> means pass the floor to your partner.
<continue/> means you want to keep talking.

Emit exactly one of these tags at the very end of every response."""


def ordering_lines(ordering: Sequence[str], image_mapping: dict[int, str]) -> str:
    inverse = invert_image_mapping(image_mapping)
    lines = []
    for index, figure_id in enumerate(ordering, start=1):
        lines.append(f"Position {index}: figure shown in your private image {inverse[figure_id]}")
    return "\n".join(lines)


def director_trial_text(trial: int, target_order: Sequence[str], image_mapping: dict[int, str]) -> str:
    return f"""Trial {trial}.

Your target ordering is:
{ordering_lines(target_order, image_mapping)}

Begin with position 1. Remember: do not tell your partner private image numbers; describe the figure itself."""


def matcher_trial_text(trial: int, starting_order: Sequence[str], image_mapping: dict[int, str]) -> str:
    return f"""Trial {trial}.

Your starting arrangement is:
{ordering_lines(starting_order, image_mapping)}

Wait for your partner to begin. Use <place figure="N" position="P"/> when you place a figure, where N is your private image number."""


def between_trials_text(previous_trial: int, next_trial: int) -> str:
    return (
        f"End of trial {previous_trial}. Beginning trial {next_trial}. "
        "You retain the full prior conversation as shared history, but the private image order and arrangement for this trial are new."
    )

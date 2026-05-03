from __future__ import annotations

from collections.abc import Sequence

from tangram.stimuli import invert_image_mapping


PROMPT_VERSION = "v2_paper_minimal"


DIRECTOR_SYSTEM = """You are participating in a communication task with a partner.

You and your partner have the same 12 abstract figures, but arranged in different orders. Your partner cannot see your arrangement, and you cannot see your partner's arrangement. You can only communicate by sending messages.

Your job is to get your partner to arrange their figures in the same order as yours, quickly and accurately. Go through the positions in order from 1 to 12. You may talk back and forth as much as needed.

Your private image numbers are only a way for you to view your own figures in this interface. Your partner's private image numbers may be different, so do not mention image numbers in your messages. Refer to the figures however you naturally would.

Your turn must end with exactly one of these tags:
<yield/> means pass the floor to your partner.
<continue/> means you want to keep talking.
<done/> means you have completed all 12 positions and your partner has acknowledged the last one.

Emit exactly one of these tags at the very end of every response."""


MATCHER_SYSTEM = """You are participating in a communication task with a partner.

You and your partner have the same 12 abstract figures, but arranged in different orders. Your partner has a target order. Your partner cannot see your arrangement, and you cannot see your partner's arrangement. You can only communicate by sending messages.

Your job is to rearrange your figures to match your partner's order, quickly and accurately. You may talk back and forth as much as needed.

Your private image numbers are only a way for you to view your own figures in this interface. Your partner's private image numbers may be different, so do not mention image numbers in your messages. Use private image numbers only inside <place/> action tags.

When you place a figure, emit a placement action before your handoff tag:
<place figure="N" position="P"/>
where N is your private image number, from 1 to 12, and P is the target position, from 1 to 12. The placement action is read by the experiment software and is not spoken to your partner.

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

Begin with position 1. Remember: do not tell your partner private image numbers."""


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

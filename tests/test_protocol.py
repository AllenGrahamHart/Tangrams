from __future__ import annotations

from tangram.protocol import infer_position_from_text, parse_model_response, swap_place, visible_partner_message


def test_parse_matcher_placement_and_handoff():
    parsed = parse_model_response(
        'Okay, placing it. <place figure="3" position="7"/><yield/>',
        "matcher",
    )
    assert parsed.text == "Okay, placing it."
    assert parsed.handoff == "yield"
    assert parsed.actions[0].figure_image_n == 3
    assert parsed.actions[0].position == 7


def test_missing_handoff_defaults_to_yield():
    parsed = parse_model_response("I think it is the upright one.", "director")
    assert parsed.handoff == "yield"
    assert parsed.parse_errors


def test_matcher_done_is_not_allowed():
    parsed = parse_model_response("Done. <done/>", "matcher")
    assert parsed.handoff == "yield"
    assert any("Matcher emitted" in error for error in parsed.parse_errors)


def test_swap_place_keeps_permutation():
    ordering = list("ABCD")
    swap_place(ordering, "C", 1)
    assert ordering == list("CBAD")


def test_private_image_numbers_are_scrubbed_for_partner():
    visible = visible_partner_message("matcher", "I think my image 4 is right.")
    assert "image 4" not in visible
    assert "[private image number omitted]" in visible


def test_infer_latest_position_from_director_text():
    assert infer_position_from_text("Good. Now let's move to position 12.") == 12

from __future__ import annotations

import base64

import pytest

from tangram.stimuli import FIGURE_IDS, load_tangrams


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def write_test_stimuli(path) -> None:
    path.mkdir(exist_ok=True)
    for figure_id in FIGURE_IDS:
        (path / f"{figure_id}.png").write_bytes(PNG_1X1)


def test_loads_twelve_tangrams(tmp_path):
    write_test_stimuli(tmp_path)
    stimuli = load_tangrams(tmp_path)
    assert set(stimuli) == set(FIGURE_IDS)
    assert base64.b64decode(stimuli["A"].data_base64) == PNG_1X1
    assert stimuli["A"].image_block()["source"]["media_type"] == "image/png"


def test_missing_tangrams_raise(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(FileNotFoundError, match="A.png"):
        load_tangrams(tmp_path)

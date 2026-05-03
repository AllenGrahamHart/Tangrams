from __future__ import annotations

import base64
import hashlib
from pathlib import Path

from pydantic import BaseModel


FIGURE_IDS: tuple[str, ...] = tuple("ABCDEFGHIJKL")


class Stimulus(BaseModel):
    figure_id: str
    path: Path
    media_type: str = "image/png"
    data_base64: str
    sha256: str

    def image_block(self) -> dict:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": self.media_type,
                "data": self.data_base64,
            },
        }


def load_tangrams(stimuli_dir: str | Path, figure_ids: tuple[str, ...] = FIGURE_IDS) -> dict[str, Stimulus]:
    directory = Path(stimuli_dir)
    missing = [figure_id for figure_id in figure_ids if not (directory / f"{figure_id}.png").exists()]
    if missing:
        expected = ", ".join(f"{figure_id}.png" for figure_id in missing)
        raise FileNotFoundError(f"Missing Tangram stimuli in {directory}: {expected}")

    stimuli: dict[str, Stimulus] = {}
    for figure_id in figure_ids:
        path = directory / f"{figure_id}.png"
        data = path.read_bytes()
        stimuli[figure_id] = Stimulus(
            figure_id=figure_id,
            path=path,
            data_base64=base64.b64encode(data).decode("ascii"),
            sha256=hashlib.sha256(data).hexdigest(),
        )
    return stimuli


def image_mapping_content(
    stimuli: dict[str, Stimulus],
    image_mapping: dict[int, str],
    heading: str = "Your private numbered images are below.",
) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": heading}]
    for image_number in sorted(image_mapping):
        figure_id = image_mapping[image_number]
        content.append({"type": "text", "text": f"Private image {image_number}:"})
        content.append(stimuli[figure_id].image_block())
    return content


def invert_image_mapping(image_mapping: dict[int, str]) -> dict[str, int]:
    return {figure_id: image_number for image_number, figure_id in image_mapping.items()}


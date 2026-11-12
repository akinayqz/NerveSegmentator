"""The models NerveSegmentator knows about.

One model segments one structure in one contrast. To publish retrained weights, export
them (`nerveseg_weights export`), upload the zip as a release asset and set `release`,
`url` and `sha256` here. To add a contrast, add a ModelSpec; nothing else changes.
"""
from dataclasses import dataclass

from .errors import InputError


@dataclass(frozen=True)
class ModelSpec:
    id: str
    structure: str                  # what it segments; also the suffix of the output file
    contrast: str
    description: str
    # Input channels in the order the network was trained with. "image" is the MR image;
    # the others are built by nervesegmentator.coords.
    channels: tuple = ("image", "coordLR", "coordCC")
    # Orientation (nibabel axis codes) of the training images. Inputs are flipped and
    # permuted to it before prediction, because nnU-Net is not orientation invariant.
    orientation: str = "LSP"
    release: str | None = None      # tag of the published weights; None: nothing published yet
    url: str | None = None          # zip made by `nerveseg_weights export`
    sha256: str | None = None


MODELS = {m.id: m for m in (
    ModelSpec("drg_t2", "drg", "t2", "Dorsal root ganglia L2-S1 on T2w"),
    ModelSpec("root_t2", "root", "t2", "Nerve roots L2-S1 on T2w"),
)}

STRUCTURES = tuple(dict.fromkeys(m.structure for m in MODELS.values()))
CONTRASTS = tuple(dict.fromkeys(m.contrast for m in MODELS.values()))


def get(structure, contrast):
    for m in MODELS.values():
        if (m.structure, m.contrast) == (structure, contrast):
            return m
    raise InputError(f"no model for structure '{structure}' in contrast '{contrast}'; available: "
                     + ", ".join(f"{m.structure}/{m.contrast}" for m in MODELS.values()))

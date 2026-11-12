"""Label values of the output segmentations.

Outputs are labelled by spinal level. The values are defined here, by level name, and
not by the networks: each model's own values (dataset.json) are mapped onto these, so a
retrained model with different or reordered labels does not change what users get.

The values currently equal those of the manual segmentations the models were trained
on, so outputs can be compared with them directly.
"""
import numpy as np

from .errors import InputError

LEVEL_VALUES = {"L2": 2, "L3": 3, "L4": 4, "L5": 5, "S1": 1}


def check(model_labels):
    """Raise if a model predicts a level that has no output value."""
    unknown = sorted(set(model_labels) - set(LEVEL_VALUES))
    if unknown:
        raise InputError(f"model labels {unknown} have no output value; "
                         f"add them to LEVEL_VALUES in {__name__}")


def to_output(seg, model_labels):
    """Map a segmentation in a model's label values (`model_labels`: level name -> value)
    to the output values."""
    check(model_labels)
    lut = np.zeros(max(model_labels.values()) + 1, np.uint8)
    for name, value in model_labels.items():
        lut[value] = LEVEL_VALUES[name]
    return lut[np.asarray(seg, dtype=np.intp)]

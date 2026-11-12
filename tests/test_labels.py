import numpy as np
import pytest

from nervesegmentator import labels
from nervesegmentator.errors import InputError


def test_model_values_are_mapped_by_level_name():
    model = {"L2": 1, "L3": 2, "L4": 3, "L5": 4, "S1": 5}      # a model with its own numbering
    seg = np.array([0, 1, 2, 3, 4, 5])
    out = labels.to_output(seg, model)
    assert out.dtype == np.uint8
    assert list(out) == [0] + [labels.LEVEL_VALUES[k] for k in ("L2", "L3", "L4", "L5", "S1")]


def test_unknown_level():
    with pytest.raises(InputError, match="S2"):
        labels.to_output(np.zeros(3, int), {"S1": 1, "S2": 2})

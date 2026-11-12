"""NerveSegmentator: segmentation of the lumbosacral (L2-S1) dorsal root ganglia and nerve roots in MRI."""

__version__ = "0.1.0"

from .api import NerveSegmentator, Result, segment  # noqa: E402  (needs __version__)

__all__ = ["NerveSegmentator", "Result", "segment", "__version__"]

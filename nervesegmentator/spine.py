"""Spine segmentation with TotalSpineSeg, the source of the coordinate channels.

TotalSpineSeg is optional: users who already have its output (for instance from
`sct_deepseg spine`) pass it in and this module is not used.
"""
import gzip
import importlib.resources
import logging
import os
import shutil
from pathlib import Path

from .errors import SpineSegmentationError

log = logging.getLogger(__name__)

# TotalSpineSeg output folder used for the channels, by step. Step 1 has everything the
# channels need (cord, canal, labelled discs) and takes one network instead of two.
OUTPUTS = {1: "step1_output", 2: "step2_output"}


def version():
    """Installed TotalSpineSeg version, or None."""
    try:
        from importlib.metadata import version as _version
        return _version("totalspineseg")
    except Exception:
        return None


def run_totalspineseg(images, out_dir, device, step=1, data_dir=None, quiet=False):
    """Segment `images` with TotalSpineSeg.

    out_dir    working folder for TotalSpineSeg
    device     torch.device; TotalSpineSeg runs on CUDA or CPU
    step       1 or 2: which TotalSpineSeg output to return
    data_dir   TotalSpineSeg weights folder (default: $TOTALSPINESEG_DATA, else inside the
               totalspineseg package); the weights are downloaded there on first use
    Returns the segmentation path of each image, None where TotalSpineSeg gave no output.
    """
    try:
        import torch
        import totalspineseg.models
        from totalspineseg.inference import inference
        from totalspineseg.init_inference import init_inference
        from totalspineseg.utils.utils import ZIP_URLS
    except ImportError as e:
        raise SpineSegmentationError(
            "TotalSpineSeg is needed to segment the spine and is not installed "
            f"({e}). Install it with: pip install \"nervesegmentator[totalspineseg]\", or pass "
            "an existing TotalSpineSeg segmentation (--spine-seg).") from e

    data_dir = Path(data_dir or os.environ.get("TOTALSPINESEG_DATA")
                    or importlib.resources.files(totalspineseg.models))
    init_inference(data_path=data_dir, dict_urls=ZIP_URLS, store_export=False, quiet=quiet)

    # TotalSpineSeg takes a folder; give each image a unique name in it
    in_dir, work_dir = Path(out_dir) / "input", Path(out_dir) / "totalspineseg"
    in_dir.mkdir(parents=True)
    names = [f"image{i:04d}.nii.gz" for i in range(len(images))]
    for image, name in zip(images, names):
        if str(image).endswith(".gz"):
            shutil.copyfile(image, in_dir / name)
        else:
            with open(image, "rb") as src, gzip.open(in_dir / name, "wb") as dst:
                shutil.copyfileobj(src, dst)

    log.info("segmenting the spine of %d image(s) with TotalSpineSeg %s", len(images), version())
    inference(
        input_path=in_dir,
        output_path=work_dir,
        data_path=data_dir,
        default_release=list(ZIP_URLS.values())[0].split("/")[-2],
        device=torch.device("cuda" if device.type == "cuda" else "cpu"),
        step1_only=step == 1,
        keep_only=[OUTPUTS[step]],
        # one worker each: its process pools stall in some host environments
        max_workers=1,
        max_workers_nnunet=1,
        quiet=quiet,
    )
    segs = [work_dir / OUTPUTS[step] / name for name in names]
    return [seg if seg.is_file() else None for seg in segs]

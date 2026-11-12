"""The pipeline: image -> spine segmentation -> coordinate channels -> networks -> labels.

    from nervesegmentator import segment
    results = segment("sub-01_T2w.nii.gz", "out")

or, to keep the networks loaded between calls,

    from nervesegmentator import NerveSegmentator
    ns = NerveSegmentator(device="cuda")
    results = ns.run(["sub-01_T2w.nii.gz", "sub-02_T2w.nii.gz"], "out")
"""
import json
import logging
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np

from . import __version__, coords, io, labels, registry, spine, weights
from .errors import CoordinateError, InputError
from .predict import Predictor, resolve_device

log = logging.getLogger(__name__)


@dataclass
class Result:
    image: Path
    report: Path                                    # <name>_nerveseg.json
    status: str = "ok"                              # 'ok', 'warning' or 'failed'
    outputs: dict = field(default_factory=dict)     # structure -> segmentation file
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)


def _paths(x):
    return [Path(p) for p in ([x] if isinstance(x, (str, Path)) else x)]


class NerveSegmentator:
    def __init__(self, structures=None, contrast="t2", device="auto", fast=False, folds=None,
                 tta=True, data_dir=None, spine_step=1, spine_data_dir=None, quiet=False):
        """
        structures      what to segment: any of registry.STRUCTURES (default: all)
        contrast        contrast of the input images
        device          'auto', 'cpu', 'cuda', 'cuda:N', 'mps' or a torch.device
        fast            use one fold of each model instead of the ensemble
        folds           folds to ensemble (default: all a model has)
        tta             test-time mirroring
        data_dir        where the weights are (see nervesegmentator.weights)
        spine_step      TotalSpineSeg output the channels are built from when it is run here: 1 or 2
        spine_data_dir  TotalSpineSeg weights folder (default: TotalSpineSeg's own)
        """
        if fast and folds is not None:
            raise InputError("give either fast or folds")
        self.device = resolve_device(device)
        self.spine_step, self.spine_data_dir, self.quiet = spine_step, spine_data_dir, quiet
        self.predictors = {}
        for structure in structures or registry.STRUCTURES:
            model = weights.require(registry.get(structure, contrast), data_dir)
            use = model.folds[:1] if fast else folds
            self.predictors[structure] = Predictor(model, self.device, use, tta, quiet)
            log.info("loaded model '%s' (%s), folds %s on %s", model.id, model.release,
                     list(self.predictors[structure].folds), self.device)

    def run(self, images, output_dir, spine_seg=None, save_channels=False):
        """Segment one image or a list of images; one Result per image.

        output_dir     receives <name>_<structure>.nii.gz and <name>_nerveseg.json per image
        spine_seg      TotalSpineSeg segmentation(s) of the image(s), one per image. Without
                       it TotalSpineSeg is run here and its output kept as <name>_spine.nii.gz
        save_channels  also write the coordinate channels (<name>_coordLR/_coordCC.nii.gz)
        """
        images, output_dir = _paths(images), Path(output_dir)
        segs = None if spine_seg is None else _paths(spine_seg)
        if segs is not None and len(segs) != len(images):
            raise InputError(f"{len(images)} image(s) but {len(segs)} spine segmentation(s)")
        for p in images + (segs or []):
            if not p.is_file():
                raise FileNotFoundError(p)
        for p in images:
            io.load(p)                              # not 3D: fail before the long steps
        names = [io.stem(p) for p in images]
        if len(set(names)) != len(names):
            raise InputError("image file names must differ: the outputs are named after them")

        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="nerveseg_") as tmp:
            source = {"source": "provided"}
            if segs is None:
                source = {"source": "totalspineseg", "version": spine.version(),
                          "output": spine.OUTPUTS[self.spine_step]}
                segs = spine.run_totalspineseg(images, tmp, self.device, self.spine_step,
                                               self.spine_data_dir, self.quiet)
                segs = [seg and Path(shutil.copyfile(seg, output_dir / f"{name}_spine.nii.gz"))
                        for name, seg in zip(names, segs)]
            return [self._run_one(image, seg, output_dir, source, save_channels)
                    for image, seg in zip(images, segs)]

    def _run_one(self, image, seg, output_dir, spine_source, save_channels):
        name = io.stem(image)
        result = Result(image=image, report=output_dir / f"{name}_nerveseg.json")
        report = {
            "nervesegmentator_version": __version__,
            "image": str(image.resolve()),
            "spine_segmentation": {**spine_source, "file": seg and str(seg.resolve())},
            "models": {s: {"id": p.model.id, "release": p.model.release, "folds": list(p.folds),
                           "test_time_mirroring": p.tta} for s, p in self.predictors.items()},
            "labels": labels.LEVEL_VALUES,
        }

        channels = None
        if seg is None:
            result.errors = ["TotalSpineSeg returned no segmentation"]
        else:
            img = io.load(image)
            try:
                channels = coords.compute(nib.load(str(seg)), img)
                report["coordinates"] = channels.report
                result.warnings = channels.report["warnings"]
            except CoordinateError as e:     # the levels cannot be anchored: no prediction
                report["coordinates"] = e.report
                result.errors = e.report["errors"]

        if channels is not None:
            if save_channels:
                io.save_like(channels.lr, img, output_dir / f"{name}_coordLR.nii.gz", np.float32)
                io.save_like(channels.cc, img, output_dir / f"{name}_coordCC.nii.gz", np.float32)
            data = {"image": img.get_fdata(dtype=np.float32),
                    "coordLR": channels.lr, "coordCC": channels.cc}
            for structure, predictor in self.predictors.items():
                t0 = time.time()
                out = output_dir / f"{name}_{structure}.nii.gz"
                io.save_like(predictor.predict(data, img.affine), img, out, np.uint8)
                result.outputs[structure] = out
                log.info("[%s] %s -> %s (%.0f s)", name, structure, out, time.time() - t0)

        result.status = "failed" if result.errors else ("warning" if result.warnings else "ok")
        for msg in result.errors:
            log.error("[%s] %s", name, msg)
        for msg in result.warnings:
            log.warning("[%s] %s", name, msg)
        report["status"] = result.status
        report["outputs"] = {s: str(p) for s, p in result.outputs.items()}
        result.report.write_text(json.dumps(report, indent=2))
        return result


def segment(images, output_dir, spine_seg=None, save_channels=False, **kwargs):
    """Segment `images` into `output_dir`; see NerveSegmentator and NerveSegmentator.run."""
    return NerveSegmentator(**kwargs).run(images, output_dir, spine_seg, save_channels)

"""Command line entry points: NerveSegmentator, nerveseg_weights, nerveseg_coords."""
import argparse
import json
import logging
import re
import sys
from pathlib import Path

from . import __version__, registry
from .errors import CoordinateError, NerveSegmentatorError


def _logging(quiet=False):
    """Show this package's log messages on stderr (and leave other packages' loggers alone)."""
    log = logging.getLogger(__package__)
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        log.addHandler(handler)
    log.setLevel(logging.WARNING if quiet else logging.INFO)


def _fold(value):
    return value if value == "all" else int(value)


def _run(parser, func):
    """Run `func`; the errors a user can act on end the program with their message."""
    try:
        return func()
    except (NerveSegmentatorError, FileNotFoundError) as e:
        parser.exit(2, f"{parser.prog}: error: {e}\n")


# ---- NerveSegmentator

def main(argv=None):
    p = argparse.ArgumentParser(
        prog="NerveSegmentator",
        description="Segment the L2-S1 dorsal root ganglia and nerve roots in lumbar spine MRI.",
        epilog="examples:\n"
               "  NerveSegmentator -i sub-01_T2w.nii.gz -o out\n"
               "  NerveSegmentator -i sub-01_T2w.nii.gz -o out --spine-seg sub-01_T2w_step1_output.nii.gz\n"
               "  NerveSegmentator -i sub-*_T2w.nii.gz -o out --structures drg --fast --device cuda\n\n"
               "Per image, the output folder gets <name>_<structure>.nii.gz (labels by level:\n"
               "L2=2, L3=3, L4=4, L5=5, S1=1) and <name>_nerveseg.json (what was run, level anchors,\n"
               "warnings). Exit status 1 means at least one image could not be segmented.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", nargs="+", type=Path, required=True, metavar="IMAGE",
                   help="image(s) to segment (.nii or .nii.gz)")
    p.add_argument("-o", "--output", type=Path, required=True, metavar="DIR", help="output folder")
    p.add_argument("-s", "--structures", nargs="+", choices=registry.STRUCTURES,
                   default=list(registry.STRUCTURES), help="what to segment (default: all)")
    p.add_argument("-c", "--contrast", choices=registry.CONTRASTS, default=registry.CONTRASTS[0],
                   help="contrast of the input (default: %(default)s)")
    p.add_argument("--spine-seg", nargs="+", type=Path, metavar="SEG",
                   help="TotalSpineSeg segmentation (step1_output or step2_output) of each input, in the "
                        "same order. Without it TotalSpineSeg is run and kept as <name>_spine.nii.gz")
    speed = p.add_mutually_exclusive_group()
    speed.add_argument("--fast", action="store_true",
                       help="one fold per model instead of the ensemble")
    speed.add_argument("--folds", nargs="+", type=_fold, metavar="N", help="folds to ensemble (default: all)")
    p.add_argument("--no-tta", action="store_true", help="no test-time mirroring (twice as fast)")
    p.add_argument("--device", default="auto",
                   help="auto (CUDA if available, else CPU), cpu, cuda, cuda:N or mps (default: %(default)s)")
    p.add_argument("--save-channels", action="store_true",
                   help="also write the coordinate channels given to the networks")
    p.add_argument("--data-dir", type=Path, metavar="DIR",
                   help="weights folder (default: $NERVESEGMENTATOR_DATA, else ~/.nervesegmentator)")
    p.add_argument("-q", "--quiet", action="store_true", help="only print warnings and errors")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    a = p.parse_args(argv)
    _logging(a.quiet)

    def run():
        from .api import NerveSegmentator
        ns = NerveSegmentator(a.structures, a.contrast, a.device, a.fast, a.folds, not a.no_tta,
                              a.data_dir, quiet=a.quiet)
        return ns.run(a.input, a.output, a.spine_seg, a.save_channels)

    results = _run(p, run)
    sys.exit(1 if any(r.status == "failed" for r in results) else 0)


# ---- nerveseg_weights

def weights_main(argv=None):
    p = argparse.ArgumentParser(
        prog="nerveseg_weights", description="Install, inspect and package NerveSegmentator model weights.")
    p.add_argument("--data-dir", type=Path, metavar="DIR",
                   help="weights folder (default: $NERVESEGMENTATOR_DATA, else ~/.nervesegmentator)")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="show the models and what is installed")
    d = sub.add_parser("download", help="download published weights")
    d.add_argument("models", nargs="*", metavar="MODEL", help=f"default: all ({', '.join(registry.MODELS)})")
    i = sub.add_parser("import", help="install weights from an nnU-Net training output folder or a model zip")
    e = sub.add_parser("export", help="package an nnU-Net training output folder as a release zip")
    for s in (i, e):
        s.add_argument("source",
                       help="nnU-Net training output folder (with plans.json and fold_*/), or the "
                            "DatasetXXX folder above it" + ("; or a zip made by export, or its URL" if s is i else ""))
        s.add_argument("-m", "--model", choices=list(registry.MODELS), required=s is e,
                       help="the model these weights are for")
        s.add_argument("--folds", nargs="+", type=_fold, metavar="N", help="folds to include (default: all)")
    i.add_argument("--release", help="tag to record (default: local-<today>)")
    e.add_argument("--release", required=True, help="release tag, e.g. r20261112")
    e.add_argument("-o", "--output", type=Path, default=Path("release"), metavar="DIR",
                   help="where to write the zip (default: %(default)s)")
    a = p.parse_args(argv)
    _logging()
    if a.command == "download" and set(a.models) - set(registry.MODELS):
        p.error(f"unknown model(s) {sorted(set(a.models) - set(registry.MODELS))}; choose from {list(registry.MODELS)}")

    def run():
        from . import weights
        if a.command == "list":
            print(f"data dir: {weights.data_dir(a.data_dir)}")
            for spec in registry.MODELS.values():
                m = weights.installed(spec.id, a.data_dir)
                state = f"installed: {m.release}, folds {list(m.folds)}" if m else "not installed"
                print(f"  {spec.id:10s} {spec.description:36s} {state}; published: {spec.release or 'no'}")
        elif a.command == "download":
            for model_id in a.models or registry.MODELS:
                weights.download(registry.MODELS[model_id], a.data_dir)
        elif a.command == "import":
            spec = registry.MODELS[a.model] if a.model else None
            weights.import_model(a.source, spec, a.data_dir, a.release, a.folds)
        else:
            zip_path, sha256 = weights.export(a.source, a.output, registry.MODELS[a.model], a.release, a.folds)
            print(f"{zip_path}\nsha256 {sha256}")

    _run(p, run)


# ---- nerveseg_coords

def coords_main(argv=None):
    p = argparse.ArgumentParser(
        prog="nerveseg_coords",
        description="Build the coordinate channels of an image from its TotalSpineSeg segmentation, as "
                    "the networks get them. For preparing training data.",
        epilog="Writes <prefix>_coordCC.nii.gz, <prefix>_coordLR.nii.gz and <prefix>_coordAnchors.json.\n"
               "Exit status 0: channels written (check WARNING lines); 1: failed, only the JSON written.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seg", required=True, help="TotalSpineSeg step1_output or step2_output")
    p.add_argument("--ref", required=True, help="image given to nnU-Net (defines the output grid)")
    p.add_argument("--prefix", help="output name prefix (default: ref name up to _T2W)")
    p.add_argument("--outdir", help="output folder (default: folder of --ref)")
    a = p.parse_args(argv)

    import nibabel as nib
    import numpy as np
    from . import coords, io

    outdir = Path(a.outdir) if a.outdir else Path(a.ref).resolve().parent
    m = re.match(r"(.*?_T2W)", io.stem(a.ref), re.IGNORECASE)
    prefix = a.prefix or (m.group(1) if m else io.stem(a.ref))
    ref = nib.load(a.ref)
    try:
        channels = coords.compute(nib.load(a.seg), ref)
        report = channels.report
    except CoordinateError as e:
        channels, report = None, e.report
    report = {"seg": str(Path(a.seg).resolve()), "ref": str(Path(a.ref).resolve()), **report}
    (outdir / f"{prefix}_coordAnchors.json").write_text(json.dumps(report, indent=2))

    if channels is None:
        for msg in report["errors"]:
            print(f"ERROR [{prefix}] {msg}", file=sys.stderr)
        print(f"ERROR [{prefix}] labels found: {report['labels_found']}", file=sys.stderr)
        sys.exit(1)
    for msg in report["warnings"]:
        print(f"WARNING [{prefix}] {msg}")
    io.save_like(channels.cc, ref, outdir / f"{prefix}_coordCC.nii.gz", np.float32)
    io.save_like(channels.lr, ref, outdir / f"{prefix}_coordLR.nii.gz", np.float32)
    discs = sorted(report["anchors"].values(), key=lambda d: d["unit"])
    print(f"OK [{prefix}] {len(discs)} discs: {', '.join(d['name'] for d in discs)} "
          f"-> {prefix}_coordCC.nii.gz, {prefix}_coordLR.nii.gz")

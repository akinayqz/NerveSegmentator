"""Model weights: where they are kept and how they get there.

An installed model is a self-describing folder,

    <data dir>/models/<model id>/
        model.json                        what the model is and what it needs (see package())
        nnunet/plans.json, dataset.json   as written by nnU-Net
        nnunet/fold_<k>/checkpoint_final.pth

made from an nnU-Net training output folder by package(). A release is that folder
zipped. The data dir is `--data-dir`, else $NERVESEGMENTATOR_DATA, else ~/.nervesegmentator.
"""
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from . import __version__, labels
from .errors import InputError, ModelNotInstalled, NerveSegmentatorError

log = logging.getLogger(__name__)

ENV_DATA = "NERVESEGMENTATOR_DATA"
META = "model.json"
NNUNET = "nnunet"
CHECKPOINT = "checkpoint_final.pth"
FORMAT = 1
# The checkpoints name the trainer they were trained with, and nnU-Net looks that class
# up when it loads them. Our trainers change the augmentation only, so the released
# checkpoints name the stock trainer and load in any nnU-Net installation.
STOCK_TRAINER = "nnUNetTrainer"


@dataclass(frozen=True)
class InstalledModel:
    path: Path
    meta: dict

    id = property(lambda self: self.meta["id"])
    structure = property(lambda self: self.meta["structure"])
    release = property(lambda self: self.meta["release"])
    channels = property(lambda self: tuple(self.meta["channels"]))
    orientation = property(lambda self: self.meta["orientation"])
    labels = property(lambda self: dict(self.meta["labels"]))
    folds = property(lambda self: tuple(self.meta["folds"]))
    nnunet_dir = property(lambda self: self.path / NNUNET)


def data_dir(path=None):
    return Path(path or os.environ.get(ENV_DATA) or Path.home() / ".nervesegmentator").expanduser()


def _open(path):
    path = Path(path)
    meta = json.loads((path / META).read_text())
    if meta.get("format") != FORMAT:
        raise NerveSegmentatorError(f"{path}: model format {meta.get('format')} is not supported by "
                                    f"nervesegmentator {__version__} (expects {FORMAT})")
    return InstalledModel(path, meta)


def installed(model_id, data=None):
    """The installed model, or None."""
    path = data_dir(data) / "models" / model_id
    return _open(path) if (path / META).is_file() else None


def require(spec, data=None):
    """The installed model for `spec`, downloading it first if it is published."""
    model = installed(spec.id, data)
    if model is None:
        if spec.url is None:
            raise ModelNotInstalled(
                f"Model '{spec.id}' is not installed in {data_dir(data)} and has no published weights yet. "
                f"Install local weights with: nerveseg_weights import <nnU-Net output folder or zip> -m {spec.id}")
        model = download(spec, data)
    elif spec.release and model.release != spec.release and not model.release.startswith("local"):
        log.warning("model '%s' is release %s; %s is available (nerveseg_weights download %s)",
                    spec.id, model.release, spec.release, spec.id)
    return model


# ---- making a model folder from an nnU-Net training output

def _trainer_dir(source):
    """`source` is an nnU-Net training output folder (the one with plans.json) or a folder
    holding exactly one, such as nnUNet_results/DatasetXXX_Name."""
    source = Path(source)
    if (source / "plans.json").is_file():
        return source
    found = sorted(p.parent for p in source.glob("*/plans.json"))
    if len(found) != 1:
        raise InputError(f"{source}: expected one nnU-Net training output folder (with plans.json), "
                         f"found {len(found)}")
    return found[0]


def _fold(name):
    return name if name == "all" else int(name)


def _clean_checkpoint(src, dst):
    """Keep what prediction needs: the weights, the configuration name and the mirror axes.
    Drops the optimizer state (half the file) and everything that is not a tensor or a
    plain value, so the file also loads with torch.load(weights_only=True)."""
    import torch
    ckpt = torch.load(src, map_location="cpu", weights_only=False)   # our own training output
    mirror = ckpt.get("inference_allowed_mirroring_axes")
    torch.save({
        "network_weights": ckpt["network_weights"],
        "trainer_name": STOCK_TRAINER,
        "init_args": {"configuration": ckpt["init_args"]["configuration"]},
        "inference_allowed_mirroring_axes": None if mirror is None else tuple(int(a) for a in mirror),
    }, dst)
    return ckpt["trainer_name"], ckpt["init_args"]["configuration"]


def package(source, dest, spec, release=None, folds=None):
    """Write the model folder `dest` from the nnU-Net training output `source`.

    spec     the registry entry this is a model for
    release  tag stored in model.json (default: local-<today>)
    folds    folds to include (default: every fold with a final checkpoint)
    """
    src = _trainer_dir(source)
    available = sorted((_fold(p.parent.name[5:]) for p in src.glob(f"fold_*/{CHECKPOINT}")), key=str)
    folds = list(available if folds is None else folds)
    if not folds or set(folds) - set(available):
        raise InputError(f"{src}: folds with a {CHECKPOINT}: {available}; asked for: {folds}")

    dataset = json.loads((src / "dataset.json").read_text())
    model_labels = {k: v for k, v in dataset["labels"].items() if k != "background"}
    labels.check(model_labels)
    if len(dataset["channel_names"]) != len(spec.channels):
        raise InputError(f"{src}: trained with {len(dataset['channel_names'])} input channels, "
                         f"but '{spec.id}' is registered with {len(spec.channels)} {spec.channels}")

    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    (dest / NNUNET).mkdir(parents=True)
    shutil.copyfile(src / "plans.json", dest / NNUNET / "plans.json")
    for key in ("training", "test"):                # the list of training cases is not needed
        dataset.pop(key, None)
    (dest / NNUNET / "dataset.json").write_text(json.dumps(dataset, indent=4))
    for fold in folds:
        (dest / NNUNET / f"fold_{fold}").mkdir()
        trainer, configuration = _clean_checkpoint(src / f"fold_{fold}" / CHECKPOINT,
                                                   dest / NNUNET / f"fold_{fold}" / CHECKPOINT)

    meta = {
        "format": FORMAT,
        "id": spec.id,
        "structure": spec.structure,
        "contrast": spec.contrast,
        "release": release or f"local-{date.today():%Y%m%d}",
        "channels": list(spec.channels),
        "orientation": spec.orientation,
        "labels": model_labels,
        "folds": folds,
        "source": {
            "dataset": json.loads((src / "plans.json").read_text()).get("dataset_name"),
            "trainer": trainer,
            "configuration": configuration,
        },
        "nervesegmentator_version": __version__,
    }
    (dest / META).write_text(json.dumps(meta, indent=4))
    return _open(dest)


def export(source, out_dir, spec, release, folds=None):
    """Package `source` and zip it as <out_dir>/<id>_<release>.zip. Returns (zip path, sha256)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{spec.id}_{release}.zip"
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        model = package(source, Path(tmp) / spec.id, spec, release, folds)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(p for p in model.path.rglob("*") if p.is_file()):
                zf.write(f, f.relative_to(tmp))
    return zip_path, _sha256(zip_path)


# ---- installing

def _install_folder(folder, data):
    """Move a model folder to its place in the data dir, replacing an installed one."""
    model = _open(folder)
    dest = data_dir(data) / "models" / model.id
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(folder), dest)
    log.info("installed model '%s' (%s) in %s", model.id, model.release, dest)
    return _open(dest)


def install_zip(zip_path, data=None):
    """Install a model from a zip made by export()."""
    models = data_dir(data) / "models"
    models.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=models) as tmp, zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        tops = {n.split("/")[0] for n in names}
        if len(tops) != 1 or any(os.path.isabs(n) or ".." in Path(n).parts for n in names):
            raise InputError(f"{zip_path} is not a NerveSegmentator model zip")
        zf.extractall(tmp)
        return _install_folder(Path(tmp) / tops.pop(), data)


def import_model(source, spec=None, data=None, release=None, folds=None):
    """Install a model from a zip made by export(), or straight from an nnU-Net training
    output folder (then `spec` says which model it is)."""
    source = Path(source)
    if source.is_file():
        return install_zip(source, data)
    if spec is None:
        raise InputError("importing an nnU-Net folder needs the model it is for (-m)")
    models = data_dir(data) / "models"
    models.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=models) as tmp:
        return _install_folder(package(source, Path(tmp) / spec.id, spec, release, folds).path, data)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(spec, data=None):
    """Download and install the published weights of `spec`."""
    if spec.url is None:
        raise ModelNotInstalled(f"model '{spec.id}' has no published weights yet")
    models = data_dir(data) / "models"
    models.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=models) as tmp:
        zip_path = Path(tmp) / re.sub(r"[^\w.-]", "_", spec.url.rsplit("/", 1)[-1])
        log.info("downloading %s", spec.url)
        with urllib.request.urlopen(spec.url) as r, open(zip_path, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        if spec.sha256 and _sha256(zip_path) != spec.sha256:
            raise NerveSegmentatorError(f"{spec.url}: checksum mismatch, the download is incomplete or not release {spec.release}")
        return install_zip(zip_path, data)

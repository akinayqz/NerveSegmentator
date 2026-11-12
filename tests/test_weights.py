import dataclasses
import json
import zipfile

import pytest

from nervesegmentator import registry, weights
from nervesegmentator.errors import InputError, ModelNotInstalled, NerveSegmentatorError

torch = pytest.importorskip("torch")

SPEC = registry.MODELS["drg_t2"]
LABELS = {"background": 0, "S1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5}


@pytest.fixture
def nnunet_output(tmp_path):
    """An nnU-Net training output as nnU-Net leaves it, with a small stand-in for the network."""
    trainer = tmp_path / "nnUNet_results" / "Dataset999_Test" / "MyTrainer__nnUNetPlans__3d_fullres"
    trainer.mkdir(parents=True)
    (trainer / "plans.json").write_text(json.dumps({"dataset_name": "Dataset999_Test"}))
    (trainer / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "T2", "1": "noNorm", "2": "noNorm"}, "labels": LABELS,
        "file_ending": ".nii.gz", "training": [{"image": "a", "label": "b"}]}))
    for fold in (0, 1):
        (trainer / f"fold_{fold}").mkdir()
        torch.save({
            "network_weights": {"conv.weight": torch.full((2, 3), float(fold))},
            "optimizer_state": {"momentum": torch.zeros(1000)},
            "trainer_name": "MyTrainer",
            "init_args": {"configuration": "3d_fullres", "device": torch.device("cpu"), "fold": fold},
            "inference_allowed_mirroring_axes": (2,),
        }, trainer / f"fold_{fold}" / "checkpoint_final.pth")
        (trainer / f"fold_{fold}" / "checkpoint_best.pth").write_bytes(b"not needed")
    return trainer


def test_import_nnunet_folder(nnunet_output, tmp_path):
    data = tmp_path / "data"
    model = weights.import_model(nnunet_output.parent, SPEC, data, release="r1")    # the DatasetXXX folder
    assert model == weights.installed("drg_t2", data)
    assert model.path == data / "models" / "drg_t2"
    assert (model.id, model.release, model.folds) == ("drg_t2", "r1", (0, 1))
    assert model.channels == SPEC.channels and model.orientation == SPEC.orientation
    assert model.labels == {k: v for k, v in LABELS.items() if k != "background"}
    assert model.meta["source"] == {"dataset": "Dataset999_Test", "trainer": "MyTrainer",
                                    "configuration": "3d_fullres"}
    assert sorted(p.name for p in model.nnunet_dir.rglob("*.pth")) == ["checkpoint_final.pth"] * 2
    assert "training" not in json.loads((model.nnunet_dir / "dataset.json").read_text())

    # the checkpoint: weights only, stock trainer, loadable without unpickling arbitrary objects
    ckpt = torch.load(model.nnunet_dir / "fold_1" / "checkpoint_final.pth", weights_only=True)
    assert set(ckpt) == {"network_weights", "trainer_name", "init_args", "inference_allowed_mirroring_axes"}
    assert ckpt["trainer_name"] == "nnUNetTrainer" and ckpt["init_args"] == {"configuration": "3d_fullres"}
    assert ckpt["inference_allowed_mirroring_axes"] == (2,)
    assert torch.equal(ckpt["network_weights"]["conv.weight"], torch.full((2, 3), 1.0))


def test_export_then_install_zip(nnunet_output, tmp_path):
    zip_path, sha256 = weights.export(nnunet_output, tmp_path / "release", SPEC, "r20260101", folds=[1])
    assert zip_path.name == "drg_t2_r20260101.zip" and sha256 == weights._sha256(zip_path)
    model = weights.import_model(zip_path, data=tmp_path / "data")
    assert (model.id, model.release, model.folds) == ("drg_t2", "r20260101", (1,))
    assert (model.nnunet_dir / "fold_1" / "checkpoint_final.pth").is_file()


def test_download(nnunet_output, tmp_path):
    zip_path, sha256 = weights.export(nnunet_output, tmp_path / "release", SPEC, "r1")
    spec = dataclasses.replace(SPEC, release="r1", url=zip_path.as_uri(), sha256=sha256)
    assert weights.require(spec, tmp_path / "data").release == "r1"
    with pytest.raises(NerveSegmentatorError, match="checksum"):
        weights.download(dataclasses.replace(spec, sha256="0" * 64), tmp_path / "data2")


def test_not_installed(tmp_path):
    assert weights.installed("drg_t2", tmp_path) is None
    with pytest.raises(ModelNotInstalled, match="nerveseg_weights import"):
        weights.require(SPEC, tmp_path)


def test_reinstall_replaces(nnunet_output, tmp_path):
    weights.import_model(nnunet_output, SPEC, tmp_path / "data", release="old")
    assert weights.import_model(nnunet_output, SPEC, tmp_path / "data", release="new", folds=[0]).folds == (0,)
    assert weights.installed("drg_t2", tmp_path / "data").release == "new"
    assert not (tmp_path / "data" / "models" / "drg_t2" / "nnunet" / "fold_1").exists()


def test_rejects_what_does_not_fit_the_model(nnunet_output, tmp_path):
    with pytest.raises(InputError, match="folds"):
        weights.import_model(nnunet_output, SPEC, tmp_path / "data", folds=[3])
    dataset = json.loads((nnunet_output / "dataset.json").read_text())
    (nnunet_output / "dataset.json").write_text(json.dumps({**dataset, "labels": {**LABELS, "S2": 6}}))
    with pytest.raises(InputError, match="S2"):
        weights.import_model(nnunet_output, SPEC, tmp_path / "data")
    (nnunet_output / "dataset.json").write_text(json.dumps({**dataset, "channel_names": {"0": "T2"}}))
    with pytest.raises(InputError, match="input channels"):
        weights.import_model(nnunet_output, SPEC, tmp_path / "data")
    assert weights.installed("drg_t2", tmp_path / "data") is None


def test_rejects_foreign_zip(tmp_path):
    with zipfile.ZipFile(tmp_path / "bad.zip", "w") as zf:
        zf.writestr("../outside.txt", "x")
    with pytest.raises(InputError):
        weights.install_zip(tmp_path / "bad.zip", tmp_path / "data")

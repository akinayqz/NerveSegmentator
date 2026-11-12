"""End to end on real data. Needs test_data/ and the weights installed in the data dir
($NERVESEGMENTATOR_DATA or ~/.nervesegmentator); skipped otherwise. Takes a few minutes on
CPU; set NERVESEG_TEST_DEVICE (e.g. to cuda or mps) to run on something faster."""
import json
import os

import nibabel as nib
import numpy as np
import pytest

from nervesegmentator import io, labels, registry, segment, weights

pytestmark = pytest.mark.slow


def dice(a, b, value):
    a, b = a == value, b == value
    return 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)


@pytest.fixture(scope="module")
def device():
    pytest.importorskip("nnunetv2")
    if weights.installed("drg_t2") is None:
        pytest.skip("model 'drg_t2' is not installed")
    return os.environ.get("NERVESEG_TEST_DEVICE", "auto")


def test_drg_against_manual_segmentation(test_data, tmp_path, device):
    d = test_data / "subj1"
    image, seg = d / "Sbj012_T2W_1mm_Mask.nii.gz", d / "Sbj012_T2W_spine_step1_output.nii.gz"
    [result] = segment(image, tmp_path, spine_seg=seg, structures=["drg"], fast=True, device=device,
                       save_channels=True, quiet=True)

    assert result.status == "ok" and list(result.outputs) == ["drg"]
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        f"Sbj012_T2W_1mm_Mask_{s}" for s in ("coordCC.nii.gz", "coordLR.nii.gz", "drg.nii.gz", "nerveseg.json")]
    report = json.loads(result.report.read_text())
    assert report["status"] == "ok" and report["spine_segmentation"]["source"] == "provided"
    assert report["models"]["drg"]["folds"] == [0] and report["coordinates"]["anchors"]["L5-S1"]["unit"] == 0

    out, ref = nib.load(result.outputs["drg"]), nib.load(image)
    assert out.shape == ref.shape and np.array_equal(out.affine, ref.affine) and out.get_data_dtype() == np.uint8
    manual = np.asanyarray(nib.load(d / "DRG_012.nii.gz").dataobj)
    predicted = np.asanyarray(out.dataobj)
    for level, value in labels.LEVEL_VALUES.items():
        assert dice(predicted, manual, value) > 0.7, level


def test_ensemble_of_folds(test_data, tmp_path, device):
    """Several folds: their logits are averaged in predict.py, not by nnU-Net's own loop."""
    d = test_data / "subj1"
    image, seg = d / "Sbj012_T2W_1mm_Mask.nii.gz", d / "Sbj012_T2W_spine_step1_output.nii.gz"
    [result] = segment(image, tmp_path, spine_seg=seg, structures=["drg"], folds=[0, 1], tta=False,
                       device=device, quiet=True)

    assert json.loads(result.report.read_text())["models"]["drg"]["folds"] == [0, 1]
    manual = np.asanyarray(nib.load(d / "DRG_012.nii.gz").dataobj)
    predicted = np.asanyarray(nib.load(result.outputs["drg"]).dataobj)
    for level, value in labels.LEVEL_VALUES.items():
        assert dice(predicted, manual, value) > 0.7, level


def test_same_result_whatever_the_orientation_of_the_input(test_data, tmp_path, device):
    d = test_data / "subj1"
    image = nib.load(d / "Sbj012_T2W_1mm_Mask.nii.gz")
    seg = d / "Sbj012_T2W_spine_step1_output.nii.gz"
    ornt = io.orientation_to(image.affine, "RAS")
    assert registry.MODELS["drg_t2"].orientation != "RAS"
    nib.save(image.as_reoriented(ornt), tmp_path / "ras.nii.gz")

    kwargs = dict(spine_seg=seg, structures=["drg"], fast=True, device=device, quiet=True)
    [native] = segment(d / "Sbj012_T2W_1mm_Mask.nii.gz", tmp_path / "native", **kwargs)
    [ras] = segment(tmp_path / "ras.nii.gz", tmp_path / "ras", **kwargs)

    a = np.asanyarray(nib.load(native.outputs["drg"]).dataobj)
    b = io.reorient(np.asanyarray(nib.load(ras.outputs["drg"]).dataobj), io.invert_orientation(ornt))
    assert all(dice(a, b, value) > 0.99 for value in labels.LEVEL_VALUES.values())

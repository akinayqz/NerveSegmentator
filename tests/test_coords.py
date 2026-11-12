import nibabel as nib
import numpy as np
import pytest

from conftest import DISC_Z, phantom, world_grid
from nervesegmentator import coords, io
from nervesegmentator.errors import CoordinateError


def nearest_voxel(img, xyz):
    return tuple(np.rint(nib.affines.apply_affine(np.linalg.inv(img.affine), xyz)).astype(int))


def test_levels_and_sides():
    image, seg = phantom()
    c = coords.compute(seg, image)
    assert c.report["status"] == "ok" and not c.report["errors"]
    assert set(c.report["anchors"]) == {"L5-S1", "L4-L5", "L3-L4", "L2-L3", "L1-L2"}
    assert c.cc.shape == c.lr.shape == image.shape and c.cc.dtype == c.lr.dtype == np.float32

    # coordCC: the disc's unit at each disc level, half way between two discs, negative below L5-S1
    for label, z in DISC_Z.items():
        assert c.cc[nearest_voxel(image, (0, 0, z))] == pytest.approx(coords.DISC_UNITS[label], abs=0.05)
    assert c.cc[nearest_voxel(image, (0, 0, -42.5))] == pytest.approx(0.5, abs=0.05)
    assert c.cc[nearest_voxel(image, (0, 0, -95))] == pytest.approx(-1.0, abs=0.05)

    # coordLR: mm / 50 from the centreline, positive on the subject's right (world +x)
    assert c.lr[nearest_voxel(image, (25, 0, 0))] == pytest.approx(0.5, abs=0.03)
    assert c.lr[nearest_voxel(image, (-25, 0, 0))] == pytest.approx(-0.5, abs=0.03)
    assert np.allclose(c.lr, world_grid(image.shape, image.affine)[..., 0] / coords.LR_SCALE, atol=0.03)


def test_same_channels_whatever_the_orientation_of_the_image():
    image, seg = phantom("LSP")
    reference = coords.compute(seg, image)
    ornt = io.orientation_to(image.affine, "RAS")
    other = coords.compute(seg, image.as_reoriented(ornt))
    back = io.invert_orientation(ornt)
    assert np.allclose(io.reorient(other.cc, back), reference.cc, atol=1e-4)
    assert np.allclose(io.reorient(other.lr, back), reference.lr, atol=1e-4)


def test_segmentation_on_another_grid():
    image, _ = phantom("LSP")
    _, seg = phantom("RAS", shape=(80, 72, 210))
    c = coords.compute(seg, image)
    assert c.cc[nearest_voxel(image, (0, 0, DISC_Z[95]))] == pytest.approx(1.0, abs=0.05)


def test_no_l5s1_fails():
    image, seg = phantom(discs={k: v for k, v in DISC_Z.items() if k != 100})
    with pytest.raises(CoordinateError) as e:
        coords.compute(seg, image)
    assert e.value.report["status"] == "failed"
    assert any("L5-S1" in msg for msg in e.value.report["errors"])


def test_no_canal_fails():
    image, seg = phantom()
    data = np.asanyarray(seg.dataobj).copy()
    data[data <= 2] = 0
    with pytest.raises(CoordinateError, match="no cord/canal"):
        coords.compute(nib.Nifti1Image(data, seg.affine), image)


def test_missing_disc_warns():
    image, seg = phantom(discs={k: v for k, v in DISC_Z.items() if k != 94})
    c = coords.compute(seg, image)
    assert c.report["status"] == "warning"
    assert any("missing discs: L3-L4" in msg for msg in c.report["warnings"])
    assert c.cc[nearest_voxel(image, (0, 0, DISC_Z[94]))] == pytest.approx(2.0, abs=0.05)   # interpolated


@pytest.mark.parametrize("subject, case", [("subj1", "DRG_012"), ("subj2", "DRG_013")])
def test_close_to_the_training_channels(test_data, subject, case):
    """The channel files the networks were trained on (_0001 = coordLR, _0002 = coordCC) are
    not reproduced exactly by the current code from the segmentations kept with them (about
    0.5 mm and 0.05 levels apart at the nerves), so this only guards against gross changes."""
    d = test_data / subject
    image = nib.load(d / f"{case}_0000.nii.gz")
    seg = nib.load(next(d.glob("*_step1_output.nii.gz")))
    nerves = np.asanyarray(nib.load(d / f"{case}.nii.gz").dataobj) > 0
    c = coords.compute(seg, image)
    lr = np.asanyarray(nib.load(d / f"{case}_0001.nii.gz").dataobj)
    cc = np.asanyarray(nib.load(d / f"{case}_0002.nii.gz").dataobj)
    assert np.median(np.abs(c.lr - lr)[nerves]) < 0.02      # 1 mm
    assert np.median(np.abs(c.cc - cc)[nerves]) < 0.1       # a tenth of a level

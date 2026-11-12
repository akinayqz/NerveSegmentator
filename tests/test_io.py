import nibabel as nib
import numpy as np
import pytest

from conftest import affine_for
from nervesegmentator import io
from nervesegmentator.errors import InputError


@pytest.mark.parametrize("source", ["RAS", "LSP", "PIR", "ASL"])
@pytest.mark.parametrize("target", ["LSP", "RAS", "IPL"])
def test_reorient(source, target):
    shape = (5, 6, 7)
    data = np.random.default_rng(0).random(shape)
    affine = affine_for(source, shape) @ np.diag([1.0, 2.0, 3.0, 1.0])    # anisotropic voxels
    ornt = io.orientation_to(affine, target)
    out = io.reorient(data, ornt)

    expected = nib.Nifti1Image(data, affine).as_reoriented(ornt)
    assert nib.aff2axcodes(expected.affine) == tuple(target)
    assert np.array_equal(out, expected.get_fdata())
    assert np.allclose(io.reorient_zooms((1, 2, 3), ornt), expected.header.get_zooms())
    assert np.array_equal(io.reorient(out, io.invert_orientation(ornt)), data)


def test_load_drops_trailing_singleton(tmp_path):
    nib.save(nib.Nifti1Image(np.zeros((4, 5, 6, 1), np.float32), np.eye(4)), tmp_path / "a.nii.gz")
    assert io.load(tmp_path / "a.nii.gz").shape == (4, 5, 6)


def test_load_rejects_4d(tmp_path):
    nib.save(nib.Nifti1Image(np.zeros((4, 5, 6, 2), np.float32), np.eye(4)), tmp_path / "a.nii.gz")
    with pytest.raises(InputError):
        io.load(tmp_path / "a.nii.gz")


def test_save_like_keeps_the_grid(tmp_path):
    ref = nib.Nifti1Image(np.full((4, 5, 6), 500.0), affine_for("LSP", (4, 5, 6)))
    ref.header["cal_max"] = 500
    nib.save(ref, tmp_path / "ref.nii.gz")
    ref = nib.load(tmp_path / "ref.nii.gz")         # the affine as a file holds it (float32)
    io.save_like(np.ones((4, 5, 6)), ref, tmp_path / "seg.nii.gz", np.uint8)
    out = nib.load(tmp_path / "seg.nii.gz")
    assert np.array_equal(out.affine, ref.affine)
    assert out.get_data_dtype() == np.uint8 and out.header["cal_max"] == 0
    assert np.array_equal(np.asanyarray(out.dataobj), np.ones((4, 5, 6)))


def test_stem():
    assert io.stem("/a/b/sub-01_T2w.nii.gz") == "sub-01_T2w"
    assert io.stem("sub-01.T2w.nii") == "sub-01.T2w"

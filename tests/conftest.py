import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from nibabel import orientations as nio

REPO = Path(__file__).resolve().parents[1]

# World positions (mm, along the canal) of the discs in the phantom, 35 mm apart
DISC_Z = {100: -60.0, 95: -25.0, 94: 10.0, 93: 45.0, 92: 80.0}


def affine_for(axcodes, shape, tilt_deg=8.0):
    """A 1 mm affine with the given axis codes, tilted about the left-right axis like a real
    acquisition, with the world origin at the centre of the grid."""
    ornt = nio.axcodes2ornt(tuple(axcodes))
    A = np.zeros((4, 4))
    for axis, (world, sign) in enumerate(ornt):
        A[int(world), axis] = sign
    t = np.radians(tilt_deg)
    R = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    A[:3, :3] = R @ A[:3, :3]
    A[:3, 3] = -A[:3, :3] @ ((np.array(shape) - 1) / 2)
    A[3, 3] = 1
    return A


def world_grid(shape, affine):
    ijk = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1)
    return nib.affines.apply_affine(affine, ijk)


def phantom(axcodes="LSP", shape=(96, 200, 64), discs=DISC_Z):
    """(image, TotalSpineSeg-like segmentation) of a straight spine along world z: a canal
    tube on the z axis holding a cord, and a disc 15 mm anterior of it at each level."""
    affine = affine_for(axcodes, shape)
    x, y, z = np.moveaxis(world_grid(shape, affine), -1, 0)
    r = np.hypot(x, y)
    seg = np.zeros(shape, np.uint8)
    seg[r < 6] = 2                                  # canal
    seg[(r < 2) & (z > 0)] = 1                      # cord, ends above the lower discs
    for label, z0 in discs.items():
        seg[(x ** 2 + (y - 15) ** 2 + ((z - z0) * 2) ** 2) < 10 ** 2] = label
    image = (100 * np.exp(-r / 40)).astype(np.float32)
    return nib.Nifti1Image(image, affine), nib.Nifti1Image(seg, affine)


@pytest.fixture
def test_data():
    path = Path(os.environ.get("NERVESEG_TEST_DATA", REPO / "test_data"))
    if not (path / "subj1").is_dir():
        pytest.skip("test_data/ is not available")
    return path

"""NIfTI helpers: loading, reorientation without resampling, saving on a reference grid."""
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel import orientations as nio

from .errors import InputError


def stem(path):
    """File name without .nii or .nii.gz."""
    return re.sub(r"\.nii(\.gz)?$", "", Path(path).name)


def load(path):
    """Load a 3D NIfTI image; trailing singleton dimensions are dropped."""
    img = nib.load(str(path))
    if len(img.shape) > 3:
        img = nib.funcs.squeeze_image(img)
    if len(img.shape) != 3:
        raise InputError(f"{path}: expected a 3D image, got shape {img.shape}")
    return img


def orientation_to(affine, axcodes):
    """Orientation transform that takes an array stored on `affine` to the axis codes
    `axcodes` (e.g. "LSP": first axis towards left, second superior, third posterior)."""
    return nio.ornt_transform(nio.io_orientation(affine), nio.axcodes2ornt(tuple(axcodes)))


def invert_orientation(ornt):
    """The transform that undoes `ornt`."""
    target = ornt[:, 0].astype(int)
    inverse = np.empty_like(ornt)
    inverse[target, 0] = np.arange(len(ornt))
    inverse[target, 1] = ornt[:, 1]
    return inverse


def reorient(data, ornt):
    """Flip and permute the three spatial axes of `data`; no resampling."""
    return nio.apply_orientation(data, ornt)


def reorient_zooms(zooms, ornt):
    """Voxel sizes after `reorient`."""
    out = np.empty(3)
    out[ornt[:, 0].astype(int)] = zooms[:3]
    return tuple(float(z) for z in out)


def save_like(data, ref, path, dtype):
    """Save `data` on the grid of `ref`. The reference header geometry is copied verbatim
    (affine=None), so spacing, origin and direction match the reference bit for bit; nnU-Net
    compares them exactly."""
    hdr = ref.header.copy()
    hdr.set_data_dtype(dtype)
    if len(ref.shape) > 3:
        hdr.set_data_shape(data.shape)
    hdr["cal_min"] = hdr["cal_max"] = 0     # the display range of the reference does not apply
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=dtype), None, hdr), str(path))

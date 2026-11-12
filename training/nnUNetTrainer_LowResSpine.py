"""
nnUNetTrainer_LowResSpine.py

nnU-Net v2 trainer (v2.5 or later, batchgeneratorsv2 transforms) that trains on
the 1 mm T2W but prepares the network for the low-resolution sagittal T2W and
for spines that are bent, tilted or twisted in the field of view. It is the
default nnUNetTrainer with these changes:

  Rotation and mirroring   in-plane rotation limited to +/-15 degrees (nnU-Net
                           uses +/-180 for our patch, so the spine is often
                           upside down) and mirroring left-right only (nnU-Net:
                           all three axes). The mirror axes are stored in the
                           checkpoint and also used for test-time mirroring at
                           prediction.
  IntensityCurveTransform  random smooth increasing intensity curve (a
                           different T2 contrast); image mean and SD kept.
  BiasFieldTransform       smooth multiplicative field (coil shading).
  SliceThicknessTransform  re-acquires the 1 mm image with thick slices along
                           one axis (Gaussian slice profile, FWHM = slice
                           spacing, random slice offset), with a random gain per
                           slice and between odd and even slices (striping of
                           interleaved acquisitions), a small in-plane shift per
                           slice (motion between slices) and noise per slice,
                           then interpolates back to 1 mm with cubic
                           interpolation, as nnU-Net does when it resamples a
                           2.5 mm image to the training spacing. Image only: the
                           target stays the 1 mm manual segmentation.
  FovBandTransform         keeps a left-right band centred on the labels and
                           blanks the rest (image 0, the value nnU-Net pads with
                           at prediction; labels background), then re-normalises
                           the MR channels in the band, as nnU-Net normalises the
                           narrow low-resolution image on its own (noNorm
                           channels keep their values). The low-resolution crops
                           are 65-75 mm wide and centred on the labels to within
                           7.5 mm (113 label maps).
  SpineBendTransform       smooth displacement along the superior-inferior
                           axis: lateral bend (scoliosis-like), anterior-
                           posterior bend and tilt, and rotation about the
                           superior-inferior axis that varies along the spine
                           (twist). Image and segmentation move together.

The intensity changes (curve, bias field, slice gain and noise, field-of-view
re-normalisation) are applied to INTENSITY_CHANNELS only; other input channels
(noNorm, e.g. coordinate maps) get only the geometric ones, slice resampling and
the zeros outside the field of view. INTENSITY_CHANNELS must be the normalised
channels of the plans; training stops with an error otherwise. nnU-Net's own augmentation (scaling, noise, blur, brightness,
contrast, low resolution, gamma) is unchanged. Elastic deformation stays off.

Axes are nnU-Net's array axes after transpose_forward. For our images (NIfTI
LSP, 256 x 256 x 50, read by SimpleITK, transpose_forward [0, 1, 2]):
0 = anterior-posterior (50 slices), 1 = superior-inferior, 2 = left-right.
The sagittal low-resolution T2W has its 2.5 mm slices along axis 2; a coronal
one (as subject 050) along axis 0. Check transpose_forward in the plans file
if the dataset changes.

Install: copy this file anywhere under nnunetv2/training/nnUNetTrainer/ of the
nnU-Net installation used for training and prediction, e.g.
    cp nnUNetTrainer_LowResSpine.py \\
       $(python -c "import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))")/training/nnUNetTrainer/variants/

Train / predict (the -tr name must match in both):
    nnUNetv2_train 145 3d_fullres 0 -tr nnUNetTrainer_LowResSpine     # folds 0-4
    nnUNetv2_predict -i IN -o OUT -d 145 -c 3d_fullres -tr nnUNetTrainer_LowResSpine -f 0

Settings are in the block below; a probability of 0 switches an augmentation
off. Run this file directly to print the augmentation pipeline and push one
random patch through it with every augmentation on.
"""

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform, ImageOnlyTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform

from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# ---- settings (axes as in the module docstring; distances in voxels = mm at 1 mm spacing;
#      (a, b) ranges are sampled uniformly per patch)
ROTATION_DEG = 15.            # in-plane rotation range, +/- degrees; None keeps nnU-Net's (180 here)
MIRROR_AXES = (2,)            # mirroring in training and at prediction; None keeps nnU-Net's (0, 1, 2),
                              # () switches it off. Left-right is safe while labels are levels, not sides
INTENSITY_CHANNELS = (0,)     # MR image channels: the normalised ones in dataset.json (not noNorm);
                              # checked against the plans when training starts
CURVE_P = 0.3                 # probability per training patch
CURVE_POINTS = 5              # control points of the intensity curve
CURVE_LOG_SLOPE_SD = 0.5      # SD of the log slope per curve segment (0.5: slopes mostly x0.6 to x1.6)
BIAS_P = 0.3
BIAS_STRENGTH = (0., 0.3)     # largest deviation of the field from 1 over the patch
BIAS_ORDER = 2                # cosine terms per axis: at most one bright-dark-bright cycle
THICK_P = 0.5
THICK_AXES = (2,)             # axes the thick slices may lie along, one picked per patch;
                              # (2, 0) also covers coronal thick-slice images
THICK_SPACING = (2.0, 3.0)    # slice spacing, voxels; slices contiguous (thickness = spacing)
THICK_GAIN_SD = (0., 0.05)    # SD of a random gain per slice
THICK_INTERLEAVE = (0., 0.05) # gain difference between odd and even slices (sign random)
THICK_MOTION_SD = (0., 0.75)  # SD of an in-plane shift per slice, voxels
THICK_NOISE_SD = (0., 0.1)    # SD of noise added to the slices, as a fraction of the image SD
FOV_P = 0.3
FOV_AXIS = 2                  # left-right
FOV_WIDTH = (65., 80.)        # band width, voxels
FOV_SHIFT = (-7.5, 7.5)       # band centre minus the labels' left-right midpoint, voxels
BEND_P = 0.3
BEND_SI_AXIS = 1              # the axis the spine runs along
BEND_LR_AXIS = 2
BEND_LR_SHIFT = (-10., 10.)   # lateral bend amplitude, voxels
BEND_AP_SHIFT = (-5., 5.)     # anterior-posterior bend amplitude, voxels
BEND_TILT = (-10., 10.)       # anterior-posterior tilt over the patch, degrees
BEND_ROTATION = (-10., 10.)   # rotation about the superior-inferior axis at the patch centre, degrees
BEND_TWIST = (-10., 10.)      # extra rotation at the superior/inferior patch edges, degrees
BEND_CYCLES = (0.25, 1.)      # bend curvature: sine cycles over the patch length
# ----


def _channels(img):
    return [c for c in INTENSITY_CHANNELS if c < img.shape[0]]


def _randn(*shape):
    """Standard normal noise from numpy: nnU-Net reseeds numpy in each data-loading
    worker but not torch, so torch noise would repeat across workers."""
    return torch.from_numpy(np.random.standard_normal(shape).astype(np.float32))


def _interp_last(values, pos, cubic):
    """Values at fractional indices pos along the last axis (edges clamped).
    Cubic is Keys' convolution kernel (a = -0.5)."""
    n = values.shape[-1]
    i0 = torch.floor(pos).long()
    t = (pos - i0).to(values.dtype)
    if not cubic:
        a = values[..., i0.clamp(0, n - 1)]
        b = values[..., (i0 + 1).clamp(0, n - 1)]
        return a + (b - a) * t
    w = [((-0.5 * t + 1.0) * t - 0.5) * t,
         (1.5 * t - 2.5) * t * t + 1.0,
         ((-1.5 * t + 2.0) * t + 0.5) * t,
         (0.5 * t - 0.5) * t * t]
    return sum(values[..., (i0 + k - 1).clamp(0, n - 1)] * w[k] for k in range(4))


class IntensityCurveTransform(ImageOnlyTransform):
    """Random increasing piecewise-linear intensity curve between the 0.5th and
    99.5th percentiles (slope 1 beyond them); image mean and SD kept."""

    def __init__(self, points=CURVE_POINTS, log_slope_sd=CURVE_LOG_SLOPE_SD):
        super().__init__()
        self.points, self.log_slope_sd = points, log_slope_sd

    def get_parameters(self, **data_dict):
        slopes = np.exp(np.random.normal(0, self.log_slope_sd, self.points - 1))
        knots = np.concatenate([[0.], np.cumsum(slopes)])
        return {"knots": torch.from_numpy(knots / knots[-1]).float()}

    def _apply_to_image(self, img, knots):
        img, m = img.clone(), len(knots) - 1
        for c in _channels(img):
            x = img[c].float()
            lo, hi = torch.quantile(x.flatten()[::7], torch.tensor([0.005, 0.995], dtype=x.dtype))
            if hi - lo < 1e-6:
                continue
            u = (x - lo) / (hi - lo) * m
            uc = u.clamp(0, m)
            j = uc.floor().clamp(0, m - 1).long()
            y = knots[j] + (uc - j) * (knots[j + 1] - knots[j]) + (u - uc) / m
            y = (y - y.mean()) / y.std().clamp_min(1e-6) * x.std() + x.mean()
            img[c] = y.to(img.dtype)
        return img


class BiasFieldTransform(ImageOnlyTransform):
    """Smooth multiplicative field: exp of a random sum of low-frequency cosines.
    Applied to the signal above the image minimum (the images are z-scored)."""

    def __init__(self, strength: RandomScalar = BIAS_STRENGTH, order=BIAS_ORDER):
        super().__init__()
        self.strength, self.order = strength, order

    def get_parameters(self, **data_dict):
        shape = data_dict["image"].shape[1:]
        k = self.order + 1
        f = np.random.normal(size=(k,) * len(shape))
        f.flat[0] = 0
        for n in shape:                     # contract one axis at a time: (k, k, k) -> (n0, n1, n2)
            f = np.tensordot(f, np.cos(np.pi * np.arange(k)[:, None] * np.linspace(0, 1, n)[None]),
                             axes=([0], [0]))
        f *= math.log1p(float(sample_scalar(self.strength))) / max(np.abs(f).max(), 1e-6)
        return {"field": torch.from_numpy(np.exp(f)).float()}

    def _apply_to_image(self, img, field):
        img = img.clone()
        for c in _channels(img):
            lo = img[c].min()
            img[c] = (lo + field * (img[c].float() - lo)).to(img.dtype)
        return img


class SliceThicknessTransform(ImageOnlyTransform):
    """Thick-slice acquisition along one axis, interpolated back to the grid."""

    def __init__(self, axes=THICK_AXES, spacing: RandomScalar = THICK_SPACING, gain_sd=THICK_GAIN_SD,
                 interleave=THICK_INTERLEAVE, motion_sd=THICK_MOTION_SD, noise_sd=THICK_NOISE_SD,
                 cubic=True):
        super().__init__()
        self.axes, self.spacing, self.cubic = tuple(axes), spacing, cubic
        self.gain_sd, self.interleave, self.motion_sd, self.noise_sd = gain_sd, interleave, motion_sd, noise_sd

    def get_parameters(self, **data_dict):
        axis = self.axes[np.random.randint(len(self.axes))]
        spacing = float(sample_scalar(self.spacing, image=data_dict["image"], dim=axis))
        u = np.random.uniform
        return {"axis": axis, "spacing": spacing, "offset": u(0, spacing), "gain_sd": u(*self.gain_sd),
                "interleave": u(*self.interleave) * np.random.choice([-1., 1.]),
                "motion_sd": u(*self.motion_sd), "noise_sd": u(*self.noise_sd)}

    def _apply_to_image(self, img, axis, spacing, offset, gain_sd, interleave, motion_sd, noise_sd):
        x = img.movedim(axis + 1, -1)                        # (C, ..., n)
        shape, n = x.shape, x.shape[-1]
        if n < 2 * spacing:
            return img
        x = x.reshape(-1, 1, n).float()
        # slice profile: Gaussian, FWHM = slice thickness = spacing (a sum of shifted copies;
        # F.conv1d is ~100x slower on CPU for this shape)
        sigma = spacing / (2 * math.sqrt(2 * math.log(2)))
        r = max(1, math.ceil(3 * sigma))
        k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=x.dtype) / sigma) ** 2)
        k = k / k.sum()
        xp = F.pad(x, (r, r), mode="replicate")[:, 0]
        x = sum(xp[:, j:j + n] * k[j] for j in range(2 * r + 1))
        # sample the slices, corrupt them as an acquisition would, then back to the 1 mm grid
        centres = torch.arange(offset, n - 1 + 1e-6, spacing, dtype=x.dtype)
        s = _interp_last(x, centres, cubic=False).reshape(*shape[:-1], len(centres))
        s = self._acquire(s, gain_sd, interleave, motion_sd, noise_sd)
        grid = (torch.arange(n, dtype=x.dtype) - offset) / spacing
        x = _interp_last(s.reshape(-1, s.shape[-1]), grid, cubic=self.cubic)
        return x.reshape(shape).movedim(-1, axis + 1).to(img.dtype)

    @staticmethod
    def _acquire(s, gain_sd, interleave, motion_sd, noise_sd):
        """Shift, gain and noise per slice; s is (C, a, b, slices)."""
        m = s.shape[-1]
        if motion_sd > 0 and s.ndim == 4:
            d = torch.from_numpy(np.random.normal(0, motion_sd, (m, 2))).float()
            theta = torch.zeros(m, 2, 3)
            theta[:, 0, 0] = theta[:, 1, 1] = 1
            theta[:, 0, 2] = 2 * d[:, 0] / max(s.shape[2] - 1, 1)       # along b
            theta[:, 1, 2] = 2 * d[:, 1] / max(s.shape[1] - 1, 1)       # along a
            sl = s.permute(3, 0, 1, 2)                                   # (slices, C, a, b)
            grid = F.affine_grid(theta, list(sl.shape), align_corners=True)
            s = F.grid_sample(sl, grid, mode="bilinear", padding_mode="border",
                              align_corners=True).permute(1, 2, 3, 0)
        if gain_sd > 0 or interleave != 0 or noise_sd > 0:
            gain = 1 + np.random.normal(0, gain_sd, m) + interleave * (-1.) ** np.arange(m)
            gain = torch.from_numpy(gain).float()
            for c in _channels(s):
                v = s[c]
                lo, sd = v.min(), v.std()
                s[c] = lo + gain * (v - lo) + noise_sd * sd * _randn(*v.shape)
        return s


class FovBandTransform(BasicTransform):
    """Field of view limited to a band along one axis, centred on the labels'
    extent along that axis (the patch centre when the patch has no labels)."""

    def __init__(self, axis=FOV_AXIS, width=FOV_WIDTH, shift=FOV_SHIFT, renormalize=True):
        super().__init__()
        self.axis, self.width, self.shift, self.renormalize = axis, width, shift, renormalize

    def get_parameters(self, **data_dict):
        seg = data_dict.get("segmentation")
        d = self.axis + 1
        n = data_dict["image"].shape[d]
        centre = (n - 1) / 2
        if seg is not None:
            idx = torch.nonzero((seg > 0).sum(dim=tuple(i for i in range(seg.ndim) if i != d))).flatten()
            if len(idx):
                centre = (idx.min().item() + idx.max().item()) / 2
        width = np.random.uniform(*self.width)
        lo = int(round(centre + np.random.uniform(*self.shift) - width / 2))
        lo = min(max(lo, 0), n - 1)
        return {"lo": lo, "hi": min(max(lo + int(round(width)), lo + 1), n)}

    def _apply_to_image(self, img, lo, hi):
        """Every channel 0 outside the band (nnU-Net pads every channel with 0 at prediction);
        inside it the MR channels re-normalised, as nnU-Net z-scores a narrow scan on its own.
        Other channels (noNorm, e.g. coordinate maps) keep their values."""
        d = self.axis + 1
        out = torch.zeros_like(img)
        out.narrow(d, lo, hi - lo).copy_(img.narrow(d, lo, hi - lo))
        if self.renormalize:
            for c in _channels(img):
                band = out[c].narrow(d - 1, lo, hi - lo)
                x = band.float()
                band.copy_(((x - x.mean()) / x.std().clamp_min(1e-6)).to(img.dtype))
        return out

    def _apply_to_segmentation(self, seg, lo, hi):
        d = self.axis + 1
        out = torch.zeros_like(seg)
        out.narrow(d, lo, hi - lo).copy_(seg.narrow(d, lo, hi - lo))
        return out


class SpineBendTransform(BasicTransform):
    """Smooth bend, tilt and twist of the spine along the superior-inferior axis."""

    def __init__(self, si_axis=BEND_SI_AXIS, lr_axis=BEND_LR_AXIS, lr_shift=BEND_LR_SHIFT,
                 ap_shift=BEND_AP_SHIFT, tilt=BEND_TILT, rotation=BEND_ROTATION,
                 twist=BEND_TWIST, cycles=BEND_CYCLES):
        super().__init__()
        self.si, self.lr = si_axis, lr_axis
        self.ap = ({0, 1, 2} - {si_axis, lr_axis}).pop()
        self.lr_shift, self.ap_shift, self.tilt = lr_shift, ap_shift, tilt
        self.rotation, self.twist, self.cycles = rotation, twist, cycles

    def get_parameters(self, **data_dict):
        u = np.random.uniform
        p = {k: u(*getattr(self, k)) for k in ("lr_shift", "ap_shift", "tilt", "rotation", "twist")}
        p.update(cyc_lr=u(*self.cycles), cyc_ap=u(*self.cycles),
                 ph_lr=u(0, 2 * math.pi), ph_ap=u(0, 2 * math.pi))
        p["shape"] = data_dict["image"].shape[1:]
        return {"grid": self._grid(**p)}

    def _grid(self, shape, lr_shift, ap_shift, tilt, rotation, twist, cyc_lr, cyc_ap, ph_lr, ph_ap):
        """grid_sample grid: for each output voxel, where to read the input."""
        n = [float(s) for s in shape]
        c = [(s - 1) / 2 for s in n]
        coords = torch.meshgrid(*[torch.arange(s, dtype=torch.float32) for s in shape], indexing="ij")
        s = (coords[self.si] - c[self.si]) / max(c[self.si], 1)          # -1 .. 1 along the spine
        a, l = coords[self.ap] - c[self.ap], coords[self.lr] - c[self.lr]

        def bend(amp, cyc, ph):        # zero at the patch centre, so the patch does not just shift
            return amp * (torch.sin(math.pi * cyc * s + ph) - math.sin(ph))

        theta = math.radians(rotation) + math.radians(twist) * s
        cos, sin = torch.cos(theta), torch.sin(theta)
        src = [None] * 3
        src[self.si] = coords[self.si]
        src[self.ap] = (c[self.ap] + cos * a - sin * l + bend(ap_shift, cyc_ap, ph_ap)
                        + math.tan(math.radians(tilt)) * s * c[self.si])
        src[self.lr] = c[self.lr] + sin * a + cos * l + bend(lr_shift, cyc_lr, ph_lr)
        # grid_sample wants (x, y, z) = (axis 2, axis 1, axis 0), normalised to -1 .. 1
        norm = [2 * src[d] / max(n[d] - 1, 1) - 1 for d in (2, 1, 0)]
        return torch.stack(norm, dim=-1)[None]

    def _sample(self, x, grid, mode):
        return F.grid_sample(x[None].float(), grid, mode=mode, padding_mode="zeros",
                             align_corners=True)[0]

    def _apply_to_image(self, img, grid):
        return self._sample(img, grid, "bilinear").to(img.dtype)

    def _apply_to_segmentation(self, seg, grid):
        return self._sample(seg, grid, "nearest").round().to(seg.dtype)


def _rotation_and_initial_patch(patch_size, do_dummy_2d_data_aug):
    """Rotation range and the larger patch to load so that rotation and scaling
    leave no empty corners, computed as nnU-Net does."""
    rot = (-math.radians(ROTATION_DEG), math.radians(ROTATION_DEG))
    initial = get_patch_size(patch_size, rot, rot, rot, (0.85, 1.25))
    if do_dummy_2d_data_aug:
        initial[0] = patch_size[0]
    return rot, initial


class nnUNetTrainer_LowResSpine(nnUNetTrainer):
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        # the intensity changes must reach exactly the normalised (MR) channels: a noNorm channel
        # (coordinate map) given them, or an MR channel missed, would differ from prediction
        schemes = list(self.configuration_manager.normalization_schemes)
        mr = [c for c, s in enumerate(schemes) if s != "NoNormalization"]
        if sorted(INTENSITY_CHANNELS) != mr:
            raise RuntimeError(f"INTENSITY_CHANNELS = {tuple(INTENSITY_CHANNELS)}, but the normalised channels "
                               f"in the plans are {tuple(mr)} ({schemes}); set INTENSITY_CHANNELS = {tuple(mr)}")
        rot, dummy_2d, initial, mirror = super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        if ROTATION_DEG is not None:
            rot, initial = _rotation_and_initial_patch(self.configuration_manager.patch_size, dummy_2d)
        if MIRROR_AXES is not None:
            mirror = tuple(MIRROR_AXES) or None
            self.inference_allowed_mirroring_axes = mirror      # test-time mirroring at prediction
        self.print_to_log_file(f"rotation +/-{math.degrees(max(rot)):.0f} deg, mirror axes {mirror}, "
                               f"initial patch {[int(i) for i in initial]}")
        return rot, dummy_2d, initial, mirror

    @staticmethod
    def get_training_transforms(patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
                                do_dummy_2d_data_aug, use_mask_for_norm=None, is_cascaded=False,
                                foreground_labels=None, regions=None, ignore_label=None):
        tr = nnUNetTrainer.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=use_mask_for_norm, is_cascaded=is_cascaded,
            foreground_labels=foreground_labels, regions=regions, ignore_label=ignore_label)
        ts = tr.transforms
        # on the final patch, after the spatial transform and before nnU-Net's noise, so the
        # slices and the field-of-view band lie along the array axes as in the low-resolution
        # scans; in acquisition order, the band (with its re-normalisation) last
        i = next(k for k, t in enumerate(ts)
                 if isinstance(t, RandomTransform) and isinstance(t.transform, GaussianNoiseTransform))
        ts[i:i] = [RandomTransform(t(), apply_probability=p) for t, p in (
            (IntensityCurveTransform, CURVE_P), (BiasFieldTransform, BIAS_P),
            (SliceThicknessTransform, THICK_P), (FovBandTransform, FOV_P)) if p > 0]
        # bend first, on the larger initial patch, so the crop that follows has no empty borders
        if BEND_P > 0:
            ts.insert(0, RandomTransform(SpineBendTransform(), apply_probability=BEND_P))
        return tr


class nnUNetTrainer_LowResSpine_100epochs(nnUNetTrainer_LowResSpine):
    # set here rather than in __init__, whose signature differs between nnU-Net versions
    def initialize(self):
        self.num_epochs = 100
        super().initialize()


class nnUNetTrainer_LowResSpine_250epochs(nnUNetTrainer_LowResSpine):
    def initialize(self):
        self.num_epochs = 250
        super().initialize()


class nnUNetTrainer_LowResSpine_500epochs(nnUNetTrainer_LowResSpine):
    def initialize(self):
        self.num_epochs = 500
        super().initialize()


if __name__ == "__main__":
    patch = (48, 224, 224)                      # our 3d_fullres plans (dummy 2D augmentation)
    rot, initial = _rotation_and_initial_patch(patch, True)
    tr = nnUNetTrainer_LowResSpine.get_training_transforms(
        patch, rot, [[1, 1, 1], [1, .5, .5]], MIRROR_AXES, True)
    print("\n".join(f"{k:2d} {t}"[:150] for k, t in enumerate(tr.transforms)))
    ours = (IntensityCurveTransform, BiasFieldTransform, SliceThicknessTransform, FovBandTransform,
            SpineBendTransform)
    for t in tr.transforms:
        if isinstance(t, RandomTransform) and isinstance(t.transform, ours):
            t.apply_probability = 1.
    seg = torch.zeros(1, *initial, dtype=torch.int16)
    mid = initial[2] // 2
    seg[0, 10:40, 60:200, mid - 35:mid - 20] = 1
    seg[0, 10:40, 60:200, mid + 20:mid + 35] = 2
    t0 = time.time()
    out = tr(image=torch.randn(1, *initial), segmentation=seg)
    print(f"initial patch {[int(i) for i in initial]} -> image {tuple(out['image'].shape)}, segmentation "
          f"{[tuple(s.shape) for s in out['segmentation']]}; {time.time() - t0:.2f} s with everything on")

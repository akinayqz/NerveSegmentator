"""
Spine-aligned coordinate channels.

Two volumes that tell the network where a voxel lies relative to the spine, built
from a TotalSpineSeg segmentation (step1_output or step2_output):

  coordCC   craniocaudal position along the canal in disc units
            (0 = L5/S1, 1 = L4/L5, 2 = L3/L4, ...; negative below L5/S1)
  coordLR   signed left-right offset from the canal centreline,
            in mm / 50 (positive = subject's right)

The same code prepares the training data (`nerveseg_coords`) and runs inside the
pipeline, so the networks see the same channels in both.
"""
from dataclasses import dataclass

import numpy as np
import nibabel as nib
from scipy.ndimage import center_of_mass, gaussian_filter1d
from scipy.spatial import cKDTree

from .errors import CoordinateError

# TotalSpineSeg label values
CORD, CANAL = 1, 2
DISC_UNITS = {100: 0.0, 95: 1.0, 94: 2.0, 93: 3.0, 92: 4.0, 91: 5.0}
DISC_NAMES = {100: 'L5-S1', 95: 'L4-L5', 94: 'L3-L4',
              93: 'L2-L3', 92: 'L1-L2', 91: 'T12-L1'}

LR_SCALE = 50.0               # mm, so the LR channel is roughly within +-1
SMOOTH_MM = 5.0               # Gaussian smoothing of the centreline
STEP_MM = 0.5                 # centreline resampling step
EXTEND_MM = 80.0              # extrapolate centreline beyond the canal ends (sacral roots)
SPACING_RANGE = (20.0, 60.0)  # plausible disc-to-disc distance along the canal, mm
MAX_DISC_DIST = 45.0          # disc centroid to centreline, mm


@dataclass
class Coordinates:
    cc: np.ndarray      # coordCC on the reference grid, float32
    lr: np.ndarray      # coordLR on the reference grid, float32
    report: dict        # anchor positions and QC flags (status: 'ok' or 'warning')


def canal_centreline(seg, A):
    """Smoothed, uniformly resampled, extended centreline of cord + canal (world mm)."""
    mask = np.isin(seg, [CORD, CANAL])
    if not mask.any():
        raise RuntimeError('no cord/canal labels (1, 2) in segmentation')

    si = int(np.argmax(np.abs(A[2, :3])))        # voxel axis closest to head-foot
    pts = []
    for k in range(seg.shape[si]):
        sl = np.take(mask, k, axis=si)
        if sl.sum() < 3:
            continue
        c = list(center_of_mass(sl))
        c.insert(si, k)
        pts.append(c)
    if len(pts) < 10:
        raise RuntimeError('canal too short: only %d slices' % len(pts))

    pts = nib.affines.apply_affine(A, np.asarray(pts, float))
    pts = pts[np.argsort(pts[:, 2])]              # inferior -> superior

    vox = float(np.linalg.norm(A[:3, si]))
    pts = gaussian_filter1d(pts, SMOOTH_MM / vox, axis=0, mode='nearest')

    # uniform resampling by arc length
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate([[True], d > 1e-6])
    pts = pts[keep]
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    if arc[-1] < 20.0:
        raise RuntimeError('canal centreline shorter than 20 mm')
    s = np.arange(0.0, arc[-1], STEP_MM)
    pts = np.column_stack([np.interp(s, arc, pts[:, j]) for j in range(3)])

    # straight-line extension at both ends along the end tangents
    n = min(int(10.0 / STEP_MM), len(pts) - 1)
    t0 = pts[0] - pts[n]
    t1 = pts[-1] - pts[-1 - n]
    t0 /= np.linalg.norm(t0)
    t1 /= np.linalg.norm(t1)
    ext = np.arange(STEP_MM, EXTEND_MM + STEP_MM / 2, STEP_MM)
    head = pts[0] + ext[::-1, None] * t0
    tail = pts[-1] + ext[:, None] * t1
    core = (len(head), len(head) + len(pts) - 1)  # index range of the real canal
    pts = np.vstack([head, pts, tail])
    arc = np.arange(len(pts)) * STEP_MM
    return pts, arc, core


def disc_anchors(seg, A, pts, arc, core, tree):
    anchors = {}
    for lab, unit in DISC_UNITS.items():
        m = seg == lab
        if not m.any():
            continue
        c = nib.affines.apply_affine(A, center_of_mass(m))
        dist, i = tree.query(c)
        anchors[lab] = {
            'name': DISC_NAMES[lab],
            'unit': unit,
            'arc_mm': float(arc[i]),
            'disc_centroid_mm': [round(float(x), 2) for x in c],
            'centreline_point_mm': [round(float(x), 2) for x in pts[i]],
            'dist_to_centreline_mm': round(float(dist), 2),
            'on_extrapolated_centreline': bool(i < core[0] or i > core[1]),
            'n_voxels': int(m.sum()),
        }
    return anchors


def check_anchors(anchors):
    """Return (fatal_errors, warnings)."""
    errors, warnings = [], []
    if 100 not in anchors:
        errors.append('L5-S1 disc (label 100) not found')
    if len(anchors) < 2:
        errors.append('fewer than two lumbar discs found')
    if errors:
        return errors, warnings

    labs = sorted(anchors, key=lambda l: anchors[l]['unit'])
    a = np.array([anchors[l]['arc_mm'] for l in labs])
    steps = np.diff(a)                           # inferior -> superior should be positive
    if np.any(steps <= 0):
        errors.append('disc order along the canal is inconsistent: %s'
                      % ', '.join('%s=%.1f' % (anchors[l]['name'], anchors[l]['arc_mm'])
                                  for l in labs))
        return errors, warnings

    units = np.array([anchors[l]['unit'] for l in labs])
    for k, st in enumerate(steps):
        per_level = st / (units[k + 1] - units[k])
        if not SPACING_RANGE[0] <= per_level <= SPACING_RANGE[1]:
            warnings.append('spacing %s to %s is %.1f mm per level'
                            % (anchors[labs[k]]['name'], anchors[labs[k + 1]]['name'], per_level))
    missing = [DISC_NAMES[l] for l in DISC_UNITS
               if l not in anchors and DISC_UNITS[l] <= max(units)]
    if missing:
        warnings.append('missing discs: ' + ', '.join(missing))
    for l in labs:
        if anchors[l]['dist_to_centreline_mm'] > MAX_DISC_DIST:
            warnings.append('%s is %.1f mm from the centreline'
                            % (anchors[l]['name'], anchors[l]['dist_to_centreline_mm']))
        if anchors[l]['on_extrapolated_centreline']:
            warnings.append('%s maps onto the extrapolated centreline (canal mask stops short)'
                            % anchors[l]['name'])
    return errors, warnings


def arc_to_units(arc, anchors):
    labs = sorted(anchors, key=lambda l: anchors[l]['unit'])
    a = np.array([anchors[l]['arc_mm'] for l in labs])
    u = np.array([anchors[l]['unit'] for l in labs])
    cc = np.interp(arc, a, u)
    lo, hi = arc < a[0], arc > a[-1]             # linear extrapolation beyond the anchors
    cc[lo] = u[0] + (arc[lo] - a[0]) * (u[1] - u[0]) / (a[1] - a[0])
    cc[hi] = u[-1] + (arc[hi] - a[-1]) * (u[-1] - u[-2]) / (a[-1] - a[-2])
    return cc


def build_channels(ref, pts, cc_line, tree):
    A = ref.affine
    shp = ref.shape[:3]
    cc = np.empty(shp, np.float32)
    lr = np.empty(shp, np.float32)
    jj, kk = np.meshgrid(np.arange(shp[1]), np.arange(shp[2]), indexing='ij')
    jj, kk = jj.ravel(), kk.ravel()
    for i in range(shp[0]):                      # slab by slab to limit memory
        ijk = np.column_stack([np.full(jj.size, i), jj, kk])
        xyz = nib.affines.apply_affine(A, ijk)
        _, near = tree.query(xyz)
        cc[i] = cc_line[near].reshape(shp[1], shp[2])
        # nibabel world space is RAS+, so +x points to the subject's right
        lr[i] = ((xyz[:, 0] - pts[near, 0]) / LR_SCALE).reshape(shp[1], shp[2])
    return cc, lr


def compute(seg_img, ref_img):
    """Channels on the grid of `ref_img` from the TotalSpineSeg segmentation `seg_img`
    (nibabel images; the two need not share a grid, only world space).

    Raises CoordinateError when the segmentation cannot anchor the levels (no canal,
    no L5-S1 disc, ...); its `report` is the same dict with the errors filled in.
    """
    seg = np.rint(np.asanyarray(seg_img.dataobj)).astype(np.int32)
    if seg.ndim > 3:
        seg = seg[..., 0]

    report = {'labels_found': sorted(int(v) for v in np.unique(seg) if v > 0)}
    try:
        pts, arc, core = canal_centreline(seg, seg_img.affine)
        tree = cKDTree(pts)
        anchors = disc_anchors(seg, seg_img.affine, pts, arc, core, tree)
        errors, warnings = check_anchors(anchors)
        report['anchors'] = {anchors[l]['name']: anchors[l] for l in sorted(anchors)}
        report['canal_length_mm'] = round(float((core[1] - core[0]) * STEP_MM), 1)
    except RuntimeError as e:
        errors, warnings = [str(e)], []

    report['errors'], report['warnings'] = errors, warnings
    report['status'] = 'failed' if errors else ('warning' if warnings else 'ok')
    if errors:
        raise CoordinateError(report)

    cc_line = arc_to_units(arc, anchors)
    cc, lr = build_channels(ref_img, pts, cc_line, tree)
    return Coordinates(cc=cc, lr=lr, report=report)

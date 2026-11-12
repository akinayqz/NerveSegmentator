#!/usr/bin/env python
"""
make_level_channels.py

Build spine-aligned coordinate channels for nnUNet from a TotalSpineSeg
segmentation (step1_output or step2_output).

Outputs, written on the grid of the reference image (the image fed to nnUNet):
  <prefix>_coordCC.nii.gz       craniocaudal position along the canal in disc units
                                (0 = L5/S1, 1 = L4/L5, 2 = L3/L4, ...; negative below L5/S1)
  <prefix>_coordLR.nii.gz       signed left-right offset from the canal centreline,
                                in mm / 50 (positive = subject's right)
  <prefix>_coordAnchors.json    anchor positions and QC flags

Usage:
  python make_level_channels.py --seg SEG.nii.gz --ref T2W.nii.gz [--prefix P] [--outdir D]

Exit code 0 = channels written (check WARNING lines), 1 = failed, nothing written
except the JSON.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import nibabel as nib
from scipy.ndimage import center_of_mass, gaussian_filter1d
from scipy.spatial import cKDTree

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


def save_like(data, ref, path):
    # Copy the reference header geometry verbatim (affine=None), so spacing, origin
    # and direction match the reference bit for bit; nnUNet compares them exactly.
    hdr = ref.header.copy()
    hdr.set_data_dtype(np.float32)
    if len(ref.shape) > 3:
        hdr.set_data_shape(data.shape)
    nib.save(nib.Nifti1Image(data, None, hdr), path)


def default_prefix(ref_path):
    base = os.path.basename(ref_path)
    base = re.sub(r'\.nii(\.gz)?$', '', base)
    m = re.match(r'(.*?_T2W)', base, re.IGNORECASE)
    return m.group(1) if m else base


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seg', required=True, help='TotalSpineSeg step1_output or step2_output')
    p.add_argument('--ref', required=True, help='image given to nnUNet (defines the output grid)')
    p.add_argument('--prefix', help='output name prefix (default: ref name up to _T2W)')
    p.add_argument('--outdir', help='output folder (default: folder of --ref)')
    args = p.parse_args()

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.ref))
    prefix = args.prefix or default_prefix(args.ref)
    out_cc = os.path.join(outdir, prefix + '_coordCC.nii.gz')
    out_lr = os.path.join(outdir, prefix + '_coordLR.nii.gz')
    out_js = os.path.join(outdir, prefix + '_coordAnchors.json')

    seg_img = nib.load(args.seg)
    seg = np.rint(np.asanyarray(seg_img.dataobj)).astype(np.int32)
    if seg.ndim > 3:
        seg = seg[..., 0]
    ref = nib.load(args.ref)

    report = {'seg': os.path.abspath(args.seg), 'ref': os.path.abspath(args.ref),
              'labels_found': sorted(int(v) for v in np.unique(seg) if v > 0)}
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
    with open(out_js, 'w') as f:
        json.dump(report, f, indent=2)

    tag = prefix
    if errors:
        for e in errors:
            print('ERROR [%s] %s' % (tag, e), file=sys.stderr)
        print('ERROR [%s] labels found: %s' % (tag, report['labels_found']), file=sys.stderr)
        sys.exit(1)
    for w in warnings:
        print('WARNING [%s] %s' % (tag, w))

    cc_line = arc_to_units(arc, anchors)
    cc, lr = build_channels(ref, pts, cc_line, tree)
    save_like(cc, ref, out_cc)
    save_like(lr, ref, out_lr)
    print('OK [%s] %d discs: %s -> %s, %s'
          % (tag, len(anchors), ', '.join(anchors[l]['name'] for l in
                                          sorted(anchors, key=lambda l: anchors[l]['unit'])),
             os.path.basename(out_cc), os.path.basename(out_lr)))


if __name__ == '__main__':
    main()

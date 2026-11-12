# NerveSegmentator

Segmentation of the lumbosacral dorsal root ganglia (DRG) and nerve roots, L2 to S1, in
lumbar spine MRI. One command takes a T2w image and writes a label map per structure, labelled
by spinal level.

It wraps nnU-Net models, in the manner of TotalSegmentator and TotalSpineSeg. The networks
get the image plus two coordinate maps that say where each voxel lies relative to the spine
(level along the canal, distance left or right of it); these are built from a
[TotalSpineSeg](https://github.com/neuropoly/totalspineseg) segmentation of the same image.

> **Status: pre-release.** The models are still being trained and no weights are published
> yet: install them from a local nnU-Net folder (see [Weights](#weights)). Only T2w is
> supported so far. The licence is not decided.

## Install

```bash
pip install -e ".[all]"          # from a clone; Python >= 3.10
```

`[all]` adds nnU-Net and TotalSpineSeg. They are extras so that a host environment can bring
its own: `".[neuropoly,totalspineseg]"` uses the nnU-Net fork that Spinal Cord Toolbox ships,
and TotalSpineSeg can be left out if you always pass an existing spine segmentation.

## Weights

```bash
nerveseg_weights import /path/to/nnUNet_results/Dataset311_DRG  -m drg_t2
nerveseg_weights import /path/to/nnUNet_results/Dataset312_root -m root_t2
nerveseg_weights list
```

`import` takes an nnU-Net training output folder (or a zip made by `nerveseg_weights export`),
keeps the final checkpoint of each fold without its optimizer state (about 120 MB per fold)
and stores it in the data folder: `--data-dir`, else `$NERVESEGMENTATOR_DATA`, else
`~/.nervesegmentator`. Once weights are published they are downloaded on first use.

## Usage

```bash
NerveSegmentator -i sub-01_T2w.nii.gz -o out
```

| Option | |
|---|---|
| `-i IMAGE [IMAGE ...]` | Image(s) to segment. The networks are loaded once for all of them. |
| `-o DIR` | Output folder. |
| `-s drg root` | Structures to segment (default: both). |
| `--spine-seg SEG [SEG ...]` | TotalSpineSeg `step1_output` or `step2_output` of each image, e.g. from `sct_deepseg spine`. Without it TotalSpineSeg is run (step 1). |
| `--fast` | One fold per model instead of the 5-fold ensemble: 5 times faster. |
| `--folds N [N ...]`, `--no-tta` | Choose the folds; switch off test-time mirroring. |
| `--device` | `auto` (CUDA if available, else CPU), `cpu`, `cuda`, `cuda:N`, `mps`. |
| `--save-channels` | Also write the two coordinate maps. |

Output, per image `<name>.nii.gz`:

| File | |
|---|---|
| `<name>_drg.nii.gz`, `<name>_root.nii.gz` | Label maps on the grid of the input. |
| `<name>_nerveseg.json` | What was run (versions, models, folds), where the discs were found, warnings. |
| `<name>_spine.nii.gz` | The TotalSpineSeg segmentation, when it was run here. Pass it back with `--spine-seg` to skip that step next time. |

Labels are spinal levels, the same in both maps: **L2 = 2, L3 = 3, L4 = 4, L5 = 5, S1 = 1**
(the values of the manual segmentations used for training; defined in
`nervesegmentator/labels.py`). Left and right share a label. The two structures come from
separate networks, so the two maps can overlap by a few voxels.

The exit status is 1 if an image could not be segmented. That happens when the levels cannot
be anchored, for instance when TotalSpineSeg finds no L5-S1 disc; the JSON says why. Level
labels are only as good as the disc labels they are anchored to, so read the warnings.

### Input

T2w, prepared as the training images were: NerveSegmentator does no preprocessing other than
bringing the image to the orientation the networks were trained in (any orientation is
accepted, and the output is on the original grid). The current models were trained on coronal
3D T2w at 1 mm isotropic (256 x 256 x 50), with augmentation towards thick-slice sagittal T2w.
nnU-Net resamples other resolutions to 1 mm.

### Speed

Measured on an Apple M2 Ultra for one 256 x 256 x 50 image. Per structure, the 5-fold
ensemble takes about 16 s on the Apple GPU (`--device mps`, PyTorch 2.15) and one fold about
80 s on CPU, so on CPU use `--fast` (about 3 minutes for both structures) unless you can wait
some 13 minutes. TotalSpineSeg adds about 2 minutes on CPU. `mps` is not chosen automatically
because it stalled in our test with PyTorch 2.2.

## Python

```python
from nervesegmentator import NerveSegmentator

ns = NerveSegmentator(structures=["drg", "root"], device="cuda")      # loads the networks
for r in ns.run(["sub-01_T2w.nii.gz", "sub-02_T2w.nii.gz"], "out"):
    print(r.image, r.status, r.outputs, r.warnings)
```

`segment(images, output_dir, ...)` does both steps in one call. When TotalSpineSeg is run
(no `spine_seg=`), call from under `if __name__ == "__main__":`, as it starts worker processes.

## How it works

1. **Spine segmentation** (`spine.py`): TotalSpineSeg, unless a segmentation is given.
2. **Coordinate channels** (`coords.py`): the canal centreline and the lumbar discs give
   `coordCC` (position along the canal in disc units, 0 at L5/S1, 1 at L4/L5, ...) and
   `coordLR` (mm / 50 right of the centreline). `nerveseg_coords` writes the same channels to
   files for preparing training data.
3. **Prediction** (`predict.py`): each model gets the channels it lists, in its training
   orientation, and its folds are ensembled by nnU-Net.
4. **Labels** (`labels.py`): the model's values are mapped to the output values by level name.

| | |
|---|---|
| `api.py` | The pipeline: `NerveSegmentator`, `segment`. |
| `registry.py` | The models: one per structure and contrast, with its channels, orientation and release URL. |
| `weights.py` | The data folder; import, export, download. An installed model is a folder with a `model.json` that says what it needs. |
| `cli.py` | `NerveSegmentator`, `nerveseg_weights`, `nerveseg_coords`. |
| `training/` | The nnU-Net trainer the models are trained with, and how to prepare the data. |

### Releasing a model, adding a contrast

```bash
nerveseg_weights export /path/to/nnUNet_results/Dataset311_DRG -m drg_t2 --release r20261112
```

writes `release/drg_t2_r20261112.zip` and prints its SHA-256. Upload the zip as a release
asset and set `release`, `url` and `sha256` of the model in `registry.py`. A retrained model
needs nothing else, also when its label values or their order change. A new contrast (T1w) is
one more `ModelSpec` in `registry.py`, selected with `--contrast`.

The exported checkpoints name nnU-Net's stock trainer, so they load in any nnU-Net
installation without our trainer class. This relies on the trainer changing the augmentation
only, not the network.

### Spinal Cord Toolbox

The package is meant to be called from `sct_deepseg` the way TotalSpineSeg is: SCT lists it
as a dependency and calls `NerveSegmentator(...).run(...)` with its own weights folder
(`data_dir=`) and the TotalSpineSeg weights it already has (`spine_data_dir=`). Prediction
is tested with SCT's pins (Python 3.10, torch 2.2, numpy 1.26, `nnunetv2-neuropoly` 2.6.2);
the adapter in SCT is still to be written. SCT's generic nnU-Net path cannot be used, because
it gives a model one image as one channel.

## Tests

```bash
pip install -e ".[all,dev]"
pytest -m "not slow"             # no weights or data needed
pytest -m slow                   # end to end: needs test_data/ and installed weights
```

# NerveSegmentator in Spinal Cord Toolbox

`sct_deepseg_nerves_lumbar_t2.patch` adds a task to `sct_deepseg`:

```bash
sct_deepseg nerves_lumbar_t2 -install
sct_deepseg spine -i t2.nii.gz                    # writes t2_totalspineseg_all.nii.gz
sct_deepseg nerves_lumbar_t2 -i t2.nii.gz -spine-seg t2_totalspineseg_all.nii.gz
```

The last command writes `t2_drg.nii.gz` and `t2_root.nii.gz` next to the image (with
`-o name.nii.gz`: `name_drg.nii.gz`, `name_root.nii.gz`), with the labels of the standalone
command. It takes one image per call.

| Option | |
|---|---|
| `-spine-seg FILE` | The `_totalspineseg_all` output of `sct_deepseg spine` for the image. Without it the spine is segmented first, and that segmentation is not kept. |
| `-single-fold` | One fold per model instead of five: 5 times faster. |
| `-test-time-aug` | Left-right mirroring. Off unless asked for, as in every `sct_deepseg` task; the standalone command has it on. |

The patch is against SCT commit `017d797` (7.4.dev0, November 2026).

## What it changes in SCT

| File | |
|---|---|
| `deepseg/models.py` | The model entry (one URL per NerveSegmentator model) and the task. Installing hands the URLs to `nervesegmentator.weights.install_url`, which puts the models in SCT's model folder, then installs the TotalSpineSeg model if it is missing. |
| `deepseg/inference.py` | `create_nervesegmentator` loads the networks; `segment_nervesegmentator` segments the spine with SCT's own TotalSpineSeg function if no segmentation was given, then calls `NerveSegmentator.run`. |
| `scripts/sct_deepseg.py` | `-spine-seg`, `-single-fold`, and the checks for this task. |
| `documentation/`, `testing/` | The task's page, and two tests that need no weights. |

SCT does not import this package unless the task is used. If the package is missing, the
task stops with the `pip` command that installs it.

## Trying it

```bash
git clone https://github.com/spinalcordtoolbox/spinalcordtoolbox && cd spinalcordtoolbox
git checkout 017d797
git apply /path/to/NerveSegmentator/sct/sct_deepseg_nerves_lumbar_t2.patch
./install_sct                                                  # as for any SCT checkout
python/envs/venv_sct/bin/pip install -e /path/to/NerveSegmentator
```

No weights are published yet, and SCT only installs from `http(s)` URLs, so serve zips made
by `nerveseg_weights export` from your own machine:

```bash
nerveseg_weights export /path/to/Dataset311_DRG  -m drg_t2  --release r20261112 -o release
nerveseg_weights export /path/to/Dataset312_root -m root_t2 --release r20261112 -o release
(cd release && python -m http.server 8765 --bind 127.0.0.1) &
sct_deepseg nerves_lumbar_t2 -install -custom-url \
    http://127.0.0.1:8765/drg_t2_r20261112.zip http://127.0.0.1:8765/root_t2_r20261112.zip
```

On an Apple M2 Ultra (SCT runs on CPU), one fold takes about 40 s per structure: both
structures take about 6.5 minutes with the default five folds and 1.5 with `-single-fold`.
Segmenting the spine first adds about 3 minutes. The output is the same, voxel for voxel, as
that of the standalone command with the same folds and no mirroring.

## Before a pull request to SCT

- **Package**: publish `nervesegmentator` on PyPI and add it to SCT's `requirements.txt`, if
  SCT is to install it by default (the name was free on PyPI on 2026-11-12).
- **Weights**: publish them and replace the two placeholder URLs (`rYYYYMMDD`) in
  `models.py`. SCT reinstalls the models whenever these URLs change.
- **Quality control**: `-qc` runs, but SCT's generic report shows small axial tiles centred
  on the ganglia, in two colours, without the levels. A report that suits these structures
  (coronal, coloured by level) is still to be written.
- **Documentation**: a figure for the task gallery in `sct_deepseg.rst` (the figures live in
  SCT's `doc-figures` repository), and the description of the training data in `models.py`.
- **Tests**: an end-to-end test needs a lumbar T2w image in `sct_testing_data`.
- **Names**: the task `nerves_lumbar_t2` and the model `model_seg_nerves_lumbar_t2` follow
  SCT's `structure_region_contrast` pattern; they are a proposal.

## Seen in SCT while testing

Not caused by the patch:

- **The nnU-Net fork predicts the first fold twice.** In `nnunetv2-neuropoly` 2.6.2,
  `nnUNetPredictor.predict_logits_from_preprocessed_data` runs the first fold, then runs it
  again and adds it (an `if` where stock nnU-Net has `if`/`else`). A one-fold model takes
  two forward passes and returns doubled logits; a five-fold ensemble counts fold 0 twice.
  SCT's own nnU-Net tasks call the same function, except `lesion_ms`, which asks for the
  folds separately. NerveSegmentator averages the folds itself (`predict.py`), so it is not
  affected; before that, its five-fold result under SCT differed from the standalone one.
- **`sct_deepseg spine` fails on uncompressed `.nii` input**: it looks for a `.nii` output
  where TotalSpineSeg wrote `.nii.gz`. Give this task `.nii.gz` images, or a `-spine-seg`.
- Runs end with `Exception ignored in: <function Timer.__del__ ...>`, which can be ignored.

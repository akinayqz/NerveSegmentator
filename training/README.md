# Training

What is needed to retrain the models NerveSegmentator runs. Nothing here is used at prediction.

- `nnUNetTrainer_LowResSpine.py`: the nnU-Net trainer (see its docstring for the augmentation
  and how to install it into nnU-Net).
- `make_level_channels.py`: the original script for the coordinate channels. `nerveseg_coords`
  has replaced it (same arguments, identical output) and is the version that prediction uses.

## Data

One case is three channels on one grid, in the orientation of `ModelSpec.orientation` in
`nervesegmentator/registry.py` (LSP), plus the label map:

| nnU-Net file | Channel | From |
|---|---|---|
| `<case>_0000.nii.gz` | image (T2w) | |
| `<case>_0001.nii.gz` | coordLR | `<prefix>_coordLR.nii.gz` |
| `<case>_0002.nii.gz` | coordCC | `<prefix>_coordCC.nii.gz` |

```bash
nerveseg_coords --seg <TotalSpineSeg step1_output or step2_output> --ref <case>_0000.nii.gz
```

In `dataset.json` the image channel is normalised and the coordinate channels are `noNorm`,
and the labels are named by level (`S1`, `L2`, ...): the names, not the values, are what
NerveSegmentator maps to its output labels.

## From a trained model to the tool

```bash
nerveseg_weights import $nnUNet_results/Dataset311_DRG -m drg_t2      # try it locally
nerveseg_weights export $nnUNet_results/Dataset311_DRG -m drg_t2 --release r20261112
```

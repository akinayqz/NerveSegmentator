import json

import nibabel as nib
import pytest

from conftest import phantom
from nervesegmentator import cli


def test_coords_cli(tmp_path, capsys):
    image, seg = phantom()
    nib.save(image, tmp_path / "sub-01_T2w_1mm.nii.gz")
    nib.save(seg, tmp_path / "seg.nii.gz")
    cli.coords_main(["--seg", str(tmp_path / "seg.nii.gz"), "--ref", str(tmp_path / "sub-01_T2w_1mm.nii.gz")])
    assert "OK [sub-01_T2w] 5 discs: L5-S1, L4-L5, L3-L4, L2-L3, L1-L2" in capsys.readouterr().out
    for suffix in ("coordCC.nii.gz", "coordLR.nii.gz", "coordAnchors.json"):       # prefix: name up to _T2W
        assert (tmp_path / f"sub-01_T2w_{suffix}").is_file()
    assert json.loads((tmp_path / "sub-01_T2w_coordAnchors.json").read_text())["status"] == "ok"


def test_coords_cli_failure_writes_only_the_report(tmp_path, capsys):
    image, _ = phantom()
    nib.save(image, tmp_path / "image.nii.gz")
    with pytest.raises(SystemExit) as e:        # an image is not a spine segmentation
        cli.coords_main(["--seg", str(tmp_path / "image.nii.gz"), "--ref", str(tmp_path / "image.nii.gz")])
    assert e.value.code == 1 and "ERROR [image]" in capsys.readouterr().err
    assert sorted(p.name for p in tmp_path.iterdir()) == ["image.nii.gz", "image_coordAnchors.json"]


def test_weights_list_with_nothing_installed(tmp_path, capsys):
    cli.weights_main(["--data-dir", str(tmp_path), "list"])
    out = capsys.readouterr().out
    assert "drg_t2" in out and "root_t2" in out and "not installed" in out


def test_missing_weights_is_a_clean_error(tmp_path, capsys):
    pytest.importorskip("torch")
    image, seg = phantom()
    nib.save(image, tmp_path / "image.nii.gz")
    nib.save(seg, tmp_path / "seg.nii.gz")
    with pytest.raises(SystemExit) as e:
        cli.main(["-i", str(tmp_path / "image.nii.gz"), "-o", str(tmp_path / "out"), "--device", "cpu",
                  "--spine-seg", str(tmp_path / "seg.nii.gz"), "--data-dir", str(tmp_path / "empty")])
    assert e.value.code == 2
    assert "not installed" in capsys.readouterr().err and not (tmp_path / "out").exists()

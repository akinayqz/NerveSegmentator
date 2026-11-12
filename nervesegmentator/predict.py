"""nnU-Net prediction for one installed model.

Everything that is specific to how the networks were trained lives here: the channel
order, the orientation of the training images, nnU-Net's array convention and the
model's own label values. Callers pass channels on the grid of the input image and get
a segmentation on that grid in the output label values.
"""
import contextlib
import os

import numpy as np

from . import io, labels
from .errors import InputError
from .weights import CHECKPOINT


def resolve_device(device="auto"):
    """'auto' (CUDA if available, else CPU), 'cpu', 'cuda', 'cuda:1', 'mps' or a torch.device."""
    import torch
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise InputError("device 'cuda' was requested but CUDA is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise InputError("device 'mps' was requested but MPS is not available")
    return device


def _nnunet_predictor_class():
    """nnU-Net's predictor, with the loop over the folds written out here.

    The loop is the one of stock nnU-Net: the mean of the folds' logits. It is repeated
    because the nnunetv2-neuropoly fork (2.6.2, the build Spinal Cord Toolbox installs)
    predicts the first fold twice and adds it twice, which costs a forward pass and
    changes the ensemble."""
    # nnU-Net warns at import when these are unset; prediction does not use them
    for var in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        os.environ.setdefault(var, "")
    import torch
    from nnunetv2.configuration import default_num_processes
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    class FoldMeanPredictor(nnUNetPredictor):
        @torch.inference_mode()
        def predict_logits_from_preprocessed_data(self, data, return_logits_per_fold=False):
            if return_logits_per_fold:              # an option of the fork; not used here
                return super().predict_logits_from_preprocessed_data(data, return_logits_per_fold=True)
            n_threads = torch.get_num_threads()
            torch.set_num_threads(min(default_num_processes, n_threads))
            try:
                network = getattr(self.network, "_orig_mod", self.network)      # compiled or not
                prediction = None
                for params in self.list_of_parameters:
                    network.load_state_dict(params)
                    logits = self.predict_sliding_window_return_logits(data).to("cpu")
                    if prediction is None:
                        prediction = logits
                    else:
                        prediction += logits
                if len(self.list_of_parameters) > 1:
                    prediction /= len(self.list_of_parameters)
                return prediction
            finally:
                torch.set_num_threads(n_threads)

    return FoldMeanPredictor


class Predictor:
    def __init__(self, model, device="auto", folds=None, tta=True, quiet=False):
        """
        model   weights.InstalledModel
        folds   folds to ensemble (default: all the model has)
        tta     test-time mirroring along the axes the model was trained to be mirrored on
        """
        self.model = model
        self.device = resolve_device(device)
        self.folds = tuple(model.folds if folds is None else folds)
        if not self.folds or set(self.folds) - set(model.folds):
            raise InputError(f"model '{model.id}' has folds {list(model.folds)}; asked for {list(self.folds)}")
        self.tta = tta
        labels.check(model.labels)

        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):   # it prints a notice off CUDA
            self._nnunet = _nnunet_predictor_class()(
                tile_step_size=0.5, use_gaussian=True, use_mirroring=tta,
                perform_everything_on_device=self.device.type == "cuda", device=self.device,
                verbose=False, verbose_preprocessing=False, allow_tqdm=not quiet)
        self._nnunet.initialize_from_trained_model_folder(
            str(model.nnunet_dir), use_folds=self.folds, checkpoint_name=CHECKPOINT)

    def predict(self, channels, affine):
        """
        channels   dict: channel name -> 3D array on the grid of the input image
        affine     voxel-to-world affine of that grid
        Returns the segmentation on the same grid, uint8, in the output label values.
        """
        missing = [c for c in self.model.channels if c not in channels]
        if missing:
            raise InputError(f"model '{self.model.id}' needs channels {missing}")
        ornt = io.orientation_to(affine, self.model.orientation)
        zooms = io.reorient_zooms(np.linalg.norm(np.asarray(affine)[:3, :3], axis=0), ornt)
        data = np.stack([io.reorient(np.asarray(channels[c], dtype=np.float32), ornt)
                         for c in self.model.channels])
        # nnU-Net trains on arrays as SimpleITK reads them: axes (z, y, x), and so the spacing
        seg = self._nnunet.predict_single_npy_array(
            np.ascontiguousarray(data.transpose(0, 3, 2, 1)), {"spacing": list(zooms[::-1])},
            None, None, False)
        seg = io.reorient(np.asarray(seg).transpose(2, 1, 0), io.invert_orientation(ornt))
        return labels.to_output(seg, self.model.labels)

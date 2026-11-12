import pytest

from nervesegmentator import predict

torch = pytest.importorskip("torch")
pytest.importorskip("nnunetv2")


def test_each_fold_is_run_once_and_the_logits_averaged():
    """Not the case in the loop of the nnunetv2-neuropoly fork (2.6.2), which Spinal Cord
    Toolbox installs: it runs the first fold twice and adds it twice."""
    cls = predict._nnunet_predictor_class()
    nnunet = cls.__new__(cls)                   # no weights: only the loop over the folds is tested
    nnunet.network = torch.nn.Linear(1, 1, bias=False)
    nnunet.list_of_parameters = [{"weight": torch.tensor([[w]])} for w in (1.0, 2.0, 6.0)]
    run = []

    @torch.inference_mode()                     # as nnU-Net's is
    def sliding_window(data):
        run.append(nnunet.network.weight.item())
        return nnunet.network.weight * data

    nnunet.predict_sliding_window_return_logits = sliding_window
    logits = nnunet.predict_logits_from_preprocessed_data(torch.ones(2, 3))
    assert run == [1.0, 2.0, 6.0]
    assert torch.equal(logits, torch.full((2, 3), 3.0))

    nnunet.list_of_parameters = nnunet.list_of_parameters[:1]
    assert torch.equal(nnunet.predict_logits_from_preprocessed_data(torch.ones(2, 3)), torch.ones(2, 3))
